//! Python bindings for `contree`.
//!
//! Converts numpy arrays to a `Dataset`, runs the search with the GIL
//! released, and returns the tree as arrays. The scikit-learn interface is in
//! `pytrees.supervised.contree`.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use contree::algorithms::{validate, ConTree, ConTreeLds};
use contree::common::{
    Cost, Deepening, Leaf, LeafLoss, Loss, LossInput, PointSelector, ScheduleKind, SearchError,
    SearchStatus, Skipping, Statistics,
};
use contree::data::view::DataView;
use contree::data::{Dataset, DatasetError};
use contree::tree::{Tree, TreeError};

use crate::dtrees::errors::{lock, raise_stored, ErrorSlot};

/// Bad input becomes a `ValueError`; a broken invariant inside the search, a
/// bug in the library, becomes a `RuntimeError`.
fn search_err(err: SearchError) -> PyErr {
    match err {
        SearchError::Tree(_) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

fn dataset_err(err: DatasetError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn tree_err(err: TreeError) -> PyErr {
    match err {
        TreeError::FeatureOutOfRange { .. } | TreeError::RaggedInput { .. } => {
            PyValueError::new_err(err.to_string())
        }
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

fn point_selector(name: &str) -> PyResult<PointSelector> {
    name.parse().map_err(PyValueError::new_err)
}

fn loss_input(name: &str) -> PyResult<LossInput> {
    match name {
        "class_counts" => Ok(LossInput::ClassCounts),
        "indices" => Ok(LossInput::Indices),
        _ => Err(PyValueError::new_err(format!(
            "unknown error_function_input {name:?}; expected class_counts or indices"
        ))),
    }
}

fn skipping(name: &str) -> PyResult<Skipping> {
    match name {
        "auto" => Ok(Skipping::Auto),
        "on" => Ok(Skipping::On),
        "off" => Ok(Skipping::Off),
        _ => Err(PyValueError::new_err(format!(
            "unknown threshold_skipping {name:?}; expected auto, on or off"
        ))),
    }
}

/// A caller's Python `error_function`, seen by the search as a `LeafLoss`.
///
/// It receives a list: the number of rows of each class in the leaf, or the
/// leaf's row indices. It returns `(cost, prediction)`, the prediction being
/// the index of a class. The first exception it raises, or the first
/// prediction that is not a class index, is kept in `failure` and raised once
/// the search returns.
struct PythonLoss {
    function: Py<PyAny>,
    n_classes: usize,
    failure: ErrorSlot,
}

/// Returned in place of a real cost once the function has failed: it makes
/// every leaf useless, so the search winds down quickly.
const FAILED: (Cost, usize) = (Cost::INFINITY, 0);

impl LeafLoss for PythonLoss {
    fn cost(&self, leaf: Leaf<'_>) -> (Cost, usize) {
        if lock(&self.failure).is_some() {
            return FAILED;
        }
        let result = Python::attach(|py| {
            let value = match leaf {
                Leaf::ClassCounts(counts) => self.function.call1(py, (counts.to_vec(),)),
                Leaf::Indices(rows) => self.function.call1(py, (rows.to_vec(),)),
            }?;
            let (cost, prediction) = value.extract::<(f64, f64)>(py)?;
            if prediction.fract() != 0.0 || prediction < 0.0 || prediction >= self.n_classes as f64
            {
                return Err(PyValueError::new_err(format!(
                    "error_function predicted {prediction}, which is not the index of one of \
                     the {} classes",
                    self.n_classes
                )));
            }
            Ok((cost, prediction as usize))
        });
        result.unwrap_or_else(|err| {
            *lock(&self.failure) = Some(err);
            FAILED
        })
    }
}

fn status_name(status: SearchStatus) -> &'static str {
    match status {
        SearchStatus::Optimal => "optimal",
        SearchStatus::TimeLimit => "time_limit",
        SearchStatus::BudgetExhausted => "budget_exhausted",
        SearchStatus::ErrorBoundReached => "error_bound_reached",
        SearchStatus::NotProven => "not_proven",
    }
}

/// The constructor arguments, kept as given.
#[derive(Clone, Copy)]
struct Params {
    min_sup: usize,
    max_depth: usize,
    max_time: f64,
    max_error: f64,
    max_gap: f64,
    point_selector: PointSelector,
    sort_by_heuristic: bool,
    fast_d2: bool,
    use_lds: bool,
    /// Seeds the split-point generator so `split_selection="random"` is
    /// reproducible. `None` draws from the OS.
    random_state: Option<u64>,
    /// The anytime search's budget schedule; ignored unless `use_lds`.
    /// `None` uses the one suited to the split selector.
    budget_schedule: Option<ScheduleKind>,
    error_function_input: LossInput,
    threshold_skipping: Skipping,
    /// SPLIT: levels searched before the greedy completion; `None` without.
    lookahead: Option<usize>,
    /// The lookahead goes one level deeper between passes.
    anytime_lookahead: bool,
    /// With the LDS budgets, the deepening is spread over the schedule.
    lookahead_sync: bool,
    /// The interval pruner runs in passes that complete nodes greedily.
    lookahead_pruning: bool,
}

impl Params {
    /// Whether the anytime search applies its LDS budgets. `anytime` is set
    /// by `fit_anytime`, which is LDS unless a lookahead is given.
    fn lds(&self, anytime: bool) -> bool {
        self.use_lds || (anytime && self.lookahead.is_none())
    }

    /// How the lookahead deepens, as `con-tree` maps its flags.
    fn deepening(&self, lds: bool) -> Deepening {
        match (self.anytime_lookahead, lds && self.lookahead_sync) {
            (false, _) => Deepening::Fixed,
            (true, true) => Deepening::Synced,
            (true, false) => Deepening::EveryPass,
        }
    }

    /// The anytime solver for these settings.
    fn anytime_solver(&self, loss: Loss, lds: bool) -> ConTreeLds {
        let mut solver = ConTreeLds::new(
            self.min_sup,
            self.max_depth,
            self.max_time,
            self.max_error,
            self.point_selector,
            self.max_gap,
            self.sort_by_heuristic,
            self.fast_d2,
        )
        .with_loss(loss)
        .with_skipping(self.threshold_skipping)
        .with_lds(lds);
        if let Some(schedule) = self.budget_schedule {
            solver = solver.with_schedule(schedule);
        }
        if let Some(lookahead) = self.lookahead {
            solver = solver
                .with_lookahead(lookahead, self.deepening(lds))
                .with_lookahead_pruning(self.lookahead_pruning);
        }
        if let Some(seed) = self.random_state {
            solver = solver.with_random_state(seed);
        }
        solver
    }
}

struct Fitted {
    tree: Tree,
    statistics: Statistics,
    status: SearchStatus,
}

#[pyclass(module = "pytrees._native.contree")]
pub struct RawConTree {
    params: Params,
    /// The caller's `error_function`; `None` for misclassification.
    error_function: Option<Py<PyAny>>,
    /// The first exception the caller's error function raised, if any.
    failure: ErrorSlot,
    fitted: Option<Fitted>,
}

impl RawConTree {
    fn fitted(&self) -> PyResult<&Fitted> {
        self.fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("this estimator has not been fitted yet"))
    }

    /// The loss of a fit on `n_classes` classes.
    fn loss(&self, py: Python<'_>, n_classes: usize) -> Loss {
        match &self.error_function {
            None => Loss::Misclassification,
            Some(function) => Loss::custom(
                Box::new(PythonLoss {
                    function: function.clone_ref(py),
                    n_classes,
                    failure: self.failure.clone(),
                }),
                self.params.error_function_input,
            ),
        }
    }

    fn dataset(x: &PyReadonlyArray2<'_, f64>, y: &PyReadonlyArray1<'_, i64>) -> PyResult<Dataset> {
        let shape = x.shape();
        let (n_rows, n_features) = (shape[0], shape[1]);
        if y.len() != n_rows {
            return Err(PyValueError::new_err(format!(
                "X has {n_rows} rows but y has {} labels",
                y.len()
            )));
        }

        let values = x
            .as_slice()
            .map_err(|_| PyValueError::new_err("X must be a contiguous C-order float64 array"))?;
        let raw = y
            .as_slice()
            .map_err(|_| PyValueError::new_err("y must be a contiguous int64 array"))?;

        let mut labels = Vec::with_capacity(raw.len());
        for (row, &label) in raw.iter().enumerate() {
            if label < 0 {
                return Err(PyValueError::new_err(format!(
                    "row {row}: label {label} is negative; encode labels as 0..n_classes"
                )));
            }
            labels.push(label as usize);
        }

        Dataset::from_rows(values, &labels, n_features).map_err(dataset_err)
    }
}

#[pymethods]
impl RawConTree {
    #[new]
    #[pyo3(signature = (
        min_sup = 1,
        max_depth = 3,
        max_time = 600.0,
        max_error = None,
        max_gap = 0.0,
        split_selection = "mid",
        sort_by_heuristic = false,
        fast_d2 = true,
        use_lds = false,
        random_state = None,
        budget_schedule = None,
        error_function = None,
        error_function_input = "class_counts",
        threshold_skipping = "auto",
        lookahead = None,
        anytime_lookahead = false,
        lookahead_sync = false,
        lookahead_pruning = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: Option<f64>,
        max_gap: f64,
        split_selection: &str,
        sort_by_heuristic: bool,
        fast_d2: bool,
        use_lds: bool,
        random_state: Option<u64>,
        budget_schedule: Option<&str>,
        error_function: Option<Py<PyAny>>,
        error_function_input: &str,
        threshold_skipping: &str,
        lookahead: Option<usize>,
        anytime_lookahead: bool,
        lookahead_sync: bool,
        lookahead_pruning: bool,
    ) -> PyResult<Self> {
        if anytime_lookahead && lookahead.is_none() {
            return Err(PyValueError::new_err(
                "anytime_lookahead=True needs a lookahead: it deepens the lookahead between passes",
            ));
        }
        Ok(Self {
            params: Params {
                min_sup,
                max_depth,
                max_time,
                max_error: max_error.unwrap_or(f64::INFINITY),
                max_gap,
                point_selector: point_selector(split_selection)?,
                sort_by_heuristic,
                fast_d2,
                use_lds,
                random_state,
                budget_schedule: budget_schedule
                    .map(str::parse)
                    .transpose()
                    .map_err(PyValueError::new_err)?,
                error_function_input: loss_input(error_function_input)?,
                threshold_skipping: skipping(threshold_skipping)?,
                lookahead,
                anytime_lookahead,
                lookahead_sync,
                lookahead_pruning,
            },
            error_function,
            failure: ErrorSlot::default(),
            fitted: None,
        })
    }

    /// Why the chosen budget schedule does not suit the split selection, if
    /// it does not, when the LDS budgets apply; see `ScheduleKind::mismatch`.
    /// `anytime` as in `fit_anytime`.
    #[pyo3(signature = (anytime = false))]
    fn schedule_warning(&self, anytime: bool) -> Option<String> {
        if !self.params.lds(anytime) {
            return None;
        }
        self.params
            .budget_schedule?
            .mismatch(self.params.point_selector)
    }

    /// Fits the tree. `y` must already be encoded as `0..n_classes`.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let dataset = Self::dataset(&x, &y)?;
        let params = self.params;
        let loss = self.loss(py, dataset.num_labels());

        // Release the GIL: the search can run for minutes.
        let outcome = py.detach(|| {
            if params.use_lds || params.lookahead.is_some() {
                params.anytime_solver(loss, params.lds(false)).fit(&dataset)
            } else {
                let mut solver = ConTree::new(
                    params.min_sup,
                    params.max_depth,
                    params.max_time,
                    params.max_error,
                    params.point_selector,
                    params.max_gap,
                    params.sort_by_heuristic,
                    params.fast_d2,
                )
                .with_loss(loss)
                .with_skipping(params.threshold_skipping);
                if let Some(seed) = params.random_state {
                    solver = solver.with_random_state(seed);
                }
                solver.fit(&dataset)
            }
        });

        raise_stored(&self.failure)?;
        let outcome = outcome.map_err(search_err)?;
        self.fitted = Some(Fitted {
            tree: outcome.tree,
            statistics: outcome.statistics,
            status: outcome.status,
        });
        Ok(())
    }

    /// Runs the anytime search, reporting each improvement as it happens.
    ///
    /// `callback(error, seconds, status)` is invoked whenever a pass finds a
    /// better tree.
    #[pyo3(signature = (x, y, callback = None))]
    fn fit_anytime(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let dataset = Self::dataset(&x, &y)?;
        let params = self.params;
        let loss = self.loss(py, dataset.num_labels());

        let mut solver = params.anytime_solver(loss, params.lds(true));
        validate(solver.config(), &dataset).map_err(search_err)?;

        let view = DataView::root(&dataset, params.sort_by_heuristic);
        let mut best = f64::INFINITY;
        loop {
            let done = py.detach(|| solver.partial_fit(&view));
            raise_stored(&self.failure)?;
            if let Some(cost) = solver.invalid_cost() {
                return Err(search_err(SearchError::InvalidLeafCost { cost }));
            }
            let statistics = *solver.statistics();

            if statistics.error < best {
                best = statistics.error;
                if let Some(callback) = &callback {
                    callback.call1(
                        py,
                        (
                            statistics.error,
                            statistics.duration,
                            status_name(solver.status()),
                        ),
                    )?;
                }
            }
            if done {
                break;
            }
        }

        let tree = solver.get_solution_tree();
        tree.validate().map_err(tree_err)?;
        self.fitted = Some(Fitted {
            tree,
            statistics: *solver.statistics(),
            status: solver.status(),
        });
        Ok(())
    }

    /// The tree as flat arrays, in the shape scikit-learn's own `tree_` uses.
    ///
    /// `children_left[i] == -1` marks a leaf, whose prediction is `value[i]`.
    fn tree_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fitted = self.fitted()?;
        let nodes = fitted.tree.nodes();

        let mut children_left = Vec::with_capacity(nodes.len());
        let mut children_right = Vec::with_capacity(nodes.len());
        let mut feature = Vec::with_capacity(nodes.len());
        let mut threshold = Vec::with_capacity(nodes.len());
        let mut value = Vec::with_capacity(nodes.len());
        let mut error = Vec::with_capacity(nodes.len());

        for node in nodes {
            match node.value.feature {
                // Index 0 is the root and also means "no child", so a leaf's
                // children are reported as -1 rather than 0.
                None => {
                    children_left.push(-1i64);
                    children_right.push(-1i64);
                    feature.push(-1i64);
                    threshold.push(f64::NAN);
                }
                Some(f) => {
                    children_left.push(node.left as i64);
                    children_right.push(node.right as i64);
                    feature.push(f as i64);
                    threshold.push(node.value.split.unwrap_or(f64::NAN));
                }
            }
            value.push(node.value.label.map_or(-1i64, |label| label as i64));
            error.push(node.value.error);
        }

        let dict = PyDict::new(py);
        dict.set_item("children_left", PyArray1::from_vec(py, children_left))?;
        dict.set_item("children_right", PyArray1::from_vec(py, children_right))?;
        dict.set_item("feature", PyArray1::from_vec(py, feature))?;
        dict.set_item("threshold", PyArray1::from_vec(py, threshold))?;
        dict.set_item("value", PyArray1::from_vec(py, value))?;
        dict.set_item("error", PyArray1::from_vec(py, error))?;
        Ok(dict)
    }

    #[getter]
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.fitted()?.statistics;
        let dict = PyDict::new(py);
        dict.set_item("error", stats.error)?;
        dict.set_item("duration", stats.duration)?;
        dict.set_item("cache_size", stats.cache_size)?;
        dict.set_item("cache_hits", stats.cache_hits)?;
        dict.set_item("general_solver_calls", stats.general_solver_call)?;
        dict.set_item("specialized_solver_calls", stats.specialized_solver_call)?;
        dict.set_item("n_samples", stats.num_samples)?;
        dict.set_item("n_features", stats.num_features)?;
        dict.set_item("skipping_violations", stats.skipping_violations)?;
        Ok(dict)
    }

    /// Why the search stopped: only `"optimal"` means the tree is proven best.
    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(status_name(self.fitted()?.status))
    }

    #[getter]
    fn error(&self) -> PyResult<f64> {
        Ok(self.fitted()?.statistics.error)
    }
}

/// Fills `pytrees._native.contree`.
pub fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RawConTree>()?;
    Ok(())
}
