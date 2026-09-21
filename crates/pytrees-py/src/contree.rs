//! Python bindings for `contree`.
//!
//! This module is deliberately thin: it converts numpy arrays to a `Dataset`,
//! runs the search with the GIL released, and hands back the tree as arrays.
//! Everything that looks like scikit-learn lives in the Python layer on top.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use contree::algorithms::{validate, ConTree, ConTreeLds};
use contree::common::{PointSelector, ScheduleKind, SearchError, SearchStatus, Statistics};
use contree::data::view::DataView;
use contree::data::{Dataset, DatasetError};
use contree::tree::{Tree, TreeError};

/// Bad input is a `ValueError`; a broken invariant inside the search is a
/// `RuntimeError`, because it is our bug and not the caller's.
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

fn status_name(status: SearchStatus) -> &'static str {
    match status {
        SearchStatus::Optimal => "optimal",
        SearchStatus::TimeLimit => "time_limit",
        SearchStatus::BudgetExhausted => "budget_exhausted",
        SearchStatus::ErrorBoundReached => "error_bound_reached",
    }
}

/// Everything the constructor was given, kept verbatim so the Python layer can
/// hand it straight back to `get_params`.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct Params {
    min_sup: usize,
    max_depth: usize,
    max_time: f64,
    max_error: usize,
    max_gap: usize,
    point_selector: PointSelector,
    sort_by_heuristic: bool,
    fast_d2: bool,
    use_lds: bool,
    /// Seeds the split-point generator so `split_selection="random"` is
    /// reproducible. `None` draws from the OS.
    random_state: Option<u64>,
    /// The anytime search's budget schedule; ignored unless `use_lds`.
    budget_schedule: ScheduleKind,
}

#[derive(Serialize, Deserialize)]
struct Fitted {
    tree: Tree,
    statistics: Statistics,
    status: SearchStatus,
    n_features: usize,
}

/// What `pickle` round-trips. Held as JSON so the wire format does not depend
/// on the memory layout of anything in the core crate.
#[derive(Serialize, Deserialize)]
struct State {
    params: Params,
    fitted: Option<Fitted>,
}

#[pyclass(module = "pytrees._native.contree")]
pub struct RawConTree {
    params: Params,
    fitted: Option<Fitted>,
}

impl RawConTree {
    fn fitted(&self) -> PyResult<&Fitted> {
        self.fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("this estimator has not been fitted yet"))
    }

    fn dataset(
        x: &PyReadonlyArray2<'_, f64>,
        y: &PyReadonlyArray1<'_, i64>,
    ) -> PyResult<(Dataset, usize)> {
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

        let dataset = Dataset::from_rows(values, &labels, n_features).map_err(dataset_err)?;
        Ok((dataset, n_features))
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
        max_gap = 0,
        split_selection = "mid",
        sort_by_heuristic = false,
        fast_d2 = true,
        use_lds = false,
        random_state = None,
        budget_schedule = "diagonal",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: Option<usize>,
        max_gap: usize,
        split_selection: &str,
        sort_by_heuristic: bool,
        fast_d2: bool,
        use_lds: bool,
        random_state: Option<u64>,
        budget_schedule: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            params: Params {
                min_sup,
                max_depth,
                max_time,
                max_error: max_error.unwrap_or(usize::MAX),
                max_gap,
                point_selector: point_selector(split_selection)?,
                sort_by_heuristic,
                fast_d2,
                use_lds,
                random_state,
                budget_schedule: budget_schedule.parse().map_err(PyValueError::new_err)?,
            },
            fitted: None,
        })
    }

    /// Fits the tree. `y` must already be encoded as `0..n_classes`.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let (dataset, n_features) = Self::dataset(&x, &y)?;
        let params = self.params;

        // The search is pure Rust and can run for minutes; holding the GIL
        // through it would block every other thread in the process.
        let outcome = py.detach(|| {
            if params.use_lds {
                let mut solver = ConTreeLds::new(
                    params.min_sup,
                    params.max_depth,
                    params.max_time,
                    params.max_error,
                    params.point_selector,
                    params.max_gap,
                    params.sort_by_heuristic,
                    params.fast_d2,
                );
                solver = solver.with_schedule(params.budget_schedule);
                if let Some(seed) = params.random_state {
                    solver = solver.with_random_state(seed);
                }
                solver.fit(&dataset)
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
                );
                if let Some(seed) = params.random_state {
                    solver = solver.with_random_state(seed);
                }
                solver.fit(&dataset)
            }
        });

        let outcome = outcome.map_err(search_err)?;
        self.fitted = Some(Fitted {
            tree: outcome.tree,
            statistics: outcome.statistics,
            status: outcome.status,
            n_features,
        });
        Ok(())
    }

    /// Runs the anytime search, reporting each improvement as it happens.
    ///
    /// `callback(error, seconds, status)` is invoked whenever a pass finds a
    /// better tree. This is the crate's distinguishing feature and the reason
    /// the LDS solver exists; without it a caller can only wait for the final
    /// answer.
    #[pyo3(signature = (x, y, callback = None))]
    fn fit_anytime(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let (dataset, n_features) = Self::dataset(&x, &y)?;
        let params = self.params;

        let mut solver = ConTreeLds::new(
            params.min_sup,
            params.max_depth,
            params.max_time,
            params.max_error,
            params.point_selector,
            params.max_gap,
            params.sort_by_heuristic,
            params.fast_d2,
        );
        solver = solver.with_schedule(params.budget_schedule);
        if let Some(seed) = params.random_state {
            solver = solver.with_random_state(seed);
        }
        validate(solver.config(), &dataset).map_err(search_err)?;

        let view = DataView::root(&dataset, params.sort_by_heuristic);
        let mut best = usize::MAX;
        loop {
            let done = py.detach(|| solver.partial_fit(&view));
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
            n_features,
        });
        Ok(())
    }

    /// Classifies a batch. The whole loop runs in Rust.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self.fitted()?;
        let shape = x.shape();
        let (n_rows, n_features) = (shape[0], shape[1]);
        if n_features != fitted.n_features {
            return Err(PyValueError::new_err(format!(
                "X has {n_features} features, but this estimator was fitted with {}",
                fitted.n_features
            )));
        }

        let values = x
            .as_slice()
            .map_err(|_| PyValueError::new_err("X must be a contiguous C-order float64 array"))?;

        let predictions = py
            .detach(|| fitted.tree.predict(values, n_features))
            .map_err(tree_err)?;

        debug_assert_eq!(predictions.len(), n_rows);
        let out: Vec<i64> = predictions.into_iter().map(|label| label as i64).collect();
        Ok(PyArray1::from_vec(py, out))
    }

    /// The path each instance takes, as node indices, one row per instance.
    fn decision_path(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<usize>>> {
        let fitted = self.fitted()?;
        let n_features = x.shape()[1];
        let values = x
            .as_slice()
            .map_err(|_| PyValueError::new_err("X must be a contiguous C-order float64 array"))?;

        values
            .chunks_exact(n_features)
            .map(|row| fitted.tree.decision_path(row).map_err(tree_err))
            .collect()
    }

    /// The tree as flat arrays, in the shape scikit-learn's own `tree_` uses.
    ///
    /// `children_left[i] == -1` marks a leaf, whose prediction is `value[i]`.
    /// This beats handing back a JSON dict: a caller can walk or plot the whole
    /// tree without parsing anything.
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
            error.push(node.value.error as i64);
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

    /// The tree as JSON, for callers that want to store or ship it.
    #[getter]
    fn tree_json(&self) -> PyResult<String> {
        let fitted = self.fitted()?;
        serde_json::to_string(&fitted.tree).map_err(|err| PyRuntimeError::new_err(err.to_string()))
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
        Ok(dict)
    }

    /// Why the search stopped: only `"optimal"` means the tree is proven best.
    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(status_name(self.fitted()?.status))
    }

    #[getter]
    fn error(&self) -> PyResult<usize> {
        Ok(self.fitted()?.statistics.error)
    }

    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        Ok(self.fitted()?.n_features)
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    // --- pickle -----------------------------------------------------------
    //
    // A `#[pyclass]` is not picklable by default, and an estimator that cannot
    // be pickled cannot be cached, sent to a worker process, or saved --
    // `sklearn.utils.estimator_checks.check_estimator` rejects it outright.

    fn __getstate__(&self) -> PyResult<String> {
        let state = State {
            params: self.params,
            fitted: self.fitted.as_ref().map(|fitted| Fitted {
                tree: fitted.tree.clone(),
                statistics: fitted.statistics,
                status: fitted.status,
                n_features: fitted.n_features,
            }),
        };
        serde_json::to_string(&state).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn __setstate__(&mut self, state: &str) -> PyResult<()> {
        let state: State = serde_json::from_str(state)
            .map_err(|err| PyValueError::new_err(format!("corrupt estimator state: {err}")))?;
        self.params = state.params;
        self.fitted = state.fitted;
        Ok(())
    }

    /// Pickle calls `__new__` with these before `__setstate__` fills the rest
    /// in, so they only have to be constructible, not correct.
    #[allow(clippy::type_complexity)]
    fn __getnewargs__(
        &self,
    ) -> (
        usize,
        usize,
        f64,
        Option<usize>,
        usize,
        String,
        bool,
        bool,
        bool,
        Option<u64>,
        String,
    ) {
        (
            self.params.min_sup,
            self.params.max_depth,
            self.params.max_time,
            (self.params.max_error != usize::MAX).then_some(self.params.max_error),
            self.params.max_gap,
            self.params.point_selector.to_string(),
            self.params.sort_by_heuristic,
            self.params.fast_d2,
            self.params.use_lds,
            self.params.random_state,
            self.params.budget_schedule.to_string(),
        )
    }
}

/// Adds the `contree` submodule to `pytrees._native`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "contree")?;
    module.add_class::<RawConTree>()?;
    parent.add_submodule(&module)?;
    // Makes `from pytrees._native.contree import ...` work, and lets pickle
    // find `RawConTree` by the module name above.
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytrees._native.contree", module)?;
    Ok(())
}
