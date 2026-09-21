//! DL8.5, the optimal search over binary features.

use dtrees_rs::algorithms::common::errors::{ErrorWrapper, NativeError};
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy,
};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, DL85};
use dtrees_rs::algorithms::optimal::rules::{Reason, Rule};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dtrees::data;
use crate::dtrees::errors::{raise_stored, ErrorSlot, PythonError};
use crate::dtrees::options;
use crate::dtrees::output;
use crate::dtrees::rules::{DiscrepancySpec, GainSpec, PuritySpec, RestartSpec, TopKSpec};

type Search = DL85<Trie, ErrorMinimizer<dyn ErrorWrapper>, dyn ErrorWrapper, dyn Heuristic>;

/// The DL8.5 search, as `pytrees.DL85Classifier` and `pytrees.DL85Cluster`
/// drive it. Options arrive as strings and rules as plain Python objects;
/// both are documented on the Python classes.
#[pyclass(module = "pytrees._native.dtrees")]
pub struct RawDL85 {
    learner: Search,
    /// The first exception the caller's error function raised, if any.
    failure: ErrorSlot,
    fitted: bool,
}

/// Wraps the caller's error function, or uses the built-in one.
fn wrap_error_function(
    py: Python<'_>,
    function: &Option<Py<PyAny>>,
    failure: &ErrorSlot,
) -> Box<dyn ErrorWrapper> {
    match function {
        Some(function) => Box::new(PythonError::new(function.clone_ref(py), failure.clone())),
        None => Box::<NativeError>::default(),
    }
}

/// A failure inside the search is a broken invariant, not bad input.
fn search_failed(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("DL8.5 failed: {err}"))
}

impl RawDL85 {
    fn fitted(&self) -> PyResult<&Search> {
        if self.fitted {
            Ok(&self.learner)
        } else {
            Err(PyRuntimeError::new_err(
                "this search has not been fitted yet",
            ))
        }
    }
}

#[pymethods]
impl RawDL85 {
    #[new]
    #[pyo3(signature = (
        min_sup=1,
        max_depth=2,
        max_error=None,
        time_limit=600.0,
        always_sort=true,
        heuristic="none",
        fast_d2=true,
        similarity_lb=true,
        dynamic_branching=true,
        error_function_input="class_counts",
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        min_sup: usize,
        max_depth: usize,
        max_error: Option<f64>,
        time_limit: f64,
        always_sort: bool,
        heuristic: &str,
        fast_d2: bool,
        similarity_lb: bool,
        dynamic_branching: bool,
        error_function_input: &str,
        discrepancy: Option<DiscrepancySpec>,
        gain: Option<GainSpec>,
        topk: Option<TopKSpec>,
        restart: Option<RestartSpec>,
        purity: Option<PuritySpec>,
        error_function: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let heuristic = options::heuristic(heuristic)?;
        let failure = ErrorSlot::default();
        let data_type = options::error_function_input(error_function_input)?;

        let mut node_rules: Vec<Box<dyn Rule>> = vec![];
        let mut search_rules: Vec<Box<dyn Rule>> = vec![];
        if let Some(rule) = discrepancy {
            search_rules.push(Box::new(rule.build()?));
        }
        if let Some(rule) = gain {
            search_rules.push(Box::new(rule.build()?));
        }
        if let Some(rule) = topk {
            search_rules.push(Box::new(rule.build()?));
        }
        if let Some(rule) = restart {
            search_rules.push(Box::new(rule.build()));
        }
        if let Some(rule) = purity {
            node_rules.push(Box::new(rule.build()));
        }

        let learner = DL85Builder::default()
            .max_depth(max_depth)
            .min_support(min_sup)
            .max_error(max_error.unwrap_or(f64::INFINITY))
            .max_time(time_limit)
            .specialization(if fast_d2 {
                OptimalDepth2Policy::Enabled
            } else {
                OptimalDepth2Policy::Disabled
            })
            .always_sort(always_sort)
            .branching_strategy(if dynamic_branching {
                BranchingPolicy::Dynamic
            } else {
                BranchingPolicy::Default
            })
            .lower_bound_strategy(if similarity_lb {
                LowerBoundPolicy::Similarity
            } else {
                LowerBoundPolicy::Disabled
            })
            .node_exposed_data(data_type)
            .cache(Box::<Trie>::default())
            .heuristic(heuristic)
            .depth2_search(Box::new(ErrorMinimizer::new(wrap_error_function(
                py,
                &error_function,
                &failure,
            ))))
            .add_search_rules(search_rules)
            .add_node_rules(node_rules)
            .error_function(wrap_error_function(py, &error_function, &failure))
            .build()
            .map_err(|err| PyValueError::new_err(format!("invalid DL8.5 configuration: {err}")))?;

        Ok(Self {
            learner,
            failure,
            fitted: false,
        })
    }

    /// Runs the search to completion, or until the time limit. `y` holds
    /// labels encoded as `0..k-1`; clustering passes none.
    #[pyo3(signature = (x, y=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<()> {
        let mut cover = data::cover(&x, y.as_ref())?;
        // The search can run for minutes; holding the GIL through it would
        // block every other thread in the process. A Python error function
        // takes the GIL back for each call.
        let learner = &mut self.learner;
        let outcome = py.detach(|| learner.fit(&mut cover));
        raise_stored(&self.failure)?;
        outcome.map_err(search_failed)?;
        self.fitted = true;
        Ok(())
    }

    /// Runs the search one pass at a time, as `fit` does, and calls
    /// `callback(error, seconds, status)` whenever a pass improves the tree.
    ///
    /// Passes only differ when rules bound them: each pass relaxes the rules
    /// until one runs unbounded. `status` is `"budget_exhausted"` while a
    /// rule still bounds the search, then `"optimal"`, or `"time_limit"`.
    fn fit_anytime(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Option<PyReadonlyArray1<'_, i64>>,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let mut cover = data::cover(&x, y.as_ref())?;
        let mut best = f64::INFINITY;
        loop {
            let learner = &mut self.learner;
            let result = py.detach(|| learner.partial_fit(&mut cover));
            raise_stored(&self.failure)?;
            let status = if self.learner.time_is_exhausted() {
                "time_limit"
            } else if result.reason == Reason::RuleReason {
                "budget_exhausted"
            } else {
                "optimal"
            };
            let error = self.learner.error();
            if error < best {
                best = error;
                callback.call1(py, (error, self.learner.elapsed_seconds(), status))?;
            }
            if status != "budget_exhausted" {
                break;
            }
        }
        self.fitted = true;
        Ok(())
    }

    /// The fitted tree as flat arrays; see `output::tree_arrays`.
    fn tree_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        output::tree_arrays(py, self.fitted()?.tree())
    }

    #[getter]
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        output::statistics(py, self.fitted()?.statistics())
    }

    /// The error of the fitted tree, as the error function measures it.
    #[getter]
    fn error(&self) -> PyResult<f64> {
        Ok(self.fitted()?.error())
    }

    /// `"optimal"` if the search ran to completion, `"time_limit"` if it
    /// stopped at `time_limit` with the best tree found so far.
    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(if self.fitted()?.time_is_exhausted() {
            "time_limit"
        } else {
            "optimal"
        })
    }
}
