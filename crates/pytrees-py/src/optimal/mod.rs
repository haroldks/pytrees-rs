use crate::common::create_cover_from_numpy;
use crate::common::errors::{raise_stored, ErrorSlot, PythonError};
use crate::common::options;
use crate::common::rules::{DiscrepancySpec, GainSpec, PuritySpec, RestartSpec, TopKSpec};
use crate::common::types::SearchOutput;
use dtrees_rs::algorithms::common::errors::{ErrorWrapper, NativeError};
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy,
};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::config::DL85Config;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, DL85};
use dtrees_rs::algorithms::optimal::rules::{Reason, Rule};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The DL8.5 search, as `pytrees.DL85Classifier` and `pytrees.DL85Cluster`
/// drive it. Options arrive as strings and rules as plain Python objects;
/// both are documented on the Python classes.
#[pyclass]
pub struct PyDL85 {
    learner: DL85<Trie, ErrorMinimizer<dyn ErrorWrapper>, dyn ErrorWrapper, dyn Heuristic>,
    config: DL85Config,
    statistics: SearchOutput,
    /// The first exception the caller's error function raised, if any.
    failure: ErrorSlot,
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

#[pymethods]
impl PyDL85 {
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
            .map_err(|e| PyValueError::new_err(format!("invalid DL8.5 configuration: {e:?}")))?;

        let config = learner.config();
        Ok(Self {
            learner,
            config,
            statistics: SearchOutput::default(),
            failure,
        })
    }

    /// Runs the search to completion, or until the time limit.
    #[pyo3(signature = (input, target=None))]
    fn fit(
        &mut self,
        input: PyReadonlyArrayDyn<f64>,
        target: Option<PyReadonlyArrayDyn<f64>>,
    ) -> PyResult<()> {
        let mut cover = create_cover_from_numpy(input, target.as_ref())?;
        let outcome = self.learner.fit(&mut cover);
        raise_stored(&self.failure)?;
        outcome.map_err(|err| PyValueError::new_err(format!("DL8.5 failed: {err:?}")))?;
        self.update_stats();
        Ok(())
    }

    /// Runs the search one pass at a time, as `fit` does, and calls
    /// `callback(error, seconds, status)` whenever a pass improves the tree.
    ///
    /// Passes only differ when rules bound them: each pass relaxes the rules
    /// until one runs unbounded. `status` is `"budget_exhausted"` while a
    /// rule still bounds the search, then `"optimal"`, or `"time_limit"`.
    #[pyo3(signature = (input, target, callback))]
    fn fit_anytime(
        &mut self,
        py: Python<'_>,
        input: PyReadonlyArrayDyn<f64>,
        target: Option<PyReadonlyArrayDyn<f64>>,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let mut cover = create_cover_from_numpy(input, target.as_ref())?;
        let mut best = f64::INFINITY;
        loop {
            let result = self.learner.partial_fit(&mut cover);
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
        self.update_stats();
        Ok(())
    }

    /// Returns comprehensive search statistics and results.
    ///
    /// This property provides access to detailed information about the search
    /// process, including the optimal tree, error metrics, and performance statistics.
    ///
    /// # Returns
    ///
    /// A `SearchOutput` object containing:
    /// - `error`: Optimal classification error achieved
    /// - `tree`: JSON representation of the current or optimal decision tree
    /// - `statistics`: Detailed search statistics (nodes explored, cache hits, etc.)
    /// - `duration`: Total search time in seconds
    ///
    /// # Example
    ///
    /// ```python
    /// classifier.fit(X_train, y_train)
    /// stats = classifier.stats
    ///
    /// print(f"Optimal error: {stats.error}")
    /// print(f"Search time: {stats.duration}s")
    /// print(f"Tree: {stats.tree}")
    /// print(f"Statistics: {stats.statistics}")
    /// ```
    #[getter]
    pub fn stats(&self) -> PyResult<SearchOutput> {
        Ok(self.statistics.clone())
    }

    /// Returns the algorithm configuration as a JSON string.
    ///
    /// This property provides access to the configuration used
    /// for the DL8.5 algorithm, useful for reproducibility and debugging.
    ///
    /// # Returns
    ///
    /// A JSON string containing all configuration parameters.
    ///
    /// # Example
    ///
    /// ```python
    /// classifier = PyDL85(max_depth=3, min_sup=5)
    /// config = classifier.config
    /// print(config)
    /// ```
    #[getter]
    pub fn config(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.config).unwrap();
        Ok(json)
    }

    /// `"optimal"` if the search ran to completion, `"time_limit"` if it
    /// stopped at `time_limit` with the best tree found so far.
    #[getter]
    fn status(&self) -> &'static str {
        if self.learner.time_is_exhausted() {
            "time_limit"
        } else {
            "optimal"
        }
    }

    /// Updates internal statistics from the current algorithm state.
    ///
    /// This method synchronizes the Python-accessible statistics with the
    /// current state of the underlying Rust algorithm. Called automatically
    /// after fitting operations.
    fn update_stats(&mut self) {
        self.statistics.error = self.learner.error();
        self.statistics.duration = self.learner.elapsed_seconds();
        self.statistics.statistics = *self.learner.statistics();
        self.statistics.tree = self.learner.tree().clone();
    }
}
