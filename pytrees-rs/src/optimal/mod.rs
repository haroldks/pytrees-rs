use crate::common::create_cover_from_numpy;
use crate::common::enums::{ExposedBranchingPolicy, ExposedDepth2Policy, ExposedHeuristic, ExposedLowerBoundPolicy, ExposedNodeDataType, };
use crate::common::errors::PythonError;
use crate::common::types::{ExposedDiscrepancyRule, ExposedGainRule, ExposedPurityRule, ExposedRestartRule, ExposedTopKRule, SearchOutput};
use dtrees_rs::algorithms::common::errors::{ErrorWrapper, NativeError};
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::config::DL85Config;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, DL85};
use dtrees_rs::algorithms::optimal::rules::{common::TimeLimitRule, DiscrepancyRule, GainRule, PurityRule, Rule, TopkRule};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use dtrees_rs::cover::Cover;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// PyDL85: Python wrapper for the DL8.5 optimal decision tree algorithm.
///
/// DL8.5 is a state-of-the-art algorithm for constructing optimal decision trees
/// that minimize classification error while respecting user-defined constraints.
/// This implementation uses modern Cover API for efficient data handling and
/// supports extensive rule-based optimization.
///
/// ## Algorithm Overview
///
/// The DL8.5 algorithm performs an exhaustive search through the space of possible
/// decision trees, using dynamic programming and caching to avoid redundant
/// computations. It guarantees finding the globally optimal tree within the
/// specified constraints.
///
/// ## Key Features
///
/// - **Optimal Solutions**: Guarantees globally optimal decision trees
/// - **Rule-based Optimization**: Support for discrepancy, gain, purity, and top-k rules
/// - **Multiple Heuristics**: Information gain, Gini index, weighted entropy for search guidance
/// - **Flexible Constraints**: Configurable depth, support, and error limits
/// - **Custom Error Functions**: Support for user-defined Python error functions
/// - **Advanced Caching**: Trie-based caching for efficient search space exploration
///
/// ## Usage Example
///
/// ```python
/// from pytreesrs.odt import PyDL85
/// from pytreesrs.enums import ExposedHeuristic
/// from pytreesrs.odt.rules import ExposedGainRule, ExposedPurityRule
///
/// # Create classifier with advanced configuration
/// classifier = PyDL85(
///     max_depth=4,
///     min_sup=10,
///     max_error=0.1,
///     time_limit=300.0,
///     heuristic=ExposedHeuristic.InformationGain,
///     gain=ExposedGainRule(min_gain=0.01, epsilon=1e-4),
///     purity=ExposedPurityRule(min_purity=0.9)
/// )
///
/// # Fit the model
/// classifier.fit(X_train, y_train)
///
/// # Access results
/// print(f"Optimal error: {classifier.stats.error}")
/// print(f"Search duration: {classifier.stats.duration}s")
/// print(f"Tree structure: {classifier.stats.tree}")
/// ```
#[pyclass]
pub struct PyDL85 {
    /// The core DL85 algorithm instance with configured components
    learner: DL85<Trie, ErrorMinimizer<dyn ErrorWrapper>, dyn ErrorWrapper, dyn Heuristic>,
    /// Algorithm configuration parameters
    config: DL85Config,
    /// Cover representing the dataset
    cover: Cover,
    /// Search statistics and results
    statistics: SearchOutput,
    /// Flag indicating whether data has been loaded
    has_data: bool,
}


#[pymethods]
impl PyDL85 {
    /// Creates a new PyDL85 instance with specified configuration.
    ///
    /// This constructor initializes the DL8.5 algorithm with comprehensive
    /// configuration options for optimal decision tree construction.
    ///
    /// # Parameters
    ///
    /// ## Basic Configuration
    /// - `min_sup`: Minimum support (number of samples) required for a split (default: 1)
    /// - `max_depth`: Maximum tree depth allowed (default: 2)
    /// - `max_error`: Maximum acceptable error for early termination (default: ∞)
    /// - `time_limit`: Maximum search time in seconds (default: 600.0)
    /// - `always_sort`: Whether to always sort features for deterministic results (default: true)
    ///
    /// ## Algorithm Configuration
    /// - `heuristic`: Search heuristic for node ordering (default: Disabled)
    /// - `depth2_policy`: Depth-2 specialization policy (default: Disabled)
    /// - `lower_bound`: Lower bound computation strategy (default: Disabled)
    /// - `branching_policy`: Branching strategy for tree construction (default: Default)
    /// - `data_type`: Node data representation type (default: ClassSupports)
    ///
    /// ## Rule-based Optimization
    /// - `discrepancy`: Limited discrepancy search rule (optional)
    /// - `gain`: Information gain-based stopping rule (optional)
    /// - `topk`: Top-k search limitation rule (optional)
    /// - `restart`: Time-based restart rule (optional)
    /// - `purity`: Node purity-based rule (optional)
    /// - `error_function`: Custom Python error function (optional)
    ///
    /// # Returns
    ///
    /// A configured PyDL85 instance ready for training.
    ///
    /// # Errors
    ///
    /// Returns `PyRuntimeError` if the algorithm configuration is invalid.
    ///
    /// # Example
    ///
    /// ```python
    /// from pytreesrs.odt import PyDL85
    /// # Basic usage
    /// classifier = PyDL85(max_depth=3, min_sup=5)
    ///
    /// # Advanced configuration with rules
    /// classifier = PyDL85(
    ///     max_depth=4,
    ///     min_sup=10,
    ///     heuristic=ExposedHeuristic.InformationGain,
    ///     gain=ExposedGainRule(min_gain=0.01),
    ///     purity=ExposedPurityRule(min_purity=0.8)
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (
        min_sup=1,
        max_depth=2,
        max_error=f64::INFINITY,
        time_limit=600.0,
        always_sort=true,
        heuristic=ExposedHeuristic::NoHeuristic,
        depth2_policy=ExposedDepth2Policy::Disabled,
        lower_bound=ExposedLowerBoundPolicy::Disabled,
        branching_policy=ExposedBranchingPolicy::Default,
        data_type=ExposedNodeDataType::ClassSupports,
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ))]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_error: f64,
        time_limit: f64,
        always_sort: bool,
        heuristic: ExposedHeuristic,
        depth2_policy: ExposedDepth2Policy,
        lower_bound: ExposedLowerBoundPolicy,
        branching_policy: ExposedBranchingPolicy,
        data_type: ExposedNodeDataType,
        // Rule parameters
        discrepancy: Option<ExposedDiscrepancyRule>,
        gain: Option<ExposedGainRule>,
        topk: Option<ExposedTopKRule>,
        restart: Option<ExposedRestartRule>,
        purity: Option<ExposedPurityRule>,
        error_function: Option<Py<PyAny>>,
    ) -> PyResult<Self> {

        let heuristic_fn: Box<dyn Heuristic> = heuristic.into();
        let depth2_policy = depth2_policy.into();
        let lower_bound_policy = lower_bound.into();
        let branching_policy = branching_policy.into();
        let data_type = data_type.into();

        let error_fn: Box<dyn ErrorWrapper> = match &error_function {
            None => Box::<NativeError>::default(),
            Some(function) => {
                let mut error: Box<dyn ErrorWrapper> = Box::<NativeError>::default();
                Python::attach(|py| {
                    error = Box::new(PythonError::new(function.clone_ref(py)))
                });
                error
            } ,
        };

        let depth2_search: Box<ErrorMinimizer<dyn ErrorWrapper>> = match &error_function {
            None => Box::new(ErrorMinimizer::new(Box::<NativeError>::default())),
            Some(function) => {
                let mut d2: Box<ErrorMinimizer<dyn ErrorWrapper>> = Box::new(ErrorMinimizer::new(Box::<NativeError>::default()));
                Python::attach(|py| {
                    d2 = Box::new(ErrorMinimizer::new(Box::new(PythonError::new(function.clone_ref(py)))))
                });
                d2
            } ,
        };

        // Configure cache
        let cache = Box::<Trie>::default();

        // Configure rules
        let mut node_rules: Vec<Box<dyn Rule>> = vec![];
        let mut search_rules: Vec<Box<dyn Rule>> = vec![];

        // Add discrepancy rule if specified
        if let Some(rule) = discrepancy {
            let discrepancy_rule = DiscrepancyRule::from(rule);
            search_rules.push(Box::new(discrepancy_rule));
        }

        // Add gain rule if specified
        if let Some(rule) = gain {
            let gain_rule = GainRule::from(rule);
            search_rules.push(Box::new(gain_rule));
        }

        // Add purity rule if specified
        if let Some(rule) = purity {
            let purity_rule = PurityRule::from(rule);
            node_rules.push(Box::new(purity_rule));
        }

        // Add topk rule if specified
        if let Some(rule) = topk {
            let topk_rule = TopkRule::from(rule);
            search_rules.push(Box::new(topk_rule));
        }

        // Create restart time limit rule
        if let Some(rule) = restart {
            let restart_rule = TimeLimitRule::from(rule);
            search_rules.push(Box::new(restart_rule));
        }

        // Build DL85 algorithm using the builder pattern
        let mut learner = DL85Builder::default()
            .max_depth(max_depth)
            .min_support(min_sup)
            .max_error(max_error)
            .max_time(time_limit)
            .specialization(depth2_policy)
            .always_sort(always_sort)
            .branching_strategy(branching_policy)
            .lower_bound_strategy(lower_bound_policy)
            .node_exposed_data(data_type)
            .cache(cache)
            .heuristic(heuristic_fn)
            .depth2_search(depth2_search)
            .add_search_rules(search_rules)
            .add_node_rules(node_rules)
            .error_function(error_fn)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build DL85: {:?}", e)))?;

        let config = learner.config();

        Ok( Self {
            learner,
            config,
            cover: Cover::new(vec![], vec![], 0),
            statistics: SearchOutput::default(),
            has_data: false
        }
        )

    }

    /// Loads training data into the classifier.
    ///
    /// This method converts NumPy arrays into the internal Cover representation
    /// used by the DL8.5 algorithm.
    ///
    /// # Parameters
    ///
    /// - `input`: Feature matrix as a NumPy array of shape (n_samples, n_features)
    /// - `target`: Optional target vector as a NumPy array of shape (n_samples,)
    ///            If None, assumes unsupervised learning or error_function is not None
    ///
    /// # Errors
    ///
    /// Returns `PyValueError` if:
    /// - Input arrays have incompatible shapes
    /// - Data contains invalid values (NaN, infinite)
    /// - Memory allocation fails during conversion
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// from pytreesrs.odt import PyDL85
    ///
    /// X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    /// y = np.array([1, 1, 0, 0])
    /// classifier = PyDL85(max_depth=3, min_sup=5)
    /// classifier.load_data(X, y)
    /// ```
    pub fn load_data(&mut self, input: PyReadonlyArrayDyn<f64>, target: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<()>  {
        let cover = create_cover_from_numpy(input, target.as_ref())?;
        self.cover = cover;
        self.has_data = true;
        Ok(())
    }


    /// Performs incremental fitting on pre-loaded data.
    ///
    /// This method continues the search from the current state, useful for
    /// implementing iterative or time-bounded optimization strategies.
    ///
    /// # Errors
    ///
    /// Returns `PyValueError` if no data has been loaded via `load_data()`.
    ///
    /// # Example
    ///
    /// ```python
    /// # Load data first
    /// classifier.load_data(X_train, y_train)
    ///
    /// # Perform incremental fitting
    /// classifier.partial_fit()
    /// ```
    pub fn partial_fit(&mut self) -> PyResult<()>{
        if self.has_data {
            return Err(PyValueError::new_err("Load data before using partial fit or use fit directly."))
        }

        self.learner.partial_fit(&mut self.cover);
        self.update_stats();
        Ok(())
    }

    /// Fits the model to the provided training data.
    ///
    /// This is the main training method that loads data and performs the complete
    /// DL8.5 search to find the optimal decision tree.
    ///
    /// # Parameters
    ///
    /// - `input`: Feature matrix as a NumPy array of shape (n_samples, n_features)
    /// - `target`: Optional target vector as a NumPy array of shape (n_samples,)
    ///
    /// # Errors
    ///
    /// Returns `PyValueError` if:
    /// - Data loading fails (see `load_data` for details)
    /// - Algorithm execution encounters an error
    /// - Search is interrupted or times out
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    ///
    /// X = np.random.rand(100, 5)
    /// y = np.random.randint(0, 2, 100)
    ///
    /// classifier.fit(X, y)
    /// print(f"Training completed with error: {classifier.stats.error}")
    /// ```
    pub fn fit(&mut self, input: PyReadonlyArrayDyn<f64>, target: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<()>{
        self.load_data(input, target).expect("Failed to load data");
        self.learner.fit(&mut self.cover).map_err(|x| PyValueError::new_err(format!("Failed to fit due to {:?}", x)))?;
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
