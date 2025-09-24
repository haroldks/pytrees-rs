use pyo3::{pyclass, pymethods, PyResult};
use dtrees_rs::algorithms::common::types::{SearchStatistics, SearchStrategy};
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, Exponential, GainRule, Luby, Monotonic, PurityRule, StepStrategy, TopkRule};
use dtrees_rs::tree::Tree;
use crate::common::enums::ExposedStepStrategy;
use dtrees_rs::algorithms::optimal::rules::common::TimeLimitRule;

/// Search output container for decision tree algorithm results.
///
/// This class encapsulates all results and statistics from decision tree
/// construction algorithms, providing a unified interface for accessing
/// tree structures, performance metrics, and detailed search statistics.
///
/// ## Fields
///
/// - `error`: Final classification error of the constructed tree
/// - `tree`: The decision tree structure in JSON format
/// - `statistics`: Detailed search statistics (nodes explored, cache performance, etc.)
/// - `duration`: Total algorithm execution time in seconds
/// - `search`: Search strategy information used during construction
///
/// ## Usage
///
/// This class is typically returned by algorithm functions and should not
/// be instantiated directly by users.
///
/// ```python
/// # Returned by algorithm functions
/// result = classifier.fit(X, y)
/// stats = classifier.stats
///
/// print(f"Error: {stats.error}")
/// print(f"Duration: {stats.duration}s")
/// print(f"Tree: {stats.tree}")
/// print(f"Statistics: {stats.statistics}")
/// ```
#[pyclass(name = "output")]
#[derive(Default, Clone)]
pub struct SearchOutput {
    #[pyo3(get, set)]
    pub(crate) error: f64,
    pub(crate) tree: Tree,
    pub(crate) statistics: SearchStatistics,
    pub(crate) duration: f64,
    pub(crate) search: SearchStrategy
}

#[pymethods]
impl SearchOutput {
    /// Returns the classification error of the constructed tree.
    ///
    /// # Returns
    ///
    /// The error rate as a float between 0.0 and 1.0, where 0.0 indicates
    /// perfect classification and 1.0 indicates completely incorrect classification.
    #[getter]
    pub fn error(&self) -> PyResult<f64> {
        Ok(self.error)
    }

    /// Returns detailed search statistics as a JSON string.
    ///
    /// The statistics include information about:
    /// - Number of nodes explored during search
    /// - Cache hit/miss ratios
    /// - Memory usage patterns
    /// - Algorithm-specific metrics
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string containing comprehensive search statistics.
    #[getter]
    pub fn statistics(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.statistics).unwrap();
        Ok(json)
    }

    /// Returns the decision tree structure as a JSON string.
    ///
    /// The tree representation includes:
    /// - Node splitting conditions
    /// - Leaf node predictions
    /// - Tree topology and depth information
    /// - Feature indices and threshold values
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string representing the complete tree structure.
    #[getter]
    pub fn tree(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.tree).unwrap();
        Ok(json)
    }

    /// Returns the total algorithm execution time.
    ///
    /// # Returns
    ///
    /// Duration in seconds as a float.
    #[getter]
    pub fn duration(&self) -> f64 {
        self.duration
    }
}

/// Limited Discrepancy Search (LDS) rule for controlling search exploration.
///
/// This rule implements limited discrepancy search, which constrains the algorithm
/// to explore only a limited number of "discrepancies" from a heuristic ordering.
/// This helps balance between search completeness and computational efficiency.
///
/// ## Algorithm Background
///
/// LDS works by allowing the search to deviate from the heuristic ordering
/// only a limited number of times. This focuses computational effort on the
/// most promising parts of the search space while still allowing exploration
/// of alternative solutions.
///
/// ## Parameters
///
/// - `initial_value`: Starting discrepancy budget (default: 0)
/// - `limit`: Maximum discrepancy budget allowed (default: unlimited)
/// - `step_strategy`: How to increase the budget over time (default: Monotonic)
/// - `base`: Base increment for step strategy (default: 1)
///
/// ## Step Strategies
///
/// - **Monotonic**: Linear increase by base amount each iteration
/// - **Exponential**: Exponential increase (base^iteration)
/// - **Luby**: Luby sequence-based increase for optimal restart behavior
///
/// ## Example
///
/// ```python
/// from pytreesrs.odt.rules import ExposedDiscrepancyRule
/// from pytreesrs.enums import ExposedStepStrategy
///
/// # Conservative LDS with slow budget increase
/// rule = ExposedDiscrepancyRule(
///     initial_value=0,
///     limit=10,
///     step_strategy=ExposedStepStrategy.Monotonic,
///     base=1
/// )
///
/// # Aggressive LDS with exponential budget increase
/// rule = ExposedDiscrepancyRule(
///     initial_value=1,
///     limit=100,
///     step_strategy=ExposedStepStrategy.Exponential,
///     base=2
/// )
/// ```
#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedDiscrepancyRule {
    initial_value: usize,
    limit: usize,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedDiscrepancyRule {
    #[new]
    #[pyo3(signature = (initial_value=0, limit=usize::MAX, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(initial_value: usize, limit:usize, step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            initial_value,
            limit,
            step_strategy,
            base,
        }
    }
}

impl From<ExposedDiscrepancyRule> for DiscrepancyRule {
    fn from(value: ExposedDiscrepancyRule) -> Self {

        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        DiscrepancyRule::new(value.limit, step).with_budget(value.initial_value)
    }
}

/// Information Gain-based stopping rule for search optimization.
///
/// This rule terminates search branches when the potential information gain
/// falls below a specified threshold, helping to prune unpromising parts
/// of the search space and improve computational efficiency.
///
/// ## Algorithm Background
///
/// Information gain measures the reduction in entropy (or other impurity measures)
/// achieved by a split. When the potential gain becomes too small, further
/// exploration is unlikely to yield significantly better trees.
///
/// ## Parameters
///
/// - `min_gain`: Minimum information gain gap required to continue search (default: 0.0)
/// - `epsilon`: Numerical step for gain gap increase strategies (default: 1e-4)
/// - `limit`: Maximum gain threshold for adaptive behavior (Expected to be the maximum depth)
/// - `step_strategy`: How to adjust thresholds over time (default: Monotonic)
/// - `base`: Base increment for step strategy (default: 1)
///
/// ## Example
///
/// ```python
/// from pytreesrs.odt.rules import ExposedGainRule
///
/// # Conservative gain-based pruning
/// rule = ExposedGainRule(
///     min_gain=0.01,
///     epsilon=1e-4,
///     limit=1.0
/// )
///
/// # Aggressive adaptive pruning
/// rule = ExposedGainRule(
///     min_gain=0.05,
///     epsilon=1e-6,
///     limit=10.0,
///     step_strategy=ExposedStepStrategy.Exponential,
///     base=2
/// )
/// ```
#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedGainRule {
    min_gain: f64,
    epsilon:f64,
    limit: f64,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedGainRule {
    #[new]
    #[pyo3(signature = (min_gain=0.0, epsilon=1e-4, limit=6.0, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(min_gain: f64, epsilon: f64, limit: f64,  step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            min_gain,
            epsilon,
            limit,
            step_strategy,
            base
        }
    }
}

impl From<ExposedGainRule> for GainRule {
    fn from(value: ExposedGainRule) -> Self {
        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        GainRule::new(value.min_gain, value.epsilon, value.limit, step)
    }
}

/// Node purity-based stopping rule for tree construction.
///
/// This rule stops further splitting of nodes when they achieve a specified
/// level of class purity.
///
/// ## Parameters
///
/// - `min_purity`: Minimum purity threshold to stop splitting (default: 0.0)
/// - `epsilon`: Numerical increase value for rule relaxing (default: 1e-4)
///
///
/// ## Usage Guidelines
///
/// - Set `min_purity` to 0.8-0.95 for moderate pruning
/// - Higher values (0.95-0.99) provide aggressive pruning
/// - Lower values (0.5-0.8) allow more tree growth
///
/// ## Example
///
/// ```python
/// from pytreesrs.odt.rules import ExposedPurityRule
///
/// # Stop when nodes are 90% pure
/// rule = ExposedPurityRule(
///     min_purity=0.9,
///     epsilon=1e-4
/// )
///
/// # Very conservative - stop at 95% purity
/// rule = ExposedPurityRule(
///     min_purity=0.95,
///     epsilon=1e-6
/// )
/// ```
#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedPurityRule {
    min_purity: f64,
    epsilon: f64
}

#[pymethods]
impl ExposedPurityRule {

    #[new]
    #[pyo3(signature = (min_purity=0.0, epsilon=1e-4))]
    pub fn new(min_purity: f64, epsilon: f64) -> Self {
        Self {
            min_purity,
            epsilon
        }
    }

}

impl From<ExposedPurityRule> for PurityRule {
    fn from(value: ExposedPurityRule) -> Self {
        PurityRule::new(value.min_purity, value.epsilon)
    }
}

/// Top-K search limitation rule for controlling search breadth.
///
/// This rule limits the search to explore only the K most promising features
/// at each level, helping to control computational complexity.
///
/// ## Parameters
///
/// - `initial_value`: Starting K value (number of nodes to explore) (default: 0)
/// - `limit`: Maximum K value allowed (default: unlimited)
/// - `step_strategy`: How to increase K over time (default: Monotonic)
/// - `base`: Base increment for step strategy (default: 1)
///
///
/// ## Example
///
/// ```python
/// from pytreesrs.odt.rules import ExposedTopKRule
/// from pytreesrs.enums import ExposedStepStrategy
///
/// # Conservative top-K search
/// rule = ExposedTopKRule(
///     initial_value=5,
///     limit=50,
///     step_strategy=ExposedStepStrategy.Monotonic,
///     base=5
/// )
///
/// # Adaptive top-K with exponential growth
/// rule = ExposedTopKRule(
///     initial_value=1,
///     limit=1000,
///     step_strategy=ExposedStepStrategy.Exponential,
///     base=2
/// )
/// ```
#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedTopKRule {
    initial_value: usize,
    limit: usize,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedTopKRule {
    #[new]
    #[pyo3(signature = (initial_value=0, limit=usize::MAX, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(initial_value: usize, limit:usize, step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            initial_value,
            limit,
            step_strategy,
            base,
        }
    }

}

impl From<ExposedTopKRule> for TopkRule {
    fn from(value: ExposedTopKRule) -> Self {

        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        TopkRule::new(value.limit, step).with_budget(value.initial_value)
    }
}

/// Time-based restart rule for iterative search algorithms.
///
/// This rule implements time-bounded search with restart capabilities,
/// allowing algorithms to restart when time
/// limits are exceeded.
///
/// ## Parameters
///
/// - `limit`: Time limit in seconds before restart (default: 1.0)
///
/// ## Example
///
/// ```python
/// from pytreesrs.odt.rules import ExposedRestartRule
///
/// # Quick restart for interactive use
/// rule = ExposedRestartRule(limit=0.5)
///
/// # Longer restart for batch processing
/// rule = ExposedRestartRule(limit=60.0)
/// ```
#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedRestartRule {
    limit: f64
}

#[pymethods]
impl ExposedRestartRule {
    #[new]
    #[pyo3(signature = (limit=1.0))]
    pub fn new(limit: f64) -> Self {
        Self {
            limit
        }
    }

}

impl From<ExposedRestartRule> for TimeLimitRule {
    fn from(value: ExposedRestartRule) -> Self {
        TimeLimitRule::new(value.limit).relaxable()
    }
}
