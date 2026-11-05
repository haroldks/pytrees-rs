mod budget_schedule;
mod loss;
mod outcome;

pub use budget_schedule::{Budget, BudgetSchedule, PassReport, ScheduleBounds, ScheduleKind};
pub use loss::{Leaf, LeafLoss, Loss, LossInput, Skipping};
pub use outcome::{FitOutcome, SearchError, SearchStatus};

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Which candidate threshold the search evaluates next inside an interval of
/// candidates.
#[derive(Default, Copy, Debug, Clone, PartialOrd, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PointSelector {
    /// The middle candidate, which bisects the interval. The default.
    #[default]
    Mid,
    /// Candidates are tried one at a time instead of by bisection, in Gini
    /// order when the heuristic is on. The anytime search limits how many are
    /// tried with its split budget.
    First,
    /// A uniformly random candidate.
    Random,
}

impl PointSelector {
    /// The spelling used on the command line and in the Python API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mid => "mid",
            Self::First => "first",
            Self::Random => "random",
        }
    }

    /// Every selector, in declaration order.
    pub const ALL: [Self; 3] = [Self::Mid, Self::First, Self::Random];
}

impl fmt::Display for PointSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PointSelector {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mid" => Ok(Self::Mid),
            "first" => Ok(Self::First),
            "random" => Ok(Self::Random),
            other => Err(format!(
                "unknown split selection strategy `{other}` (expected one of: mid, first, random)"
            )),
        }
    }
}

/// The cost of a tree: the sum of the costs of its leaves. For the built-in
/// loss, the number of misclassified instances.
pub type Cost = f64;

/// Settings of a search, and the per-node state derived from them.
#[derive(Copy, Clone, Debug)]
pub struct SearchConfig {
    /// Maximum depth of the tree (remaining depth, below the root).
    pub max_depth: usize,
    /// Minimum number of instances in each leaf.
    pub min_sup: usize,
    /// Time limit in seconds.
    pub max_time: f64,
    /// Error gap tolerated with respect to the optimum; 0 for an exact search.
    pub max_gap: Cost,
    /// Initial upper bound on the error.
    pub max_error: Cost,
    /// The smallest difference between two different costs: 1 for
    /// misclassification, whose costs are whole numbers, 0 for a custom loss.
    /// A search that finds no tree within a bound `ub` then proves the cost
    /// is at least `ub + cost_step`.
    pub cost_step: Cost,
    /// Whether thresholds are skipped; see [`Skipping`].
    pub skipping: Skipping,
    /// Whether this configuration is the root's.
    pub is_root: bool,
    /// Order features and thresholds by Gini impurity.
    pub use_heuristic: bool,
    /// Use the specialised solver for depth-2 subtrees.
    pub fast_d2: bool,
    /// How thresholds are picked inside an interval.
    pub point_selector: PointSelector,
    /// Number of passes run so far (anytime search).
    pub nb_runs: usize,
    /// Discrepancy used on the path to this node (anytime search).
    pub discrepancy: usize,
    /// Discrepancy budget of the current pass (anytime search).
    pub budget: usize,
}

impl SearchConfig {
    /// A root configuration with the given settings.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_gap: Cost,
        max_error: Cost,
        use_heuristic: bool,
        fast_d2: bool,
        split_strategy: PointSelector,
    ) -> Self {
        Self {
            max_depth,
            min_sup,
            max_time,
            max_gap,
            max_error,
            cost_step: 1.0,
            skipping: Skipping::Auto,
            is_root: true,
            use_heuristic,
            fast_d2,
            point_selector: split_strategy,
            nb_runs: 0,
            discrepancy: 0,
            budget: 0,
        }
    }

    /// The cost step for a loss: misclassification costs are counts of
    /// instances, so whole numbers. A custom loss's costs may be any number,
    /// so there is no smallest difference between two of them.
    pub fn cost_step_for(loss: &Loss) -> Cost {
        if loss.is_misclassification() {
            1.0
        } else {
            0.0
        }
    }

    /// The configuration of the first child searched: one level shallower,
    /// with part of the gap.
    pub fn derive_left(&self) -> Self {
        let mut left_config = *self;
        left_config.max_depth -= 1;
        left_config.is_root = false;
        left_config.max_gap = ((self.max_gap - (self.max_gap / 2.0).ceil()) / 2.0).floor();
        left_config
    }

    /// The configuration of the second child searched, given the gap already
    /// handed to the first.
    pub fn derive_right(&self, left_gap: Cost) -> Self {
        let mut right_config = *self;
        right_config.max_depth -= 1;
        right_config.is_root = false;
        right_config.max_gap = (self.max_gap - (self.max_gap / 2.0).ceil() - left_gap).max(0.0);
        right_config
    }
}

/// Counters collected during a search.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Statistics {
    /// Number of cached subproblems.
    pub cache_size: usize,
    /// Subproblems answered from the cache.
    pub cache_hits: usize,
    /// Calls to the general search.
    pub general_solver_call: usize,
    /// Calls to the depth-2 solver.
    pub specialized_solver_call: usize,
    /// Number of training instances.
    pub num_samples: usize,
    /// Number of features.
    pub num_features: usize,
    /// Training cost of the best tree.
    pub error: Cost,
    /// Moves of instances in the depth-2 scan that broke the assumption
    /// threshold skipping makes about the loss. Only counted when skipping
    /// is forced on for a custom loss; see [`Skipping`].
    pub skipping_violations: usize,
    /// Search time in seconds.
    pub duration: f64,
}

/// `(misclassifications, majority class)` of a leaf with the given class
/// counts. Ties go to the highest class index.
pub fn classification_error(classes_support: &[usize]) -> (Cost, usize) {
    let mut max_idx = 0;
    let mut max_value = 0;
    let mut total = 0;
    for (idx, &value) in classes_support.iter().enumerate() {
        total += value;
        if value >= max_value {
            max_value = value;
            max_idx = idx;
        }
    }
    ((total - max_value) as Cost, max_idx)
}
