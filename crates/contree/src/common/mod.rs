mod budget_schedule;

mod outcome;

pub use budget_schedule::{Budget, BudgetSchedule, PassReport, ScheduleBounds, ScheduleKind};
pub use outcome::{FitOutcome, SearchError, SearchStatus};

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// How a split threshold is chosen once the search has settled on the interval
/// between two consecutive distinct values of a feature.
#[derive(Default, Copy, Debug, Clone, PartialOrd, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PointSelector {
    /// The midpoint of the interval. The usual choice, and the default.
    #[default]
    Mid,
    /// The lower endpoint of the interval.
    First,
    /// A uniformly random point inside the interval.
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

    /// Every selector, in declaration order. Lets front ends enumerate the
    /// choices without hard-coding them.
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
#[derive(Copy, Clone, Debug)]
pub struct SearchConfig {
    pub max_depth: usize,
    pub min_sup: usize,
    pub max_time: f64,
    pub max_gap: usize,
    pub max_error: usize,
    pub is_root: bool,
    pub use_heuristic: bool,
    pub fast_d2: bool,
    pub point_selector: PointSelector,
    pub nb_runs: usize,
    pub discrepancy: usize,
    pub budget: usize,
}

impl SearchConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_gap: usize,
        max_error: usize,
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
            is_root: true,
            use_heuristic,
            fast_d2,
            point_selector: split_strategy,
            nb_runs: 0,
            discrepancy: 0,
            budget: 0,
        }
    }

    pub fn derive_left(&self) -> Self {
        let mut left_config = *self;
        left_config.max_depth -= 1;
        left_config.is_root = false;
        left_config.max_gap = (self.max_gap - (self.max_gap + 1) / 2) / 2;
        left_config
    }

    pub fn derive_right(&self, left_gap: usize) -> Self {
        let mut right_config = *self;
        right_config.max_depth -= 1;
        right_config.is_root = false;
        right_config.max_gap = (self.max_gap - (self.max_gap + 1) / 2).saturating_sub(left_gap);
        right_config
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Statistics {
    pub cache_size: usize,
    pub cache_hits: usize,
    pub general_solver_call: usize,
    pub specialized_solver_call: usize,
    pub num_samples: usize,
    pub num_features: usize,
    pub error: usize,
    pub duration: f64,
}

pub fn classification_error(classes_support: &[usize]) -> (usize, usize) {
    let mut max_idx = 0;
    let mut max_value = 0;
    let mut total = 0;
    for (idx, value) in classes_support.iter().enumerate() {
        total += value;
        if *value >= max_value {
            max_value = *value;
            max_idx = idx;
        }
    }
    let error = total - max_value;
    (error, max_idx)
}
