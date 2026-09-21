mod continuous_tree;
mod contree_lds;
mod depth2;
mod interval_pruner;
mod shared;

use crate::common::{
    FitOutcome, PointSelector, ScheduleKind, SearchConfig, SearchError, SearchStatus, Statistics,
};
use crate::data::view::DataView;
use crate::data::Dataset;
use crate::tree::Tree;
pub use continuous_tree::ConTree;
pub use contree_lds::ConTreeLds;

/// The sub-range of `possible_splits` whose splits leave at least `min_sup`
/// instances on both sides.
///
/// `possible_splits` holds positions into a column sorted by value, so a split
/// at position `p` puts `p` instances on the left and `view_len - p` on the
/// right, and the feasible positions form one contiguous run. Returning the
/// range up front is both cheaper and safer than testing each candidate as it
/// comes up: a search that discovers a violation mid-interval has to throw the
/// whole interval away, taking the feasible splits inside it with it.
///
/// The returned range is empty when no split satisfies the constraint.
pub(crate) fn support_feasible_splits(
    possible_splits: &[usize],
    view_len: usize,
    min_sup: usize,
) -> std::ops::Range<usize> {
    let min_sup = min_sup.max(1);
    if view_len < 2 * min_sup {
        return 0..0;
    }
    let first = possible_splits.partition_point(|&p| p < min_sup);
    let last = possible_splits.partition_point(|&p| view_len - p.min(view_len) >= min_sup);
    first..last.max(first)
}

/// Checks the things a caller can get wrong before any work starts.
///
/// These used to be `debug_assert!`s, which are compiled out of exactly the
/// build a wheel would ship.
pub fn validate(config: &SearchConfig, dataset: &Dataset) -> Result<(), SearchError> {
    if dataset.count() == 0 {
        return Err(SearchError::EmptyDataset);
    }
    if dataset.num_features() == 0 {
        return Err(SearchError::NoFeatures);
    }
    if !dataset.is_prepared() {
        return Err(SearchError::UnpreparedDataset);
    }
    if config.min_sup == 0 {
        return Err(SearchError::InvalidParameter {
            name: "min_sup",
            reason: "must be at least 1".to_string(),
        });
    }
    if 2 * config.min_sup > dataset.count() {
        return Err(SearchError::InvalidParameter {
            name: "min_sup",
            reason: format!(
                "{} leaves no room for a split with n_samples={}",
                config.min_sup,
                dataset.count()
            ),
        });
    }
    if config.max_time.is_nan() || config.max_time <= 0.0 {
        return Err(SearchError::InvalidParameter {
            name: "max_time",
            reason: "must be a positive number of seconds".to_string(),
        });
    }
    Ok(())
}

pub enum GenericConTree {
    Normal(ConTree),
    LDS(ConTreeLds),
}

impl GenericConTree {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: usize,
        split_selection_strategy: PointSelector,
        max_gap: usize,
        use_heuristic: bool,
        fast_d2: bool,
        use_lds: bool,
    ) -> Self {
        match use_lds {
            true => Self::LDS(ConTreeLds::new(
                min_sup,
                max_depth,
                max_time,
                max_error,
                split_selection_strategy,
                max_gap,
                use_heuristic,
                fast_d2,
            )),
            false => Self::Normal(ConTree::new(
                min_sup,
                max_depth,
                max_time,
                max_error,
                split_selection_strategy,
                max_gap,
                use_heuristic,
                fast_d2,
            )),
        }
    }

    /// Chooses the anytime search's budget schedule. No effect on the
    /// exhaustive search, which has no budget.
    pub fn with_budget_schedule(self, schedule: ScheduleKind) -> Self {
        match self {
            GenericConTree::LDS(solver) => GenericConTree::LDS(solver.with_schedule(schedule)),
            other => other,
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) -> Result<FitOutcome, SearchError> {
        match self {
            GenericConTree::Normal(solver) => solver.fit(dataset),
            GenericConTree::LDS(solver) => solver.fit(dataset),
        }
    }

    /// Runs one pass of the anytime search, returning whether it is done.
    ///
    /// The exhaustive solver has no notion of a partial fit, so it runs to
    /// completion and reports that it is done.
    pub fn partial_fit(&mut self, view: &DataView<'_>) -> Result<bool, SearchError> {
        match self {
            GenericConTree::Normal(_) => Ok(true),
            GenericConTree::LDS(solver) => Ok(solver.partial_fit(view)),
        }
    }

    pub fn stats(&self) -> Statistics {
        match self {
            GenericConTree::Normal(solver) => solver.statistics(),
            GenericConTree::LDS(solver) => *solver.statistics(),
        }
    }

    pub fn status(&self) -> SearchStatus {
        match self {
            GenericConTree::Normal(solver) => solver.status(),
            GenericConTree::LDS(solver) => solver.status(),
        }
    }

    /// The tree found by the last `fit`. Empty if `fit` has not run.
    pub fn tree(&mut self) -> Tree {
        match self {
            GenericConTree::Normal(solver) => solver.tree.clone(),
            GenericConTree::LDS(solver) => solver.get_solution_tree(),
        }
    }
}
