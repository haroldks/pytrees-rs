//! What a fit produced, and why it stopped.
//!
//! Every invariant the search relies on used to be a `debug_assert!`, which is
//! compiled out of the release build that a wheel would ship. The ones that
//! guard caller-supplied input are checked here instead, at the `fit`
//! boundary, and reported rather than assumed.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::common::Statistics;
use crate::data::DatasetError;
use crate::tree::{Tree, TreeError};

/// A fit that could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchError {
    /// The dataset has no instances.
    EmptyDataset,
    /// The dataset has no feature columns.
    NoFeatures,
    /// `sort_features` and `compute_unique_feature_values` have not both run.
    ///
    /// The search compares unique-value indices, so an unprepared dataset does
    /// not fail loudly -- it makes every fit return a single leaf. Build
    /// datasets through [`Dataset::from_rows`](crate::data::Dataset::from_rows)
    /// or [`DataReader`](crate::reader::data_reader::DataReader), which both
    /// prepare them.
    UnpreparedDataset,
    /// A parameter whose value cannot produce a meaningful search.
    InvalidParameter { name: &'static str, reason: String },
    /// The dataset itself is malformed.
    Data(DatasetError),
    /// The search produced a tree that does not hold the tree invariant. This
    /// is a bug in the crate, not in the caller's input.
    Tree(TreeError),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::EmptyDataset => write!(f, "the dataset has no instances"),
            SearchError::NoFeatures => write!(f, "the dataset has no features"),
            SearchError::UnpreparedDataset => write!(
                f,
                "the dataset has not been sorted and indexed; build it with \
                 Dataset::from_rows or DataReader"
            ),
            SearchError::InvalidParameter { name, reason } => {
                write!(f, "invalid {name}: {reason}")
            }
            SearchError::Data(err) => write!(f, "{err}"),
            SearchError::Tree(err) => write!(f, "the search produced an invalid tree: {err}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<DatasetError> for SearchError {
    fn from(err: DatasetError) -> Self {
        SearchError::Data(err)
    }
}

impl From<TreeError> for SearchError {
    fn from(err: TreeError) -> Self {
        SearchError::Tree(err)
    }
}

/// Why the search stopped.
///
/// Only `Optimal` means the tree is proven best for the given depth and
/// support; the others mean the search ran out of something first. `Statistics`
/// alone could never say which, so a caller had no way to tell an optimum from
/// a timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStatus {
    /// The search space was exhausted: no better tree exists.
    Optimal,
    /// The time limit ran out first.
    TimeLimit,
    /// The anytime search ran out of budget to widen with.
    BudgetExhausted,
    /// A tree within `max_gap` of the caller's error bound was found, and the
    /// search stopped there rather than proving optimality.
    ErrorBoundReached,
}

impl SearchStatus {
    /// Whether the returned tree is proven optimal.
    pub fn is_optimal(self) -> bool {
        matches!(self, SearchStatus::Optimal)
    }
}

impl fmt::Display for SearchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SearchStatus::Optimal => "optimal",
            SearchStatus::TimeLimit => "time limit",
            SearchStatus::BudgetExhausted => "budget exhausted",
            SearchStatus::ErrorBoundReached => "error bound reached",
        })
    }
}

/// The result of a fit.
#[derive(Clone, Debug)]
pub struct FitOutcome {
    pub tree: Tree,
    pub statistics: Statistics,
    pub status: SearchStatus,
}

impl FitOutcome {
    /// Training-set misclassifications of the returned tree.
    pub fn error(&self) -> usize {
        self.statistics.error
    }
}
