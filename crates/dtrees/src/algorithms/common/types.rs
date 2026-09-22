//! Enums and small types shared by the searches and their front ends.

use crate::algorithms::common::heuristics::{
    GiniIndex, Heuristic, InformationGain, NoHeuristic, WeightedEntropy,
};
use crate::algorithms::optimal::Reason;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Formatter;

/// Why a search could not produce a tree.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FitError {
    /// The depth is not supported by this search.
    InvalidDepth(usize),
    /// The minimum support is invalid.
    InvalidMinSupport(usize),
    /// No feature can split the instances.
    EmptyCandidates,
    /// The search failed.
    AlgorithmError,
    /// Too few instances to learn from.
    InsufficientData,
    /// No split beats a single leaf.
    EmptyTree,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            FitError::InvalidDepth(depth) => write!(f, "Invalid depth: {}", depth),
            FitError::InvalidMinSupport(support) => {
                write!(f, "Invalid minimum support threshold: {}", support)
            }
            FitError::EmptyCandidates => write!(f, "Empty candidates list found"),
            FitError::AlgorithmError => write!(f, "Error in algorithm execution"),
            FitError::InsufficientData => write!(f, "Insufficient data for training"),
            FitError::EmptyTree => write!(f, "Search produced an empty tree"),
        }
    }
}

impl std::error::Error for FitError {}

/// Counters collected during a search.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct SearchStatistics {
    /// Number of cached subproblems.
    pub cache_size: usize,
    /// Subproblems found in the cache.
    pub cache_hits: usize,
    /// Number of passes (the first one included).
    pub restarts: usize,
    /// Second branches skipped because the first one used up the bound.
    pub sibling_pruning: usize,
    /// Number of nodes visited.
    pub search_space_size: usize,
    /// Training error of the best tree.
    pub tree_error: f64,
    /// Search time in seconds.
    pub duration: f64,
    /// Number of features.
    pub num_attributes: usize,
    /// Number of training instances.
    pub num_samples: usize,
}

impl SearchStatistics {
    pub fn increment_search_space(&mut self) {
        self.search_space_size += 1;
    }
    pub fn increment_cache_hits(&mut self) {
        self.cache_hits += 1;
    }

    pub fn increment_sibling_pruning(&mut self) {
        self.sibling_pruning += 1;
    }

    pub fn increment_restarts(&mut self) {
        self.restarts += 1;
    }
    /// Number of passes so far.
    pub fn restarts(&self) -> usize {
        self.restarts
    }
}

/// Outcome of searching one node.
#[derive(Copy, Clone, Default, Debug)]
pub struct SearchResult {
    /// Best error found for the node.
    pub error: f64,
    /// Whether the cover was branched on the node's item, so that
    /// backtracking must undo it.
    pub has_intersected: bool,
    /// Why the search of the node stopped.
    pub reason: Reason,
}

/// Which set of rules to evaluate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum RuleType {
    /// Rules checked on entering a node.
    Node,
    /// Rules checked before branching on a feature.
    Search,
    /// The time limit.
    Time,
    /// The similarity lower bound.
    Similarity,
}

/// Whether to use the similarity lower bound of DL8.5.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum LowerBoundPolicy {
    /// Derive lower bounds from the most similar solved sibling. Only valid
    /// when each row adds at most 1 to the error, as with the
    /// misclassification error.
    Similarity,
    /// Only use the bounds stored in the cache.
    Disabled,
}

/// Which branch of a feature the search explores first.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum BranchingPolicy {
    /// The branch with the higher known lower bound, which leaves a tighter
    /// bound for the other one.
    Dynamic,
    /// Always the left branch.
    Default,
}

/// Whether DL8.5 uses the depth-2 solver for the last two levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum OptimalDepth2Policy {
    /// Use the depth-2 solver.
    Enabled,
    /// Search the last two levels like any other.
    Disabled,
}

/// What the error function receives for a node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
pub enum NodeDataType {
    /// The number of instances of each class.
    ClassesSupport,
    /// The ids of the instances, for custom error functions.
    Tids,
}

/// `(branch searched first, its lower bound, the other branch's lower bound)`.
pub type BranchingChoice = (usize, f64, f64);

/// Heuristic used to order the features.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SearchHeuristic {
    /// Keep the features in their original order.
    NoHeuristic,
    /// Lowest weighted Gini impurity first.
    GiniIndex,
    /// Highest information gain first.
    InformationGain,
    /// Lowest weighted entropy of the children first.
    WeightedEntropy,
}

impl From<SearchHeuristic> for Box<dyn Heuristic> {
    fn from(value: SearchHeuristic) -> Self {
        match value {
            SearchHeuristic::InformationGain => Box::<InformationGain>::default(),
            SearchHeuristic::WeightedEntropy => Box::<WeightedEntropy>::default(),
            SearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
            SearchHeuristic::NoHeuristic => Box::<NoHeuristic>::default(),
        }
    }
}

/// How the budget of a relaxable rule grows between passes. See
/// [`StepStrategy`](crate::algorithms::optimal::rules::StepStrategy).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SearchStepStrategy {
    /// Grow by a constant step.
    Monotonic,
    /// Multiply by a constant factor.
    Exponential,
    /// Follow the Luby sequence.
    Luby,
}
