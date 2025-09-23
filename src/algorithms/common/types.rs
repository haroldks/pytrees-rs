use crate::algorithms::optimal::Reason;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Formatter;
use crate::algorithms::common::heuristics::{GiniIndex, Heuristic, InformationGain, NoHeuristic, WeightedEntropy};

#[derive(Debug, Clone, Copy, Default,  Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum SearchStrategy {
    Depth2ErrorMinimizer,
    Depth2InfoGainMaximizer,
    LGDTErrorMinimizer,
    LGDTInfoGainMaximizer,
    #[default]
    DL85
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FitError {
    InvalidDepth(usize),
    InvalidMinSupport(usize),
    EmptyCandidates,
    AlgorithmError,
    InsufficientData,
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

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct SearchStatistics {
    pub cache_size: usize,
    pub cache_hits: usize,
    pub restarts: usize,
    pub sibling_pruning: usize,
    pub search_space_size: usize,
    pub tree_error: f64,
    pub duration: f64,
    pub num_attributes: usize,
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
    pub fn restarts(&self) -> usize {
        self.restarts
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct SearchResult {
    pub error: f64,
    pub has_intersected: bool,
    pub reason: Reason,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum RuleType {
    Node,
    Search,
    Time,
    Similarity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum LowerBoundPolicy {
    Similarity,
    Disabled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum BranchingPolicy {
    Dynamic,
    Default,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum OptimalDepth2Policy {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
pub enum NodeDataType {
    ClassesSupport,
    Tids,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum CacheType{
    Trie,
    Hashmap,
}



#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum CacheInitStrategy {
    DynamicAllocation,
    UserAllocation,
    Disabled,
}

pub type BranchingChoice = (usize, f64, f64);


#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum SearchHeuristic{
    NoHeuristic,
    GiniIndex,
    InformationGain,
    WeightedEntropy,
}


impl From<SearchHeuristic> for Box<dyn Heuristic> {
    fn from(value: SearchHeuristic) -> Self {
        match value {
            SearchHeuristic::InformationGain => Box::new(InformationGain::default()),
            SearchHeuristic::WeightedEntropy => Box::new(WeightedEntropy::default()),
            SearchHeuristic::GiniIndex => Box::new(GiniIndex::default()),
            SearchHeuristic::NoHeuristic => Box::new(NoHeuristic::default()),
        }
    }
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, Eq, PartialEq)]
pub enum SearchStepStrategy {
    Monotonic,
    Exponential,
    Luby
}
