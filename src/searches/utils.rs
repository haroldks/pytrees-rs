use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Constraints {
    pub max_depth: usize,
    pub min_sup: usize,
    pub max_error: f64,
    pub max_time: usize,
    pub one_time_sort: bool,
    pub node_exposed_data: NodeExposedData,
    pub specialization: Specialization,
    pub lower_bound_strategy: LowerBoundStrategy,
    pub branching_strategy: BranchingStrategy,
    pub cache_init_strategy: CacheInitStrategy,
    pub search_strategy: SearchStrategy,
    pub cache_init_size: usize,
    pub discrepancy_budget: usize,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            max_depth: 2,
            min_sup: 1,
            max_error: <f64>::INFINITY,
            max_time: 600,
            one_time_sort: false,
            node_exposed_data: NodeExposedData::ClassesSupport,
            specialization: Specialization::None_,
            lower_bound_strategy: LowerBoundStrategy::None_,
            branching_strategy: BranchingStrategy::None_,
            cache_init_strategy: CacheInitStrategy::None_,
            search_strategy: SearchStrategy::None_,
            cache_init_size: 0,
            discrepancy_budget: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Statistics {
    pub cache_size: usize,
    pub cache_callbacks: usize,
    pub search_space_size: usize,
    pub tree_error: f64,
    pub duration: Duration,
    pub num_attributes: usize,
    pub num_samples: usize,
    pub constraints: Constraints,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            cache_size: 0,
            cache_callbacks: 0,
            search_space_size: 0,
            tree_error: 0.0,
            duration: Duration::default(),
            num_attributes: 0,
            num_samples: 0,
            constraints: Constraints::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeExposedData {
    ClassesSupport,
    Tids,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum Specialization {
    Murtree,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum LowerBoundStrategy {
    Similarity,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum BranchingStrategy {
    Dynamic,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum CacheType {
    Trie,
    Hashmap,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum CacheInitStrategy {
    DynamicAllocation,
    UserAllocation,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchStrategy {
    DiscrepancySearchMonotonic,
    DiscrepancySearchLuby,
    DiscrepancySearchExponential,
    LessGreedyMurtree,
    LessGreedyInfoGain,
    RestartTimeout,
    PurityLimit,
    NormalDL85,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StopReason {
    Done,
    TimeLimitReached,
    LowerBoundConstrained,
    MaxDepthReached,
    NotEnoughSupport,
    PureNode,
    PureEnough,
    FromSpecializedAlgorithm,
    BranchBudgetExhausted,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum SearchHeuristic {
    InformationGain,
    InformationGainRatio,
    GiniIndex,
    None_,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum D2Objective {
    Error,
    InformationGain,
}
