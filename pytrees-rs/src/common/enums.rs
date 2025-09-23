use dtrees_rs::algorithms::common::heuristics::{GiniIndex, Heuristic, InformationGain, NoHeuristic, WeightedEntropy};
use dtrees_rs::algorithms::common::types::{BranchingPolicy, LowerBoundPolicy, NodeDataType, OptimalDepth2Policy};
use pyo3::pyclass;

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedSearchStrategy {
    DiscrepancySearch,
    LGDTErrorMinimizer,
    LGDTInfoGainMaximizer,
    None_,
}


#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedHeuristic {
    InformationGain,
    WeightedEntropy,
    GiniIndex,
    Disabled,
}

impl From<ExposedHeuristic> for Box<dyn Heuristic> {
    fn from(value: ExposedHeuristic) -> Self {
        match value {
            ExposedHeuristic::InformationGain => Box::<InformationGain>::default(),
            ExposedHeuristic::WeightedEntropy => Box::new(WeightedEntropy),
            ExposedHeuristic::GiniIndex => Box::new(GiniIndex),
            ExposedHeuristic::Disabled => Box::new(NoHeuristic),
        }
    }
}


#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedDepth2Policy {
    Enabled,
    Disabled,
}

impl From<ExposedDepth2Policy> for OptimalDepth2Policy {
    fn from(value: ExposedDepth2Policy) -> Self {
        match value {
            ExposedDepth2Policy::Enabled => OptimalDepth2Policy::Enabled,
            ExposedDepth2Policy::Disabled => OptimalDepth2Policy::Disabled
        }
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedBranchingPolicy {
    Dynamic,
    Default,
}

impl From<ExposedBranchingPolicy> for BranchingPolicy {
    fn from(value: ExposedBranchingPolicy) -> Self {
        match value {
            ExposedBranchingPolicy::Dynamic => BranchingPolicy::Dynamic,
            ExposedBranchingPolicy::Default => BranchingPolicy::Default
        }
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedLowerBoundPolicy {
    Similarity,
    Disabled,
}


impl From<ExposedLowerBoundPolicy> for LowerBoundPolicy {
    fn from(value: ExposedLowerBoundPolicy) -> Self {
        match value {
            ExposedLowerBoundPolicy::Similarity => LowerBoundPolicy::Similarity,
            ExposedLowerBoundPolicy::Disabled => LowerBoundPolicy::Disabled
        }
    }
}


#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedNodeDataType {
    ClassSupports,
    Tids,
}

impl From<ExposedNodeDataType> for NodeDataType {
    fn from(value: ExposedNodeDataType) -> Self {
        match value {
            ExposedNodeDataType::ClassSupports => NodeDataType::ClassesSupport,
            ExposedNodeDataType::Tids => NodeDataType::Tids
        }
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedStepStrategy {
    Monotonic,
    Exponential,
    Luby
}




#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedCacheType {
    Trie,
    Hashmap,
    None_,
}







#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedCacheInitStrategy {
    DynamicAllocation,
    UserAllocation,
    None_,
}
