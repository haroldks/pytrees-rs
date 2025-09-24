use serde::{Deserialize, Serialize};
use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::types::{
    BranchingPolicy, CacheInitStrategy, LowerBoundPolicy, NodeDataType, OptimalDepth2Policy,
};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct DL85Config {
    pub(crate) base: BaseSearchConfig,
    pub(crate) always_sort: bool,
    pub(crate) cache_init_size: usize,
    pub(crate) cache_init_strategy: CacheInitStrategy,
    pub(crate) optimal_depth2policy: OptimalDepth2Policy,
    pub(crate) lower_bound_policy: LowerBoundPolicy,
    pub(crate) branching_policy: BranchingPolicy,
    pub(crate) data_type: NodeDataType,
}

impl Default for DL85Config {
    fn default() -> Self {
        Self {
            base: BaseSearchConfig::default(),
            always_sort: false,
            cache_init_size: 0,
            cache_init_strategy: CacheInitStrategy::Disabled,
            optimal_depth2policy: OptimalDepth2Policy::Disabled,
            lower_bound_policy: LowerBoundPolicy::Disabled,
            data_type: NodeDataType::ClassesSupport,
            branching_policy: BranchingPolicy::Default,
        }
    }
}

impl DL85Config {
    pub fn use_depth2_optimization(&self) -> bool {
        self.optimal_depth2policy == OptimalDepth2Policy::Enabled
    }

    pub fn use_dynamic_branching(&self) -> bool {
        self.branching_policy == BranchingPolicy::Dynamic
    }

    pub fn use_similarity_lb(&self) -> bool {
        self.lower_bound_policy == LowerBoundPolicy::Similarity
    }
}
