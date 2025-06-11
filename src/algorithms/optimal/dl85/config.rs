use crate::algorithms::common::config::BaseSearchConfig;
use crate::searches::{BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization};

pub struct DL85Config {

    pub(crate) base : BaseSearchConfig,
    pub(crate) always_sort: bool,
    pub(crate) cache_init_size: usize,
    pub(crate) cache_init_strategy: CacheInitStrategy,
    pub(crate) specialization: Specialization,
    pub(crate) lower_bound_strategy: LowerBoundStrategy,
    pub(crate) branching_strategy: BranchingStrategy,
    pub(crate) node_exposed_data: NodeExposedData,

}

impl Default for DL85Config {

    fn default() -> Self {
        Self {
            always_sort : false,
            cache_init_size: 0,
            cache_init_strategy: CacheInitStrategy::None_,
            specialization: Specialization::None_,
            lower_bound_strategy: LowerBoundStrategy::None_,
            branching_strategy: BranchingStrategy::None_,
            ..Default::default()
        }
    }
}

