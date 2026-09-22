use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, NodeDataType, OptimalDepth2Policy,
};
use serde::{Deserialize, Serialize};

/// Settings of a [`DL85`](super::DL85) search. Set them through
/// [`DL85Builder`](super::DL85Builder).
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct DL85Config {
    pub(crate) base: BaseSearchConfig,
    /// Sort the features by the heuristic at every node, not only at the root.
    pub(crate) always_sort: bool,
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
            optimal_depth2policy: OptimalDepth2Policy::Disabled,
            lower_bound_policy: LowerBoundPolicy::Disabled,
            data_type: NodeDataType::ClassesSupport,
            branching_policy: BranchingPolicy::Default,
        }
    }
}

impl DL85Config {
    /// Whether nodes two levels from the bottom use the depth-2 solver.
    ///
    /// The solver works from class counts, so it is skipped when the error
    /// function takes row indices ([`NodeDataType::Tids`]).
    pub fn use_depth2_optimization(&self) -> bool {
        self.optimal_depth2policy == OptimalDepth2Policy::Enabled
            && self.data_type == NodeDataType::ClassesSupport
    }

    /// Whether the branch with the higher lower bound is searched first.
    pub fn use_dynamic_branching(&self) -> bool {
        self.branching_policy == BranchingPolicy::Dynamic
    }

    /// Whether the similarity lower bound is used.
    pub fn use_similarity_lb(&self) -> bool {
        self.lower_bound_policy == LowerBoundPolicy::Similarity
    }
}
