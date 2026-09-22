use serde::{Deserialize, Serialize};

/// Settings shared by every search.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct BaseSearchConfig {
    /// Minimum number of instances in each leaf.
    pub min_support: usize,
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Only trees with a lower error are accepted.
    pub max_error: f64,
    /// Time limit in seconds.
    pub max_time: f64,
}

impl Default for BaseSearchConfig {
    fn default() -> Self {
        Self {
            min_support: 1,
            max_depth: 1,
            max_error: f64::INFINITY,
            max_time: f64::INFINITY,
        }
    }
}
