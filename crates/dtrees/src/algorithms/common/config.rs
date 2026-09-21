use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct BaseSearchConfig {
    pub min_support: usize,
    pub max_depth: usize,
    pub max_error: f64,
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
