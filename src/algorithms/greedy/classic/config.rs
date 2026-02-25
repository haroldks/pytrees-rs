use crate::algorithms::common::config::BaseSearchConfig;

#[derive(Copy, Clone, Debug, Default)]
pub struct GreedyConfig {
    pub(crate) base: BaseSearchConfig,
    pub(crate) lambda: f64,
}
