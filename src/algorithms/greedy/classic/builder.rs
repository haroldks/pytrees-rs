use crate::algorithms::common::errors::ErrorWrapper;
use crate::algorithms::common::heuristics::Heuristic;
use crate::algorithms::greedy::classic::config::GreedyConfig;
use crate::algorithms::greedy::classic::Greedy;

pub struct GreedyBuilder<H>
where
    H: Heuristic + ?Sized,
{
    config: GreedyConfig,
    heuristic_fn: Option<Box<H>>,
}

impl<H> Default for GreedyBuilder<H>
where
    H: Heuristic + ?Sized,
{
    fn default() -> Self {
        Self {
            config: GreedyConfig::default(),
            heuristic_fn: None,
        }
    }
}

impl<H> GreedyBuilder<H>
where
    H: Heuristic + ?Sized,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn min_support(mut self, value: usize) -> Self {
        self.config.base.min_support = value;
        self
    }

    pub fn max_depth(mut self, value: usize) -> Self {
        self.config.base.max_depth = value;
        self
    }

    pub fn max_error(mut self, value: f64) -> Self {
        self.config.base.max_error = value;
        self
    }

    pub fn max_time(mut self, value: f64) -> Self {
        self.config.base.max_time = value;
        self
    }

    pub fn regularization(mut self, value: f64) -> Self {
        self.config.lambda = value;
        self
    }

    pub fn heuristic(mut self, value: Box<H>) -> Self {
        self.heuristic_fn = Some(value);
        self
    }

    pub fn build(self) -> Result<Greedy<H>, String> {
        let heuristic_fn = self.heuristic_fn.ok_or("Heuristic function is required.")?;
        Ok(Greedy::new(self.config, heuristic_fn))
    }
}
