mod config;

use crate::algorithms::optimal::dl85::config::DL85Config;
use crate::cache::Caching;
use crate::heuristics::Heuristic;
use crate::searches::errors::ErrorWrapper;
use crate::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};

struct DL85Builder<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    config: DL85Config,
    cache: Option<Box<C>>,
    error_fn: Option<Box<E>>,
    heuristic_fn: Option<Box<H>>,
}

impl<C, E, H> Default for DL85Builder<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    fn default() -> Self {
        Self {
            cache: None,
            error_fn: None,
            heuristic_fn: None,
            ..Default::default()
        }
    }
}

impl<C, E, H> DL85Builder<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized,
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

    pub fn always_sort(mut self, value: bool) -> Self {
        self.config.always_sort = value;
        self
    }

    pub fn cache_init_size(mut self, value: usize) -> Self {
        self.config.cache_init_size = value;
        self
    }

    pub fn cache_init_strategy(mut self, value: CacheInitStrategy) -> Self {
        self.config.cache_init_strategy = value;
        self
    }

    pub fn specialization(mut self, value: Specialization) -> Self {
        self.config.specialization = value;
        self
    }

    pub fn lower_bound_strategy(mut self, value: LowerBoundStrategy) -> Self {
        self.config.lower_bound_strategy = value;
        self
    }

    pub fn branching_strategy(mut self, value: BranchingStrategy) -> Self {
        self.config.branching_strategy = value;
        self
    }

    pub fn node_exposed_data(mut self, value: NodeExposedData) -> Self {
        self.config.node_exposed_data = value;
        self
    }

    pub fn cache(mut self, value: Box<C>) -> Self {
        self.cache = Some(value);
        self
    }

    pub fn error_function(mut self, value: Box<E>) -> Self {
        self.error_fn = Some(value);
        self
    }

    pub fn heuristic(mut self, value: Box<H>) -> Self {
        self.heuristic_fn = Some(value);
        self
    }

    pub fn build(self) -> Result<(), String> {
        let cache = self.cache.ok_or("Cache is required")?;
        let error_function = self.error_fn.ok_or("Error function is required")?;
        let heuristic = self.heuristic_fn.ok_or("Heuristic is required")?;

        Ok(())
    }
}
