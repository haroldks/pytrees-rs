use crate::algorithms::common::errors::ErrorWrapper;
use crate::algorithms::common::heuristics::Heuristic;
use crate::algorithms::common::types::{
    BranchingPolicy, CacheInitStrategy, LowerBoundPolicy, NodeDataType, OptimalDepth2Policy,
};
use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
use crate::algorithms::optimal::dl85::config::DL85Config;
use crate::algorithms::optimal::dl85::DL85;
use crate::algorithms::optimal::rules::common::{
    LowerBoundRule, MaxDepthRule, MinSupportRule, PureNodeRule, TimeLimitRule, UsableNodeRule,
};
use crate::algorithms::optimal::rules::{Rule, RuleManager};
use crate::caching::Caching;

pub struct DL85Builder<C, D, E, H   >
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    config: DL85Config,
    cache: Option<Box<C>>,
    depth2_search: Option<Box<D>>,
    error_fn: Option<Box<E>>,
    heuristic_fn: Option<Box<H>>,
    nodes_rules: RuleManager,
    search_rules: RuleManager,
    time_rule: TimeLimitRule,
}

impl<C, D, E, H> Default for DL85Builder<C, D, E, H>
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    fn default() -> Self {
        let builder = Self {
            config: DL85Config::default(),
            cache: None,
            depth2_search: None,
            error_fn: None,
            heuristic_fn: None,
            nodes_rules: RuleManager::new(),
            search_rules: RuleManager::new(),
            time_rule: TimeLimitRule::new(0.0),
        };
        builder.default_rules()
    }
}

impl<C, D, E, H> DL85Builder<C, D, E, H>
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn min_support(mut self, value: usize) -> Self {
        self.config.base.min_support = value;
        self.nodes_rules
            .add_rule(Box::new(MinSupportRule::new(value)));
        self
    }

    pub fn max_depth(mut self, value: usize) -> Self {
        self.config.base.max_depth = value;
        self.nodes_rules
            .add_rule(Box::new(MaxDepthRule::new(value)));
        self
    }

    pub fn max_error(mut self, value: f64) -> Self {
        self.config.base.max_error = value;
        self
    }

    pub fn max_time(mut self, value: f64) -> Self {
        self.config.base.max_time = value;
        self.time_rule = TimeLimitRule::new(value);
        self
    }

    pub fn default_rules(mut self) -> Self {
        self.nodes_rules.add_rule(Box::new(UsableNodeRule::new()));
        self.nodes_rules.add_rule(Box::new(PureNodeRule::new()));
        self.nodes_rules.add_rule(Box::new(LowerBoundRule::new()));
        self
    }

    pub fn depth2_search(mut self, search: Box<D>) -> Self {
        self.depth2_search = Some(search);
        self
    }

    pub fn add_node_rule(mut self, rule: Box<dyn Rule>) -> Self {
        self.nodes_rules.add_rule(rule);
        self
    }

    pub fn add_node_rules(mut self, rules: Vec<Box<dyn Rule>>) -> Self {
        for rule in rules {
            self.nodes_rules.add_rule(rule)
        }
        self
    }

    pub fn add_search_rule(mut self, rule: Box<dyn Rule>) -> Self {
        self.search_rules.add_rule(rule);
        self
    }

    pub fn add_search_rules(mut self, rules: Vec<Box<dyn Rule>>) -> Self {
        for rule in rules {
            self.search_rules.add_rule(rule)
        }
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

    pub fn specialization(mut self, value: OptimalDepth2Policy) -> Self {
        self.config.optimal_depth2policy = value;
        self
    }

    pub fn lower_bound_strategy(mut self, value: LowerBoundPolicy) -> Self {
        self.config.lower_bound_policy = value;
        self
    }

    pub fn branching_strategy(mut self, value: BranchingPolicy) -> Self {
        self.config.branching_policy = value;
        self
    }

    pub fn node_exposed_data(mut self, value: NodeDataType) -> Self {
        self.config.data_type = value;
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

    pub fn build(self) -> Result<DL85<C, D, E, H>, String> {
        let cache = self.cache.ok_or("Cache is required")?;
        let depth2 = self.depth2_search.ok_or("Neee depth 2 algorithm")?;
        let error_function = self.error_fn.ok_or("Error function is required")?;
        let heuristic = self.heuristic_fn.ok_or("Heuristic is required")?;

        Ok(DL85::new(
            self.config,
            cache,
            depth2,
            error_function,
            heuristic,
            self.nodes_rules,
            self.search_rules,
            self.time_rule,
        ))
    }
}
