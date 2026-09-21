use crate::algorithms::optimal::rules::core::Reason;
use crate::algorithms::optimal::rules::helpers::{Monotonic, StepStrategy};
use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};

pub struct DiscrepancyRule {
    limit: usize,
    budget: usize,
    increment: Box<dyn StepStrategy>,
    priority: u8,
    delay: u8,
    state: RuleState,
    relaxable: bool,
}

impl Default for DiscrepancyRule {
    fn default() -> Self {
        Self {
            limit: usize::MAX,
            budget: 0,
            increment: Box::new(Monotonic::new(1)),
            priority: 100,
            delay: 0,
            state: RuleState::Disabled,
            relaxable: true,
        }
    }
}

impl DiscrepancyRule {
    pub fn new(limit: usize, increment: Box<dyn StepStrategy>) -> Self {
        Self {
            limit,
            budget: 0,
            increment,
            priority: 100,
            delay: 0,
            state: RuleState::Disabled,
            relaxable: true,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_delay(mut self, delay: u8) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    pub fn update_to_true_limit(&mut self, nb_candidates: usize, remaining_depth: usize) {
        let mut max_discrepancy = nb_candidates;
        for i in 1..remaining_depth {
            max_discrepancy += nb_candidates.saturating_sub(i);
        }
        self.limit = self.limit.min(max_discrepancy);
        // println!("Limit : {}", self.limit);
    }
}

impl Rule for DiscrepancyRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.discrepancy > self.budget {
            RuleResult::stop_with_bound(f64::INFINITY, Reason::RuleReason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Discrepancy rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.state
    }

    fn activate(&mut self) {
        self.state = RuleState::Active
    }

    fn deactivate(&mut self) {
        self.state = RuleState::Disabled
    }

    fn relax(&mut self) {
        if !self.is_active() {
            return;
        }
        // println!("Bud {} limit {}", self.budget, self.limit);
        if self.is_relaxable() && self.budget >= self.limit {
            self.deactivate();
            return;
        }
        self.budget = self.increment.next();
        if self.budget >= self.limit {
            self.budget = self.limit;
        }
    }

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn delay(&self) -> u8 {
        self.delay
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
