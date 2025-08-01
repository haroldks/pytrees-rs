use crate::algorithms::optimal::rules::core::Reason;
use crate::algorithms::optimal::rules::helpers::StepStrategy;
use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};

pub struct TopkRule {
    limit: usize,
    budget: usize,
    increment: Box<dyn StepStrategy>,
    priority: u8,
    state: RuleState,
    relaxable: bool,
    delay: u8,
}

impl TopkRule {
    pub fn new(limit: usize, increment: Box<dyn StepStrategy>) -> Self {
        Self {
            limit,
            budget: 0,
            increment,
            priority: 90,
            state: RuleState::Active,
            relaxable: true,
            delay: 0,
        }
    }

    pub fn with_delay(mut self, delay: u8) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }
}

impl Rule for TopkRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.position > self.budget {
            RuleResult::stop_with_bound(f64::INFINITY, Reason::RuleReason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "TopK rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.state
    }

    fn is_active(&self) -> bool {
        self.state == RuleState::Active
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

pub struct DecreasingTopkRule {
    limit: usize,
    budget: usize,
    increment: Box<dyn StepStrategy>,
    priority: u8,
    state: RuleState,
    relaxable: bool,
    delay: u8,
}

impl DecreasingTopkRule {
    pub fn new(limit: usize, increment: Box<dyn StepStrategy>) -> Self {
        Self {
            limit,
            budget: 0,
            increment,
            priority: 90,
            state: RuleState::Active,
            relaxable: true,
            delay: 0,
        }
    }

    pub fn with_delay(mut self, delay: u8) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }
}

impl Rule for DecreasingTopkRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        let depth_budget = (self.budget / (2.0_f64.powi(context.depth as i32) as usize)).max(1);
        if context.position > depth_budget {
            RuleResult::stop_with_bound(f64::INFINITY, Reason::RuleReason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "TopK rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.state
    }

    fn is_active(&self) -> bool {
        self.state == RuleState::Active
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
