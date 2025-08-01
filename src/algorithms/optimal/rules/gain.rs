use crate::algorithms::optimal::rules::core::Reason;
use crate::algorithms::optimal::rules::helpers::StepStrategy;
use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};

pub struct GainRule {
    gap: f64,
    delta: f64,
    limit: f64,
    increment: Box<dyn StepStrategy>,
    priority: u8,
    delay: u8,
    relaxable: bool,
    state: RuleState,
}

impl GainRule {
    pub fn new(gap: f64, delta: f64, limit: f64, increment: Box<dyn StepStrategy>) -> Self {
        Self {
            gap,
            delta,
            limit,
            increment,
            priority: 91,
            delay: 0,
            relaxable: true,
            state: RuleState::Disabled,
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

    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }

    pub fn with_limit(mut self, limit: f64) -> Self {
        self.limit = limit;
        self
    }

    pub fn update_gap_delta(&mut self, delta: f64) {
        if delta <= 0.0 {
            return;
        }
        self.delta = delta;
    }

    pub fn update_limit(&mut self, limit: f64) {
        self.limit = self.limit.min(limit);
    }
}

impl Rule for GainRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if !self.is_active() {
            return RuleResult::continue_search();
        }
        // println!("gap {} gain {}", self.gap, context.gain);
        if context.gain > self.gap {
            RuleResult::stop_with_bound(f64::INFINITY, Reason::RuleReason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Gain rule".to_string()
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

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn relax(&mut self) {
        if !self.is_active() {
            return;
        }
        if self.is_relaxable() && self.gap >= self.limit {
            self.deactivate();
            return;
        }
        self.gap = self.delta * self.increment.next() as f64;
        if self.gap >= self.limit {
            self.gap = self.limit
        }
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
