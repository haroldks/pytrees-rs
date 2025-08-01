use crate::algorithms::optimal::rules::core::Reason;
use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};

pub struct PurityRule {
    delta: f64,
    threshold: f64,
    priority: u8,
    state: RuleState,
    relaxable: bool,
}

impl PurityRule {
    pub fn new(initial_threshold: f64, delta: f64) -> Self {
        Self {
            delta,
            threshold: initial_threshold,
            priority: 90,
            state: RuleState::Disabled,
            relaxable: true,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

impl Rule for PurityRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        let purity = 1.0 - context.error.min(context.leaf_error) / context.support as f64;
        if purity >= self.threshold {
            RuleResult::stop_with_bound(f64::INFINITY, Reason::RuleReason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Purity rule".to_string()
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

        if self.is_relaxable() && self.threshold >= 1.0 {
            self.deactivate();
            return;
        }
        self.threshold += self.delta;
        if self.threshold >= 1.0 {
            self.threshold = 1.0
        }
    }

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
