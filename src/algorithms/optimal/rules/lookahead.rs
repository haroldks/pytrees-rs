use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};
use crate::algorithms::optimal::Reason;
use std::any::Any;

#[derive(Debug)]
pub struct LookaheadRule {
    priority: u8,
    lookahead_depth: usize,
    depth_limit: usize,
    state: RuleState,
    relaxable: bool,
    delay: u8,
    activation_count: u8,
}

impl LookaheadRule {
    pub fn new(lookahead_depth: usize, depth_limit: usize, relaxable: bool) -> Self {
        Self {
            priority: 97,
            lookahead_depth,
            depth_limit,
            state: RuleState::Disabled,
            relaxable,
            delay: 0,
            activation_count: 0,
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

    pub fn is_delayed(&self) -> bool {
        self.activation_count < self.delay
    }

    pub fn update_delay(&mut self, delay: u8) {
        self.delay = delay
    }

    pub fn depth(&self) -> usize {
        self.lookahead_depth
    }
}

impl Rule for LookaheadRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.depth >= self.lookahead_depth {
            if self.lookahead_depth >= self.depth_limit {
                return RuleResult::stop_search(Reason::LookaheadDepthReachedDone);
            }
            return RuleResult::stop_search(Reason::LookaheadDepthReached);
        }
        RuleResult::continue_search()
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        format!("Lookahead depth : {}", self.lookahead_depth)
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

        if self.is_delayed() {
            self.activation_count += 1;
            return;
        } else {
            self.activation_count = 0;
        }

        if self.is_relaxable() && self.lookahead_depth >= self.depth_limit {
            self.deactivate();
            return;
        }
        self.lookahead_depth += 1;
    }

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
