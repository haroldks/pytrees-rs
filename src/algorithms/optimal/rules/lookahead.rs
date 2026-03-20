use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};
use crate::algorithms::optimal::Reason;
use std::any::Any;

#[derive(Debug)]
pub struct LookaheadRule {
    priority: u8,
    lookahead_depth: usize,
    current_state: RuleState,
}

impl LookaheadRule {
    pub fn new(lookahead_depth: usize) -> Self {
        Self {
            priority: 97,
            lookahead_depth,
            current_state: RuleState::Disabled,
        }
    }
}

impl Rule for LookaheadRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.depth >= self.lookahead_depth {
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
        self.current_state
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
