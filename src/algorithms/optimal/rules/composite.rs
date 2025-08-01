use crate::algorithms::optimal::rules::core::{Rule, RuleContext, RuleResult, RuleState};
use std::fmt::Debug;

pub struct AndRule {
    rules: Vec<Box<dyn Rule>>,
    priority: u8,
    description: String,
    delays: Vec<u8>,
    activation_counts: Vec<u8>,
    delay: u8,
    relaxable: bool,
}

impl Default for AndRule {
    fn default() -> Self {
        Self {
            rules: vec![],
            priority: 100,
            description: "".to_string(),
            delays: vec![],
            activation_counts: vec![],
            delay: 0,
            relaxable: true,
        }
    }
}

impl AndRule {
    pub fn with_rule(mut self, rule: Box<dyn Rule>) -> Self {
        let delay = rule.delay();
        self.rules.push(rule);
        self.delays.push(delay);
        self.activation_counts.push(0);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_delay(mut self, delay: u8) -> Self {
        self.delay = delay;
        self
    }

    pub fn activate_rule(&mut self, index: usize) {
        self.rules[index].activate();
    }

    pub fn is_rule_active(&self, index: usize) {
        self.rules[index].is_active();
    }

    pub fn deactivate_rule(&mut self, index: usize) {
        self.rules[index].deactivate();
    }

    pub fn relax_rule(&mut self, index: usize) {
        if self.rules[index].is_active() && self.activation_counts[index] >= self.delays[index] {
            self.rules[index].relax();
            self.activation_counts[index] = 0;
        }
    }

    pub fn rule_state(&self, index: usize) -> RuleState {
        self.rules[index].state()
    }

    pub fn reset_rule(&mut self, index: usize) {
        self.rules[index].reset()
    }

    pub fn is_delayed(&self, index: usize) -> bool {
        self.activation_counts[index] < self.delays[index]
    }
}

impl Rule for AndRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        for index in 0..self.rules.len() {
            let rule_result = self.rules[index].evaluate(context);
            if !rule_result.continue_search {
                return rule_result;
            }
        }

        RuleResult::continue_search()
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        let mut description = "And Rule of".to_string();
        for rule in &self.rules {
            description.push_str(" ");
            description.push_str(&rule.description())
        }
        description
    }

    fn state(&self) -> RuleState {
        for rule in &self.rules {
            if !rule.is_active() {
                return RuleState::Disabled;
            }
        }
        RuleState::Active
    }

    fn activate(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.activate()
        }
    }

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn deactivate(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.deactivate()
        }
    }

    fn relax(&mut self) {
        for index in 0..self.rules.len() {
            if self.is_delayed(index) {
                self.activation_counts[index] += 1;
            } else {
                self.relax_rule(index);
                self.activation_counts[index] = 0;
            }
        }
    }

    fn reset(&mut self) {
        for index in 0..self.rules.len() {
            self.reset_rule(index)
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
