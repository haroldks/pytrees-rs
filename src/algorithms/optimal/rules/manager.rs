use crate::algorithms::optimal::rules::core::{Rule, RuleContext, RuleResult};

pub struct RuleManager {
    rules: Vec<Box<dyn Rule>>,
}

impl RuleManager {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn activate_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.activate();
        }
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    pub fn relax_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.relax();
        }
    }

    pub fn reset_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.reset();
        }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    pub fn is_active(&self) -> bool {
        for rule in &self.rules {
            if rule.is_relaxable() && rule.is_active() {
                return true;
            }
        }
        false
    }

    pub fn evaluate(&self, context: &RuleContext) -> RuleResult {
        for rule in &self.rules {
            let result = rule.evaluate(context);
            // println!("Context : {:?} Rule : {:?} result {:?}", context, rule.description(), result);
            if !result.continue_search {
                return result;
            }
        }
        RuleResult::continue_search()
    }

    pub fn get_rule_states(&self) -> Vec<(String, String)> {
        self.rules
            .iter()
            .map(|rule| (rule.description().to_string(), format!("{}", rule.state())))
            .collect()
    }

    pub fn get_rule_mut<T: 'static>(&mut self) -> Option<&mut T> {
        for rule in &mut self.rules {
            if let Some(concrete_rule) = rule.as_any_mut().downcast_mut::<T>() {
                return Some(concrete_rule);
            }
        }
        None
    }
}
