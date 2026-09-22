use crate::algorithms::optimal::rules::core::{Rule, RuleContext, RuleResult};

/// An ordered set of rules, evaluated by decreasing priority.
pub struct RuleManager {
    rules: Vec<Box<dyn Rule>>,
}

impl Default for RuleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleManager {
    /// An empty set of rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Activates every rule.
    pub fn activate_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.activate();
        }
    }

    /// Adds a rule, keeping the set sorted by priority.
    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        self.rules.push(rule);
        self.rules
            .sort_by_key(|rule| std::cmp::Reverse(rule.priority()));
    }

    /// Relaxes every rule before a new pass.
    pub fn relax_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.relax();
        }
    }

    /// Resets every rule.
    pub fn reset_all(&mut self) {
        for rule in self.rules.iter_mut() {
            rule.reset();
        }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Removes every rule.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Whether some relaxable rule is still active, i.e. whether the search
    /// may still be restricted.
    pub fn is_active(&self) -> bool {
        for rule in &self.rules {
            if rule.is_relaxable() && rule.is_active() {
                return true;
            }
        }
        false
    }

    /// Evaluates the rules in order and returns the first one that stops the
    /// search, or a result that lets it continue.
    pub fn evaluate(&self, context: &RuleContext) -> RuleResult {
        for rule in &self.rules {
            let result = rule.evaluate(context);
            if !result.continue_search {
                return result;
            }
        }
        RuleResult::continue_search()
    }

    /// `(description, state)` of every rule.
    pub fn get_rule_states(&self) -> Vec<(String, String)> {
        self.rules
            .iter()
            .map(|rule| (rule.description(), format!("{}", rule.state())))
            .collect()
    }

    /// The first rule of type `T`, if any.
    pub fn get_rule_mut<T: 'static>(&mut self) -> Option<&mut T> {
        for rule in &mut self.rules {
            if let Some(concrete_rule) = rule.as_any_mut().downcast_mut::<T>() {
                return Some(concrete_rule);
            }
        }
        None
    }
}
