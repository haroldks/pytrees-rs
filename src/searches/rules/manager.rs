use crate::cache::CacheEntry;
use crate::searches::rules::Rule;
use crate::searches::StopReason;

/// A manager for a collection of rules that can be checked against a context
pub struct RuleManager<C> {
    rules: Vec<Box<dyn Rule<C> + 'static>>,
}

impl<C> RuleManager<C> {
    /// Create a new, empty rule manager
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to this manager
    pub fn add_rule<R>(&mut self, rule: R)
    where
        R: Rule<C> + 'static,
    {
        self.rules.push(Box::new(rule))
    }

    /// Check all rules against the context and node
    /// Returns (should_stop, reason) for the first rule that says to stop, or (false, None) if all pass
    pub fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        for rule in &self.rules {
            let (stop, reason) = rule.check(context, node);
            if stop {
                return (stop, reason);
            }
        }
        (false, StopReason::None)
    }

    /// Return true if this manager has no rules
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}
