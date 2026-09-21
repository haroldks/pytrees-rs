// use crate::algorithms::optimal::rules::core::{Rule, RuleContext, RuleResult, RuleState};
// use std::fmt::Debug;
//
// /// A wrapper around a rule that allows it to be relaxed at a specific relaxation level
// #[derive(Debug)]
// pub struct RelaxableRule<R: Rule> {
//     /// The inner rule being wrapped
//     inner: R,
//
//     /// The relaxation threshold at which this rule becomes relaxed
//     relaxation_threshold: usize,
//
//     /// Current state of the rule (Active, Relaxed, or Disabled)
//     state: RuleState,
//
//     /// Whether to preserve the inner rule's priority or use a custom one
//     priority: Option<u8>,
//
//     /// Description prefix to add to the inner rule's description
//     description_prefix: Option<String>,
// }
//
// impl<R: Rule> RelaxableRule<R> {
//     /// Create a new relaxable rule with the given rule and relaxation threshold
//     pub fn new(rule: R, relaxation_threshold: usize) -> Self {
//         Self {
//             inner: rule,
//             relaxation_threshold,
//             state: RuleState::Active,
//             priority: None,
//             description_prefix: None,
//         }
//     }
//
//     /// Set a custom priority for this relaxable rule
//     pub fn with_priority(mut self, priority: u8) -> Self {
//         self.priority = Some(priority);
//         self
//     }
//
//     /// Add a prefix to the inner rule's description
//     pub fn with_description_prefix(mut self, prefix: impl Into<String>) -> Self {
//         self.description_prefix = Some(prefix.into());
//         self
//     }
//
//     /// Update the state based on the current relaxation level
//     pub fn update_state(&mut self, relaxation_level: usize) {
//         if relaxation_level >= self.relaxation_threshold {
//             self.state = RuleState::Relaxed;
//         } else {
//             self.state = RuleState::Active;
//         }
//     }
//
//     /// Disable this rule entirely
//     pub fn disable(&mut self) {
//         self.state = RuleState::Disabled;
//     }
//
//     /// Enable this rule (make it active)
//     pub fn enable(&mut self) {
//         self.state = RuleState::Active;
//     }
//
//     /// Get a reference to the inner rule
//     pub fn inner(&self) -> &R {
//         &self.inner
//     }
//
//     /// Get a mutable reference to the inner rule
//     pub fn inner_mut(&mut self) -> &mut R {
//         &mut self.inner
//     }
//
//     /// Get the current relaxation threshold
//     pub fn relaxation_threshold(&self) -> usize {
//         self.relaxation_threshold
//     }
//
//     /// Set a new relaxation threshold
//     pub fn set_relaxation_threshold(&mut self, threshold: usize) {
//         self.relaxation_threshold = threshold;
//     }
// }
//
// impl<R: Rule> Rule for RelaxableRule<R> {
//     fn evaluate(&self, context: &RuleContext) -> RuleResult {
//         match self.state {
//             RuleState::Active => {
//                 // Rule is active, evaluate normally
//                 self.inner.evaluate(context)
//             }
//             RuleState::Relaxed | RuleState::Disabled => {
//                 // Rule is relaxed or disabled, always continue search
//                 RuleResult::continue_search().with_reason(format!("{} is {}", self.description(), self.state))
//             }
//         }
//     }
//
//     fn priority(&self) -> u8 {
//         // Use custom priority if set, otherwise use inner rule's priority
//         self.priority.unwrap_or_else(|| self.inner.priority())
//     }
//
//     fn description(&self) -> &str {
//         // We can't easily combine the descriptions, so just use the inner rule's description
//         // In a more complex implementation, we could store the combined description
//         self.inner.description()
//     }
//
//     fn is_relaxed(&self, relaxation_level: usize) -> bool {
//         relaxation_level >= self.relaxation_threshold
//     }
//
//     fn relaxation_threshold(&self) -> usize {
//         self.relaxation_threshold
//     }
//
//     fn state(&self) -> RuleState {
//         self.state
//     }
// }
//
// /// A manager for handling rule relaxation as search progresses
// #[derive(Debug)]
// pub struct RelaxationManager {
//     /// Current relaxation level
//     current_level: usize,
//
//     /// Maximum relaxation level
//     max_level: usize,
//
//     /// Whether to automatically increment relaxation level when search fails
//     auto_increment: bool,
//
//     /// Number of failures before incrementing relaxation level (when auto_increment is true)
//     failures_until_relaxation: usize,
//
//     /// Current number of failures at this relaxation level
//     current_failures: usize,
// }
//
// impl RelaxationManager {
//     /// Create a new relaxation manager with default settings
//     pub fn new() -> Self {
//         Self {
//             current_level: 0,
//             max_level: usize::MAX,
//             auto_increment: true,
//             failures_until_relaxation: 10, // Default: increment after 10 failures
//             current_failures: 0,
//         }
//     }
//
//     /// Create a new relaxation manager with the specified maximum level
//     pub fn with_max_level(max_level: usize) -> Self {
//         Self {
//             current_level: 0,
//             max_level,
//             auto_increment: true,
//             failures_until_relaxation: 10,
//             current_failures: 0,
//         }
//     }
//
//     /// Set whether to automatically increment relaxation level
//     pub fn with_auto_increment(mut self, auto_increment: bool) -> Self {
//         self.auto_increment = auto_increment;
//         self
//     }
//
//     /// Set the number of failures needed before incrementing relaxation level
//     pub fn with_failures_until_relaxation(mut self, failures: usize) -> Self {
//         self.failures_until_relaxation = failures;
//         self
//     }
//
//     /// Get the current relaxation level
//     pub fn current_level(&self) -> usize {
//         self.current_level
//     }
//
//     /// Manually increment the relaxation level
//     pub fn increment_level(&mut self) -> bool {
//         if self.current_level < self.max_level {
//             self.current_level += 1;
//             self.current_failures = 0;
//             true
//         } else {
//             false
//         }
//     }
//
//     /// Reset relaxation level to zero
//     pub fn reset(&mut self) {
//         self.current_level = 0;
//         self.current_failures = 0;
//     }
//
//     /// Report a search failure and possibly increment relaxation level if auto_increment is enabled
//     pub fn report_failure(&mut self) -> bool {
//         self.current_failures += 1;
//
//         if self.auto_increment && self.current_failures >= self.failures_until_relaxation {
//             return self.increment_level();
//         }
//
//         false
//     }
//
//     /// Check if relaxation level can be further increased
//     pub fn can_relax_further(&self) -> bool {
//         self.current_level < self.max_level
//     }
// }
//
// /// A rule that automatically adjusts its behavior based on the relaxation level
// #[derive(Debug)]
// pub struct AdaptiveRule<F>
// where
//     F: Fn(&RuleContext, usize) -> RuleResult + Debug
// {
//     /// The evaluation function that adapts to relaxation level
//     evaluation_fn: F,
//
//     /// Priority of this rule
//     priority: u8,
//
//     /// Description of this rule
//     description: String,
// }
//
// impl<F> AdaptiveRule<F>
// where
//     F: Fn(&RuleContext, usize) -> RuleResult + Debug
// {
//     /// Create a new adaptive rule with the given evaluation function
//     pub fn new(evaluation_fn: F, description: impl Into<String>) -> Self {
//         Self {
//             evaluation_fn,
//             priority: 20,
//             description: description.into(),
//         }
//     }
//
//     /// Set the priority of this rule
//     pub fn with_priority(mut self, priority: u8) -> Self {
//         self.priority = priority;
//         self
//     }
// }
//
// impl<F> Rule for AdaptiveRule<F>
// where
//     F: Fn(&RuleContext, usize) -> RuleResult + Debug
// {
//     fn evaluate(&self, context: &RuleContext) -> RuleResult {
//         // Call the evaluation function with current relaxation level
//         (self.evaluation_fn)(context, context.relaxation_level)
//     }
//
//     fn priority(&self) -> u8 {
//         self.priority
//     }
//
//     fn description(&self) -> &str {
//         &self.description
//     }
//
//     fn is_relaxed(&self, _relaxation_level: usize) -> bool {
//         // Adaptive rules handle relaxation internally in their evaluation function
//         false
//     }
//
//     fn relaxation_threshold(&self) -> usize {
//         // Not applicable for adaptive rules
//         usize::MAX
//     }
//
//     fn state(&self) -> RuleState {
//         // Adaptive rules are always considered active
//         RuleState::Active
//     }
// }
