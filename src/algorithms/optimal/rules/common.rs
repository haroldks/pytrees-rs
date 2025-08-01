use crate::algorithms::optimal::rules::core::Reason;
use crate::algorithms::optimal::rules::{Rule, RuleContext, RuleResult, RuleState};
use crate::globals::float_is_null;
use std::time::Instant;

#[derive(Debug)]
pub struct MaxDepthRule {
    max_depth: usize,
    priority: u8,
}

impl MaxDepthRule {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            priority: 98,
        }
    }
}

impl Rule for MaxDepthRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.depth >= self.max_depth {
            RuleResult::stop_with_bound(context.upper_bound, Reason::MaxDepthReached)
                .optimal()
                .leaf()
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        format!("Max depth {}", self.max_depth)
    }

    fn state(&self) -> RuleState {
        RuleState::Active
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct MinSupportRule {
    min_support: usize,
    priority: u8,
}

impl MinSupportRule {
    pub fn new(min_support: usize) -> Self {
        Self {
            min_support,
            priority: 97,
        }
    }
}

impl Rule for MinSupportRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if context.support < self.min_support {
            RuleResult::stop_with_bound(context.upper_bound, Reason::NotEnoughSupport)
                .optimal()
                .leaf()
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        format!("Min support {}", self.min_support)
    }

    fn state(&self) -> RuleState {
        RuleState::Active
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct TimeLimitRule {
    time_limit: f64,
    priority: u8,
    start_time: Instant,
    current_state: RuleState,
    relaxable: bool,
}

impl Default for TimeLimitRule {
    fn default() -> Self {
        Self::new(f64::INFINITY)
    }
}

impl TimeLimitRule {
    /// Create a new rule that stops search after the specified time limit (in seconds)
    pub fn new(time_limit: f64) -> Self {
        Self {
            time_limit,
            priority: 100,
            start_time: Instant::now(),
            current_state: RuleState::Disabled,
            relaxable: false,
        }
    }
    pub fn relaxable(mut self) -> Self {
        self.relaxable = true;
        self
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn remaining_seconds(&self) -> f64 {
        (self.time_limit - self.elapsed_seconds()).max(0.0)
    }

    pub fn exhausted(&self) -> bool {
        self.elapsed_seconds() >= self.time_limit
    }
}

impl Rule for TimeLimitRule {
    fn evaluate(&self, _context: &RuleContext) -> RuleResult {
        if !self.is_active() {
            return RuleResult::continue_search();
        }

        if self.exhausted() {
            let reason = match self.is_relaxable() {
                true => Reason::RuleReason,
                false => Reason::TimeLimitReached,
            };
            RuleResult::stop_with_bound(f64::INFINITY, reason)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Time limit rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.current_state
    }

    fn activate(&mut self) {
        self.current_state = RuleState::Active;
        self.reset();
    }

    fn deactivate(&mut self) {
        self.current_state = RuleState::Disabled;
    }

    fn reset(&mut self) {
        self.start_time = Instant::now()
    }

    fn is_relaxable(&self) -> bool {
        self.relaxable
    }

    fn relax(&mut self) {
        if self.is_relaxable() {
            self.reset()
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct LowerBoundRule {
    priority: u8,
    current_state: RuleState,
}

impl Default for LowerBoundRule {
    fn default() -> Self {
        Self::new()
    }
}

impl LowerBoundRule {
    pub fn new() -> Self {
        Self {
            priority: 100,
            current_state: RuleState::Disabled,
        }
    }
}

impl Rule for LowerBoundRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if !self.is_active() {
            return RuleResult::continue_search();
        }

        if context.upper_bound <= context.node_lower_bound || float_is_null(context.upper_bound) {
            RuleResult::stop_search(Reason::LowerBoundConstrained)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Lower bound rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.current_state
    }

    fn activate(&mut self) {
        self.current_state = RuleState::Active;
    }

    fn deactivate(&mut self) {
        self.current_state = RuleState::Disabled;
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct UsableNodeRule {
    priority: u8,
    current_state: RuleState,
}

impl Default for UsableNodeRule {
    fn default() -> Self {
        Self::new()
    }
}

impl UsableNodeRule {
    pub fn new() -> Self {
        Self {
            priority: 101,
            current_state: RuleState::Disabled,
        }
    }
}

impl Rule for UsableNodeRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if !self.is_active() {
            return RuleResult::continue_search();
        }

        if context.error.is_finite() && context.node_upper_bound.is_finite() {
            RuleResult::stop_search(Reason::Done)
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Usable node rule".to_string()
    }

    fn state(&self) -> RuleState {
        RuleState::Active
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct PureNodeRule {
    priority: u8,
}

impl Default for PureNodeRule {
    fn default() -> Self {
        Self::new()
    }
}

impl PureNodeRule {
    pub fn new() -> Self {
        Self { priority: 99 }
    }
}

impl Rule for PureNodeRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if float_is_null(context.error) {
            RuleResult::stop_with_bound(context.upper_bound, Reason::PureNode)
                .optimal()
                .leaf()
        } else {
            RuleResult::continue_search()
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Pure node rule".to_string()
    }

    fn state(&self) -> RuleState {
        RuleState::Active
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct SimilarityLowerBoundRule {
    priority: u8,
    current_state: RuleState,
}

impl Default for SimilarityLowerBoundRule {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityLowerBoundRule {
    pub fn new() -> Self {
        Self {
            priority: 100,
            current_state: RuleState::Disabled,
        }
    }
}

impl Rule for SimilarityLowerBoundRule {
    fn evaluate(&self, context: &RuleContext) -> RuleResult {
        if !self.is_active() {
            return RuleResult::continue_search();
        }
        if context.node_lower_bound >= context.upper_bound {
            return RuleResult::stop_search(Reason::LowerBoundConstrained);
        }
        if context.error <= context.node_lower_bound {
            return RuleResult::stop_search(Reason::PureNode).leaf();
        }
        RuleResult::continue_search()
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn description(&self) -> String {
        "Similarity Lower Bound Rule".to_string()
    }

    fn state(&self) -> RuleState {
        self.current_state
    }

    fn is_active(&self) -> bool {
        self.current_state == RuleState::Active
    }

    fn activate(&mut self) {
        self.current_state = RuleState::Active
    }

    fn deactivate(&mut self) {
        self.current_state = RuleState::Disabled
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
