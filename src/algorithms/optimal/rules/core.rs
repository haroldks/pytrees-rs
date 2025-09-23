use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, Eq, PartialEq)]
pub enum Reason {
    Done,
    TimeLimitReached,
    LowerBoundConstrained,
    MaxDepthReached,
    NotEnoughSupport,
    NoCandidates,
    PureNode,
    FromSpecializedAlgorithm,
    RuleReason,
    #[default]
    None,
}

#[derive(Debug, Clone)]
pub struct RuleResult {
    pub continue_search: bool,
    pub modified_bound: Option<f64>,
    pub reason: Reason,
    pub optimal: Option<bool>,
    pub leaf: Option<bool>,
}

impl RuleResult {
    /// Create a new result that allows search to continue
    pub fn continue_search() -> Self {
        Self {
            continue_search: true,
            modified_bound: None,
            reason: Reason::None,
            optimal: None,
            leaf: None,
        }
    }

    /// Create a new result that stops the search
    pub fn stop_search(reason: Reason) -> Self {
        Self {
            continue_search: false,
            modified_bound: None,
            reason,
            optimal: None,
            leaf: None,
        }
    }

    /// Create a new result that continues search with a modified bound
    pub fn stop_with_bound(bound: f64, reason: Reason) -> Self {
        Self {
            continue_search: false,
            modified_bound: Some(bound),
            reason,
            optimal: None,
            leaf: None,
        }
    }

    pub fn optimal(mut self) -> Self {
        self.optimal = Some(true);
        self
    }

    pub fn leaf(mut self) -> Self {
        self.leaf = Some(true);
        self
    }

    /// Add a reason to this result
    pub fn with_reason(mut self, reason: Reason) -> Self {
        self.reason = reason;
        self
    }
}

/// State of a rule (active, relaxed, or disabled)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleState {
    Active,
    Relaxed,
    Disabled,
}

impl Display for RuleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleState::Active => write!(f, "Active"),
            RuleState::Relaxed => write!(f, "Relaxed"),
            RuleState::Disabled => write!(f, "Disabled"),
        }
    }
}

#[derive(Debug)]
pub struct RuleContext {
    pub depth: usize,
    pub upper_bound: f64,
    pub node_lower_bound: f64,
    pub node_upper_bound: f64,
    pub item: usize,
    pub support: usize,
    pub position: usize,
    pub discrepancy: usize,
    pub gain: f64,
    pub error: f64,
    pub leaf_error: f64,
}

impl Default for RuleContext {
    fn default() -> Self {
        Self {
            depth: 0,
            upper_bound: 0.0,
            node_lower_bound: 0.0,
            node_upper_bound: f64::INFINITY,
            item: 0,
            support: 0,
            position: 0,
            discrepancy: 0,
            gain: 0.0,
            error: f64::INFINITY,
            leaf_error: f64::INFINITY,
        }
    }
}

impl RuleContext {
    /// Create a new rule context
    pub fn new(
        depth: usize,
        upper_bound: f64,
        node_lower_bound: f64,
        node_upper_bound: f64,
        item: usize,
        support: usize,
        position: usize,
        gain: f64,
        error: f64,
    ) -> Self {
        Self {
            depth,
            upper_bound,
            node_lower_bound,
            node_upper_bound,
            item,
            support,
            position,
            discrepancy: 0,
            gain,
            error,
            leaf_error: f64::INFINITY,
        }
    }

    pub fn depth(&mut self, depth: usize) {
        self.depth = depth;
    }
    pub fn upper_bound(&mut self, upper_bound: f64) {
        self.upper_bound = upper_bound;
    }

    pub fn node_lower_bound(&mut self, node_lower_bound: f64) {
        self.node_lower_bound = node_lower_bound;
    }

    pub fn node_upper_bound(&mut self, node_upper_bound: f64) {
        self.node_upper_bound = node_upper_bound;
    }

    pub fn item(&mut self, item: usize) {
        self.item = item;
    }

    pub fn support(&mut self, support: usize) {
        self.support = support;
    }

    pub fn position(&mut self, position: usize) {
        self.position = position;
    }

    pub fn gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    pub fn error(&mut self, error: f64) {
        self.error = error;
    }

    pub fn leaf_error(&mut self, error: f64) {
        self.leaf_error = error;
    }

    pub fn discrepancy(&mut self, discrepancy: usize) {
        self.discrepancy = discrepancy;
    }
}

/// A rule that can be applied during tree search
pub trait Rule: std::any::Any + Send + Sync{
    fn evaluate(&self, context: &RuleContext) -> RuleResult;

    fn priority(&self) -> u8;

    fn description(&self) -> String;

    fn state(&self) -> RuleState;

    fn is_active(&self) -> bool {
        self.state() == RuleState::Active
    }

    fn activate(&mut self) {}

    fn is_relaxable(&self) -> bool {
        true
    }

    fn deactivate(&mut self) {}

    fn relax(&mut self) {}

    fn reset(&mut self) {} // Think rule should be check if there is a need to reset for example time limit rule

    fn delay(&self) -> u8 {
        0
    }

    fn as_any(&self) -> &dyn std::any::Any;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
