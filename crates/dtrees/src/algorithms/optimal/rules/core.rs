use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Why the search stopped at a node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, Eq, PartialEq)]
pub enum Reason {
    /// The node was fully explored.
    Done,
    /// The time limit was reached.
    TimeLimitReached,
    /// The node's lower bound reaches its upper bound: no subtree can help.
    LowerBoundConstrained,
    /// The node is at the maximum depth and becomes a leaf.
    MaxDepthReached,
    /// The node has too few instances to be split.
    NotEnoughSupport,
    /// No feature can split the node.
    NoCandidates,
    /// The node has zero error as a leaf.
    PureNode,
    /// The node was solved by the depth-2 solver.
    FromSpecializedAlgorithm,
    /// A relaxable search rule cut the search; a later pass may go further.
    RuleReason,
    /// No reason recorded.
    #[default]
    None,
}

/// What a rule decided for a node.
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// Whether the search goes on below the node.
    pub continue_search: bool,
    /// A new upper bound to record on the node, if any.
    pub modified_bound: Option<f64>,
    /// Why the search stops, when it does.
    pub reason: Reason,
    /// Mark the node as solved optimally.
    pub optimal: Option<bool>,
    /// Turn the node into a leaf.
    pub leaf: Option<bool>,
}

impl RuleResult {
    /// Lets the search continue.
    pub fn continue_search() -> Self {
        Self {
            continue_search: true,
            modified_bound: None,
            reason: Reason::None,
            optimal: None,
            leaf: None,
        }
    }

    /// Stops the search at the node.
    pub fn stop_search(reason: Reason) -> Self {
        Self {
            continue_search: false,
            modified_bound: None,
            reason,
            optimal: None,
            leaf: None,
        }
    }

    /// Stops the search at the node and records `bound` as its upper bound.
    pub fn stop_with_bound(bound: f64, reason: Reason) -> Self {
        Self {
            continue_search: false,
            modified_bound: Some(bound),
            reason,
            optimal: None,
            leaf: None,
        }
    }

    /// Also marks the node as solved optimally.
    pub fn optimal(mut self) -> Self {
        self.optimal = Some(true);
        self
    }

    /// Also turns the node into a leaf.
    pub fn leaf(mut self) -> Self {
        self.leaf = Some(true);
        self
    }

    /// Replaces the reason.
    pub fn with_reason(mut self, reason: Reason) -> Self {
        self.reason = reason;
        self
    }
}

/// Whether a rule is applied.
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

/// What the rules know about the node being evaluated.
#[derive(Debug)]
pub struct RuleContext {
    /// Depth of the node (the root is at 0).
    pub depth: usize,
    /// Error the subtree must beat to be of use to its parent.
    pub upper_bound: f64,
    /// Proven lower bound on the node's error.
    pub node_lower_bound: f64,
    /// Upper bound the node was last solved under (infinite if never).
    pub node_upper_bound: f64,
    /// The item (feature and branch) leading to the node.
    pub item: usize,
    /// Number of instances in the node.
    pub support: usize,
    /// Rank of the node's feature among its siblings.
    pub position: usize,
    /// Sum of the ranks along the path from the root.
    pub discrepancy: usize,
    /// Accumulated heuristic score lost along the path by not taking the
    /// best-ranked feature.
    pub gain: f64,
    /// Best error found so far for the node.
    pub error: f64,
    /// Error of the node as a leaf.
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

/// A condition checked at every node of the search.
pub trait Rule: std::any::Any + Send + Sync {
    /// Decides whether the search continues below the node.
    fn evaluate(&self, context: &RuleContext) -> RuleResult;

    /// Rules with a higher priority are evaluated first.
    fn priority(&self) -> u8;

    /// A short human-readable name.
    fn description(&self) -> String;

    /// The current state of the rule.
    fn state(&self) -> RuleState;

    /// Whether the rule is applied.
    fn is_active(&self) -> bool {
        self.state() == RuleState::Active
    }

    /// Starts applying the rule; called before the first pass.
    fn activate(&mut self) {}

    /// Whether the rule restricts the search only temporarily. A search whose
    /// pass was cut by a relaxable rule is not finished.
    fn is_relaxable(&self) -> bool {
        true
    }

    /// Stops applying the rule.
    fn deactivate(&mut self) {}

    /// Widens the rule's budget before the next pass, or deactivates it once
    /// the budget is at its limit.
    fn relax(&mut self) {}

    /// Resets any internal state, such as a timer.
    fn reset(&mut self) {}

    /// Number of passes to wait before the rule takes effect.
    fn delay(&self) -> u8 {
        0
    }

    /// Upcast used by [`RuleManager::get_rule_mut`](super::RuleManager::get_rule_mut).
    fn as_any(&self) -> &dyn std::any::Any;

    /// Mutable upcast used by [`RuleManager::get_rule_mut`](super::RuleManager::get_rule_mut).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
