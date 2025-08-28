pub mod context;
pub mod manager;

use crate::cache::CacheEntry;
use crate::globals::float_is_null;
use crate::searches::optimal::Discrepancy;
use crate::searches::rules::context::{BaseContext, DiscrepancyContext, PurityContext};
use crate::searches::StopReason;

/// A rule that can be checked against a context and node
pub trait Rule<C> {
    /// Check if this rule applies to the given context and node
    /// Returns (should_stop, reason) where should_stop indicates whether to stop searching
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason);

    /// Get the name of this rule
    fn name(&self) -> &'static str;

    /// Whether this rule can be relaxed (defaults to false)
    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self);
}

/// Optimality rule: stop if the node is optimal and has a better or equal bound
pub struct OptimalityRule;

impl<C: BaseContext> Rule<C> for OptimalityRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if node.is_optimal && node.upper_bound >= context.upper_bound() {
            return (true, StopReason::Done);
        } else if node.is_optimal && node.upper_bound < context.upper_bound() {
            // Node is optimal but has worse bound than current best
            node.is_optimal = false;
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Optimality"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Time limit rule: stop if we've exceeded the time limit
pub struct TimeLimitRule;

impl<C: BaseContext> Rule<C> for TimeLimitRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if context.current_time().as_secs_f64() >= context.max_time() {
            node.to_leaf();
            return (true, StopReason::TimeLimitReached);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Time Limit"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Max depth rule: stop if we've reached the maximum depth
pub struct MaxDepthRule;

impl<C: BaseContext> Rule<C> for MaxDepthRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if context.current_depth() >= context.max_depth() {
            node.is_optimal = true;
            node.to_leaf();
            return (true, StopReason::MaxDepthReached);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Max Depth"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Min support rule: stop if we have less than the minimum support
pub struct MinSupportRule;

impl<C: BaseContext> Rule<C> for MinSupportRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if context.support() < context.min_sup() {
            node.to_leaf();
            node.is_optimal = true;
            return (true, StopReason::NotEnoughSupport);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Min Support"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Discrepancy rule: specialized for contexts that track discrepancy
pub struct DiscrepancyRule {
    max_discrepancy: usize,
    function: Box<dyn Discrepancy>,
}

impl<C: DiscrepancyContext> Rule<C> for DiscrepancyRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if node.is_optimal && node.discrepancy >= self.max_discrepancy {
            if node.upper_bound >= context.upper_bound() {
                return (true, StopReason::Done);
            } else if node.upper_bound < context.upper_bound() {
                node.is_optimal = false;
            }
        }

        if context.discrepancy_budget() <= 0 {
            node.to_leaf();
            return (true, StopReason::BranchBudgetExhausted);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Discrepancy"
    }

    fn is_relaxable(&self) -> bool {
        true
    }

    fn relax(&mut self) {
        self.max_discrepancy = self.function.next();
    }
}

/// Lower bound rule: stop if our current best is better than node's lower bound
pub struct LowerBoundRule;

impl<C: BaseContext> Rule<C> for LowerBoundRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        if context.upper_bound() <= node.lower_bound || float_is_null(context.upper_bound()) {
            return (true, StopReason::LowerBoundConstrained);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Lower Bound"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Pure node rule: stop if the node is pure (error is 0 or very small)
pub struct PureNodeRule;

impl<C: BaseContext> Rule<C> for PureNodeRule {
    fn check(&self, _context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        let error = <f64>::min(node.leaf_error, node.error);
        if float_is_null(error) {
            node.to_leaf();
            node.is_optimal = true;
            return (true, StopReason::PureNode);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Pure Node"
    }

    fn is_relaxable(&self) -> bool {
        false
    }

    fn relax(&mut self) {}
}

/// Purity rule: specialized for contexts that have a purity threshold
pub struct PurityRule {
    min_purity: f64,
    delta: f64,
}

impl<C: PurityContext> Rule<C> for PurityRule {
    fn check(&self, context: &mut C, node: &mut CacheEntry) -> (bool, StopReason) {
        let error = f64::min(node.leaf_error, node.error);

        // Pure node check (error == lower_bound)
        if float_is_null(error) {
            node.to_leaf();
            node.is_optimal = true;
            return (true, StopReason::PureNode);
        }

        let purity = 1.0 - <f64>::min(node.leaf_error, node.error) / node.size as f64;

        // Purity threshold check
        if purity >= context.purity_threshold() {
            // node.to_leaf();
            // node.is_optimal = true;
            return (true, StopReason::PureEnough);
        }

        (false, StopReason::None)
    }

    fn name(&self) -> &'static str {
        "Purity"
    }

    fn is_relaxable(&self) -> bool {
        true
    }

    fn relax(&mut self) {
        self.min_purity = <f64>::min(self.min_purity + self.delta, 1.0)
    }
}
