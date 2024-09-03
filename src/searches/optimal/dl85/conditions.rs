use crate::cache::CacheEntry;
use crate::globals::float_is_null;
use crate::searches::utils::StopReason;
use std::time::Duration;

#[derive(Default)]
pub struct StopConditions;

impl StopConditions {
    pub(crate) fn check(
        &self,
        node: &mut CacheEntry,
        support: usize,
        min_sup: usize,
        current_depth: usize,
        max_depth: usize,
        current_time: Duration,
        max_time: usize,
        upper_bound: f64,
        discrepancy: Option<usize>,
        discrepancy_budget: Option<usize>,
    ) -> (bool, StopReason) {
        if self.time_limit_reached(current_time, max_time, node) {
            if let Some(dis) = discrepancy {
                node.discrepancy = dis
            }
            return (true, StopReason::TimeLimitReached);
        }

        if self.max_depth_reached(current_depth, max_depth, node) {
            if let Some(dis) = discrepancy {
                node.discrepancy = dis
            }
            return (true, StopReason::MaxDepthReached);
        }

        if self.not_enough_support(support, min_sup, node) {
            if let Some(dis) = discrepancy {
                node.discrepancy = dis
            }
            return (true, StopReason::NotEnoughSupport);
        }

        if self.pure_node(node) {
            return (true, StopReason::PureNode);
        }

        if self.lower_bound_constrained(upper_bound, node) {
            if let Some(dis) = discrepancy_budget {
                if node.discrepancy >= dis {
                    return (true, StopReason::PureNode); // TODO : Change this to another condition
                }
            }
            return (true, StopReason::LowerBoundConstrained);
        }

        (false, StopReason::None)
    }

    pub(crate) fn stop_from_lower_bound(
        &self,
        node: &mut CacheEntry,
        actual_upper_bound: f64,
    ) -> (bool, StopReason) {
        if node.lower_bound >= actual_upper_bound {
            return (true, StopReason::LowerBoundConstrained);
        }
        if node.leaf_error <= node.lower_bound {
            node.to_leaf();
            return (true, StopReason::PureNode);
        }
        (false, StopReason::None)
    }

    fn time_limit_reached(
        &self,
        current_time: Duration,
        max_time: usize,
        node: &mut CacheEntry,
    ) -> bool {
        current_time.as_secs() as usize >= max_time && {
            node.to_leaf();
            true
        }
    }

    fn lower_bound_constrained(&self, actual_upper_bound: f64, node: &mut CacheEntry) -> bool {
        node.lower_bound >= actual_upper_bound || float_is_null(actual_upper_bound)
    }

    fn max_depth_reached(&self, depth: usize, max_depth: usize, node: &mut CacheEntry) -> bool {
        depth == max_depth && {
            node.to_leaf();
            true
        }
    }

    fn not_enough_support(&self, support: usize, min_sup: usize, node: &mut CacheEntry) -> bool {
        support < min_sup * 2 && {
            node.to_leaf();
            true
        }
    }

    fn pure_node(&self, node: &mut CacheEntry) -> bool {
        float_is_null(node.leaf_error - node.lower_bound) && {
            node.to_leaf();
            true
        }
    }

    // fn discrepancy_lower_bound(&self, budget: usize, actual_upper_bound: usize,  node: &mut CacheEntry) -> bool {
    //     match Some(budget){
    //         None => false,
    //         Some(dis) => node.discrepancy >= dis && self.lower_bound_constrained(actual_upper_bound, node)
    //     }
    // }
}
