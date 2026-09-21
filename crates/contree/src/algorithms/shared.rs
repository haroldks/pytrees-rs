//! Pieces both searches use verbatim.
//!
//! `ConTree` and `ConTreeLds` are the same branch and bound, one of them with a
//! budget. Where the two genuinely differ -- how a feature or a split is
//! admitted -- they still have their own code; everything here was duplicated
//! character for character between them, which is exactly the kind of thing
//! that drifts apart under maintenance.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::Rng;

use crate::algorithms::interval_pruner::Bound;
use crate::caching::{Cache, Entry};
use crate::common::PointSelector;
use crate::tree::{NodeInfos, Tree, TreeNode};

/// Picks the split to probe inside an interval.
pub(crate) fn select_point(selector: PointSelector, rng: &mut StdRng, bound: &Bound) -> usize {
    match selector {
        PointSelector::Mid => (bound.left_bound + bound.right_bound) / 2,
        PointSelector::First => bound.left_bound,
        PointSelector::Random => rng.random_range(bound.left_bound..=bound.right_bound),
    }
}

pub(crate) fn elapsed_time(runtime: &Instant) -> f64 {
    runtime.elapsed().as_secs_f64()
}

pub(crate) fn time_remains(runtime: &Instant, max_time: f64) -> bool {
    elapsed_time(runtime) < max_time
}

/// Converts a cache entry into a tree node.
///
/// `Entry` marks a leaf with the sentinels `feature == usize::MAX` and
/// `split == INFINITY`; the tree marks one with `feature: None`. This is where
/// the two meet.
fn node_from_entry(entry: &Entry) -> NodeInfos {
    let is_leaf = entry.feature == usize::MAX || !entry.split.is_finite();
    NodeInfos {
        feature: (!is_leaf).then_some(entry.feature),
        split: (!is_leaf).then_some(entry.split),
        error: entry.error,
        label: Some(entry.label),
    }
}

/// Reconstructs the solution tree from the cache, in canonical leaf form.
///
/// Empty when the cache holds no root -- that is, when `fit` has not run.
pub(crate) fn build_solution_tree(cache: &Cache) -> Tree {
    let mut solution = Tree::new();
    if let Some(root) = cache.root() {
        if let Some(tree_idx) = root.tree_idx {
            if let Some(tree) = cache.get_tree(tree_idx) {
                solution = tree.clone();
            }
        } else {
            let infos = node_from_entry(root);
            let root_index = solution.add_root(TreeNode::new(infos));
            expand_children(cache, &mut solution, root_index, cache.root_index());
        }
    }
    solution.normalize_leaves();
    solution
}

fn expand_children(cache: &Cache, solution: &mut Tree, parent: usize, cache_index: usize) {
    let branches = cache.get_children(cache_index);
    for (branch, &child_index) in branches.iter().enumerate() {
        // A cache index of 0 means "no child": index 0 is the root.
        if child_index == 0 {
            continue;
        }
        let Some(entry) = cache.get(child_index) else {
            continue;
        };

        match entry.tree_idx {
            // The depth-2 solver stores a whole subtree rather than cache
            // entries, so it is grafted in one piece.
            Some(tree_idx) => {
                if let Some(sub_tree) = cache.get_tree(tree_idx) {
                    let infos = sub_tree.root_details();
                    let child = solution.add_node(parent, branch == 0, TreeNode::new(infos));
                    solution.update_subtree(child, sub_tree, sub_tree.get_root_index());
                }
            }
            None => {
                let infos = node_from_entry(entry);
                let child = solution.add_node(parent, branch == 0, TreeNode::new(infos));
                if !entry.is_leaf {
                    expand_children(cache, solution, child, child_index);
                }
            }
        }
    }
}
