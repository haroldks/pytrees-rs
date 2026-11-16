//! The greedy completion of SPLIT: below its lookahead, a node is completed
//! by a tree grown top-down, each node taking the threshold with the lowest
//! weighted Gini impurity.

use crate::algorithms::{shared, support_feasible_splits};
use crate::common::{Cost, Loss};
use crate::data::view::DataView;
use crate::tree::{NodeInfos, Tree, TreeNode};

/// A greedy subtree before it is written into a [`Tree`].
enum Grown {
    Leaf {
        cost: Cost,
        label: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        cost: Cost,
        label: usize,
        left: Box<Grown>,
        right: Box<Grown>,
    },
}

impl Grown {
    fn cost(&self) -> Cost {
        match self {
            Grown::Leaf { cost, .. } | Grown::Split { cost, .. } => *cost,
        }
    }
}

/// The greedy tree of depth at most `depth` over `view`, and its cost.
///
/// Every leaf holds at least `min_sup` instances. A split is kept only when
/// its two subtrees cost less than a single leaf, so the tree never costs
/// more than the leaf. Leaf costs go through [`shared::checked_leaf`], which
/// records the first invalid cost in `invalid`.
pub(crate) fn greedy_tree(
    view: &DataView<'_>,
    depth: usize,
    min_sup: usize,
    loss: &Loss,
    invalid: &mut Option<Cost>,
) -> (Tree, Cost) {
    let grown = grow(view, depth, min_sup, loss, invalid);
    let cost = grown.cost();
    let mut tree = Tree::new();
    write(&grown, &mut tree, 0, false);
    tree.normalize_leaves();
    (tree, cost)
}

fn grow(
    view: &DataView<'_>,
    depth: usize,
    min_sup: usize,
    loss: &Loss,
    invalid: &mut Option<Cost>,
) -> Grown {
    let (cost, label) = shared::checked_leaf(
        loss,
        view.get_labels_freqs(),
        || view.instance_ids(),
        invalid,
    );
    let leaf = Grown::Leaf { cost, label };
    if depth == 0 || cost == 0.0 {
        return leaf;
    }
    let Some((feature, position)) = best_split(view, min_sup) else {
        return leaf;
    };

    let column = view.get_sorted_feature(feature);
    let ids = view.get_feature_indices(feature);
    let threshold = shared::threshold_between(
        column[ids[position - 1]].value(),
        column[ids[position]].value(),
    );
    let (left_view, right_view) = view.split(feature, position);
    let left = grow(&left_view, depth - 1, min_sup, loss, invalid);
    let right = grow(&right_view, depth - 1, min_sup, loss, invalid);

    let split_cost = left.cost() + right.cost();
    if split_cost >= cost {
        return leaf;
    }
    Grown::Split {
        feature,
        threshold,
        cost: split_cost,
        label,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// The `(feature, position)` of the threshold with the lowest weighted Gini
/// among those leaving at least `min_sup` instances on each side; the first
/// one found on a tie. `None` when no threshold qualifies.
fn best_split(view: &DataView<'_>, min_sup: usize) -> Option<(usize, usize)> {
    let num_labels = view.get_num_labels();
    let total = view.len();
    let mut left = vec![0usize; num_labels];
    let mut right = vec![0usize; num_labels];
    let mut best: Option<(f64, usize, usize)> = None;

    for feature in 0..view.get_feature_number() {
        let candidates = view.get_possible_split_indices(feature);
        let feasible = support_feasible_splits(candidates, total, min_sup);
        if feasible.is_empty() {
            continue;
        }
        let column = view.get_sorted_feature(feature);
        let ids = view.get_feature_indices(feature);
        left.fill(0);
        right.copy_from_slice(view.get_labels_freqs());
        let mut moved = 0;
        for &position in &candidates[feasible] {
            for &id in &ids[moved..position] {
                let label = column[id].label() as usize;
                left[label] += 1;
                right[label] -= 1;
            }
            moved = position;
            let gini =
                DataView::compute_gini_from_frequencies(&left, &right, position, total - position);
            if best.map_or(true, |(lowest, _, _)| gini < lowest) {
                best = Some((gini, feature, position));
            }
        }
    }
    best.map(|(_, feature, position)| (feature, position))
}

/// Appends `grown` to `tree` under `parent`, or as the root of an empty tree.
fn write(grown: &Grown, tree: &mut Tree, parent: usize, is_left: bool) {
    match grown {
        Grown::Leaf { cost, label } => {
            tree.add_node(
                parent,
                is_left,
                TreeNode::new(NodeInfos {
                    feature: None,
                    split: None,
                    error: *cost,
                    label: Some(*label),
                }),
            );
        }
        Grown::Split {
            feature,
            threshold,
            cost,
            label,
            left,
            right,
        } => {
            let index = tree.add_node(
                parent,
                is_left,
                TreeNode::new(NodeInfos {
                    feature: Some(*feature),
                    split: Some(*threshold),
                    error: *cost,
                    label: Some(*label),
                }),
            );
            write(left, tree, index, true);
            write(right, tree, index, false);
        }
    }
}
