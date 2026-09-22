//! Item encoding and small numeric helpers.
//!
//! An item is a feature together with a branch: item `2 * f` is the branch
//! where feature `f` is 0 (left), and item `2 * f + 1` the branch where it is
//! 1 (right).

use crate::tree::Tree;
use float_cmp::{ApproxEq, F64Margin};

/// The feature of an item.
pub fn attribute(item: usize) -> usize {
    item / 2
}

/// The branch of an item: 0 for left, 1 for right.
pub fn item_type(item: usize) -> usize {
    item % 2
}

/// The item of a feature and a branch.
pub fn item(attribute: usize, item_type: usize) -> usize {
    attribute * 2 + item_type
}

/// Whether a value is zero, up to two units in the last place.
pub fn float_is_null(value: f64) -> bool {
    value.approx_eq(
        0.0,
        F64Margin {
            ulps: 2,
            epsilon: 0.0,
        },
    )
}

/// Shannon entropy (base 2) of a class distribution.
pub fn compute_entropy(classes_support: &[usize]) -> f64 {
    let support = classes_support.iter().sum::<usize>();
    let mut entropy = 0f64;
    for class_support in classes_support {
        let p = match support {
            0 => 0f64,
            _ => *class_support as f64 / support as f64,
        };

        let mut log_val = 0f64;
        if p > 0. {
            log_val = p.log2();
        }
        entropy += -p * log_val;
    }
    entropy
}

/// The metric stored at the root of `tree`, 0 if none.
pub fn get_tree_root_gain(tree: &Tree) -> f64 {
    tree.get_node(tree.get_root_index())
        .map_or(0.0, |node| node.value.metric.map_or(0.0, |v| v))
}

/// The error of the root of `tree`, infinite if the tree is empty.
pub fn get_tree_root_error(tree: &Tree) -> f64 {
    tree.get_node(tree.get_root_index())
        .map_or(<f64>::INFINITY, |node| node.value.error)
}
