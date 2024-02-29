use crate::tree::Tree;
use float_cmp::{ApproxEq, F64Margin};

// Start: Items and Attributes switchers
pub fn attribute(item: usize) -> usize {
    item / 2
}

// Get if item is left: 0 or right: 1
pub fn item_type(item: usize) -> usize {
    item % 2
}

pub fn item(attribute: usize, item_type: usize) -> usize {
    attribute * 2 + item_type
}

// End: Items and Attributes switchers

pub fn float_is_null(value: f64) -> bool {
    value.approx_eq(
        0.0,
        F64Margin {
            ulps: 2,
            epsilon: 0.0,
        },
    )
}

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

// * TODO : Add this to a macro and all to get info about a node
pub fn get_tree_root_gain(tree: &Tree) -> f64 {
    tree.get_node(tree.get_root_index())
        .map_or(0.0, |node| node.value.metric.map_or(0.0, |v| v))
}

pub fn get_tree_root_error(tree: &Tree) -> f64 {
    tree.get_node(tree.get_root_index())
        .map_or(<f64>::INFINITY, |node| node.value.error)
}
