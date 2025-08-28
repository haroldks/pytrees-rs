use crate::structures::Structure;
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

pub fn information_gain(
    attribute: usize,
    structure: &mut dyn Structure,
    root_classes_support: &[usize],
    parent_entropy: f64,
    ratio: bool,
) -> f64 {
    let _ = structure.push(item(attribute, 0));
    let left_classes_supports = structure.labels_support().to_vec();
    structure.backtrack();

    let right_classes_support = root_classes_support
        .iter()
        .enumerate()
        .map(|(idx, val)| *val - left_classes_supports[idx])
        .collect::<Vec<usize>>();

    let actual_size = root_classes_support.iter().sum::<usize>();
    let left_split_size = left_classes_supports.iter().sum::<usize>();
    let right_split_size = right_classes_support.iter().sum::<usize>();

    let left_weight = match actual_size {
        0 => 0f64,
        _ => left_split_size as f64 / actual_size as f64,
    };

    let right_weight = match actual_size {
        0 => 0f64,
        _ => right_split_size as f64 / actual_size as f64,
    };

    let mut split_info = 0f64;
    if ratio {
        if left_weight > 0. {
            split_info = -left_weight * left_weight.log2();
        }
        if right_weight > 0. {
            split_info += -right_weight * right_weight.log2();
        }
    }
    if split_info.approx_eq(
        0.,
        F64Margin {
            ulps: 2,
            epsilon: 0.0,
        },
    ) {
        split_info = 1f64;
    }

    let left_split_entropy = compute_entropy(&left_classes_supports);
    let right_split_entropy = compute_entropy(&right_classes_support);

    let info_gain =
        parent_entropy - (left_weight * left_split_entropy + right_weight * right_split_entropy);
    if ratio {
        return info_gain / split_info;
    }
    info_gain
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
