//! When no split beats predicting one class for everything, every search must
//! still return a tree: a single leaf, or a split that is no worse.

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::NoHeuristic;
use dtrees_rs::algorithms::common::types::OptimalDepth2Policy;
use dtrees_rs::algorithms::greedy::LGDTBuilder;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::bitsets::{BitCollection, Bitset, BitsetInit};
use dtrees_rs::caching::Trie;
use dtrees_rs::cover::Cover;
use dtrees_rs::tree::Tree;

/// A cover with one bitset per feature column and per label.
fn cover(features: &[[bool; 2]], labels: &[usize], num_labels: usize) -> Cover {
    let rows = features.len();
    let mut attributes = vec![Bitset::new(BitsetInit::Empty(rows)); 2];
    let mut label_sets = vec![Bitset::new(BitsetInit::Empty(rows)); num_labels];
    for (row, (values, &label)) in features.iter().zip(labels).enumerate() {
        for (feature, &value) in values.iter().enumerate() {
            if value {
                attributes[feature].set(row);
            }
        }
        label_sets[label].set(row);
    }
    Cover::new(attributes, label_sets, rows)
}

const GRID: [[bool; 2]; 8] = [
    [false, false],
    [false, true],
    [true, false],
    [true, true],
    [false, false],
    [false, true],
    [true, false],
    [true, true],
];

/// Every row in the same class.
fn single_class() -> Cover {
    cover(&GRID, &[0; 8], 1)
}

/// The label is the XOR of the two features, so no single split helps:
/// at depth 1 the best tree is a leaf with half the rows wrong.
fn xor() -> Cover {
    let labels: Vec<usize> = GRID.iter().map(|[a, b]| (a ^ b) as usize).collect();
    cover(&GRID, &labels, 2)
}

fn dl85(depth: usize, cover: &mut Cover) -> Tree {
    let error_fn = Box::<NativeError>::default();
    let mut dl85 = DL85Builder::default()
        .max_depth(depth)
        .min_support(1)
        .specialization(OptimalDepth2Policy::Enabled)
        .cache(Box::<Trie>::default())
        .heuristic(Box::<NoHeuristic>::default())
        .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
        .error_function(error_fn)
        .build()
        .expect("a valid configuration");
    dl85.fit(cover).expect("the data is valid");
    dl85.tree().clone()
}

fn lgdt(depth: usize, cover: &mut Cover) -> Tree {
    let mut lgdt = LGDTBuilder::<ErrorMinimizer<NativeError>>::with_default_error_minimizer()
        .min_support(1)
        .max_depth(depth)
        .build()
        .expect("a valid configuration");
    lgdt.fit(cover)
        .unwrap_or_else(|err| panic!("depth {depth}: {err:?}"));
    lgdt.tree().clone()
}

/// The predictions of the leaves reachable from the root.
fn leaf_outputs(tree: &Tree) -> Vec<Option<f64>> {
    let mut outputs = vec![];
    let mut stack = vec![tree.get_root_index()];
    while let Some(node) = stack.pop() {
        match (tree.node_test(node), tree.node_children(node)) {
            (Some(_), (left, right)) => stack.extend([left, right]),
            (None, _) => outputs.push(tree.node_output(node)),
        }
    }
    outputs
}

fn assert_solved(tree: &Tree, error: f64) {
    assert!(!tree.is_empty(), "the search returned no tree");
    assert_eq!(tree.root_error(), error);
    assert!(
        leaf_outputs(tree).iter().all(Option::is_some),
        "every leaf must predict a class"
    );
}

#[test]
fn dl85_solves_a_single_class() {
    for depth in [1, 2, 3] {
        let tree = dl85(depth, &mut single_class());
        assert_solved(&tree, 0.0);
        assert!(leaf_outputs(&tree).iter().all(|&out| out == Some(0.0)));
    }
}

#[test]
fn lgdt_solves_a_single_class() {
    for depth in [1, 2, 3] {
        let tree = lgdt(depth, &mut single_class());
        assert_solved(&tree, 0.0);
        assert!(leaf_outputs(&tree).iter().all(|&out| out == Some(0.0)));
    }
}

#[test]
fn dl85_solves_data_where_no_split_helps() {
    assert_solved(&dl85(1, &mut xor()), 4.0);
}

#[test]
fn lgdt_solves_data_where_no_split_helps() {
    assert_solved(&lgdt(1, &mut xor()), 4.0);
}

#[test]
fn a_split_that_helps_is_still_found() {
    // At depth 2, XOR is solved exactly.
    assert_eq!(dl85(2, &mut xor()).root_error(), 0.0);
    assert_eq!(lgdt(2, &mut xor()).root_error(), 0.0);
}
