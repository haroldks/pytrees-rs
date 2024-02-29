use crate::globals::{
    compute_entropy, float_is_null, get_tree_root_error, get_tree_root_gain, item,
};
use crate::searches::deduce_sibling_error;
use crate::searches::errors::{ErrorWrapper, NativeError};
use crate::searches::optimal::d2::{Depth2Algorithm, MAX_ERROR};
use crate::structures::Structure;
use crate::tree::Tree;

#[derive(Default)]
pub struct InfoGainDT {
    error_function: NativeError,
}

impl Depth2Algorithm for InfoGainDT {
    fn fit<S: Structure>(&self, min_sup: usize, depth: usize, structure: &mut S) -> Tree {
        match depth {
            1 => self.depth_one(min_sup, structure),
            2 => self.depth_two(min_sup, structure),
            _ => {
                panic!("Depth must be 1 or 2")
            }
        }
    }
}

impl InfoGainDT {
    fn depth_one<S: Structure>(&self, min_sup: usize, structure: &mut S) -> Tree {
        let candidates = self.generate_candidates_list(structure, min_sup);

        if candidates.is_empty() {
            return Tree::empty_tree(1);
        }

        let mut left_index = 0;
        let mut right_index = 0;

        let mut tree = Tree::empty_tree(1);

        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            root.value.test = Some(candidates[0]);
            left_index = root.left;
            right_index = root.right;
        }

        let mut node_error = MAX_ERROR;

        for branch in [0usize, 1].iter() {
            structure.push(item(candidates[0], *branch));
            let classes_support = structure.labels_support();
            let error = self.error_function.compute(classes_support);

            let index = match *branch == 0 {
                true => left_index,
                false => right_index,
            };

            if let Some(node) = tree.get_node_mut(index) {
                node.value.error = error.0;
                node.value.out = Some(error.1);
                node_error += error.0;
            }
            structure.backtrack();
        }

        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            root.value.error = node_error;
        }
        tree
    }

    fn depth_two<S: Structure>(&self, min_sup: usize, structure: &mut S) -> Tree {
        let candidates = self.generate_candidates_list(structure, min_sup);
        if candidates.is_empty() {
            return Tree::empty_tree(2);
        }
        if candidates.len() < 2 {
            return self.depth_one(min_sup, structure);
        }

        let matrix = self.build_depth_two_matrix(structure, &candidates);

        let classes_support = structure.labels_support();
        let support = classes_support.iter().sum::<usize>();

        let parent_entropy = compute_entropy(classes_support);

        if float_is_null(parent_entropy) {
            return self.depth_one(min_sup, structure);
        }

        let mut tree = Tree::empty_tree(2);

        for (i, first) in candidates.iter().enumerate().take(candidates.len()) {
            let mut root_tree = Tree::empty_tree(2);
            let mut left_index = 0;
            let mut right_index = 0;

            if let Some(root_node) = root_tree.get_node_mut(root_tree.get_root_index()) {
                root_node.value.test = Some(*first);
                left_index = root_node.left;
                right_index = root_node.right;
            }

            for (j, second) in candidates.iter().enumerate() {
                if i == j {
                    continue;
                }

                let mut left_leaves = vec![];
                let mut right_leaves = vec![];
                let mut root_error = 0.0;
                let mut root_gain = 0f64;

                for val in [0usize, 1].iter() {
                    let mut left_leaf_index = 0;
                    let mut right_leaf_index = 0;
                    let mut node_gain = parent_entropy;

                    left_leaves = Self::deduce_leaves_classes_support(
                        &matrix,
                        (i, *val),
                        (j, 0),
                        classes_support,
                    );

                    let weight = match support {
                        0 => 0f64,
                        _ => left_leaves.iter().sum::<usize>() as f64 / support as f64,
                    };

                    node_gain -= compute_entropy(&left_leaves) * weight;
                    let left_leaves_error = self.error_function.compute(&left_leaves);

                    right_leaves = Self::deduce_leaves_classes_support(
                        &matrix,
                        (i, *val),
                        (j, 1),
                        classes_support,
                    );

                    let weight = match support {
                        0 => 0f64,
                        _ => right_leaves.iter().sum::<usize>() as f64 / support as f64,
                    };

                    node_gain -= compute_entropy(&right_leaves) * weight;
                    let right_leaves_error = self.error_function.compute(&right_leaves);

                    let node_error = left_leaves_error.0 + right_leaves_error.0;

                    let mut past_info_gain = 0f64;
                    let index = match *val == 0 {
                        true => left_index,
                        false => right_index,
                    };

                    // TODO : Replace by a macro call
                    if let Some(node) = root_tree.get_node(index) {
                        if node.value.metric.is_some() {
                            past_info_gain = node.value.metric.unwrap();
                        }
                    }

                    if node_gain > past_info_gain {
                        if let Some(node) = root_tree.get_node_mut(index) {
                            node.value.test = Some(*second);
                            node.value.error = node_error;
                            node.value.metric = Some(node_gain);
                            left_leaf_index = node.left;
                            right_leaf_index = node.right;
                        }
                        if let Some(left_leaf_ref) = root_tree.get_node_mut(left_leaf_index) {
                            left_leaf_ref.value.error = left_leaves_error.0;
                            left_leaf_ref.value.out = Some(left_leaves_error.1);
                        }

                        if let Some(right_leaf_ref) = root_tree.get_node_mut(right_leaf_index) {
                            right_leaf_ref.value.error = right_leaves_error.0;
                            right_leaf_ref.value.out = Some(right_leaves_error.1);
                        }
                    }
                    if let Some(node) = root_tree.get_node_mut(index) {
                        root_error += node.value.error;
                        root_gain += node.value.metric.unwrap();
                    }
                }
                if let Some(root_node) = root_tree.get_node_mut(root_tree.get_root_index()) {
                    root_node.value.error = root_error;
                    root_node.value.metric = Some(root_gain);
                    if float_is_null(root_node.value.error) {
                        break;
                    }
                }
            }

            // TODO Replace by a macro
            if get_tree_root_gain(&root_tree) > get_tree_root_gain(&tree) {
                tree = root_tree;
            }
            if float_is_null(get_tree_root_error(&tree)) {
                break;
            }
        }
        tree
    }

    fn deduce_leaves_classes_support(
        matrix: &[Vec<Vec<usize>>],
        first: (usize, usize),
        second: (usize, usize),
        root_classes_support: &[usize],
    ) -> Vec<usize> {
        let i = first.0;
        let j = second.0;
        let is_left_i = first.1 == 0;
        let is_left_j = second.1 == 0;

        let i_right_sc = &matrix[i][i];
        let j_right_sc = &matrix[j][j];
        let i_right_j_right_sc = &matrix[i][j];

        let i_left_j_right_sc = deduce_sibling_error(j_right_sc, i_right_j_right_sc);

        match is_left_i {
            true => {
                match is_left_j {
                    true => {
                        // i_left_j_left
                        let i_left_sc = deduce_sibling_error(root_classes_support, i_right_sc);
                        deduce_sibling_error(&i_left_sc, &i_left_j_right_sc) // i_left_j_left_sc
                    }

                    false => {
                        // i_left_j_right
                        i_left_j_right_sc.to_vec()
                    }
                }
            }
            false => {
                match is_left_j {
                    true => {
                        // i_right_j_left
                        deduce_sibling_error(i_right_sc, i_right_j_right_sc) // i_right_j_left_sc
                    }

                    false => {
                        // i_right_j_right
                        i_right_j_right_sc.to_vec()
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod info_gain_odt_test {
    use crate::data::{BinaryData, FileReader};
    use crate::searches::errors::NativeError;
    use crate::searches::optimal::d2::{Depth2Algorithm, InfoGainDT};
    use crate::structures::Bitset;

    fn run_small_data() {
        let data = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = Bitset::new(&data);
        let method = InfoGainDT::default();
        let tree = method.fit(1, 2, &mut structure);
        let error = tree.get_node(0).unwrap().value.error;
        assert_eq!(error, 0.0)
    }

    #[test]
    fn run_anneal_data() {
        let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = Bitset::new(&data);
        let method = InfoGainDT::default();
        let tree = method.fit(1, 2, &mut structure);
        let error = tree.get_node(0).unwrap().value.error;
        assert_eq!(error, 151.0)
    }
}
