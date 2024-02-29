use crate::globals::{float_is_null, item};
use crate::searches::deduce_sibling_error;
use crate::searches::errors::{ErrorWrapper, NativeError};
use crate::searches::optimal::d2::{Depth2Algorithm, MAX_ERROR};
use crate::structures::Structure;
use crate::tree::Tree;

#[derive(Default)]
pub struct Murtree {
    error_function: NativeError,
}

impl Depth2Algorithm for Murtree {
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

impl Murtree {
    fn depth_one<S: Structure>(&self, min_sup: usize, structure: &mut S) -> Tree {
        let candidates = self.generate_candidates_list(structure, min_sup);
        if candidates.is_empty() {
            return Tree::empty_tree(1);
        }
        let mut tree = Tree::empty_tree(1);
        let mut left_index = 0;
        let mut right_index = 0;
        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            left_index = root.left;
            right_index = root.right
        }
        for candidate in candidates.iter() {
            structure.push(item(*candidate, 0));
            let classes_support = structure.labels_support();
            let left_error = self.error_function.compute(classes_support);
            structure.backtrack();

            structure.push(item(*candidate, 1));
            let classes_support = structure.labels_support();
            let right_error = self.error_function.compute(classes_support);
            structure.backtrack();

            let error = left_error.0 + right_error.0;
            let mut past_error = MAX_ERROR;

            if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
                past_error = root.value.error;
            }
            if error < past_error {
                if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
                    root.value.error = error;
                    root.value.test = Some(*candidate);
                }
                if let Some(left) = tree.get_node_mut(left_index) {
                    left.value.error = left_error.0;
                    left.value.out = Some(left_error.1);
                }
                if let Some(right) = tree.get_node_mut(right_index) {
                    right.value.error = right_error.0;
                    right.value.out = Some(right_error.1);
                }
            }
        }
        tree
    }

    fn depth_two<S: Structure>(&self, min_sup: usize, structure: &mut S) -> Tree {
        // TODO : depth attribute
        let candidates = self.generate_candidates_list(structure, min_sup);

        if candidates.is_empty() {
            return Tree::empty_tree(2);
        }
        if candidates.len() < 2 {
            return self.depth_one(min_sup, structure);
        }
        let matrix = self.build_depth_two_matrix(structure, &candidates);

        let mut tree = Tree::empty_tree(2);
        let classes_support = structure.labels_support().to_vec();
        let support = structure.get_support();

        let before_creation_error = self.error_function.compute(&classes_support);

        for (i, first) in candidates.iter().enumerate() {
            // LEFT PART
            let i_right_classes_support = &matrix[i][i];
            let i_left_classes_support =
                deduce_sibling_error(&classes_support, i_right_classes_support);

            let root_right_support = matrix[i][i].iter().sum::<usize>();
            let root_left_support = support - root_right_support;
            if root_left_support < min_sup || root_right_support < min_sup {
                continue;
            }

            // TODO : Should check if there is enough support for the leaf as root_left_support should be >= 2 * min_sup otherwise it is a leaf.

            let mut root_tree = Tree::empty_tree(2);
            let mut left_index = 0;
            let mut right_index = 0;

            if let Some(root_node) = root_tree.get_node_mut(root_tree.get_root_index()) {
                root_node.value.test = Some(*first);
                root_node.value.error = before_creation_error.0;
                left_index = root_node.left;
                right_index = root_node.right;
            }

            // TODO : Check if support in enough 2 * min_sup

            if root_left_support < 2 * min_sup {
                let error = self.error_function.compute(&i_left_classes_support);
                if let Some(left_node) = root_tree.get_node_mut(left_index) {
                    left_node.value.error = error.0;
                    left_node.value.out = Some(error.1);
                    left_node.left = 0;
                    left_node.right = 0;
                }
                if let Some(node) = tree.get_node(tree.get_root_index()) {
                    if node.value.error < error.0 {
                        continue;
                    }
                }
            } else {
                let mut feat_error = MAX_ERROR;
                if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
                    feat_error = root.value.error;
                }
                let error = self.error_function.compute(&i_left_classes_support);

                if error.0 < feat_error {
                    if let Some(left_node) = root_tree.get_node_mut(left_index) {
                        left_node.value.error = error.0;
                    }
                }

                for (j, second) in candidates.iter().enumerate().take(candidates.len()) {
                    if i == j {
                        // TODO: Can be replaced by first == second
                        continue;
                    }

                    // TODO: Add upper bound for left or right error based on the best tree so far

                    // Here is if the left part of the tree so the classes support of i__left -> j__right
                    let i_left_j_right_classes_support =
                        deduce_sibling_error(&matrix[j][j], &matrix[i][j]);
                    let j_right_support = matrix[j][j].iter().sum::<usize>();
                    let i_right_j_right_support = matrix[i][j].iter().sum::<usize>();
                    let i_left_j_right_support = j_right_support - i_right_j_right_support; // Important
                    let i_left_j_left_support = root_left_support - i_left_j_right_support; // Important

                    if i_left_j_right_support < min_sup || i_left_j_left_support < min_sup {
                        // Not enough support for the left part of the tree
                        continue;
                    }

                    let right_leaf_error =
                        self.error_function.compute(&i_left_j_right_classes_support);

                    // TODO Upper bound control here
                    if right_leaf_error.0 >= feat_error {
                        continue;
                    }

                    let i_left_j_left_classes_support = deduce_sibling_error(
                        &i_left_classes_support,
                        &i_left_j_right_classes_support,
                    );
                    let left_leaf_error =
                        self.error_function.compute(&i_left_j_left_classes_support);

                    // TODO Upper bound control here
                    if (left_leaf_error.0 + right_leaf_error.0) >= feat_error {
                        continue;
                    }

                    let mut left_leaf_index = 0;
                    let mut right_leaf_index = 0;

                    if let Some(left_node) = root_tree.get_node_mut(left_index) {
                        left_node.value.test = Some(*second);
                        left_node.value.error = left_leaf_error.0 + right_leaf_error.0;
                        left_leaf_index = left_node.left;
                        right_leaf_index = left_node.right;
                    }

                    if let Some(left_leaf) = root_tree.get_node_mut(left_leaf_index) {
                        left_leaf.value.error = left_leaf_error.0;
                        left_leaf.value.out = Some(left_leaf_error.1);
                    }

                    if let Some(right_leaf) = root_tree.get_node_mut(right_leaf_index) {
                        right_leaf.value.error = right_leaf_error.0;
                        right_leaf.value.out = Some(right_leaf_error.1);
                    }

                    feat_error = left_leaf_error.0 + right_leaf_error.0;

                    if float_is_null(feat_error) {
                        break;
                    }
                }
            }

            if root_right_support < 2 * min_sup {
                let error = self.error_function.compute(i_right_classes_support);
                if let Some(right_node) = root_tree.get_node_mut(right_index) {
                    right_node.value.error = error.0;
                    right_node.value.out = Some(error.1);
                    right_node.left = 0;
                    right_node.right = 0;
                }

                let mut current_left_error = MAX_ERROR;
                let best_error = tree.get_node(tree.get_root_index()).unwrap().value.error;

                if let Some(node) = root_tree.get_node(left_index) {
                    current_left_error = node.value.error;
                }
                if current_left_error > best_error || error.0 >= best_error - current_left_error {
                    // TODO Danger here
                    continue;
                }
            } else {
                let mut feat_error = MAX_ERROR;
                if let Some(root) = tree.get_node(tree.get_root_index()) {
                    feat_error = root.value.error;
                }
                let mut current_left_error = MAX_ERROR;
                if let Some(left_node) = root_tree.get_node(left_index) {
                    current_left_error = left_node.value.error;
                }

                let error = self.error_function.compute(i_right_classes_support);

                if current_left_error > feat_error || error.0 < feat_error - current_left_error {
                    if let Some(right_node) = root_tree.get_node_mut(right_index) {
                        right_node.value.error = error.0;
                        right_node.value.out = Some(error.1);
                    }
                }

                for (j, second) in candidates.iter().enumerate().take(candidates.len()) {
                    if i == j {
                        // TODO: Can be replaced by first == second
                        continue;
                    }

                    // Here is if the right part of the tree so the classes support of i__right -> j__right
                    let i_right_j_right_classes_support = &matrix[i][j];
                    let i_right_j_left_classes_support =
                        deduce_sibling_error(&matrix[i][i], &matrix[i][j]);
                    let i_right_j_right_support = matrix[i][j].iter().sum::<usize>();
                    let i_right_j_left_support = root_right_support - i_right_j_right_support; // Important

                    if i_right_j_left_support < min_sup || i_right_j_right_support < min_sup {
                        // Not enough support for the right part of the tree
                        continue;
                    }

                    let left_leaf_error =
                        self.error_function.compute(&i_right_j_left_classes_support);

                    // TODO Upper bound control here
                    if left_leaf_error.0 >= feat_error {
                        continue;
                    }

                    let right_leaf_error =
                        self.error_function.compute(i_right_j_right_classes_support);

                    // TODO Upper bound control here
                    if (left_leaf_error.0 + right_leaf_error.0) >= feat_error {
                        continue;
                    }

                    let mut left_leaf_index = 0;
                    let mut right_leaf_index = 0;

                    if let Some(right_node) = root_tree.get_node_mut(right_index) {
                        right_node.value.test = Some(*second);
                        right_node.value.error = left_leaf_error.0 + right_leaf_error.0;
                        left_leaf_index = right_node.left;
                        right_leaf_index = right_node.right;
                    }

                    if let Some(left_leaf) = root_tree.get_node_mut(left_leaf_index) {
                        left_leaf.value.error = left_leaf_error.0;
                        left_leaf.value.out = Some(left_leaf_error.1);
                    }

                    if let Some(right_leaf) = root_tree.get_node_mut(right_leaf_index) {
                        right_leaf.value.error = right_leaf_error.0;
                        right_leaf.value.out = Some(right_leaf_error.1);
                    }

                    feat_error = left_leaf_error.0 + right_leaf_error.0;
                    if float_is_null(feat_error) {
                        break;
                    }
                }

                let mut feat_error = MAX_ERROR;
                if let Some(node) = root_tree.get_node(left_index) {
                    feat_error = node.value.error;
                }
                if let Some(node) = root_tree.get_node(right_index) {
                    feat_error += node.value.error;
                }

                if let Some(feat_tree) = root_tree.get_node_mut(tree.get_root_index()) {
                    feat_tree.value.error = feat_error;
                }

                let mut best_error = MAX_ERROR;
                if let Some(node) = tree.get_node(tree.get_root_index()) {
                    best_error = node.value.error;
                }
                if best_error > feat_error {
                    tree = root_tree;
                }

                if float_is_null(feat_error) {
                    break;
                }
            }
        }

        tree
    }
}
#[cfg(test)]
mod murtree_test {
    use crate::data::{BinaryData, FileReader};
    use crate::searches::errors::NativeError;
    use crate::searches::optimal::d2::{Depth2Algorithm, Murtree};
    use crate::structures::Bitset;

    #[test]
    fn run_small_data() {
        let data = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = Bitset::new(&data);
        let method = Murtree::default();
        let tree = method.fit(1, 2, &mut structure);
        let error = tree.get_node(0).unwrap().value.error;
        assert_eq!(error, 0.0)
    }

    #[test]
    fn run_anneal_data() {
        let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = Bitset::new(&data);
        let method = Murtree::default();
        let tree = method.fit(1, 2, &mut structure);
        let error = tree.get_node(0).unwrap().value.error;
        assert_eq!(error, 137.0)
    }
}
