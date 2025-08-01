use crate::algorithms::common::errors::{ErrorWrapper, NativeError};
use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::{
    build_labels_count_distribution_matrix, deduce_sibling_error, deduce_sibling_error_with_buffer,
};
use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
use crate::cover::Cover;
use crate::globals::{float_is_null, item};
use crate::tree::Tree;

pub struct ErrorMinimizer<E>
where
    E: ErrorWrapper,
{
    error_fn: Box<E>,
}

impl Default for ErrorMinimizer<NativeError> {
    fn default() -> Self {
        Self {
            error_fn: Box::<NativeError>::default(),
        }
    }
}

impl<E> OptimalDepth2Tree for ErrorMinimizer<E>
where
    E: ErrorWrapper,
{
    fn find_optimal_depth_one_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError> {
        let candidates = self.get_candidates(cover, min_sup, provided_candidates);

        if candidates.is_empty() {
            return Err(FitError::EmptyCandidates);
        }

        let parent_labels_count = cover.labels_count();
        let mut tree = Tree::empty_tree(1);
        let mut left_index = 0;
        let mut right_index = 0;

        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            left_index = root.left;
            right_index = root.right
        }

        let mut best_error = <f64>::INFINITY;

        for &candidate in candidates.iter() {
            let _ = cover.branch_on(item(candidate, 0));
            let left_labels_count = cover.labels_count();
            let left_error = self.error_fn.compute(&left_labels_count);
            cover.backtrack();

            let right_labels_count = deduce_sibling_error(&parent_labels_count, &left_labels_count);
            let right_error = self.error_fn.compute(&right_labels_count);

            let total_error = left_error.0 + right_error.0;

            if total_error < best_error {
                best_error = total_error;
                tree.update_root()
                    .map(|updater| updater.test(candidate).error(total_error));
                tree.update_node(left_index)
                    .map(|updater| updater.error(left_error.0).output(left_error.1));
                tree.update_node(right_index)
                    .map(|updater| updater.error(right_error.0).output(right_error.1));
            }
        }

        Ok(tree)
    }

    fn find_optimal_depth_two_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError> {
        let candidates = self.get_candidates(cover, min_sup, provided_candidates);
        if candidates.is_empty() {
            return Err(FitError::EmptyCandidates);
        }
        if candidates.len() < 2 {
            return self.find_optimal_depth_one_tree(min_sup, cover, Some(&candidates));
        }

        let matrix = build_labels_count_distribution_matrix(cover, &candidates);

        let mut best_tree = Tree::empty_tree(2);

        let classes_distribution = cover.labels_count();
        let total_support = cover.count();
        let base_error = self.error_fn.compute(&classes_distribution);

        let mut left_distribution = vec![0; cover.num_labels];

        for (i, &first_attribute) in candidates.iter().enumerate() {
            let right_distribution = &matrix[i][i];
            let right_support = right_distribution.iter().sum::<usize>();

            deduce_sibling_error_with_buffer(
                &classes_distribution,
                right_distribution,
                &mut left_distribution,
            );
            let left_support = total_support - right_support;

            if left_support < min_sup || right_support < min_sup {
                continue;
            }

            let mut candidate_tree = Tree::empty_tree(2);

            let (left_index, right_index) =
                candidate_tree.update_root().map_or((0, 0), |updater| {
                    updater
                        .test(first_attribute)
                        .error(base_error.0)
                        .get_children()
                });

            let left_error = self.error_fn.compute(&left_distribution);

            // The left does not have enough support to be split further
            if left_support < 2 * min_sup {
                candidate_tree
                    .update_node(left_index)
                    .map(|updater| updater.error(left_error.0).output(left_error.1).leaf());

                if best_tree.root_error() < left_error.0 {
                    continue;
                }
            } else {
                let mut feature_error = candidate_tree.root_error();
                if left_error.0 < feature_error {
                    candidate_tree
                        .update_node(left_index)
                        .map(|updater| updater.error(left_error.0));
                }

                for (j, &second_attribute) in candidates.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    // Here is if the left part of the tree so the classes support of i__left -> j__right
                    let i_left_j_right_classes_support =
                        deduce_sibling_error(&matrix[j][j], &matrix[i][j]);
                    let j_right_support = matrix[j][j].iter().sum::<usize>();
                    let i_right_j_right_support = matrix[i][j].iter().sum::<usize>();
                    let i_left_j_right_support = j_right_support - i_right_j_right_support; // Important
                    let i_left_j_left_support = left_support - i_left_j_right_support; // Important

                    if i_left_j_right_support < min_sup || i_left_j_left_support < min_sup {
                        // Not enough support for the left part of the tree
                        continue;
                    }
                    let right_leaf_error = self.error_fn.compute(&i_left_j_right_classes_support);

                    if right_leaf_error.0 >= feature_error {
                        continue;
                    }

                    // Here is if the left part of the tree so the classes support of i__left -> j__right
                    let i_left_j_right_classes_support =
                        deduce_sibling_error(&matrix[j][j], &matrix[i][j]);
                    let j_right_support = matrix[j][j].iter().sum::<usize>();
                    let i_right_j_right_support = matrix[i][j].iter().sum::<usize>();
                    let i_left_j_right_support = j_right_support - i_right_j_right_support; // Important
                    let i_left_j_left_support = left_support - i_left_j_right_support; // Important

                    if i_left_j_right_support < min_sup || i_left_j_left_support < min_sup {
                        // Not enough support for the left part of the tree
                        continue;
                    }

                    let right_leaf_error = self.error_fn.compute(&i_left_j_right_classes_support);

                    // TODO Upper bound control here
                    if right_leaf_error.0 >= feature_error {
                        continue;
                    }

                    let i_left_j_left_classes_support =
                        deduce_sibling_error(&left_distribution, &i_left_j_right_classes_support);
                    let left_leaf_error = self.error_fn.compute(&i_left_j_left_classes_support);

                    // TODO Upper bound control here
                    let branch_error = left_leaf_error.0 + right_leaf_error.0;
                    if branch_error >= feature_error {
                        continue;
                    }
                    feature_error = branch_error;

                    let (left_leaf_index, right_leaf_index) = candidate_tree
                        .update_node(left_index)
                        .map_or((0, 0), |updater| {
                            updater
                                .test(second_attribute)
                                .error(feature_error)
                                .get_children()
                        });

                    candidate_tree
                        .update_leaf_node(left_leaf_index, left_leaf_error)
                        .update_leaf_node(right_leaf_index, right_leaf_error);

                    if float_is_null(feature_error) {
                        break;
                    }
                }
            }

            let right_error = self.error_fn.compute(right_distribution);
            if right_support < 2 * min_sup {
                candidate_tree
                    .update_node(right_index)
                    .map(|updater| updater.error(right_error.0).output(right_error.1).leaf());

                let best_error = best_tree.root_error();
                let current_left_error = candidate_tree.node_error(left_index);
                if current_left_error > best_error
                    || right_error.0 >= best_error - current_left_error
                {
                    // TODO : Quite not clear
                    continue;
                }
            } else {
                let mut feature_error = best_tree.root_error();
                let current_left_error = candidate_tree.node_error(left_index);

                if current_left_error > feature_error
                    || right_error.0 < feature_error - current_left_error
                {
                    // TODO : Not clear
                    candidate_tree
                        .update_node(right_index)
                        .map(|updater| updater.error(right_error.0).output(right_error.1));
                }

                let mut i_right_j_left_classes_support = vec![0; cover.num_labels];

                for (j, &second_attribute) in candidates.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    // Here is if the right part of the tree so the classes support of i__right -> j__right
                    let i_right_j_right_classes_support = &matrix[i][j];
                    deduce_sibling_error_with_buffer(
                        &matrix[i][i],
                        &matrix[i][j],
                        &mut i_right_j_left_classes_support,
                    );
                    let i_right_j_right_support = matrix[i][j].iter().sum::<usize>();
                    let i_right_j_left_support = right_support - i_right_j_right_support; // Important

                    if i_right_j_left_support < min_sup || i_right_j_right_support < min_sup {
                        // Not enough support for the right part of the tree
                        continue;
                    }

                    let left_leaf_error = self.error_fn.compute(&i_right_j_left_classes_support);

                    // TODO Upper bound control here
                    if left_leaf_error.0 >= feature_error {
                        continue;
                    }

                    let right_leaf_error = self.error_fn.compute(i_right_j_right_classes_support);

                    let branch_error = left_leaf_error.0 + right_leaf_error.0;

                    // TODO Upper bound control here
                    if branch_error >= feature_error {
                        continue;
                    }

                    feature_error = branch_error;

                    let (left_leaf_index, right_leaf_index) = candidate_tree
                        .update_node(right_index)
                        .map_or((0, 0), |updater| {
                            updater
                                .test(second_attribute)
                                .error(feature_error)
                                .get_children()
                        });

                    candidate_tree
                        .update_leaf_node(left_leaf_index, left_leaf_error)
                        .update_leaf_node(right_leaf_index, right_leaf_error);

                    if float_is_null(feature_error) {
                        break;
                    }
                }

                let feature_error =
                    candidate_tree.node_error(left_index) + candidate_tree.node_error(right_index);
                candidate_tree
                    .update_root()
                    .map(|updater| updater.error(feature_error));

                if best_tree.root_error() > feature_error {
                    best_tree = candidate_tree
                }
                if float_is_null(feature_error) {
                    break;
                }
            }
        }
        Ok(best_tree)
    }

    fn error(&self, distribution: &[usize]) -> (f64, f64) {
        self.error_fn.compute(distribution)
    }
}

impl<E> ErrorMinimizer<E>
where
    E: ErrorWrapper,
{
    pub fn new(error_function: Box<E>) -> Self {
        Self {
            error_fn: error_function,
        }
    }
}

#[cfg(test)]
mod error_minimizer {
    use crate::algorithms::optimal::depth2::error_minimizer::ErrorMinimizer;
    use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
    use crate::reader::data_reader::DataReader;
    use std::path::Path;

    #[test]
    fn run_small_data() {
        let reader = DataReader::default();
        let path = Path::new("test_data/anneal.txt");
        let cover_result = reader.read_file(path);

        let mut cover = match cover_result {
            Ok(cover) => cover,
            Err(_) => panic!("oops"),
        };

        let error_minimizer = ErrorMinimizer::default();
        let tree = error_minimizer.fit(1, 2, &mut cover, None);

        if let Ok(t) = tree {
            println!("Error {}", t.root_error());
            t.print()
        }
    }
}
