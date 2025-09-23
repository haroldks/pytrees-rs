use crate::algorithms::common::errors::{ErrorWrapper, NativeError};
use crate::algorithms::common::heuristics::helpers::{entropy, information_gain};
use crate::algorithms::common::heuristics::{Heuristic, InformationGain};
use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::{
    build_labels_count_distribution_matrix, deduce_sibling_error,
};
use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
use crate::cover::Cover;
use crate::globals::{float_is_null, get_tree_root_error, item};
use crate::tree::Tree;

pub struct InfoGainMaximizer<E>
where
    E: ErrorWrapper,
{
    error_fn: Box<E>,
    heuristic_fn: InformationGain,
}

impl Default for InfoGainMaximizer<NativeError> {
    fn default() -> Self {
        Self {
            error_fn: Box::<NativeError>::default(),
            heuristic_fn: InformationGain,
        }
    }
}

impl<E> OptimalDepth2Tree for InfoGainMaximizer<E>
where
    E: ErrorWrapper,
{
    fn find_optimal_depth_one_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError> {
        let mut candidates = self.get_candidates(cover, min_sup, provided_candidates);
        if candidates.is_empty() {
            return Err(FitError::EmptyCandidates);
        }

        if let Some(best_attr) = self.find_best_attribute(cover, &mut candidates) {
            let root_distribution = cover.labels_count();
            cover.branch_on(item(best_attr, 0));
            let left_distribution = cover.labels_count();
            cover.backtrack();
            let right_distribution = deduce_sibling_error(&root_distribution, &left_distribution);

            let left_error = self.error_fn.compute(&left_distribution);
            let right_error = self.error_fn.compute(&right_distribution);

            let mut tree = Tree::new();

            let (left, right) = tree.node_children(tree.get_root_index());

            tree.update_leaf_node(left, left_error);
            tree.update_leaf_node(right, right_error);

            tree.update_root()
                .map(|updater| updater.test(best_attr).error(left_error.0 + right_error.0));

            if tree.root_error().is_infinite() {
                return Err(FitError::EmptyTree);
            }
            return Ok(tree);
        }

        Err(FitError::EmptyTree)
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
            return self.find_optimal_depth_one_tree(min_sup, cover, provided_candidates);
        }

        let matrix = build_labels_count_distribution_matrix(cover, &candidates);

        let root_distribution = cover.labels_count();

        let parent_entropy = entropy(&root_distribution);

        if float_is_null(parent_entropy) {
            return self.find_optimal_depth_one_tree(min_sup, cover, provided_candidates);
        }

        let mut best_tree = Tree::empty_tree(2);

        for (i, &first_attr) in candidates.iter().enumerate() {
            let mut candidate_tree = Tree::empty_tree(2);

            let (left_index, right_index) = candidate_tree
                .update_root()
                .map_or((0, 0), |updater| updater.test(first_attr).get_children());

            for (j, &second_attr) in candidates.iter().enumerate() {
                if i == j {
                    continue;
                }

                let mut root_error = 0.0;
                let mut root_gain = 0f64;

                for &val in [0usize, 1].iter() {
                    let left_leaf_distribution = Self::deduce_leaves_classes_support(
                        &matrix,
                        (i, val),
                        (j, 0),
                        &root_distribution,
                    );
                    let right_leaf_distribution = Self::deduce_leaves_classes_support(
                        &matrix,
                        (i, val),
                        (j, 1),
                        &root_distribution,
                    );

                    // TODO : The weights are from the distribution not the global support from the parent

                    let branch_index = if val == 0 { left_index } else { right_index };

                    let branch_gain = information_gain(
                        &root_distribution,
                        &left_leaf_distribution,
                        &right_leaf_distribution,
                        parent_entropy,
                    );

                    let left_leaf_error = self.error_fn.compute(&left_leaf_distribution);
                    let right_leaf_error = self.error_fn.compute(&right_leaf_distribution);

                    let stored_gain = candidate_tree
                        .node_metric(branch_index)
                        .map_or(0.0, |metric| metric);

                    if branch_gain > stored_gain {
                        let (left_leaf, righ_leaf) = candidate_tree
                            .update_node(branch_index)
                            .map_or((0, 0), |updater| {
                                updater
                                    .test(second_attr)
                                    .error(left_leaf_error.0 + right_leaf_error.0)
                                    .metric(branch_gain)
                                    .get_children()
                            });

                        candidate_tree.update_leaf_node(left_leaf, left_leaf_error);
                        candidate_tree.update_leaf_node(righ_leaf, right_leaf_error);
                    }

                    root_error += candidate_tree.node_error(branch_index);
                    root_gain += candidate_tree.node_metric(branch_index).unwrap_or(0.0);
                }

                candidate_tree
                    .update_root()
                    .map(|updater| updater.error(root_error).metric(root_gain));

                if float_is_null(root_error) {
                    break;
                }
            }

            if candidate_tree
                .node_metric(candidate_tree.get_root_index())
                .unwrap_or(0.0)
                > best_tree
                    .node_metric(best_tree.get_root_index())
                    .unwrap_or(0.0)
            {
                best_tree = candidate_tree
            }

            if float_is_null(get_tree_root_error(&best_tree)) {
                break;
            }
        }

        if best_tree.root_error().is_infinite() {
            return Err(FitError::EmptyTree);
        }

        Ok(best_tree)
    }

    fn error(&self, distribution: &[usize]) -> (f64, f64) {
        self.error_fn.compute(distribution)
    }
}

impl<E> InfoGainMaximizer<E>
where
    E: ErrorWrapper,
{
    fn find_best_attribute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        self.heuristic_fn.compute(cover, candidates);
        candidates.first().copied()
    }

    fn deduce_leaves_classes_support(
        matrix: &[Vec<Vec<usize>>],
        first: (usize, usize),
        second: (usize, usize),
        root_classes_support: &[usize],
    ) -> Vec<usize> {
        let (attr1, is_left1) = (first.0, first.1 == 0);
        let (attr2, is_left2) = (second.0, second.1 == 0);

        let attr1_right_dist = &matrix[attr1][attr1];
        let attr1_right_attr2_right_dist = &matrix[attr1][attr2];

        match (is_left1, is_left2) {
            (true, true) => {
                let attr1_left_dist = deduce_sibling_error(root_classes_support, attr1_right_dist);
                let attr1_left_attr2_right_dist =
                    deduce_sibling_error(&matrix[attr2][attr2], attr1_right_attr2_right_dist);
                deduce_sibling_error(&attr1_left_dist, &attr1_left_attr2_right_dist)
            }
            (true, false) => {
                deduce_sibling_error(&matrix[attr2][attr2], attr1_right_attr2_right_dist)
            }
            (false, true) => deduce_sibling_error(attr1_right_dist, attr1_right_attr2_right_dist),
            (false, false) => attr1_right_attr2_right_dist.to_vec(),
        }
    }
}

#[cfg(test)]
mod info_gain_maximizer {
    use crate::algorithms::optimal::depth2::info_gain_maximizer::InfoGainMaximizer;
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

        let info_gain_maximizer = InfoGainMaximizer::default();
        let tree = info_gain_maximizer.fit(1, 2, &mut cover, None);

        if let Ok(t) = tree {
            println!("Error {}", t.root_error());
            t.print()
        }
    }
}
