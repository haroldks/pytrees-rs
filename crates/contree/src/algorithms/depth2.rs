use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::algorithms::shared::threshold_between;
use crate::algorithms::support_feasible_splits;
use crate::caching::Entry;
use crate::common::{SearchConfig, Statistics};
use crate::data::view::DataView;
use crate::data::DataPoint;
use crate::tree::Tree;
use std::collections::VecDeque;

/// Helper struct to track state while finding optimal depth-1 trees
struct SubtreeLeafScores {
    // Best split found so far
    classification_score: usize,
    best_feature_index: usize,
    best_threshold: f64,
    best_left_label: Option<usize>,
    best_right_label: Option<usize>,

    best_left_error: usize,  // Error for the left leaf
    best_right_error: usize, // Error for the right leaf

    // State tracking during feature iteration
    previous_value: f64,
    previous_unique_value_index: Option<usize>,
    is_zero: bool,
    can_skip: usize,
    current_element_count: usize,

    // Tree properties
    size: usize,
    min_sup: usize,
    max_label_frequency: usize,
    max_label: usize,

    // Label frequency vectors
    label_frequency: Vec<usize>,
    current_label_frequency: Vec<usize>,
}

impl SubtreeLeafScores {
    fn new(size: usize, class_number: usize, min_sup: usize) -> Self {
        Self {
            classification_score: 0,
            best_feature_index: 0,
            best_threshold: -1.0,
            best_left_label: None,
            best_right_label: None,
            best_left_error: 0,
            best_right_error: 0,
            previous_value: 0.0,
            previous_unique_value_index: None,
            is_zero: false,
            can_skip: 0,
            current_element_count: 0,
            size,
            min_sup: min_sup.max(1),
            max_label_frequency: 0,
            max_label: 0,
            label_frequency: vec![0; class_number],
            current_label_frequency: vec![0; class_number],
        }
    }

    fn reset_label_frequency(&mut self) {
        self.current_label_frequency.fill(0);
        self.previous_value = 0.0;
        self.previous_unique_value_index = None;
        self.is_zero = false;
        self.can_skip = 0;
        self.current_element_count = 0;
    }
}

#[derive(Default)]
pub struct ConTreeDepth2;

impl ConTreeDepth2 {
    pub fn fit(
        &self,
        view: &DataView<'_>,
        search_config: &SearchConfig,
        entry: &mut Entry,
        upper_bound: usize,
        stats: &mut Statistics,
    ) -> Tree {
        debug_assert!(search_config.max_depth <= 2, "Depth should be at most 2");

        let num_features = view.get_feature_number();
        let mut tree = Tree::empty_tree(2);
        tree.update_root().map(|updater| {
            updater
                .error(entry.error)
                .feature(entry.feature)
                .label(entry.label)
        });

        for feature in 0..num_features {
            self.expand_feature_subtree(
                view,
                feature,
                search_config,
                entry,
                entry.error.min(upper_bound),
                &mut tree,
                stats,
            );
            if entry.error <= search_config.max_gap {
                break;
            }
        }
        tree.normalize_leaves();
        // Whether this is an exact answer depends on the budget the caller
        // passed in -- the search above prunes against it -- so the caller
        // decides, through `Entry::finalize_lower_bound`.
        tree
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_feature_subtree(
        &self,
        view: &DataView<'_>,
        feature: usize,
        config: &SearchConfig,
        entry: &mut Entry,
        // The budget the parent handed down: no split here scoring at or
        // above it is any use to the caller. Upstream's depth-2 node search
        // prunes against it; this solver used to accept it and ignore it.
        upper_bound: usize,
        tree: &mut Tree,
        stats: &mut Statistics,
    ) {
        let feature_column = view.get_sorted_feature(feature);
        let feature_column_ids = view.get_feature_indices(feature);
        let possible_split_indices = view.get_possible_split_indices(feature);

        if possible_split_indices.is_empty() {
            return;
        }

        // Minimum support applies to the root split of this depth-2 subtree.
        // Restricting the interval up front keeps every candidate the search
        // ever looks at feasible.
        let feasible = support_feasible_splits(
            possible_split_indices,
            view.get_dataset_size(),
            config.min_sup,
        );
        if feasible.is_empty() {
            return;
        }

        let mut pruner =
            IntervalsPruner::new(possible_split_indices, config.max_gap, config.min_sup);

        let mut queue = VecDeque::new();
        let init_bound = Bound::new(feasible.start, feasible.end - 1, None, None);
        queue.push_back(init_bound);

        while !queue.is_empty() {
            let mut current_bound = queue.pop_front().unwrap();
            if pruner.subinterval_pruning(&current_bound, entry.error.min(upper_bound)) {
                continue;
            }

            pruner.interval_shrinking(&mut current_bound, entry.error.min(upper_bound));
            if !current_bound.is_valid() {
                continue;
            }

            let mid = (current_bound.left_bound + current_bound.right_bound) / 2;
            let split_point = possible_split_indices[mid];

            let threshold_value = if mid > 0 {
                let previous = feature_column_ids[possible_split_indices[mid - 1]];
                let point = feature_column_ids[split_point];
                threshold_between(
                    feature_column[previous].value(),
                    feature_column[point].value(),
                )
            } else {
                let point = feature_column_ids[split_point];
                threshold_between(
                    feature_column[feature_column_ids[0]].value(),
                    feature_column[point].value(),
                )
            };

            let mut left_tree = Tree::empty_tree(1);
            let mut right_tree = Tree::empty_tree(1);
            self.get_leaves_score(
                view,
                feature,
                split_point,
                threshold_value,
                &mut left_tree,
                &mut right_tree,
                entry.error,
                config.min_sup,
            );
            stats.specialized_solver_call += 1;
            let current_best_error = left_tree.root_error() + right_tree.root_error();
            if current_best_error < entry.error {
                entry.error = current_best_error;
                tree.update_root().map(|updater| {
                    updater
                        .error(entry.error)
                        .feature(feature)
                        .split(threshold_value)
                });

                entry.feature = tree.root_feature().map_or(usize::MAX, |v| v);
                entry.split = tree.root_split().map_or(f64::INFINITY, |v| v);

                let (left, right) = tree.node_children(tree.get_root_index());
                tree.update_subtree(left, &left_tree, left_tree.get_root_index());

                tree.update_subtree(right, &right_tree, right_tree.get_root_index());

                if current_best_error == 0 {
                    return;
                }
            }

            pruner.add_result(
                mid,
                Some(left_tree.root_error()),
                Some(right_tree.root_error()),
            );
            if current_bound.left_bound == current_bound.right_bound {
                continue;
            }

            let score_difference = (left_tree.root_error() + right_tree.root_error())
                .saturating_sub(entry.error.min(upper_bound));
            let (left_bound, right_bound) = pruner.neighbourhood_pruning(
                score_difference,
                current_bound.left_bound,
                current_bound.right_bound,
                mid,
            );
            if left_bound <= current_bound.right_bound {
                queue.push_back(Bound {
                    left_bound,
                    right_bound: current_bound.right_bound,
                    last_split_left_index: Some(mid),
                    last_split_right_index: current_bound.last_split_right_index,
                });
            }
            if current_bound.left_bound <= right_bound {
                queue.push_back(Bound {
                    left_bound: current_bound.left_bound,
                    right_bound,
                    last_split_left_index: current_bound.last_split_left_index,
                    last_split_right_index: Some(mid),
                });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get_leaves_score(
        &self,
        view: &DataView<'_>,
        feature: usize,
        split_point: usize,
        threshold_value: f64,
        left_tree: &mut Tree,
        right_tree: &mut Tree,
        best_error: usize,
        min_sup: usize,
    ) {
        let split_feature = view.get_sorted_feature(feature);
        let feature_column_ids = view.get_feature_indices(feature);
        let mut split_feature_split_indices = vec![0; view.total_instances];
        let mut split_index: Option<usize> = None;

        let size = feature_column_ids.len();
        for i in 0..size {
            let data_point = &split_feature[feature_column_ids[i]];
            split_feature_split_indices[data_point.tid()] = data_point.unique_value_id();
            if split_index.is_none() && data_point.value() > threshold_value {
                split_index = Some(data_point.unique_value_id());
            }
        }

        debug_assert!(split_index.is_some(), "Split not found");

        let dataset_size = view.get_dataset_size();
        let class_number = view.get_num_labels();

        debug_assert!(
            split_point > 0 && split_point < dataset_size,
            "left and right subtree need to be non-empty"
        );

        let mut left_leaves = SubtreeLeafScores::new(split_point, class_number, min_sup);
        let mut right_leaves =
            SubtreeLeafScores::new(dataset_size - split_point, class_number, min_sup);

        view.initialize_split_parameters(
            feature,
            split_point,
            &mut left_leaves.label_frequency,
            &mut right_leaves.label_frequency,
        );
        let mut upper_bound = best_error;

        for label in 0..class_number {
            if left_leaves.label_frequency[label] > left_leaves.max_label_frequency {
                left_leaves.max_label_frequency = left_leaves.label_frequency[label];
                left_leaves.max_label = label;
            }
            if right_leaves.label_frequency[label] > right_leaves.max_label_frequency {
                right_leaves.max_label_frequency = right_leaves.label_frequency[label];
                right_leaves.max_label = label;
            }
        }
        left_leaves.classification_score = left_leaves
            .classification_score
            .max(left_leaves.max_label_frequency);
        right_leaves.classification_score = right_leaves
            .classification_score
            .max(right_leaves.max_label_frequency);

        for current_feature_index in 0..view.get_feature_number() {
            if current_feature_index == feature {
                self.process_depth_one_feature::<true>(
                    view,
                    feature,
                    split_point,
                    current_feature_index,
                    split_index.unwrap(),
                    &mut left_leaves,
                    &mut right_leaves,
                    &split_feature_split_indices,
                    &mut upper_bound,
                );
            } else {
                self.process_depth_one_feature::<false>(
                    view,
                    feature,
                    split_point,
                    current_feature_index,
                    split_index.unwrap(),
                    &mut left_leaves,
                    &mut right_leaves,
                    &split_feature_split_indices,
                    &mut upper_bound,
                );
            }
            if left_leaves.classification_score + right_leaves.classification_score == dataset_size
            {
                break;
            }
        }

        if left_leaves.classification_score == left_leaves.max_label_frequency {
            // Make a leaf with the majority class
            left_tree.update_root().map(|updater| {
                updater
                    .label(left_leaves.max_label)
                    .error(left_leaves.size - left_leaves.classification_score)
                    .leaf()
            });
        } else {
            left_tree.update_root().map(|updater| {
                updater
                    .feature(left_leaves.best_feature_index)
                    .split(left_leaves.best_threshold)
                    .error(left_leaves.size - left_leaves.classification_score)
            });

            let (ll, lr) = left_tree.node_children(left_tree.get_root_index());

            left_tree.update_node(ll).map(|updater| {
                updater
                    .label(left_leaves.best_left_label.unwrap())
                    .error(left_leaves.best_left_error)
                    .leaf()
            });

            left_tree.update_node(lr).map(|updater| {
                updater
                    .label(left_leaves.best_right_label.unwrap())
                    .error(left_leaves.best_right_error)
                    .leaf()
            });
        }
        debug_assert!(
            left_leaves.classification_score <= left_leaves.size,
            "LR - Left tree error should be non-negative"
        );

        if right_leaves.classification_score == right_leaves.max_label_frequency {
            // Make a leaf with the majority class
            right_tree.update_root().map(|updater| {
                updater
                    .label(right_leaves.max_label)
                    .error(right_leaves.size - right_leaves.classification_score)
                    .leaf()
            });
        } else {
            // Make a split with two leaf children
            right_tree.update_root().map(|updater| {
                updater
                    .feature(right_leaves.best_feature_index)
                    .split(right_leaves.best_threshold)
                    .error(right_leaves.size - right_leaves.classification_score)
            });

            let (ll, lr) = right_tree.node_children(right_tree.get_root_index());

            right_tree.update_node(ll).map(|updater| {
                updater
                    .label(right_leaves.best_left_label.unwrap())
                    .error(right_leaves.best_left_error)
                    .leaf()
            });

            right_tree.update_node(lr).map(|updater| {
                updater
                    .label(right_leaves.best_right_label.unwrap())
                    .error(right_leaves.best_right_error)
                    .leaf()
            });
        }
        debug_assert!(
            right_leaves.classification_score <= right_leaves.size,
            "LR - Right tree error should be non-negative"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn process_depth_one_feature<const IS_SAME_FEATURE: bool>(
        &self,
        view: &DataView<'_>,
        _feature: usize,
        split_point: usize,
        current_feature_index: usize,
        split_index: usize,
        left_tree: &mut SubtreeLeafScores,
        right_tree: &mut SubtreeLeafScores,
        split_feature_split_indices: &[usize],
        upper_bound: &mut usize,
    ) {
        let current_feature = view.get_sorted_feature(current_feature_index);
        let feature_column_ids = view.get_feature_indices(current_feature_index);
        let class_number = view.get_num_labels();
        let dataset_size = view.get_dataset_size();

        left_tree.reset_label_frequency();
        right_tree.reset_label_frequency();

        let mut index = 0;

        for i in 0..feature_column_ids.len() {
            let current_feature_data = &current_feature[feature_column_ids[i]];

            // Determine which tree and process it
            if IS_SAME_FEATURE {
                // When processing the same feature, data is already sorted
                if index < split_point {
                    Self::process_single_point(
                        current_feature_data,
                        left_tree,
                        class_number,
                        current_feature_index,
                    );
                } else {
                    Self::process_single_point(
                        current_feature_data,
                        right_tree,
                        class_number,
                        current_feature_index,
                    );
                }
                index += 1;
            } else {
                // When processing different feature, lookup which side
                let cur_split_index = split_feature_split_indices[current_feature_data.tid()];
                if cur_split_index < split_index {
                    Self::process_single_point(
                        current_feature_data,
                        left_tree,
                        class_number,
                        current_feature_index,
                    );
                } else {
                    Self::process_single_point(
                        current_feature_data,
                        right_tree,
                        class_number,
                        current_feature_index,
                    );
                }
            }

            // Update upper bound (now both trees are accessible)
            *upper_bound = (*upper_bound).min(
                dataset_size - (left_tree.classification_score + right_tree.classification_score),
            );

            // Early termination checks
            if left_tree.is_zero && right_tree.is_zero {
                break;
            }

            if left_tree.can_skip >= (left_tree.size - left_tree.current_element_count)
                && right_tree.can_skip >= (right_tree.size - right_tree.current_element_count)
            {
                break;
            }
        }
    }

    // Helper function to process a single data point for one tree
    fn process_single_point(
        current_feature_data: &DataPoint, // Adjust type to match your data structure
        tree: &mut SubtreeLeafScores,
        class_number: usize,
        current_feature_index: usize,
    ) {
        if tree.is_zero {
            return;
        }

        tree.can_skip = tree.can_skip.saturating_sub(1);

        // Skip if same value or in skip mode
        if Some(current_feature_data.unique_value_id()) == tree.previous_unique_value_index
            || tree.can_skip > 0
        {
            tree.current_element_count += 1;
            tree.current_label_frequency[current_feature_data.label() as usize] += 1;
            tree.previous_value = current_feature_data.value();
            tree.previous_unique_value_index = Some(current_feature_data.unique_value_id());
            return;
        }

        // Evaluate split at this threshold
        let mut left_classification_score = -1i32;
        let mut right_classification_score = -1i32;
        let mut left_label = 0;
        let mut right_label = 0;

        for label_value in 0..class_number {
            let current_freq = tree.current_label_frequency[label_value] as i32;
            if current_freq > left_classification_score {
                left_classification_score = current_freq;
                left_label = label_value;
            }

            let remaining = (tree.label_frequency[label_value]
                - tree.current_label_frequency[label_value]) as i32;
            if remaining > right_classification_score {
                right_classification_score = remaining;
                right_label = label_value;
            }
        }

        let total_score = left_classification_score + right_classification_score;

        let left_size = tree.current_element_count;
        let right_size = tree.size - tree.current_element_count;
        let has_support = left_size >= tree.min_sup && right_size >= tree.min_sup;

        // Update if improved
        if has_support && total_score > 0 && total_score as usize > tree.classification_score {
            debug_assert!(tree.classification_score <= tree.size);
            tree.classification_score = total_score as usize;
            tree.best_feature_index = current_feature_index;
            tree.best_threshold =
                threshold_between(tree.previous_value, current_feature_data.value());

            tree.best_left_label = Some(left_label);
            tree.best_right_label = Some(right_label);
            tree.best_left_error =
                (tree.current_element_count as i32 - left_classification_score) as usize;
            tree.best_right_error = ((tree.size - tree.current_element_count) as i32
                - right_classification_score) as usize;
        } else {
            tree.can_skip = tree
                .classification_score
                .saturating_sub(total_score as usize);
        }

        // Early termination check for this tree
        let remaining_size = tree.size - tree.current_element_count;
        tree.is_zero |= (right_classification_score == remaining_size as i32)
            || (tree.can_skip >= remaining_size);

        // Update state
        tree.current_element_count += 1;
        tree.current_label_frequency[current_feature_data.label() as usize] += 1;
        tree.previous_value = current_feature_data.value();
        tree.previous_unique_value_index = Some(current_feature_data.unique_value_id());
    }
}

#[cfg(test)]
mod d2_test {
    use crate::algorithms::depth2::ConTreeDepth2;
    use crate::caching::Entry;
    use crate::common::{classification_error, PointSelector, SearchConfig, Statistics};
    use crate::data::view::DataView;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;

    use crate::tests::fixture;

    #[test]
    fn depth_two_beats_the_majority_class_leaf() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("hepatitis.txt"))?;
        dataset.sort_features();

        let root_view = DataView::root(&dataset, true);
        let (leaf_error, leaf_label) = classification_error(root_view.get_labels_freqs());

        let d2 = ConTreeDepth2;
        // Seeded the way the searches seed it: the majority-class leaf is the
        // incumbent the solver has to beat.
        let mut entry = Entry {
            error: leaf_error,
            label: leaf_label,
            ..Entry::default()
        };
        let config = SearchConfig {
            max_depth: 2,
            min_sup: 1,
            max_time: 10.0,
            max_gap: 0,
            max_error: usize::MAX,
            is_root: true,
            use_heuristic: true,
            fast_d2: true,
            point_selector: PointSelector::default(),
            nb_runs: 0,
            discrepancy: 0,
            budget: 0,
        };

        // No budget from a parent: this is the root.
        let tree = d2.fit(
            &root_view,
            &config,
            &mut entry,
            usize::MAX,
            &mut Statistics::default(),
        );

        assert!(
            entry.error <= leaf_error,
            "a depth-2 tree scored {} where a single leaf scores {leaf_error}",
            entry.error
        );
        assert!(!tree.is_empty(), "the specialized solver returned no tree");
        assert_eq!(tree.root_error(), entry.error);

        Ok(())
    }
}
