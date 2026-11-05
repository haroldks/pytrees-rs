use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::algorithms::shared::{checked_leaf, threshold_between};
use crate::algorithms::support_feasible_splits;
use crate::caching::Entry;
use crate::common::{classification_error, Cost, Loss, SearchConfig, Statistics};
use crate::data::view::DataView;
use crate::data::DataPoint;
use crate::tree::Tree;
use std::collections::VecDeque;

/// The best depth-1 tree found so far for one side of a depth-2 root split,
/// and the scan state used to find it.
///
/// With misclassification the scan compares scores, the number of correctly
/// classified instances, so higher is better. With a custom loss it compares
/// costs.
struct SubtreeLeafScores {
    // Best split found so far.
    classification_score: usize,
    best_cost: Cost,
    best_feature_index: usize,
    best_threshold: f64,
    best_left_label: Option<usize>,
    best_right_label: Option<usize>,
    best_left_error: Cost,
    best_right_error: Cost,

    // Scan state for the feature being processed.
    previous_value: f64,
    previous_unique_value_index: Option<usize>,
    /// No split on this feature can improve the score any more.
    is_zero: bool,
    /// Instances that have to move to the left before a threshold can beat
    /// the score: moving an instance changes the score by at most 1, so the
    /// thresholds before that are skipped.
    can_skip: Cost,
    current_element_count: usize,
    /// Costs of both leaves at the threshold evaluated last, and the number
    /// of instances then left of it. Used to check a custom loss under
    /// forced skipping.
    last_evaluated: Option<(Cost, Cost, usize)>,
    /// Scratch space for the class counts right of the threshold.
    remaining_frequency: Vec<usize>,

    // The side itself.
    size: usize,
    min_sup: usize,
    max_label_frequency: usize,
    max_label: usize,
    /// Cost and label of the side as a single leaf.
    leaf_cost: Cost,
    leaf_label: usize,
    label_frequency: Vec<usize>,
    current_label_frequency: Vec<usize>,
}

impl SubtreeLeafScores {
    fn new(size: usize, class_number: usize, min_sup: usize) -> Self {
        Self {
            classification_score: 0,
            best_cost: Cost::INFINITY,
            best_feature_index: 0,
            best_threshold: -1.0,
            best_left_label: None,
            best_right_label: None,
            best_left_error: 0.0,
            best_right_error: 0.0,
            previous_value: 0.0,
            previous_unique_value_index: None,
            is_zero: false,
            can_skip: 0.0,
            current_element_count: 0,
            last_evaluated: None,
            remaining_frequency: vec![0; class_number],
            size,
            min_sup: min_sup.max(1),
            max_label_frequency: 0,
            max_label: 0,
            leaf_cost: Cost::INFINITY,
            leaf_label: 0,
            label_frequency: vec![0; class_number],
            current_label_frequency: vec![0; class_number],
        }
    }

    fn reset_label_frequency(&mut self) {
        self.current_label_frequency.fill(0);
        self.previous_value = 0.0;
        self.previous_unique_value_index = None;
        self.is_zero = false;
        self.can_skip = 0.0;
        self.current_element_count = 0;
        self.last_evaluated = None;
    }

    /// Instances still to scan.
    fn remaining(&self) -> usize {
        self.size - self.current_element_count
    }

    /// Whether no threshold still to come can beat the best one: each needs
    /// `can_skip` more instances moved to the left, and fewer remain.
    fn cannot_improve(&self) -> bool {
        self.can_skip >= self.remaining() as Cost
    }

    /// Adds the instance to the left of the thresholds still to come.
    fn move_left(&mut self, data_point: &DataPoint) {
        self.current_element_count += 1;
        self.current_label_frequency[data_point.label() as usize] += 1;
        self.previous_value = data_point.value();
        self.previous_unique_value_index = Some(data_point.unique_value_id());
    }

    /// Instances of classes other than `label` among the scanned instances
    /// (`left`) or the instances still to scan (`!left`): the error of a
    /// leaf predicting `label` there.
    fn other_classes(&self, label: usize, left: bool) -> Cost {
        let others: usize = (0..self.label_frequency.len())
            .filter(|&l| l != label)
            .map(|l| {
                if left {
                    self.current_label_frequency[l]
                } else {
                    self.label_frequency[l] - self.current_label_frequency[l]
                }
            })
            .sum();
        others as Cost
    }
}

/// Why the best labels of a side are set when they are read: its score starts
/// at the leaf's score and only rises when a split also sets the labels, and
/// they are read only when the score is above the leaf's.
const LABELS_SET: &str = "the labels are set whenever the score beats the leaf";

/// The loss of a depth-2 search and what the search reports about it.
pub(crate) struct Depth2Loss<'a> {
    pub loss: &'a Loss,
    /// Whether thresholds are skipped; see [`Skipping`](crate::common::Skipping).
    pub skipping: bool,
    /// The first negative or NaN cost the loss returned.
    pub invalid: &'a mut Option<Cost>,
    /// Moves that broke the skipping assumption; added to the statistics.
    violations: usize,
}

impl<'a> Depth2Loss<'a> {
    pub fn new(loss: &'a Loss, skipping: bool, invalid: &'a mut Option<Cost>) -> Self {
        Self {
            loss,
            skipping,
            invalid,
            violations: 0,
        }
    }

    fn is_custom(&self) -> bool {
        !self.loss.is_misclassification()
    }

    /// The `(cost, label)` of a leaf with these class counts. A loss that
    /// reads instance ids never reaches the depth-2 solver.
    fn leaf(&mut self, class_counts: &[usize]) -> (Cost, usize) {
        checked_leaf(
            self.loss,
            class_counts,
            || unreachable!("the depth-2 solver is off for losses that read instance ids"),
            self.invalid,
        )
    }
}

/// Specialised solver for trees of depth at most 2.
///
/// For each root threshold it finds the best depth-1 tree on both sides in a
/// single scan per feature, reusing the class counts of the scan, instead of
/// recursing.
#[derive(Default)]
pub struct ConTreeDepth2;

impl ConTreeDepth2 {
    /// Finds the best depth-2 tree on `view` that beats `entry.error` and
    /// `upper_bound`, updating `entry` when it does. Whether the result is
    /// exact depends on `upper_bound`, so the caller decides, through
    /// `Entry::finalize_lower_bound`.
    pub fn fit(
        &self,
        view: &DataView<'_>,
        search_config: &SearchConfig,
        entry: &mut Entry,
        upper_bound: Cost,
        stats: &mut Statistics,
        costs: &mut Depth2Loss<'_>,
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
                costs,
            );
            if entry.error <= search_config.max_gap {
                break;
            }
        }
        stats.skipping_violations += std::mem::take(&mut costs.violations);
        tree.normalize_leaves();
        tree
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_feature_subtree(
        &self,
        view: &DataView<'_>,
        feature: usize,
        config: &SearchConfig,
        entry: &mut Entry,
        // Splits scoring at or above this bound are of no use to the caller.
        upper_bound: Cost,
        tree: &mut Tree,
        stats: &mut Statistics,
        costs: &mut Depth2Loss<'_>,
    ) {
        let feature_column = view.get_sorted_feature(feature);
        let feature_column_ids = view.get_feature_indices(feature);
        let possible_split_indices = view.get_possible_split_indices(feature);

        if possible_split_indices.is_empty() {
            return;
        }

        // Only consider root splits that leave `min_sup` instances per side.
        let feasible = support_feasible_splits(
            possible_split_indices,
            view.get_dataset_size(),
            config.min_sup,
        );
        if feasible.is_empty() {
            return;
        }

        let mut pruner = IntervalsPruner::new(
            possible_split_indices,
            config.max_gap,
            config.min_sup,
            costs.skipping,
        );

        let mut queue = VecDeque::new();
        let init_bound = Bound::new(feasible.start, feasible.end - 1, None, None);
        queue.push_back(init_bound);

        while let Some(mut current_bound) = queue.pop_front() {
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
                costs,
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

                entry.feature = tree.root_feature().unwrap_or(usize::MAX);
                entry.split = tree.root_split().unwrap_or(f64::INFINITY);

                let (left, right) = tree.node_children(tree.get_root_index());
                tree.update_subtree(left, &left_tree, left_tree.get_root_index());

                tree.update_subtree(right, &right_tree, right_tree.get_root_index());

                if current_best_error == 0.0 {
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

            let score_difference = (left_tree.root_error() + right_tree.root_error()
                - entry.error.min(upper_bound))
            .max(0.0);
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
        best_error: Cost,
        min_sup: usize,
        costs: &mut Depth2Loss<'_>,
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

        // The threshold lies between two values of the feature, so at least
        // one value is above it.
        let split_index = split_index.expect("a value lies above the threshold");

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
        for leaves in [&mut left_leaves, &mut right_leaves] {
            if costs.is_custom() {
                (leaves.leaf_cost, leaves.leaf_label) = costs.leaf(&leaves.label_frequency);
                leaves.best_cost = leaves.leaf_cost;
            } else {
                leaves.leaf_cost = classification_error(&leaves.label_frequency).0;
                leaves.leaf_label = leaves.max_label;
            }
        }

        for current_feature_index in 0..view.get_feature_number() {
            if current_feature_index == feature {
                self.process_depth_one_feature::<true>(
                    view,
                    feature,
                    split_point,
                    current_feature_index,
                    split_index,
                    &mut left_leaves,
                    &mut right_leaves,
                    &split_feature_split_indices,
                    &mut upper_bound,
                    costs,
                );
            } else {
                self.process_depth_one_feature::<false>(
                    view,
                    feature,
                    split_point,
                    current_feature_index,
                    split_index,
                    &mut left_leaves,
                    &mut right_leaves,
                    &split_feature_split_indices,
                    &mut upper_bound,
                    costs,
                );
            }
            let perfect = if costs.is_custom() {
                left_leaves.best_cost == 0.0 && right_leaves.best_cost == 0.0
            } else {
                left_leaves.classification_score >= left_leaves.size
                    && right_leaves.classification_score >= right_leaves.size
            };
            if perfect {
                break;
            }
        }

        if left_leaves.best_left_label.is_none() {
            left_tree.update_root().map(|updater| {
                updater
                    .label(left_leaves.leaf_label)
                    .error(left_leaves.leaf_cost)
                    .leaf()
            });
        } else {
            left_tree.update_root().map(|updater| {
                updater
                    .feature(left_leaves.best_feature_index)
                    .split(left_leaves.best_threshold)
                    .error(left_leaves.best_left_error + left_leaves.best_right_error)
            });

            let (ll, lr) = left_tree.node_children(left_tree.get_root_index());

            left_tree.update_node(ll).map(|updater| {
                updater
                    .label(left_leaves.best_left_label.expect(LABELS_SET))
                    .error(left_leaves.best_left_error)
                    .leaf()
            });

            left_tree.update_node(lr).map(|updater| {
                updater
                    .label(left_leaves.best_right_label.expect(LABELS_SET))
                    .error(left_leaves.best_right_error)
                    .leaf()
            });
        }

        if right_leaves.best_left_label.is_none() {
            right_tree.update_root().map(|updater| {
                updater
                    .label(right_leaves.leaf_label)
                    .error(right_leaves.leaf_cost)
                    .leaf()
            });
        } else {
            right_tree.update_root().map(|updater| {
                updater
                    .feature(right_leaves.best_feature_index)
                    .split(right_leaves.best_threshold)
                    .error(right_leaves.best_left_error + right_leaves.best_right_error)
            });

            let (ll, lr) = right_tree.node_children(right_tree.get_root_index());

            right_tree.update_node(ll).map(|updater| {
                updater
                    .label(right_leaves.best_left_label.expect(LABELS_SET))
                    .error(right_leaves.best_left_error)
                    .leaf()
            });

            right_tree.update_node(lr).map(|updater| {
                updater
                    .label(right_leaves.best_right_label.expect(LABELS_SET))
                    .error(right_leaves.best_right_error)
                    .leaf()
            });
        }
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
        upper_bound: &mut Cost,
        costs: &mut Depth2Loss<'_>,
    ) {
        let current_feature = view.get_sorted_feature(current_feature_index);
        let feature_column_ids = view.get_feature_indices(current_feature_index);
        let class_number = view.get_num_labels();
        let total_size = left_tree.size + right_tree.size;

        left_tree.reset_label_frequency();
        right_tree.reset_label_frequency();

        let mut index = 0;

        for i in 0..feature_column_ids.len() {
            let current_feature_data = &current_feature[feature_column_ids[i]];

            if IS_SAME_FEATURE {
                // Same feature as the root split: position decides the side.
                if index < split_point {
                    Self::process_single_point(
                        current_feature_data,
                        left_tree,
                        class_number,
                        current_feature_index,
                        costs,
                    );
                } else {
                    Self::process_single_point(
                        current_feature_data,
                        right_tree,
                        class_number,
                        current_feature_index,
                        costs,
                    );
                }
                index += 1;
            } else {
                // Other feature: look up the side from the root split's index.
                let cur_split_index = split_feature_split_indices[current_feature_data.tid()];
                if cur_split_index < split_index {
                    Self::process_single_point(
                        current_feature_data,
                        left_tree,
                        class_number,
                        current_feature_index,
                        costs,
                    );
                } else {
                    Self::process_single_point(
                        current_feature_data,
                        right_tree,
                        class_number,
                        current_feature_index,
                        costs,
                    );
                }
            }

            if !costs.is_custom() {
                *upper_bound = (*upper_bound).min(total_size.saturating_sub(
                    left_tree.classification_score + right_tree.classification_score,
                ) as Cost);
            }

            if left_tree.is_zero && right_tree.is_zero {
                break;
            }

            if left_tree.cannot_improve() && right_tree.cannot_improve() {
                break;
            }
        }
    }

    /// Advances the scan of one side by one instance, first evaluating the
    /// threshold just before it if its value differs from the previous one.
    fn process_single_point(
        current_feature_data: &DataPoint,
        tree: &mut SubtreeLeafScores,
        class_number: usize,
        current_feature_index: usize,
        costs: &mut Depth2Loss<'_>,
    ) {
        if costs.is_custom() {
            Self::process_single_point_custom(
                current_feature_data,
                tree,
                current_feature_index,
                costs,
            );
            return;
        }
        if tree.is_zero {
            return;
        }

        // The instance scanned last is now left of this threshold.
        tree.can_skip = (tree.can_skip - 1.0).max(0.0);

        // No threshold between equal values, or inside a skipped stretch.
        if Some(current_feature_data.unique_value_id()) == tree.previous_unique_value_index
            || tree.can_skip > 0.0
        {
            tree.move_left(current_feature_data);
            return;
        }

        let mut left_classification_score = 0;
        let mut right_classification_score = 0;
        let mut left_label = None;
        let mut right_label = None;

        for label_value in 0..class_number {
            let current_freq = tree.current_label_frequency[label_value];
            if left_label.is_none() || current_freq > left_classification_score {
                left_classification_score = current_freq;
                left_label = Some(label_value);
            }

            let remaining =
                tree.label_frequency[label_value] - tree.current_label_frequency[label_value];
            if right_label.is_none() || remaining > right_classification_score {
                right_classification_score = remaining;
                right_label = Some(label_value);
            }
        }
        let left_label = left_label.unwrap_or(0);
        let right_label = right_label.unwrap_or(0);

        let total_score = left_classification_score + right_classification_score;

        let left_size = tree.current_element_count;
        let right_size = tree.remaining();
        let has_support = left_size >= tree.min_sup && right_size >= tree.min_sup;

        if has_support && total_score > 0 && total_score > tree.classification_score {
            tree.classification_score = total_score;
            tree.best_feature_index = current_feature_index;
            tree.best_threshold =
                threshold_between(tree.previous_value, current_feature_data.value());

            tree.best_left_label = Some(left_label);
            tree.best_right_label = Some(right_label);
            tree.best_left_error = tree.other_classes(left_label, true);
            tree.best_right_error = tree.other_classes(right_label, false);
        } else if costs.skipping {
            tree.can_skip = tree.classification_score.saturating_sub(total_score) as Cost;
        }

        if costs.skipping {
            // A pure right side only loses score as the threshold moves right,
            // so no later threshold beats this one. That ends the scan only if
            // this one could be chosen: while the left side is below the
            // minimum support, a later threshold that meets it can still beat
            // the best.
            let right_is_pure = right_classification_score == right_size;
            tree.is_zero |= (right_is_pure && left_size >= tree.min_sup) || tree.cannot_improve();
        }

        tree.move_left(current_feature_data);
    }

    /// `process_single_point` for a custom loss, which compares costs.
    ///
    /// Skipping and the early stops hold under the assumption `Skipping`
    /// states, so they apply only when skipping is on. Then each evaluated
    /// threshold is also checked against that assumption: since the one
    /// evaluated before it, the left leaf gained, and the right leaf lost,
    /// between 0 and the number of instances moved across.
    fn process_single_point_custom(
        current_feature_data: &DataPoint,
        tree: &mut SubtreeLeafScores,
        current_feature_index: usize,
        costs: &mut Depth2Loss<'_>,
    ) {
        if tree.is_zero {
            return;
        }
        tree.can_skip = (tree.can_skip - 1.0).max(0.0);
        if Some(current_feature_data.unique_value_id()) == tree.previous_unique_value_index
            || tree.can_skip > 0.0
        {
            tree.move_left(current_feature_data);
            return;
        }

        for label in 0..tree.label_frequency.len() {
            tree.remaining_frequency[label] =
                tree.label_frequency[label] - tree.current_label_frequency[label];
        }
        let (left_cost, left_label) = costs.leaf(&tree.current_label_frequency);
        let (right_cost, right_label) = costs.leaf(&tree.remaining_frequency);

        if costs.skipping {
            if let Some((previous_left, previous_right, previous_count)) = tree.last_evaluated {
                let moved = (tree.current_element_count - previous_count) as Cost;
                let within = |change: Cost| (0.0..=moved).contains(&change);
                if !(within(left_cost - previous_left) && within(previous_right - right_cost)) {
                    costs.violations += 1;
                }
            }
            tree.last_evaluated = Some((left_cost, right_cost, tree.current_element_count));
        }

        let total_cost = left_cost + right_cost;
        let left_size = tree.current_element_count;
        let right_size = tree.remaining();
        let has_support = left_size >= tree.min_sup && right_size >= tree.min_sup;

        if has_support && total_cost < tree.best_cost {
            tree.best_cost = total_cost;
            tree.best_feature_index = current_feature_index;
            tree.best_threshold =
                threshold_between(tree.previous_value, current_feature_data.value());
            tree.best_left_label = Some(left_label);
            tree.best_right_label = Some(right_label);
            tree.best_left_error = left_cost;
            tree.best_right_error = right_cost;
        } else if costs.skipping {
            tree.can_skip = (total_cost - tree.best_cost).max(0.0);
        }

        if costs.skipping {
            tree.is_zero |=
                (right_cost == 0.0 && left_size >= tree.min_sup) || tree.cannot_improve();
        }

        tree.move_left(current_feature_data);
    }
}

#[cfg(test)]
mod d2_test {
    use crate::algorithms::depth2::{ConTreeDepth2, Depth2Loss};
    use crate::caching::Entry;
    use crate::common::{
        classification_error, Cost, Loss, PointSelector, SearchConfig, Statistics,
    };
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
        // The majority-class leaf is the incumbent to beat.
        let mut entry = Entry {
            error: leaf_error,
            label: leaf_label,
            ..Entry::default()
        };
        let config = SearchConfig {
            max_depth: 2,
            min_sup: 1,
            max_time: 10.0,
            max_gap: 0.0,
            max_error: Cost::INFINITY,
            cost_step: 1.0,
            skipping: crate::common::Skipping::Auto,
            is_root: true,
            use_heuristic: true,
            fast_d2: true,
            point_selector: PointSelector::default(),
            nb_runs: 0,
            discrepancy: 0,
            budget: 0,
        };

        // No upper bound from a parent: this is the root.
        let loss = Loss::default();
        let mut invalid = None;
        let tree = d2.fit(
            &root_view,
            &config,
            &mut entry,
            Cost::INFINITY,
            &mut Statistics::default(),
            &mut Depth2Loss::new(&loss, true, &mut invalid),
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
