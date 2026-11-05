use std::collections::HashMap;

use crate::common::Cost;

/// An interval of candidate thresholds still to be searched, as indices into
/// the list of possible splits.
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    /// First candidate of the interval (inclusive).
    pub left_bound: usize,
    /// Last candidate of the interval (inclusive).
    pub right_bound: usize,
    /// The evaluated split just left of the interval, if any.
    pub last_split_left_index: Option<usize>,
    /// The evaluated split just right of the interval, if any.
    pub last_split_right_index: Option<usize>,
}

impl Bound {
    pub fn new(
        left: usize,
        right: usize,
        last_left: Option<usize>,
        last_right: Option<usize>,
    ) -> Self {
        Self {
            left_bound: left,
            right_bound: right,
            last_split_left_index: last_left,
            last_split_right_index: last_right,
        }
    }

    /// Whether the interval still contains a candidate.
    pub fn is_valid(&self) -> bool {
        self.left_bound <= self.right_bound
    }

    /// Empties the interval.
    pub fn invalidate(&mut self) {
        self.left_bound = 1;
        self.right_bound = 0;
        self.last_split_left_index = None;
        self.last_split_right_index = None;
    }
}

/// Prunes intervals of candidate thresholds for one feature.
///
/// Moving a threshold to the right only adds instances to the left child and
/// removes them from the right child, so child errors are monotone in the
/// threshold. From the errors of the thresholds already evaluated, the pruner
/// derives which neighbouring thresholds cannot beat the incumbent, following
/// the ConTree algorithm (Brită et al., AAAI 2025).
///
/// Moving a threshold past `n` instances changes the optimum of either side
/// by at most `n`. A threshold is therefore pruned when no more instances
/// than the score difference lie between it and an evaluated one.
pub struct IntervalsPruner<'a> {
    /// Sorted positions of the candidate thresholds.
    possible_split_indexes: &'a [usize],
    /// Whether the pruning rules apply. See [`IntervalsPruner::is_sound`].
    sound: bool,
    pub rightmost_zero_index: Option<usize>,
    pub leftmost_zero_index: Option<usize>,
    max_gap: Cost,
    /// Maps an evaluated split index to its `(left_score, right_score)`.
    evaluated_indices_record: HashMap<usize, (Cost, Cost)>,
}

impl<'a> IntervalsPruner<'a> {
    /// Creates a pruner over the candidate thresholds of one feature.
    ///
    /// # Arguments
    /// * `possible_split_indexes` - Sorted positions of the candidate thresholds.
    /// * `max_gap` - Error gap tolerated with respect to the optimum.
    /// * `min_sup` - Minimum number of instances per leaf.
    /// * `skipping` - Whether the loss allows threshold skipping; see
    ///   [`Skipping`](crate::common::Skipping).
    pub fn new(
        possible_split_indexes: &'a [usize],
        max_gap: Cost,
        min_sup: usize,
        skipping: bool,
    ) -> Self {
        let possible_split_size = possible_split_indexes.len();
        let mut evaluated_indices_record = HashMap::new();
        evaluated_indices_record.reserve(possible_split_size / 8);

        Self {
            possible_split_indexes,
            sound: skipping && Self::is_sound(min_sup),
            rightmost_zero_index: None,
            leftmost_zero_index: None,
            max_gap,
            evaluated_indices_record,
        }
    }

    /// Whether the pruning rules hold for a given minimum support.
    ///
    /// Every rule here rests on the same argument: for a split further to the
    /// right, the left child is a *superset* of the left child of a split
    /// already evaluated, so its optimal error cannot be smaller. That holds
    /// because the optimal tree for the superset, restricted to the subset,
    /// is a tree for the subset with no more errors.
    ///
    /// A minimum support constraint breaks it. The restricted tree can have a
    /// leaf that met the support threshold on the superset and misses it on
    /// the subset, so it is not a legal tree there and proves nothing. The
    /// subset optimum can then be *higher* than the superset optimum, and the
    /// pruner concludes an interval is hopeless when it is not.
    ///
    /// The pruning rules are therefore only used when the minimum support is
    /// at most 1.
    fn is_sound(min_sup: usize) -> bool {
        min_sup <= 1
    }

    /// Whether the whole interval can be discarded, given the scores of the
    /// splits on either side of it.
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds of the interval being evaluated
    /// * `current_best_score` - The best score obtained so far
    ///
    /// # Returns
    /// `true` if the subinterval can be pruned, `false` otherwise
    pub fn subinterval_pruning(&self, current_bounds: &Bound, current_best_score: Cost) -> bool {
        if !self.sound {
            return false;
        }
        let left_bound_score_left = current_bounds
            .last_split_left_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(left_score, _)| left_score)
            .unwrap_or(0.0);

        let right_bound_score_right = current_bounds
            .last_split_right_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(_, right_score)| right_score)
            .unwrap_or(0.0);

        left_bound_score_left + right_bound_score_right + self.max_gap >= current_best_score
    }

    /// Narrows the interval to the thresholds that may still beat
    /// `current_best_score`. The interval is emptied if none can.
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds to be updated by shrinking
    /// * `current_best_score` - The best score to compare against
    pub fn interval_shrinking(&self, current_bounds: &mut Bound, current_best_score: Cost) {
        if !self.sound {
            return;
        }
        // A zero-error child on either side rules out everything beyond it.
        if let Some(leftmost) = self.leftmost_zero_index {
            current_bounds.left_bound = current_bounds.left_bound.max(leftmost + 1);
        }

        if let Some(rightmost) = self.rightmost_zero_index {
            current_bounds.right_bound =
                current_bounds.right_bound.min(rightmost.saturating_sub(1));
        }

        // The adjustments above can push a bound past the end of the
        // candidate list (e.g. `leftmost + 1` when the zero is the last
        // candidate); clamp before indexing.
        if current_bounds.right_bound >= self.possible_split_indexes.len() {
            current_bounds.right_bound = self.possible_split_indexes.len().saturating_sub(1);
        }
        if self.possible_split_indexes.is_empty()
            || current_bounds.left_bound >= self.possible_split_indexes.len()
            || !current_bounds.is_valid()
        {
            current_bounds.invalidate();
            return;
        }

        if let Some(last_left_idx) = current_bounds.last_split_left_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_left_idx)
            {
                let sum = left_score + right_score + self.max_gap;
                if sum >= current_best_score {
                    // Viable thresholds lie more than the difference in
                    // instances to the right of the evaluated one.
                    let updated_score_difference = sum - current_best_score;
                    let limit = self.position(last_left_idx) + updated_score_difference;

                    if self.position(current_bounds.right_bound) > limit {
                        let new_left_bound = self.first_above(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            limit,
                        );
                        current_bounds.left_bound = current_bounds.left_bound.max(new_left_bound);
                    } else {
                        current_bounds.invalidate();
                        return;
                    }
                }
            }
        }

        if let Some(last_right_idx) = current_bounds.last_split_right_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_right_idx)
            {
                let sum = left_score + right_score + self.max_gap;
                if sum >= current_best_score {
                    // Mirror image: viable thresholds lie more than the
                    // difference in instances to the left.
                    let updated_score_difference = sum - current_best_score;
                    let limit = self.position(last_right_idx) - updated_score_difference;

                    if self.position(current_bounds.left_bound) < limit {
                        let new_right_bound = self.first_at_least(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            limit,
                        );
                        current_bounds.right_bound =
                            current_bounds.right_bound.min(new_right_bound);
                    } else {
                        current_bounds.invalidate();
                    }
                }
            }
        }
    }

    /// Computes the thresholds around `split_index` that cannot beat the
    /// incumbent, given how much the evaluated split exceeded it.
    ///
    /// # Arguments
    /// * `score_difference` - The difference in scores used to determine pruning
    /// * `left` - The left boundary of the interval
    /// * `right` - The right boundary of the interval
    /// * `split_index` - The index at which the split is evaluated
    ///
    /// # Returns
    /// `(new_left_bound, new_right_bound)`: the search continues on
    /// `[new_left_bound, right]` and `[left, new_right_bound]`.
    pub fn neighbourhood_pruning(
        &self,
        score_difference: Cost,
        left: usize,
        right: usize,
        split_index: usize,
    ) -> (usize, usize) {
        if !self.sound {
            // Prune only the point just evaluated.
            return (split_index + 1, split_index.saturating_sub(1));
        }
        let score_difference = score_difference + self.max_gap;

        if score_difference == 0.0 {
            return (split_index + 1, split_index);
        }

        let mut new_bound_left = split_index + 1;
        if let Some(leftmost) = self.leftmost_zero_index {
            new_bound_left = new_bound_left.max(leftmost + 1);
        }

        // A threshold to the right is viable when more than the difference
        // in instances lies between it and the evaluated one.
        let right_limit = self.position(split_index) + score_difference;
        if self.position(right) <= right_limit {
            new_bound_left = right + 1;
        } else {
            new_bound_left = self.first_above(new_bound_left, right, right_limit);
        }

        let mut new_bound_right = split_index.saturating_sub(1);
        if let Some(rightmost) = self.rightmost_zero_index {
            new_bound_right = new_bound_right.min(rightmost.saturating_sub(1));
        }

        // Mirror image of the test above.
        let left_limit = self.position(split_index) - score_difference;
        if self.position(left) >= left_limit {
            new_bound_right = left.saturating_sub(1);
        } else {
            new_bound_right = self.first_at_least(left, new_bound_right, left_limit);
        }

        (new_bound_left, new_bound_right)
    }

    /// Records the outcome of evaluating one split.
    ///
    /// A score is `None` when that side gives no usable lower bound, e.g.
    /// the right subtree was never explored because the left side alone
    /// already exceeded the upper bound. Such a side counts as 0, the only
    /// value that cannot prune away a better split.
    pub fn add_result(
        &mut self,
        index: usize,
        left_score: Option<Cost>,
        right_score: Option<Cost>,
    ) {
        if left_score == Some(0.0) {
            self.leftmost_zero_index = Some(
                self.leftmost_zero_index
                    .map(|v| v.max(index))
                    .unwrap_or(index),
            );
        }

        if right_score == Some(0.0) {
            self.rightmost_zero_index = Some(
                self.rightmost_zero_index
                    .map(|v| v.min(index))
                    .unwrap_or(index),
            );
        }

        self.evaluated_indices_record.insert(
            index,
            (left_score.unwrap_or(0.0), right_score.unwrap_or(0.0)),
        );
    }

    /// The position of candidate `index`, as a cost: moving the threshold
    /// past one instance changes a cost by at most 1.
    fn position(&self, index: usize) -> Cost {
        self.possible_split_indexes[index] as Cost
    }

    /// Index of the first candidate in `left..=right` whose position is
    /// `> value`.
    fn first_above(&self, left: usize, mut right: usize, value: Cost) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        left + slice.partition_point(|&p| p as Cost <= value)
    }

    /// Index of the first candidate in `left..=right` whose position is
    /// `>= value`.
    fn first_at_least(&self, left: usize, mut right: usize, value: Cost) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        left + slice.partition_point(|&p| (p as Cost) < value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruner_creation() {
        let splits = vec![10, 20, 30, 40, 50];
        let pruner = IntervalsPruner::new(&splits, 2.0, 1, true);

        assert_eq!(pruner.possible_split_indexes.len(), 5);
        assert_eq!(pruner.max_gap, 2.0);
        assert_eq!(pruner.leftmost_zero_index, None);
        assert_eq!(pruner.rightmost_zero_index, None);
    }

    #[test]
    fn test_bound_creation() {
        let bound = Bound::new(0, 10, None, None);
        assert_eq!(bound.left_bound, 0);
        assert_eq!(bound.right_bound, 10);
        assert_eq!(bound.last_split_left_index, None);
        assert_eq!(bound.last_split_right_index, None);
        assert!(bound.is_valid());
    }

    #[test]
    fn test_bound_invalidation() {
        let mut bound = Bound::new(0, 10, Some(5), Some(7));
        assert!(bound.is_valid());

        bound.invalidate();
        assert!(!bound.is_valid());
        assert_eq!(bound.last_split_left_index, None);
        assert_eq!(bound.last_split_right_index, None);
    }
}
