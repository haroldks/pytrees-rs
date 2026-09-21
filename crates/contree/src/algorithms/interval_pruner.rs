use std::collections::HashMap;

/// Bound structure representing an interval with metadata about previous splits
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub left_bound: usize,
    pub right_bound: usize,
    pub last_split_left_index: Option<usize>,
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

    /// Check if the interval is valid (left <= right)
    pub fn is_valid(&self) -> bool {
        self.left_bound <= self.right_bound
    }

    /// Mark interval as invalid
    pub fn invalidate(&mut self) {
        self.left_bound = 1;
        self.right_bound = 0;
        self.last_split_left_index = None;
        self.last_split_right_index = None;
    }
}

/// IntervalsPruner performs branch-and-bound pruning on split intervals
/// to reduce the search space when finding optimal decision tree splits.
pub struct IntervalsPruner<'a> {
    possible_split_indexes: &'a [usize],
    /// Every rule here rests on the same monotonicity argument, and a minimum
    /// support constraint breaks it. See [`IntervalsPruner::is_sound`].
    sound: bool,
    pub rightmost_zero_index: Option<usize>,
    pub leftmost_zero_index: Option<usize>,
    max_gap: usize,
    /// Maps split index -> (left_score, right_score)
    evaluated_indices_record: HashMap<usize, (usize, usize)>,
}

impl<'a> IntervalsPruner<'a> {
    /// Creates a new IntervalsPruner
    ///
    /// # Arguments
    /// * `possible_split_indexes` - Reference to vector of possible split indices
    /// * `max_gap` - Maximum allowable gap for suboptimal solutions
    pub fn new(possible_split_indexes: &'a [usize], max_gap: usize, min_sup: usize) -> Self {
        let possible_split_size = possible_split_indexes.len();
        let mut evaluated_indices_record = HashMap::new();
        evaluated_indices_record.reserve(possible_split_size / 8);

        Self {
            possible_split_indexes,
            sound: Self::is_sound(min_sup),
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
    /// Upstream ConTree does not hit this: it constrains tree size with an
    /// additive complexity cost, which survives the restriction. Minimum
    /// support is this port's own addition, so the rules are switched off
    /// where they do not apply.
    fn is_sound(min_sup: usize) -> bool {
        min_sup <= 1
    }

    /// Performs subinterval pruning to determine if an interval can be entirely pruned
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds of the interval being evaluated
    /// * `current_best_score` - The best score obtained so far
    ///
    /// # Returns
    /// `true` if the subinterval can be pruned, `false` otherwise
    pub fn subinterval_pruning(&self, current_bounds: &Bound, current_best_score: usize) -> bool {
        if !self.sound {
            return false;
        }
        let left_bound_score_left = current_bounds
            .last_split_left_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(left_score, _)| left_score)
            .unwrap_or(0);

        let right_bound_score_right = current_bounds
            .last_split_right_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(_, right_score)| right_score)
            .unwrap_or(0);

        left_bound_score_left
            .saturating_add(right_bound_score_right)
            .saturating_add(self.max_gap)
            >= current_best_score
    }

    /// Performs interval shrinking by narrowing the bounds based on current best score
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds to be updated by shrinking
    /// * `current_best_score` - The best score to compare against
    pub fn interval_shrinking(&self, current_bounds: &mut Bound, current_best_score: usize) {
        if !self.sound {
            return;
        }
        // Update bounds based on zero indices
        if let Some(leftmost) = self.leftmost_zero_index {
            current_bounds.left_bound = current_bounds.left_bound.max(leftmost + 1);
        }

        if let Some(rightmost) = self.rightmost_zero_index {
            current_bounds.right_bound =
                current_bounds.right_bound.min(rightmost.saturating_sub(1));
        }

        // Both adjustments above can push a bound past the end of the
        // candidate list -- `leftmost + 1` when the zero is the last candidate,
        // for one -- and everything below indexes with them. That was an
        // out-of-bounds panic on ordinary input, not a debug assertion.
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

        // Shrink from left side
        if let Some(last_left_idx) = current_bounds.last_split_left_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_left_idx)
            {
                let sum = left_score
                    .saturating_add(right_score)
                    .saturating_add(self.max_gap);
                if sum >= current_best_score {
                    let updated_score_difference = sum - current_best_score;
                    let extended_left_bound =
                        self.possible_split_indexes[last_left_idx] + updated_score_difference + 1;

                    if extended_left_bound
                        <= self.possible_split_indexes[current_bounds.right_bound]
                    {
                        let new_left_bound = self.lower_bound(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            extended_left_bound,
                        );
                        current_bounds.left_bound = current_bounds.left_bound.max(new_left_bound);
                    } else {
                        current_bounds.invalidate();
                        return;
                    }
                }
            }
        }

        // Shrink from right side
        if let Some(last_right_idx) = current_bounds.last_split_right_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_right_idx)
            {
                let sum = left_score
                    .saturating_add(right_score)
                    .saturating_add(self.max_gap);
                if sum >= current_best_score {
                    let updated_score_difference = sum - current_best_score;
                    let extended_right_bound = self.possible_split_indexes[last_right_idx]
                        .saturating_sub(updated_score_difference + 1);

                    if extended_right_bound
                        >= self.possible_split_indexes[current_bounds.left_bound]
                    {
                        let new_right_bound = self.upper_bound(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            extended_right_bound,
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

    /// Performs neighborhood pruning by evaluating the interval around a split index
    ///
    /// # Arguments
    /// * `score_difference` - The difference in scores used to determine pruning
    /// * `left` - The left boundary of the interval
    /// * `right` - The right boundary of the interval
    /// * `split_index` - The index at which the split is evaluated
    ///
    /// # Returns
    /// A tuple `(new_left_bound, new_right_bound)` representing the pruned interval
    pub fn neighbourhood_pruning(
        &self,
        score_difference: usize,
        left: usize,
        right: usize,
        split_index: usize,
    ) -> (usize, usize) {
        if !self.sound {
            // Prune only the point just evaluated.
            return (split_index + 1, split_index.saturating_sub(1));
        }
        let score_difference = score_difference.saturating_add(self.max_gap);

        if score_difference == 0 {
            return (split_index + 1, split_index);
        }

        // Calculate new left bound
        let mut new_bound_left = split_index + 1;
        if let Some(leftmost) = self.leftmost_zero_index {
            new_bound_left = new_bound_left.max(leftmost + 1);
        }

        let minimum_right_value =
            self.possible_split_indexes[split_index].saturating_add(score_difference) + 1;

        // The candidate at `right` is viable when its position is at least
        // `minimum_right_value` -- *at least*, so the equal case is viable and
        // the interval must be narrowed rather than dropped. Written the other
        // way round this discarded a whole half-interval whenever the boundary
        // fell exactly on a candidate.
        if minimum_right_value > self.possible_split_indexes[right] {
            new_bound_left = right + 1;
        } else {
            new_bound_left = self.lower_bound(new_bound_left, right, minimum_right_value);
        }

        // Calculate new right bound
        let mut new_bound_right = split_index.saturating_sub(1);
        if let Some(rightmost) = self.rightmost_zero_index {
            new_bound_right = new_bound_right.min(rightmost.saturating_sub(1));
        }

        let minimum_left_value =
            self.possible_split_indexes[split_index].saturating_sub(score_difference + 1);

        // Mirror image of the test above.
        if minimum_left_value < self.possible_split_indexes[left] {
            new_bound_right = left.saturating_sub(1);
        } else {
            new_bound_right = self.upper_bound(left, new_bound_right, minimum_left_value);
        }

        (new_bound_left, new_bound_right)
    }

    /// Records the outcome of evaluating one split.
    ///
    /// `right_score` is `None` when the right subtree was never explored --
    /// the left side alone already exhausted the upper bound, so the split
    /// cannot beat the incumbent. Every bound this pruner derives is a *lower*
    /// bound on what a split can score, so an unexplored side contributes 0:
    /// the only value that cannot cause a better split to be pruned away.
    pub fn add_result(
        &mut self,
        index: usize,
        left_score: Option<usize>,
        right_score: Option<usize>,
    ) {
        if left_score == Some(0) {
            self.leftmost_zero_index = Some(
                self.leftmost_zero_index
                    .map(|v| v.max(index))
                    .unwrap_or(index),
            );
        }

        if right_score == Some(0) {
            self.rightmost_zero_index = Some(
                self.rightmost_zero_index
                    .map(|v| v.min(index))
                    .unwrap_or(index),
            );
        }

        self.evaluated_indices_record
            .insert(index, (left_score.unwrap_or(0), right_score.unwrap_or(0)));
    }

    /// Helper: Find first element >= value (like std::lower_bound)
    fn lower_bound(&self, left: usize, mut right: usize, value: usize) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        match slice.binary_search(&value) {
            Ok(pos) => left + pos,
            Err(pos) => left + pos,
        }
    }

    /// Helper: Find first element > value (like std::upper_bound)
    fn upper_bound(&self, left: usize, mut right: usize, value: usize) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        let pos = slice.partition_point(|&x| x <= value);
        left + pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruner_creation() {
        let splits = vec![10, 20, 30, 40, 50];
        let pruner = IntervalsPruner::new(&splits, 2, 1);

        assert_eq!(pruner.possible_split_indexes.len(), 5);
        assert_eq!(pruner.max_gap, 2);
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
