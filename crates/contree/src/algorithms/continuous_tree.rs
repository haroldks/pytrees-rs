use crate::algorithms::depth2::ConTreeDepth2;
use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::algorithms::shared;
use crate::algorithms::support_feasible_splits;
use crate::caching::{Cache, Entry};
use crate::common::{
    classification_error, FitOutcome, PointSelector, SearchConfig, SearchError, SearchStatus,
    Statistics,
};
use crate::data::view::DataView;
use crate::data::Dataset;
use crate::tree::Tree;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::time::Instant;

/// Exact ConTree search for optimal decision trees on continuous features.
///
/// A depth-first branch-and-bound search with caching. For each feature it
/// explores candidate thresholds by bisection over intervals, and uses the
/// errors already found to prune whole intervals of thresholds at once.
pub struct ConTree {
    config: SearchConfig,
    statistics: Statistics,
    cache: Cache,
    specialized: ConTreeDepth2,
    runtime: Instant,
    rng: StdRng,
    status: SearchStatus,
    /// Every improvement of the root incumbent, as `(seconds, error)`.
    trajectory: Vec<(f64, usize)>,
    pub tree: Tree,
}

impl ConTree {
    /// Builds a solver from a [`SearchConfig`].
    ///
    /// This is the preferred constructor; [`Self::new`] takes the same
    /// settings as positional arguments.
    pub fn with_config(config: SearchConfig) -> Self {
        let mut solver = Self::new(
            config.min_sup,
            config.max_depth,
            config.max_time,
            config.max_error,
            config.point_selector,
            config.max_gap,
            config.use_heuristic,
            config.fast_d2,
        );
        solver.config = config;
        solver
    }

    /// Builds a solver from positional settings. See [`Self::with_config`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: usize,
        split_selection_strategy: PointSelector,
        max_gap: usize,
        use_heuristic: bool,
        fast_d2: bool,
    ) -> Self {
        Self {
            cache: Cache::default(),
            config: SearchConfig::new(
                min_sup,
                max_depth,
                max_time,
                max_gap,
                max_error,
                use_heuristic,
                fast_d2,
                split_selection_strategy,
            ),
            statistics: Statistics::default(),
            specialized: ConTreeDepth2,
            runtime: Instant::now(),
            rng: StdRng::from_os_rng(),
            status: SearchStatus::Optimal,
            trajectory: Vec::new(),
            tree: Tree::default(),
        }
    }

    /// Seeds the random generator used by `PointSelector::Random`, making
    /// runs reproducible. Unseeded solvers draw their seed from the OS.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Searches for the optimal tree.
    ///
    /// Returns the tree, the search counters, and why the search stopped.
    /// Only `SearchStatus::Optimal` means the tree is proven best for the
    /// given depth and minimum support.
    pub fn fit(&mut self, dataset: &Dataset) -> Result<FitOutcome, SearchError> {
        crate::algorithms::validate(&self.config, dataset)?;

        let root_view = DataView::root(dataset, self.config.use_heuristic);

        self.cache = Cache::new(self.config.max_depth, root_view.total_instances);

        let root_index = self.cache.init();

        self.statistics.num_features = root_view.get_feature_number();
        self.statistics.num_samples = root_view.total_instances;

        let (error, label) = classification_error(root_view.get_labels_freqs());
        let mut entry = Entry {
            error,
            label,
            ..Entry::default()
        };

        if let Some(entry) = self.cache.get_mut(root_index) {
            self.config.max_error = self.config.max_error.min(error);
            entry.error = error;
            entry.label = label;
        }

        let root_config = self.config;
        self.runtime = Instant::now();
        self.trajectory.clear();
        self.expand_node_with_view(
            &root_view,
            &root_config,
            &mut entry,
            0,
            true,
            root_config.max_error,
        );

        self.statistics.error = entry.error;
        self.statistics.cache_size = self.cache.len();
        self.statistics.duration = self.elapsed_time();
        self.get_solution_tree();

        self.status = if !self.time_remains() {
            SearchStatus::TimeLimit
        } else if entry.error <= self.config.max_gap && self.config.max_gap > 0 {
            SearchStatus::ErrorBoundReached
        } else {
            SearchStatus::Optimal
        };
        self.tree.validate()?;

        Ok(FitOutcome {
            tree: self.tree.clone(),
            statistics: self.statistics,
            status: self.status,
        })
    }

    /// The anytime profile of the last fit: `(seconds, error)` at every
    /// improvement of the root incumbent.
    pub fn trajectory(&self) -> &[(f64, usize)] {
        &self.trajectory
    }

    /// Why the last `fit` stopped.
    pub fn status(&self) -> SearchStatus {
        self.status
    }

    /// Looks a child subproblem up in the cache, seeding a new entry with the
    /// error of its majority-class leaf.
    ///
    /// `child_depth` is the child's remaining depth. Returns whether the entry
    /// is new, its cache index, and a copy of it.
    fn cache_child(&mut self, view: &DataView<'_>, child_depth: usize) -> (bool, usize, Entry) {
        let (is_new, index) = self.cache.insert(&view.bitset, child_depth);
        let depth = self.config.max_depth - child_depth;
        let mut entry = Entry::default();

        if let Some(slot) = self.cache.get_mut(index) {
            if is_new {
                let (error, label) = classification_error(view.get_labels_freqs());
                slot.error = error;
                slot.label = label;
                slot.depth = depth;
                slot.lower_bound = 0;
                slot.is_valid = true;
            }
            entry = *slot;
        }
        (is_new, index, entry)
    }

    fn expand_node_with_view(
        &mut self,
        view: &DataView<'_>,
        config: &SearchConfig,
        current_best: &mut Entry,
        parent_index: usize,
        is_new: bool,
        upper_bound: usize,
    ) {
        #[cfg(feature = "profiling")]
        coz::progress!();
        if current_best.error == 0 || view.is_empty() {
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.is_leaf = true;
                entry.mark_exact();
                *current_best = *entry;
            }
            return;
        }

        // Only an entry proven exact can replace a new search. An entry that
        // only showed "no better than the bound it ran under" says nothing
        // about a later, looser bound.
        if !is_new && current_best.is_optimal {
            self.statistics.cache_hits += 1;
            return;
        }

        if config.max_depth == 0 {
            current_best.is_leaf = true;
            current_best.mark_exact();

            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.is_leaf = true;
                entry.mark_exact();
                *current_best = *entry;
            }

            return;
        }

        if current_best.error <= config.max_gap || view.len() <= 1 {
            // Within the allowed gap, or too few instances to split: the leaf
            // is the answer.
            current_best.mark_exact();
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.mark_exact();
            }
            return;
        }

        // Only at depth exactly 2: `ConTreeDepth2` always builds two levels of
        // tests, which would exceed a depth-1 request.
        if config.fast_d2 && config.max_depth == 2 {
            let tree = self.specialized.fit(
                view,
                config,
                current_best,
                upper_bound,
                &mut self.statistics,
            );
            current_best.finalize_lower_bound(upper_bound);
            current_best.is_optimal = current_best.is_valid;
            let tree_index = self.cache.insert_tree(tree);
            if let Some(entry) = self.cache.get_mut(parent_index) {
                *entry = *current_best;
                entry.tree_idx = Some(tree_index);
            }
            self.statistics.specialized_solver_call += 1;

            return;
        }

        let num_features = view.get_feature_number();
        let heuristics_data = view.features_best_score();
        debug_assert!(
            num_features == heuristics_data.len(),
            "Missmatch with heuristics and features number"
        );
        for &(_, feat) in heuristics_data.iter().take(num_features) {
            self.expand_on_feature(
                view,
                feat,
                parent_index,
                config,
                current_best,
                upper_bound.min(current_best.error),
            );

            if current_best.error == 0 {
                current_best.mark_exact();
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    entry.mark_exact();
                }

                return;
            }
            if !self.time_remains() {
                return;
            }
        }

        // The entry becomes reusable only if the search found a tree within
        // the upper bound it was given. Otherwise it only proves that the
        // bound cannot be met here.
        current_best.finalize_lower_bound(upper_bound);
        current_best.is_optimal = current_best.is_valid;
        if let Some(entry) = self.cache.get_mut(parent_index) {
            entry.lower_bound = current_best.lower_bound;
            entry.is_valid = current_best.is_valid;
            entry.is_optimal = current_best.is_valid;
        }
    }

    fn expand_on_feature(
        &mut self,
        view: &DataView<'_>,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) {
        #[cfg(feature = "profiling")]
        coz::scope!("expand_on_feature");
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        if config.point_selector == PointSelector::First {
            self.expand_on_feature_gini_priority(
                view,
                feature_index,
                cache_index,
                config,
                current_best,
                upper_bound,
            );
            return;
        }

        let possible_index_split = view.get_possible_split_indices(feature_index);
        if possible_index_split.is_empty() {
            return;
        }

        // Restrict the search to splits that meet the minimum support. The
        // per-candidate check below skips a whole interval, so it cannot be
        // relied on to find the feasible splits inside it.
        let feasible =
            support_feasible_splits(possible_index_split, view.len(), self.config.min_sup);
        if feasible.is_empty() {
            return;
        }

        let mut pruner = IntervalsPruner::new(possible_index_split, config.max_gap, config.min_sup);
        let mut queue = VecDeque::new();
        let init_bound = Bound::new(feasible.start, feasible.end - 1, None, None);
        queue.push_back(init_bound);

        while let Some(mut current_bound) = queue.pop_front() {
            if !self.time_remains() {
                return;
            }

            // Prune against the tighter of the incumbent and the parent's upper
            // bound: a split that cannot beat the parent's bound is of no use
            // to the parent.
            if pruner.subinterval_pruning(&current_bound, current_best.error.min(upper_bound)) {
                continue;
            }

            pruner.interval_shrinking(&mut current_bound, current_best.error.min(upper_bound));
            if !current_bound.is_valid() {
                continue;
            }

            let selected_point = self.select_point(config, &current_bound);
            let split_point = possible_index_split[selected_point];
            let int_half_distance = split_point
                .saturating_sub(possible_index_split[current_bound.left_bound])
                .max(possible_index_split[current_bound.right_bound].saturating_sub(split_point));

            let threshold_value = if selected_point > 0 {
                let previous = feature_column_ids[possible_index_split[selected_point - 1]];
                let point = feature_column_ids[split_point];
                shared::threshold_between(
                    feature_column[previous].value(),
                    feature_column[point].value(),
                )
            } else {
                let point = feature_column_ids[split_point];
                shared::threshold_between(
                    feature_column[feature_column_ids[0]].value(),
                    feature_column[point].value(),
                )
            };

            let (left_view, right_view) = view.split(feature_index, split_point);

            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                continue;
            }

            // Search the larger child first: its error tightens the bound
            // passed to the smaller one.
            let process_left_first = left_view.len() >= right_view.len();

            let left_config = config.derive_left();
            let mut left_entry = Entry::default();
            let mut right_entry = Entry::default();

            // A cache index of 0 means "no child", kept by a side the search
            // does not descend into.
            let (mut left_index, mut right_index) = (0, 0);
            let (mut left_is_new, mut right_is_new);

            let larger_upper_bound = current_best.error.min(upper_bound.saturating_add(1));
            self.statistics.general_solver_call += 1;

            if process_left_first {
                (left_is_new, left_index, left_entry) =
                    self.cache_child(&left_view, left_config.max_depth);

                self.expand_node_with_view(
                    &left_view,
                    &left_config,
                    &mut left_entry,
                    left_index,
                    left_is_new,
                    larger_upper_bound,
                );
                left_entry.finalize_lower_bound(larger_upper_bound);
            } else {
                (right_is_new, right_index, right_entry) =
                    self.cache_child(&right_view, left_config.max_depth);

                self.expand_node_with_view(
                    &right_view,
                    &left_config,
                    &mut right_entry,
                    right_index,
                    right_is_new,
                    larger_upper_bound,
                );
                right_entry.finalize_lower_bound(larger_upper_bound);
            }

            // The bound left for the smaller child comes from what the larger
            // child was proved to cost, i.e. its lower bound.
            let larger_error = if process_left_first {
                left_entry.lower_bound
            } else {
                right_entry.lower_bound
            };
            // Widen that bound by the interval half-distance so that one
            // search can prune the whole interval around this split.
            let budget =
                current_best.error.min(upper_bound.saturating_add(1)) as i64 - larger_error as i64;
            let smaller_ub = budget + int_half_distance as i64;
            let smaller_upper_bound = smaller_ub.max(0) as usize;
            // Stays `None` when the smaller child is not searched because the
            // larger one alone already exceeds the upper bound.
            let mut right_error: Option<usize> = None;

            // Search the smaller child unless the budget is negative. A budget
            // of exactly zero is still explored, since a zero-error subtree
            // may exist.
            if smaller_ub > 0 || budget == 0 {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);

                if process_left_first {
                    right_entry.error = current_best.error;
                    (right_is_new, right_index, right_entry) =
                        self.cache_child(&right_view, right_config.max_depth);

                    self.expand_node_with_view(
                        &right_view,
                        &right_config,
                        &mut right_entry,
                        right_index,
                        right_is_new,
                        smaller_upper_bound,
                    );
                    right_entry.finalize_lower_bound(smaller_upper_bound);
                } else {
                    left_entry.error = current_best.error;
                    (left_is_new, left_index, left_entry) =
                        self.cache_child(&left_view, right_config.max_depth);

                    self.expand_node_with_view(
                        &left_view,
                        &right_config,
                        &mut left_entry,
                        left_index,
                        left_is_new,
                        smaller_upper_bound,
                    );
                    left_entry.finalize_lower_bound(smaller_upper_bound);
                }

                // Use lower bounds rather than errors: when a bound cut a
                // child search short, its `error` is only an upper bound.
                right_error = Some(right_entry.lower_bound);

                let feature_best = left_entry.lower_bound + right_entry.lower_bound;
                if left_entry.is_valid && right_entry.is_valid && feature_best < current_best.error
                {
                    current_best.error = feature_best;
                    if config.is_root {
                        self.trajectory.push((self.elapsed_time(), feature_best));
                    }
                    current_best.feature = feature_index;
                    current_best.split = threshold_value;
                    current_best.left = left_index;
                    current_best.right = right_index;

                    let is_optimal = feature_best == 0;
                    current_best.is_optimal = is_optimal;

                    if let Some(entry) = self.cache.get_mut(cache_index) {
                        *entry = *current_best;
                    }

                    if feature_best == 0 {
                        return;
                    }
                }
            }

            // `None` for a child the bound invalidated: its lower bound may be
            // zero without the subtree being error-free, and the pruner would
            // read a zero as "nothing beyond this split can do better".
            let left_score = left_entry.is_valid.then_some(left_entry.lower_bound);
            pruner.add_result(selected_point, left_score, right_error);
            if current_bound.left_bound == current_bound.right_bound {
                continue;
            }

            let score_difference = left_entry
                .lower_bound
                .saturating_add(right_error.unwrap_or(0))
                .saturating_sub(current_best.error.min(upper_bound));

            let (left_bound, right_bound) = pruner.neighbourhood_pruning(
                score_difference,
                current_bound.left_bound,
                current_bound.right_bound,
                selected_point,
            );

            if left_bound <= current_bound.right_bound {
                queue.push_back(Bound {
                    left_bound,
                    right_bound: current_bound.right_bound,
                    last_split_left_index: Some(selected_point),
                    last_split_right_index: current_bound.last_split_right_index,
                });
            }

            if current_bound.left_bound <= right_bound {
                queue.push_back(Bound {
                    left_bound: current_bound.left_bound,
                    right_bound,
                    last_split_left_index: current_bound.last_split_left_index,
                    last_split_right_index: Some(selected_point),
                });
            }
        }

        if let Some(entry) = self.cache.get_mut(cache_index) {
            entry.is_optimal = true;
        }
    }

    fn expand_on_feature_gini_priority(
        &mut self,
        view: &DataView<'_>,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) {
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        let possible_splits = view.get_possible_split_indices(feature_index);

        if possible_splits.is_empty() {
            return;
        }

        let sorted_by_heuristic_indices = view.ordered_possible_splits(feature_index);

        let mut pruner = IntervalsPruner::new(possible_splits, config.max_gap, config.min_sup);

        let mut queue = VecDeque::new();
        let init_bound = Bound::new(0, possible_splits.len() - 1, None, None);
        queue.push_back(init_bound);

        let mut pruned = vec![false; possible_splits.len()];

        for &split_idx in sorted_by_heuristic_indices {
            if !self.time_remains() {
                return;
            }

            if pruned[split_idx] {
                continue;
            }

            // The interval around `split_idx` that no pruned threshold
            // separates from it.
            let mut current_left = split_idx;
            while current_left > 0 && pruned[current_left - 1] {
                current_left -= 1;
            }
            current_left = current_left.saturating_sub(1);

            let mut current_right = split_idx;
            while current_right < possible_splits.len() - 1 && pruned[current_right + 1] {
                current_right += 1;
            }
            if current_right < possible_splits.len() - 1 {
                current_right += 1;
            }

            let split_point = possible_splits[split_idx];

            let threshold_value = if split_idx > 0 {
                let previous = feature_column_ids[possible_splits[split_idx - 1]];
                let point = feature_column_ids[split_point];
                shared::threshold_between(
                    feature_column[previous].value(),
                    feature_column[point].value(),
                )
            } else {
                let point = feature_column_ids[split_point];
                shared::threshold_between(
                    feature_column[feature_column_ids[0]].value(),
                    feature_column[point].value(),
                )
            };

            let (left_view, right_view) = view.split(feature_index, split_point);

            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                pruned[split_idx] = true;
                continue;
            }

            // Search the larger child first: its error tightens the bound
            // passed to the smaller one.
            let process_left_first = left_view.len() >= right_view.len();

            let left_config = config.derive_left();
            let mut left_entry = Entry::default();
            let mut right_entry = Entry::default();

            // A cache index of 0 means "no child", kept by a side the search
            // does not descend into.
            let (mut left_index, mut right_index) = (0, 0);
            let (mut left_is_new, mut right_is_new);

            let int_half_distance = split_point
                .saturating_sub(possible_splits[0])
                .max(possible_splits[possible_splits.len() - 1].saturating_sub(split_point));

            let larger_upper_bound = current_best.error.min(upper_bound.saturating_add(1));
            self.statistics.general_solver_call += 1;

            if process_left_first {
                (left_is_new, left_index, left_entry) =
                    self.cache_child(&left_view, left_config.max_depth);

                self.expand_node_with_view(
                    &left_view,
                    &left_config,
                    &mut left_entry,
                    left_index,
                    left_is_new,
                    larger_upper_bound,
                );
                left_entry.finalize_lower_bound(larger_upper_bound);
            } else {
                (right_is_new, right_index, right_entry) =
                    self.cache_child(&right_view, left_config.max_depth);

                self.expand_node_with_view(
                    &right_view,
                    &left_config,
                    &mut right_entry,
                    right_index,
                    right_is_new,
                    larger_upper_bound,
                );
                right_entry.finalize_lower_bound(larger_upper_bound);
            }

            // The bound left for the smaller child comes from what the larger
            // child was proved to cost, i.e. its lower bound.
            let larger_error = if process_left_first {
                left_entry.lower_bound
            } else {
                right_entry.lower_bound
            };
            // Widen that bound by the interval half-distance so that one
            // search can prune the whole interval around this split.
            let budget =
                current_best.error.min(upper_bound.saturating_add(1)) as i64 - larger_error as i64;
            let smaller_ub = budget + int_half_distance as i64;
            let smaller_upper_bound = smaller_ub.max(0) as usize;
            let mut right_error: Option<usize> = None;

            // Search the smaller child unless the budget is negative. A budget
            // of exactly zero is still explored, since a zero-error subtree
            // may exist.
            if smaller_ub > 0 || budget == 0 {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);

                if process_left_first {
                    right_entry.error = current_best.error;
                    (right_is_new, right_index, right_entry) =
                        self.cache_child(&right_view, right_config.max_depth);

                    self.expand_node_with_view(
                        &right_view,
                        &right_config,
                        &mut right_entry,
                        right_index,
                        right_is_new,
                        smaller_upper_bound,
                    );
                    right_entry.finalize_lower_bound(smaller_upper_bound);
                } else {
                    left_entry.error = current_best.error;
                    (left_is_new, left_index, left_entry) =
                        self.cache_child(&left_view, right_config.max_depth);

                    self.expand_node_with_view(
                        &left_view,
                        &right_config,
                        &mut left_entry,
                        left_index,
                        left_is_new,
                        smaller_upper_bound,
                    );
                    left_entry.finalize_lower_bound(smaller_upper_bound);
                }

                // Use lower bounds rather than errors: when a bound cut a
                // child search short, its `error` is only an upper bound.
                right_error = Some(right_entry.lower_bound);
                let feature_best = left_entry.lower_bound + right_entry.lower_bound;
                if left_entry.is_valid && right_entry.is_valid && feature_best < current_best.error
                {
                    current_best.error = feature_best;
                    if config.is_root {
                        self.trajectory.push((self.elapsed_time(), feature_best));
                    }
                    current_best.feature = feature_index;
                    current_best.split = threshold_value;
                    current_best.left = left_index;
                    current_best.right = right_index;

                    let is_optimal = feature_best == 0;
                    current_best.is_optimal = is_optimal;

                    if let Some(entry) = self.cache.get_mut(cache_index) {
                        *entry = *current_best;
                    }
                }
            }

            // `None` for a child the bound invalidated: its lower bound may be
            // zero without the subtree being error-free, and the pruner would
            // read a zero as "nothing beyond this split can do better".
            let left_score = left_entry.is_valid.then_some(left_entry.lower_bound);
            pruner.add_result(split_idx, left_score, right_error);

            let score_difference = left_entry
                .lower_bound
                .saturating_add(right_error.unwrap_or(0))
                .saturating_sub(current_best.error.min(upper_bound));
            let (new_left_bound, new_right_bound) = pruner.neighbourhood_pruning(
                score_difference,
                0,
                possible_splits.len() - 1,
                split_idx,
            );

            // `neighbourhood_pruning` leaves two intervals to search,
            // `[current_left, new_right_bound]` and `[new_left_bound, current_right]`.
            // Only the thresholds strictly between them can be skipped.
            let skip_from = new_right_bound.saturating_add(1).max(current_left);
            if skip_from < split_idx {
                pruned[skip_from..split_idx].fill(true);
            }
            let skip_to = new_left_bound.min(current_right + 1);
            if split_idx + 1 < skip_to {
                pruned[split_idx + 1..skip_to].fill(true);
            }

            if current_best.error == 0 {
                break;
            }
        }

        if let Some(entry) = self.cache.get_mut(cache_index) {
            entry.is_optimal = true;
        }
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    fn time_remains(&self) -> bool {
        shared::time_remains(&self.runtime, self.config.max_time)
    }

    fn elapsed_time(&self) -> f64 {
        shared::elapsed_time(&self.runtime)
    }

    pub fn select_point(&mut self, config: &SearchConfig, bound: &Bound) -> usize {
        shared::select_point(config.point_selector, &mut self.rng, bound)
    }

    /// Rebuilds `self.tree` from the cache.
    pub fn get_solution_tree(&mut self) {
        self.tree = shared::build_solution_tree(&self.cache);
    }
}

#[cfg(test)]
mod contree_test {
    use crate::algorithms::continuous_tree::ConTree;
    use crate::common::PointSelector;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;
    use crate::tests::fixture;

    #[test]
    fn deeper_search_never_scores_worse() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("avila_1k.txt"))?;
        dataset.sort_features();

        let mut previous = usize::MAX;
        for depth in 1..=3 {
            let mut contree: ConTree = ConTree::new(
                1,
                depth,
                100.0,
                usize::MAX,
                PointSelector::Mid,
                0,
                false,
                true,
            );
            contree.fit(&dataset).expect("fit");

            let error = contree.statistics().error;
            assert!(
                error <= previous,
                "depth {depth} scored {error}, worse than depth {} at {previous}",
                depth - 1
            );
            assert!(
                error < dataset.count(),
                "the search reported a nonsense error"
            );
            previous = error;
        }

        Ok(())
    }

    #[test]
    fn the_depth_two_specialization_agrees_with_the_general_search() -> Result<(), DataReaderError>
    {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("hepatitis.txt"))?;
        dataset.sort_features();

        // Both are exact at depth 2, so they must agree on the error even if
        // they break ties towards different trees.
        let mut general: ConTree =
            ConTree::new(1, 2, 100.0, usize::MAX, PointSelector::Mid, 0, false, false);
        general.fit(&dataset).expect("fit");

        let mut specialized: ConTree =
            ConTree::new(1, 2, 100.0, usize::MAX, PointSelector::Mid, 0, false, true);
        specialized.fit(&dataset).expect("fit");

        assert_eq!(general.statistics().error, specialized.statistics().error);

        Ok(())
    }
}
