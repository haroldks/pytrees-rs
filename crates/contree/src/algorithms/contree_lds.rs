use crate::algorithms::depth2::ConTreeDepth2;
use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::algorithms::shared;
use crate::algorithms::support_feasible_splits;
use crate::caching::{Cache, Entry, SearchedUnder};
use crate::common::{
    classification_error, Budget, BudgetSchedule, FitOutcome, PassReport, PointSelector,
    ScheduleBounds, ScheduleKind, SearchConfig, SearchError, SearchStatus, Statistics,
};
use crate::data::view::DataView;
use crate::data::{Dataset, Feature};
use crate::tree::Tree;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::time::Instant;

/// Anytime ConTree search based on limited discrepancy search.
///
/// The search runs in passes. Each pass explores the tree space under a
/// discrepancy budget (how far down the Gini ranking of features it may go)
/// and, for some point selectors, a split budget (how many thresholds per
/// feature it may try). A [`BudgetSchedule`] widens both budgets between
/// passes, so a good tree is available early and the last complete pass
/// proves optimality.
pub struct ConTreeLds {
    config: SearchConfig,
    statistics: Statistics,
    cache: Cache,
    specialized: ConTreeDepth2,
    runtime: Instant,
    rng: StdRng,
    max_discrepancy: usize,
    split_budget: usize,
    schedule_kind: ScheduleKind,
    schedule: Option<Box<dyn BudgetSchedule>>,
    /// Which budgets truncated the pass in progress, read by the schedule.
    pass_report: PassReport,
    status: SearchStatus,
    /// Every improvement of the root incumbent, as `(seconds, error)`.
    trajectory: Vec<(f64, usize)>,
}

impl ConTreeLds {
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
        let config = SearchConfig::new(
            min_sup,
            max_depth,
            max_time,
            max_gap,
            max_error,
            use_heuristic,
            fast_d2,
            split_selection_strategy,
        );

        Self {
            cache: Cache::default(),
            config,
            statistics: Statistics::default(),
            specialized: ConTreeDepth2,
            runtime: Instant::now(),
            rng: StdRng::from_os_rng(),
            max_discrepancy: usize::MAX,
            split_budget: 1,
            schedule_kind: ScheduleKind::default(),
            schedule: None,
            pass_report: PassReport::default(),
            status: SearchStatus::Optimal,
            trajectory: Vec::new(),
        }
    }

    /// Seeds the random generator used by `PointSelector::Random`, making
    /// runs reproducible. Unseeded solvers draw their seed from the OS.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Runs the anytime search until it stops.
    ///
    /// Each pass widens the discrepancy and split budgets. The search ends
    /// when a pass completes without being truncated (the tree is optimal),
    /// when the schedule has no larger budget to offer, or when the time
    /// limit is reached.
    pub fn fit(&mut self, dataset: &Dataset) -> Result<FitOutcome, SearchError> {
        crate::algorithms::validate(&self.config, dataset)?;

        let root_view = DataView::root(dataset, self.config.use_heuristic);
        while !self.partial_fit(&root_view) {}

        let tree = self.get_solution_tree();
        tree.validate()?;

        Ok(FitOutcome {
            tree,
            statistics: self.statistics,
            status: self.status,
        })
    }

    /// Chooses how the budget widens from pass to pass. See [`ScheduleKind`].
    pub fn with_schedule(mut self, schedule: ScheduleKind) -> Self {
        self.schedule_kind = schedule;
        self
    }

    /// The budget schedule this solver uses.
    pub fn schedule(&self) -> ScheduleKind {
        self.schedule_kind
    }

    fn apply(&mut self, budget: Budget) {
        self.config.budget = budget.discrepancy;
        self.split_budget = budget.split_budget();
    }

    /// The configuration this solver was built with.
    pub fn config(&self) -> &SearchConfig {
        &self.config
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

    /// Runs one pass of the search and returns `true` once the search is
    /// over. The first call initialises the cache and the budget schedule.
    pub fn partial_fit(&mut self, root_view: &DataView<'_>) -> bool {
        self.config.nb_runs += 1;
        let mut root_index = 0;
        let mut entry = Entry::default();
        if self.config.nb_runs <= 1 {
            self.cache = Cache::new(self.config.max_depth, root_view.total_instances);

            self.max_discrepancy = self.max_discrepancy.min(Self::discrepancy_limit(
                root_view.get_feature_number(),
                self.config.max_depth,
            ));
            let mut schedule = self.schedule_kind.build(ScheduleBounds {
                max_discrepancy: self.max_discrepancy,
                max_splits: root_view.get_max_splits(),
                split_applies: self.config.point_selector == PointSelector::First,
            });
            let budget = schedule.first();
            self.apply(budget);
            self.schedule = Some(schedule);
            root_index = self.cache.init();

            self.statistics.num_features = root_view.get_feature_number();
            self.statistics.num_samples = root_view.total_instances;

            let (error, label) = classification_error(root_view.get_labels_freqs());
            entry.error = error;
            entry.label = label;

            if let Some(entry) = self.cache.get_mut(root_index) {
                self.config.max_error = self.config.max_error.min(error);
                entry.error = error;
                entry.label = label;
                entry.lower_bound = 0;
                entry.is_valid = true;
            }

            self.runtime = Instant::now();
            self.trajectory.clear();
        }

        let mut is_optimal = false;

        if let Some(e) = self.cache.get(root_index) {
            entry = *e
        }
        if entry.is_optimal {
            return true;
        }

        let mut config = self.config;

        config.discrepancy = 0;

        self.pass_report = PassReport::default();
        let error = entry.error;
        let stopped = self.expand_node_with_view(root_view, &config, &mut entry, 0, true, error);
        self.pass_report.improved = entry.error < error;

        // Once the schedule has no larger budget, another pass would repeat
        // the same truncated search, so the search stops here.
        let next_budget = self
            .schedule
            .as_mut()
            .and_then(|schedule| schedule.next(&self.pass_report));
        let search_exhausted = next_budget.is_none();

        if !stopped || !self.time_remains() || search_exhausted {
            entry.is_optimal = true;
            is_optimal = true;
            self.status = if !stopped {
                SearchStatus::Optimal
            } else if !self.time_remains() {
                SearchStatus::TimeLimit
            } else {
                SearchStatus::BudgetExhausted
            };
        }

        if let Some(budget) = next_budget {
            self.apply(budget);
        }

        self.statistics.error = entry.error;
        self.statistics.cache_size = self.cache.len();
        self.statistics.duration = self.elapsed_time();
        is_optimal
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
    ) -> bool {
        if current_best.error == 0 || view.is_empty() {
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.mark_exact();
                *current_best = *entry;
            }
            return false;
        }

        // The budgets this visit runs under. The split budget only applies to
        // the `first` selector and to the first pass of every selector.
        let under = SearchedUnder {
            discrepancy: config.budget.saturating_sub(config.discrepancy),
            split_budget: if config.point_selector == PointSelector::First
                || self.config.nb_runs <= 1
            {
                self.split_budget
            } else {
                usize::MAX
            },
            upper_bound,
            truncated: false,
        };

        if !is_new {
            if current_best.is_optimal {
                self.statistics.cache_hits += 1;
                return false;
            }
            if let Some(previous) = current_best.searched_under {
                // A complete earlier search proved its lower bound. If that
                // bound already reaches the parent's upper bound, no subtree
                // here can help the parent.
                if !previous.truncated && current_best.lower_bound >= upper_bound {
                    self.statistics.cache_hits += 1;
                    return false;
                }
                // An earlier search with at least this much budget and upper
                // bound found everything this visit could, and was truncated
                // exactly when this visit would be.
                if previous.covers(&under, current_best.is_valid) {
                    self.statistics.cache_hits += 1;
                    return previous.truncated;
                }
            }
        }

        if config.max_depth == 0 {
            current_best.is_leaf = true;
            current_best.mark_exact();

            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.is_leaf = true;
                entry.mark_exact();
                *current_best = *entry;
            }

            return false;
        }

        if current_best.error <= config.max_gap || view.len() <= 1 {
            return false;
        }

        // Only at depth exactly 2: `ConTreeDepth2` always builds two levels of
        // tests, which would exceed a depth-1 request.
        if config.fast_d2 && config.max_depth == 2 {
            // Every pass revisits the same depth-2 subproblems, so each one is
            // solved exactly once, without the parent's upper bound, and then
            // reused. The parent still compares the result to its own bound.
            let tree =
                self.specialized
                    .fit(view, config, current_best, usize::MAX, &mut self.statistics);
            current_best.mark_exact();
            let tree_index = self.cache.insert_tree(tree);
            if let Some(entry) = self.cache.get_mut(parent_index) {
                *entry = *current_best;
                entry.tree_idx = Some(tree_index);
            }
            self.statistics.specialized_solver_call += 1;

            return false;
        }

        let num_features = view.get_feature_number();
        let heuristics_data = view.features_best_score();
        debug_assert!(
            num_features == heuristics_data.len(),
            "Missmatch with heuristics and features number"
        );
        let mut stopped = false;
        for (it, &(_, feat)) in heuristics_data.iter().take(num_features).enumerate() {
            let feat_discrepancy = config.discrepancy + it;
            if feat_discrepancy > config.budget {
                // Features come in Gini order, so every later one is over the
                // budget too. Fall through to record what this search covered.
                self.pass_report.cut_by_discrepancy = true;
                stopped = true;
                break;
            }

            let mut node_config = *config;
            node_config.discrepancy = feat_discrepancy;

            stopped |= self.expand_on_feature(
                view,
                feat,
                parent_index,
                &node_config,
                current_best,
                upper_bound.min(current_best.error),
            );

            if current_best.error == 0 {
                current_best.mark_exact();
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    entry.mark_exact();
                }

                return false;
            }

            if !self.time_remains() {
                return true;
            }
        }
        // A result is exact only if the search was neither truncated by a
        // budget nor cut short by the upper bound: a search completed under a
        // tight bound says nothing about a looser one.
        current_best.finalize_lower_bound(upper_bound);
        current_best.searched_under = Some(SearchedUnder {
            truncated: stopped,
            ..under
        });
        if let Some(entry) = self.cache.get_mut(parent_index) {
            entry.lower_bound = current_best.lower_bound;
            entry.is_valid = current_best.is_valid;
            entry.is_optimal = !stopped && current_best.is_valid;
            entry.searched_under = current_best.searched_under;
        }
        current_best.is_optimal = !stopped && current_best.is_valid;

        stopped
    }

    fn expand_on_feature(
        &mut self,
        view: &DataView<'_>,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) -> bool {
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        if config.point_selector == PointSelector::First || self.config.nb_runs <= 1 {
            let stopped = self.expand_on_feature_gini_priority(
                view,
                feature_index,
                cache_index,
                config,
                current_best,
                upper_bound,
            );
            return stopped;
        }

        let possible_index_split = view.get_possible_split_indices(feature_index);
        if possible_index_split.is_empty() {
            return false;
        }

        // Restrict the search to splits that meet the minimum support. The
        // per-candidate check below skips a whole interval, so it cannot be
        // relied on to find the feasible splits inside it.
        let feasible =
            support_feasible_splits(possible_index_split, view.len(), self.config.min_sup);
        if feasible.is_empty() {
            return false;
        }

        let mut pruner = IntervalsPruner::new(possible_index_split, config.max_gap, config.min_sup);
        let mut queue = VecDeque::new();
        let init_bound = Bound::new(feasible.start, feasible.end - 1, None, None);
        queue.push_back(init_bound);

        let mut stopped = false;

        while let Some(mut current_bound) = queue.pop_front() {
            if !self.time_remains() {
                return true;
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

                stopped |= self.expand_node_with_view(
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

                stopped |= self.expand_node_with_view(
                    &right_view,
                    &left_config,
                    &mut right_entry,
                    right_index,
                    right_is_new,
                    larger_upper_bound,
                );
                right_entry.finalize_lower_bound(larger_upper_bound);
            }

            let larger_error = if process_left_first {
                left_entry.lower_bound
            } else {
                right_entry.lower_bound
            };
            // The smaller child's bound is what the larger child left of the
            // incumbent, widened by the interval half-distance so that one
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

                    stopped |= self.expand_node_with_view(
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

                    stopped |= self.expand_node_with_view(
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

                    if let Some(entry) = self.cache.get_mut(cache_index) {
                        *entry = *current_best;
                    }

                    if feature_best == 0 {
                        return false;
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
        stopped
    }

    /// Tries the thresholds of one feature one at a time, in Gini order when
    /// the heuristic is on and in position order otherwise, up to the split
    /// budget. Returns `true` if the budget truncated the search.
    fn expand_on_feature_gini_priority(
        &mut self,
        view: &DataView<'_>,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) -> bool {
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        let possible_splits = view.get_possible_split_indices(feature_index);

        if possible_splits.is_empty() {
            return false;
        }

        let mut pruner = IntervalsPruner::new(possible_splits, config.max_gap, config.min_sup);

        // Thresholds already evaluated or ruled out by the pruner.
        let mut pruned = vec![false; possible_splits.len()];
        let mut stopped = false;

        let local_split_budget = possible_splits.len().min(self.split_budget);

        if config.use_heuristic {
            let sorted_indices = view.ordered_possible_splits(feature_index);
            for (idx, &split_idx) in sorted_indices.iter().enumerate() {
                if self.config.nb_runs <= 1 && idx > 0 {
                    self.pass_report.cut_by_split = true;
                    return true;
                }

                if idx > local_split_budget - 1 {
                    self.pass_report.cut_by_split = true;
                    return true;
                }

                if !self.time_remains() {
                    return true;
                }

                if pruned[split_idx] {
                    continue;
                }

                stopped |= self.evaluate_split_gini_priority(
                    view,
                    feature_index,
                    split_idx,
                    possible_splits,
                    feature_column,
                    feature_column_ids,
                    config,
                    current_best,
                    cache_index,
                    upper_bound,
                    &mut pruner,
                    &mut pruned,
                );

                if current_best.error == 0 {
                    return false;
                }
            }
        } else {
            for split_idx in 0..possible_splits.len() {
                if self.config.nb_runs <= 1 && split_idx > 0 {
                    self.pass_report.cut_by_split = true;
                    return true;
                }

                if split_idx > local_split_budget - 1 {
                    self.pass_report.cut_by_split = true;
                    return true;
                }

                if !self.time_remains() {
                    return true;
                }

                if pruned[split_idx] {
                    continue;
                }

                stopped |= self.evaluate_split_gini_priority(
                    view,
                    feature_index,
                    split_idx,
                    possible_splits,
                    feature_column,
                    feature_column_ids,
                    config,
                    current_best,
                    cache_index,
                    upper_bound,
                    &mut pruner,
                    &mut pruned,
                );

                if current_best.error == 0 {
                    return false;
                }
            }
        }

        if local_split_budget < possible_splits.len() {
            self.pass_report.cut_by_split = true;
            stopped = true;
        }

        stopped
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_split_gini_priority(
        &mut self,
        view: &DataView<'_>,
        feature_index: usize,
        split_idx: usize,
        possible_splits: &[usize],
        feature_column: &Feature,
        feature_column_ids: &[usize],
        config: &SearchConfig,
        current_best: &mut Entry,
        cache_index: usize,
        upper_bound: usize,
        pruner: &mut IntervalsPruner<'_>,
        pruned: &mut [bool],
    ) -> bool {
        // The interval around `split_idx` that no pruned threshold separates
        // from it.
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
            return false;
        }

        // Search the larger child first: its error tightens the bound passed
        // to the smaller one.
        let process_left_first = left_view.len() >= right_view.len();

        let left_config = config.derive_left();
        let mut left_entry = Entry::default();
        let mut right_entry = Entry::default();

        // A cache index of 0 means "no child", kept by a side the search does
        // not descend into.
        let (mut left_index, mut right_index) = (0, 0);
        let (mut left_is_new, mut right_is_new);
        let mut stopped = false;

        let int_half_distance = split_point
            .saturating_sub(possible_splits[0])
            .max(possible_splits[possible_splits.len() - 1].saturating_sub(split_point));

        let larger_upper_bound = current_best.error.min(upper_bound.saturating_add(1));
        self.statistics.general_solver_call += 1;

        if process_left_first {
            (left_is_new, left_index, left_entry) =
                self.cache_child(&left_view, left_config.max_depth);

            stopped |= self.expand_node_with_view(
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

            stopped |= self.expand_node_with_view(
                &right_view,
                &left_config,
                &mut right_entry,
                right_index,
                right_is_new,
                larger_upper_bound,
            );
            right_entry.finalize_lower_bound(larger_upper_bound);
        }

        let larger_error = if process_left_first {
            left_entry.lower_bound
        } else {
            right_entry.lower_bound
        };
        // The smaller child's bound is what the larger child left of the
        // incumbent, widened by the interval half-distance so that one search
        // can prune the whole interval around this split.
        let budget =
            current_best.error.min(upper_bound.saturating_add(1)) as i64 - larger_error as i64;
        let smaller_ub = budget + int_half_distance as i64;
        let smaller_upper_bound = smaller_ub.max(0) as usize;
        let mut right_error: Option<usize> = None;

        // Search the smaller child unless the budget is negative. A budget of
        // exactly zero is still explored, since a zero-error subtree may exist.
        if smaller_ub > 0 || budget == 0 {
            self.statistics.general_solver_call += 1;
            let right_config = config.derive_right(left_config.max_gap);

            if process_left_first {
                right_entry.error = current_best.error;
                (right_is_new, right_index, right_entry) =
                    self.cache_child(&right_view, right_config.max_depth);

                stopped |= self.expand_node_with_view(
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

                stopped |= self.expand_node_with_view(
                    &left_view,
                    &right_config,
                    &mut left_entry,
                    left_index,
                    left_is_new,
                    smaller_upper_bound,
                );
                left_entry.finalize_lower_bound(smaller_upper_bound);
            }

            // Use lower bounds rather than errors: when a bound cut a child
            // search short, its `error` is only an upper bound.
            right_error = Some(right_entry.lower_bound);

            let feature_best = left_entry.lower_bound + right_entry.lower_bound;
            if left_entry.is_valid && right_entry.is_valid && feature_best < current_best.error {
                current_best.error = feature_best;
                if config.is_root {
                    self.trajectory.push((self.elapsed_time(), feature_best));
                }
                current_best.feature = feature_index;
                current_best.split = threshold_value;
                current_best.left = left_index;
                current_best.right = right_index;

                if let Some(entry) = self.cache.get_mut(cache_index) {
                    *entry = *current_best;
                }

                if feature_best == 0 {
                    return false;
                }
            }
        }

        // `None` for a child the bound invalidated: its lower bound may be
        // zero without the subtree being error-free, and the pruner would read
        // a zero as "nothing beyond this split can do better".
        let left_score = left_entry.is_valid.then_some(left_entry.lower_bound);
        pruner.add_result(split_idx, left_score, right_error);
        pruned[split_idx] = true;

        let score_difference = left_entry
            .lower_bound
            .saturating_add(right_error.unwrap_or(0))
            .saturating_sub(current_best.error.min(upper_bound));
        let (new_left_bound, new_right_bound) =
            pruner.neighbourhood_pruning(score_difference, current_left, current_right, split_idx);

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
        stopped
    }

    fn discrepancy_limit(nb_candidates: usize, remaining_depth: usize) -> usize {
        let mut max_discrepancy = nb_candidates;
        for i in 1..remaining_depth {
            max_discrepancy += nb_candidates.saturating_sub(i);
        }

        max_discrepancy
    }

    pub fn statistics(&self) -> &Statistics {
        &self.statistics
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

    /// The tree found by the passes run so far.
    pub fn get_solution_tree(&mut self) -> Tree {
        shared::build_solution_tree(&self.cache)
    }
}

#[cfg(test)]
mod contree_lds_test {
    use crate::algorithms::{ConTree, ConTreeLds};
    use crate::common::PointSelector;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;
    use crate::tests::fixture;

    #[test]
    fn lds_is_never_better_than_the_exhaustive_optimum() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("hepatitis.txt"))?;
        dataset.sort_features();

        let mut exhaustive: ConTree =
            ConTree::new(1, 2, 100.0, usize::MAX, PointSelector::Mid, 0, false, true);
        exhaustive.fit(&dataset).expect("fit");

        let mut lds: ConTreeLds =
            ConTreeLds::new(1, 2, 100.0, usize::MAX, PointSelector::Mid, 0, false, true);
        lds.fit(&dataset).expect("fit");

        // LDS restricts the search, so it can only tie or lose against the
        // exhaustive optimum. Beating it would mean the exhaustive search is
        // pruning something it should not.
        assert!(
            lds.statistics.error >= exhaustive.statistics().error,
            "LDS reported {} against an exhaustive optimum of {}",
            lds.statistics.error,
            exhaustive.statistics().error
        );
        assert!(lds.cache.len() > 0, "the cache was never written to");

        Ok(())
    }

    #[test]
    fn the_anytime_loop_terminates_without_leaning_on_the_time_limit() {
        // With no time limit, the search must still stop once the budget
        // schedule runs out.
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("small.txt")).unwrap();
        dataset.sort_features();

        let mut lds: ConTreeLds = ConTreeLds::new(
            1,
            3,
            f64::INFINITY,
            usize::MAX,
            PointSelector::Mid,
            0,
            false,
            false,
        );
        lds.fit(&dataset).expect("fit");

        assert!(lds.statistics.error <= dataset.count());
    }
}
