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
    /// What the pass in progress has run into; handed to the schedule.
    pass_report: PassReport,
    status: SearchStatus,
    /// Every improvement of the root incumbent, as `(seconds, error)`: the
    /// anytime profile of the search, from which a primal integral is taken.
    trajectory: Vec<(f64, usize)>,
}

impl ConTreeLds {
    /// Builds a solver from an assembled [`SearchConfig`].
    ///
    /// Prefer this over `new` when you are threading a configuration through:
    /// eight positional parameters, two of which are `bool`, is a call site
    /// nobody can read.
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

    /// The positional convenience constructor. See [`Self::with_config`].
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

    /// Seeds the split-point generator, making `PointSelector::Random`
    /// reproducible.
    ///
    /// Without this the search draws from a thread-local generator that cannot
    /// be seeded, so a random-split run could never be repeated -- and the
    /// generator is `!Send`, which kept the whole solver off a worker thread.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Runs the anytime search to completion.
    ///
    /// Each pass widens the discrepancy and split budgets; the loop ends when
    /// a pass finishes untruncated, the budgets run out, or time does.
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

        // The anytime loop can only make progress while the schedule has a
        // budget left to offer. Once it does not, a truncated search will never
        // become less truncated, and re-running it would spin until the time
        // limit -- or forever, when there is none.
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

    /// Looks a child subproblem up in the cache, seeding a fresh entry with the
    /// error of its majority-class leaf.
    ///
    /// This block appeared four times in each of the two searches, character
    /// for character. `child_depth` is the child's remaining depth, which is
    /// both the cache's second key and what the entry records: `derive_left`
    /// and `derive_right` both subtract one from the same parent, so the two
    /// were always the same number.
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

        // What this visit runs under: the discrepancy left below this node,
        // and the split budget where it applies -- the `first` selector, and
        // every selector's first pass.
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
                // A search the budget did not cut short proved its lower
                // bound. If that already rules out anything under the bound
                // the parent needs, there is nothing to look for.
                if !previous.truncated && current_best.lower_bound >= upper_bound {
                    self.statistics.cache_hits += 1;
                    return false;
                }
                // Searched before under at least this much budget and bound:
                // that result is at least as good as this visit could find,
                // and was cut short exactly when this visit would be.
                //
                // This replaces reusing whatever improved earlier in the same
                // pass, which reported "not cut short" even for a search that
                // was, and could let a pass count as complete when it was not.
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

        // `== 2`, not `<= 2`: `ConTreeDepth2` always builds a tree with two
        // levels of tests, so applying it to a depth-1 request returned a
        // depth-2 tree and an error below the depth-1 optimum.
        if config.fast_d2 && config.max_depth == 2 {
            // Solved without the parent's budget, unlike in the exhaustive
            // search. There a subproblem is visited once and pruning it against
            // the budget is pure gain. Here every pass revisits the same
            // depth-2 subproblems, and a budget-limited result is not exact, so
            // it used to be re-solved -- under the same budget, for the same
            // answer -- on every pass. Solved exactly once, it is reused
            // thereafter; the parent still judges it against its own budget.
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
                // budget too. Finish through the end of the function, which
                // records what this search ran under.
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
        // Reusable only when the search was neither truncated nor stopped
        // short by the bound. `!stopped` alone was not enough: a pass that ran
        // to completion under a tight bound proves nothing about a looser one.
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

        // Restrict the interval to splits that meet minimum support before the
        // search starts. The per-candidate check further down stays as a
        // guard, but on its own it is not enough: it `continue`s, which drops
        // the whole current interval and the feasible splits inside it.
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

        while !queue.is_empty() {
            if !self.time_remains() {
                return true;
            }

            let mut current_bound = queue.pop_front().unwrap();

            // Prune against the tighter of the incumbent and the budget the
            // parent handed down, as upstream does: a split that cannot come
            // in under the parent's budget is of no use to the parent, and
            // `finalize_lower_bound` records what that proved.
            // Prune against the tighter of the incumbent and the budget the
            // parent handed down, as upstream does: a split that cannot come
            // in under the parent's budget is of no use to the parent, and
            // `finalize_lower_bound` records what that proved.
            if pruner.subinterval_pruning(&current_bound, current_best.error.min(upper_bound)) {
                continue;
            }

            pruner.interval_shrinking(&mut current_bound, current_best.error.min(upper_bound));
            if !current_bound.is_valid() {
                continue;
            }

            // TODO : Allow to use other points as the best split and random

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

            // TODO : Do larger and smaller tree comparison and take the first

            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                continue;
            }

            // Determine which dataset is larger
            let process_left_first = left_view.len() >= right_view.len();

            // Always derive both configs
            let left_config = config.derive_left();
            let mut left_entry = Entry::default();
            let mut right_entry = Entry::default();

            // A cache index of 0 means "no child"; whichever side the search does
            // not descend into keeps it. The `is_new` flags are always written by
            // `Cache::insert` before they are read.
            let (mut left_index, mut right_index) = (0, 0);
            let (mut left_is_new, mut right_is_new);

            // Process LARGER dataset first with left_config
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
                // What the bound actually proved about this child.
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
                // What the bound actually proved about this child.
                right_entry.finalize_lower_bound(larger_upper_bound);
            }

            let larger_error = if process_left_first {
                left_entry.lower_bound
            } else {
                right_entry.lower_bound
            };
            // Upstream computes this as a signed quantity and *adds* the
            // interval half-distance, widening the budget so a whole interval
            // can be pruned at once; saturating at zero and taking the maximum
            // instead gives a tighter bound than the argument supports.
            let budget =
                current_best.error.min(upper_bound.saturating_add(1)) as i64 - larger_error as i64;
            let smaller_ub = budget + int_half_distance as i64;
            let smaller_upper_bound = smaller_ub.max(0) as usize;
            // `None` means the second subtree was never explored: the first
            // one alone already exhausted the upper bound, so this split
            // cannot beat the incumbent.
            let mut right_error: Option<usize> = None;

            // Process SMALLER dataset second with right_config
            // Search the second subtree unless the budget is genuinely
            // negative -- a budget of exactly zero still has to be explored,
            // since the first subtree may already account for the whole
            // incumbent.
            if smaller_ub > 0 || budget == 0 {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);

                if process_left_first {
                    // Right is smaller - process it second with right_config
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
                    // What the bound actually proved about this child.
                    right_entry.finalize_lower_bound(smaller_upper_bound);
                } else {
                    // Left is smaller - process it second with right_config
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
                    // What the bound actually proved about this child.
                    left_entry.finalize_lower_bound(smaller_upper_bound);
                }

                // Lower bounds, not errors: `error` is an upper bound once a
                // bound has cut the child search short, and everything below
                // this point reasons about what a split *cannot* beat.
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
            // zero without the subtree being error-free, and a zero here tells
            // the pruner that everything past this point is redundant.
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

    /// Explore splits prioritized by gini quality while using pruner
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

        // Get position-sorted splits for pruner
        let possible_splits = view.get_possible_split_indices(feature_index);

        if possible_splits.is_empty() {
            return false;
        }

        // Initialize pruner with position-sorted data
        let mut pruner = IntervalsPruner::new(possible_splits, config.max_gap, config.min_sup);

        // Track which split indices have been pruned
        let mut pruned = vec![false; possible_splits.len()];
        let mut stopped = false;

        let local_split_budget = possible_splits.len().min(self.split_budget);

        if config.use_heuristic {
            // Use gini-sorted order
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
            // Use normal position order
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

                // Skip if this split has been pruned
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
        // Find current bounds
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

        // Calculate threshold value
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

        // Split the view
        let (left_view, right_view) = view.split(feature_index, split_point);

        // Check minimum support
        if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
            pruned[split_idx] = true;
            return false;
        }

        // Determine which dataset is larger
        let process_left_first = left_view.len() >= right_view.len();

        // Always derive both configs
        let left_config = config.derive_left();
        let mut left_entry = Entry::default();
        let mut right_entry = Entry::default();

        // A cache index of 0 means "no child"; whichever side the search does
        // not descend into keeps it. The `is_new` flags are always written by
        // `Cache::insert` before they are read.
        let (mut left_index, mut right_index) = (0, 0);
        let (mut left_is_new, mut right_is_new);
        let mut stopped = false;

        let int_half_distance = split_point
            .saturating_sub(possible_splits[0])
            .max(possible_splits[possible_splits.len() - 1].saturating_sub(split_point));

        // Process LARGER dataset first with left_config
        let larger_upper_bound = current_best.error.min(upper_bound.saturating_add(1));
        self.statistics.general_solver_call += 1;

        if process_left_first {
            // Left is larger - process it first with left_config
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
            // What the bound actually proved about this child.
            left_entry.finalize_lower_bound(larger_upper_bound);
        } else {
            // Right is larger - process it first with left_config
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
            // What the bound actually proved about this child.
            right_entry.finalize_lower_bound(larger_upper_bound);
        }

        // Calculate upper bound for SMALLER dataset
        let larger_error = if process_left_first {
            left_entry.lower_bound
        } else {
            right_entry.lower_bound
        };
        // Upstream computes this as a signed quantity and *adds* the
        // interval half-distance, widening the budget so a whole interval
        // can be pruned at once; saturating at zero and taking the maximum
        // instead gives a tighter bound than the argument supports.
        let budget =
            current_best.error.min(upper_bound.saturating_add(1)) as i64 - larger_error as i64;
        let smaller_ub = budget + int_half_distance as i64;
        let smaller_upper_bound = smaller_ub.max(0) as usize;
        let mut right_error: Option<usize> = None;

        // Process SMALLER dataset second with right_config
        // Search the second subtree unless the budget is genuinely
        // negative -- a budget of exactly zero still has to be explored,
        // since the first subtree may already account for the whole
        // incumbent.
        if smaller_ub > 0 || budget == 0 {
            self.statistics.general_solver_call += 1;
            let right_config = config.derive_right(left_config.max_gap);

            if process_left_first {
                // Right is smaller - process it second with right_config
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
                // What the bound actually proved about this child.
                right_entry.finalize_lower_bound(smaller_upper_bound);
            } else {
                // Left is smaller - process it second with right_config
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
                // What the bound actually proved about this child.
                left_entry.finalize_lower_bound(smaller_upper_bound);
            }

            // Lower bounds, not errors: `error` is an upper bound once a
            // bound has cut the child search short, and everything below this
            // point reasons about what a split *cannot* beat.
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

        // Record result in pruner
        // `None` for a child the bound invalidated: its lower bound may be
        // zero without the subtree being error-free, and a zero here tells
        // the pruner that everything past this point is redundant.
        let left_score = left_entry.is_valid.then_some(left_entry.lower_bound);
        pruner.add_result(split_idx, left_score, right_error);
        pruned[split_idx] = true;

        // Use pruner to mark neighbors as pruned
        let score_difference = left_entry
            .lower_bound
            .saturating_add(right_error.unwrap_or(0))
            .saturating_sub(current_best.error.min(upper_bound));
        let (new_left_bound, new_right_bound) =
            pruner.neighbourhood_pruning(score_difference, current_left, current_right, split_idx);

        // `neighbourhood_pruning` leaves two intervals standing,
        // `[current_left, new_right_bound]` and `[new_left_bound, current_right]`,
        // exactly as the interval-queue search uses them. Only the splits strictly
        // between them can be skipped. This used to mark from `current_left` and to
        // `current_right`, covering both surviving intervals as well -- a split that
        // merely tied the incumbent pruned both its neighbours, and the search
        // could finish its last pass and call a suboptimal tree optimal.
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
        // `is_search_exhausted` could never fire, so once the budget iterator
        // ran out the same truncated search repeated until `max_time`. With no
        // time limit that is a hang. This test would not return.
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
