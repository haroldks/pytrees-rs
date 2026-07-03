use std::time::Instant;

use crate::algorithms::common::errors::NativeError;
use crate::algorithms::common::heuristics::NoHeuristic;
use crate::algorithms::common::types::OptimalDepth2Policy;
use crate::algorithms::optimal::depth2::ErrorMinimizer;
use crate::algorithms::optimal::dl85::DL85Builder;
use crate::algorithms::TreeSearchAlgorithm;
use crate::caching::Trie;
use crate::reader::Dataset;
use crate::tree::Tree;

use super::{ImpurityMetric, MIDDiscretizer};
use super::schedule::ThresholdSchedule;

/// Configuration for the DL8.5 learner used inside each MID iteration.
#[derive(Clone, Debug)]
pub struct DL85Config {
    pub max_depth: usize,
    pub min_sup: usize,
    pub max_error: f64,
}

impl Default for DL85Config {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_sup: 1,
            max_error: f64::INFINITY,
        }
    }
}

/// Anytime MID search: runs DL8.5 with increasing numbers of binary features
/// until the total time budget or schedule limit is exhausted.
///
/// At each iteration, a fresh `Cover` is built from the top-N MID thresholds,
/// and a fresh DL8.5 instance is run with the remaining global time budget.
/// The best tree found across all iterations is kept.
pub struct MIDSearch {
    discretizer: MIDDiscretizer,
    schedule: ThresholdSchedule,
    dl85_config: DL85Config,
    time_limit: f64,
    start_time: Option<Instant>,
    best_tree: Tree,
    best_error: f64,
    pub iteration: usize,
}

impl MIDSearch {
    /// Create a new `MIDSearch` from a `Dataset` and immediately fit the `MIDDiscretizer`.
    ///
    /// The schedule's `limit` is clamped to the number of thresholds MID can produce
    /// so that the schedule always reflects achievable bounds.
    pub fn new(
        dataset: &Dataset,
        metric: ImpurityMetric,
        dl85_config: DL85Config,
        mut schedule: ThresholdSchedule,
        time_limit: f64,
    ) -> Self {
        let discretizer = MIDDiscretizer::fit(&dataset.features, &dataset.labels, metric);

        // Clamp schedule limit to what MID actually computed.
        let available = discretizer.len();
        if schedule.limit == 0 || schedule.limit > available {
            schedule.limit = available;
        }

        Self {
            discretizer,
            schedule,
            dl85_config,
            time_limit,
            start_time: None,
            best_tree: Tree::default(),
            best_error: f64::INFINITY,
            iteration: 0,
        }
    }

    /// Run one iteration: build a cover with the current N thresholds, run DL8.5,
    /// update the best result, then advance the schedule.
    ///
    /// Returns `true` if the iteration ran and more iterations may follow.
    /// Returns `false` if the search is exhausted (schedule done or time budget used up).
    ///
    /// The global clock starts on the first call and is shared across all subsequent
    /// `partial_fit` calls, so the time budget covers the whole sequence.
    pub fn partial_fit(&mut self) -> bool {
        let start = self.start_time.get_or_insert_with(Instant::now);
        let elapsed = start.elapsed().as_secs_f64();

        if elapsed >= self.time_limit || self.schedule.exhausted(self.discretizer.len()) {
            return false;
        }

        let n = self.schedule.current();
        let time_remaining = self.time_limit - elapsed;
        let mut cover = self.discretizer.get_cover(n);

        let max_error = if self.best_error.is_finite() {
            self.best_error
        } else {
            self.dl85_config.max_error
        };

        let error_fn = Box::<NativeError>::default();
        let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));

        let mut learner = DL85Builder::default()
            .max_depth(self.dl85_config.max_depth)
            .min_support(self.dl85_config.min_sup)
            .max_error(max_error)
            .max_time(time_remaining)
            .specialization(OptimalDepth2Policy::Enabled)
            .cache(Box::<Trie>::default())
            .heuristic(Box::<NoHeuristic>::default())
            .depth2_search(depth2)
            .error_function(error_fn)
            .build()
            .expect("DL85 builder should succeed with complete configuration");

        if learner.fit(&mut cover).is_ok() {
            let err = learner.error();
            if err < self.best_error {
                self.best_error = err;
                self.best_tree = learner.tree().clone();
            }
        }

        self.schedule.step();
        self.iteration += 1;

        let now_elapsed = self.start_time.unwrap().elapsed().as_secs_f64();
        !self.schedule.exhausted(self.discretizer.len()) && now_elapsed < self.time_limit
    }

    /// Run all iterations until the time budget or schedule is exhausted.
    ///
    /// Resets the schedule and clock, then calls `partial_fit` in a loop.
    pub fn fit(&mut self) {
        self.schedule.reset();
        self.start_time = Some(Instant::now());
        self.iteration = 0;

        loop {
            let elapsed = self.start_time.unwrap().elapsed().as_secs_f64();
            if elapsed >= self.time_limit || self.schedule.exhausted(self.discretizer.len()) {
                break;
            }
            self.partial_fit();
        }
    }

    /// The best tree found across all iterations.
    pub fn tree(&self) -> &Tree {
        &self.best_tree
    }

    /// The error of the best tree found (raw misclassification count).
    pub fn error(&self) -> f64 {
        self.best_error
    }

    /// Elapsed seconds since the search started (first `partial_fit` or `fit` call).
    /// Returns `0.0` if the search has not started yet.
    pub fn elapsed(&self) -> f64 {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// The underlying discretizer (for inspection of computed thresholds).
    pub fn discretizer(&self) -> &MIDDiscretizer {
        &self.discretizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Dataset;

    fn toy_dataset() -> Dataset {
        let f0: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let f1: Vec<f64> = (0..10).map(|i| (i * 2) as f64).collect();
        let y: Vec<usize> = (0..10).map(|i| if i < 5 { 0 } else { 1 }).collect();
        Dataset::new(vec![f0, f1], y)
    }

    #[test]
    fn fit_runs_without_panic() {
        let dataset = toy_dataset();
        let schedule = ThresholdSchedule::new(1, 1, 100); // limit > available; will be clamped
        let config = DL85Config { max_depth: 2, min_sup: 1, max_error: f64::INFINITY };
        let mut search = MIDSearch::new(&dataset, ImpurityMetric::Gini, config, schedule, 5.0);
        search.fit();
        assert!(search.error().is_finite());
    }

    #[test]
    fn schedule_limit_is_clamped() {
        let dataset = toy_dataset();
        let schedule = ThresholdSchedule::new(1, 1, 9999);
        let config = DL85Config::default();
        let search = MIDSearch::new(&dataset, ImpurityMetric::Gini, config, schedule, 1.0);
        assert!(search.schedule.limit <= search.discretizer.len());
    }

    #[test]
    fn partial_fit_runs_one_iteration() {
        let dataset = toy_dataset();
        let schedule = ThresholdSchedule::new(1, 1, 3);
        let config = DL85Config { max_depth: 2, min_sup: 1, max_error: f64::INFINITY };
        let mut search = MIDSearch::new(&dataset, ImpurityMetric::Gini, config, schedule, 10.0);
        // First call should run (schedule starts at 1, limit=3).
        let more = search.partial_fit();
        // After one step schedule is at 2; there is still more to do.
        assert!(more || search.error().is_finite());
    }
}
