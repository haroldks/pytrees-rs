//! Heuristics that rank candidate features, best first.

pub mod helpers;

use crate::algorithms::common::heuristics::helpers::{
    entropy, gini_index, information_gain, weighted_entropy,
};
use crate::algorithms::common::utils::deduce_sibling_error_with_buffer;
use crate::cover::Cover;
use crate::globals::item;

/// Scores a split from the class counts of the parent and both children.
type ScoreFn = Box<dyn Fn(&[usize], &[usize], &[usize], f64) -> f64>;

/// Ranks candidate features at a node.
pub trait Heuristic: Send + Sync {
    /// Sorts `candidates` best first and returns their scores in the same
    /// order (empty when the heuristic does not score).
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) -> Vec<f64>;

    /// Scores each candidate with `scorer` and sorts them, ascending when
    /// `lower_is_better` and descending otherwise.
    fn compute_with_scorer(
        &self,
        parent_entropy: f64,
        cover: &mut Cover,
        candidates: &mut Vec<usize>,
        scorer: ScoreFn,
        lower_is_better: bool,
    ) -> Vec<f64> {
        if candidates.is_empty() {
            return vec![];
        }

        let root_distribution = cover.labels_count();
        let mut left_distribution = vec![0; cover.num_labels];
        let mut right_distribution = vec![0; cover.num_labels];

        let mut scores: Vec<_> = candidates
            .iter()
            .map(|&attr| {
                cover.branch_on(item(attr, 0));
                cover.labels_count_with_buffer(&mut left_distribution);
                cover.backtrack();

                deduce_sibling_error_with_buffer(
                    &root_distribution,
                    &left_distribution,
                    &mut right_distribution,
                );

                let score = scorer(
                    &root_distribution,
                    &left_distribution,
                    &right_distribution,
                    parent_entropy,
                );

                (attr, score)
            })
            .collect();

        if lower_is_better {
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        candidates.clear();
        let (sorted_candidates, scores): (Vec<usize>, Vec<f64>) = scores.into_iter().unzip();
        candidates.extend(sorted_candidates);
        scores
    }
}

/// Keeps the candidates in their original order.
#[derive(Default)]
pub struct NoHeuristic;

impl Heuristic for NoHeuristic {
    fn compute(&self, _cover: &mut Cover, _candidates: &mut Vec<usize>) -> Vec<f64> {
        vec![]
    }
}

/// Lowest weighted Gini impurity of the children first.
#[derive(Default)]
pub struct GiniIndex;

impl Heuristic for GiniIndex {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) -> Vec<f64> {
        self.compute_with_scorer(0.0, cover, candidates, Box::new(gini_index), true)
    }
}

/// Highest information gain first.
#[derive(Default)]
pub struct InformationGain;

impl Heuristic for InformationGain {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) -> Vec<f64> {
        if candidates.is_empty() {
            return vec![];
        }

        let parent_distribution = cover.labels_count();
        let parent_entropy = entropy(&parent_distribution);

        self.compute_with_scorer(
            parent_entropy,
            cover,
            candidates,
            Box::new(information_gain),
            false,
        )
    }
}

/// Lowest weighted entropy of the children first.
#[derive(Default)]
pub struct WeightedEntropy;

impl Heuristic for WeightedEntropy {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) -> Vec<f64> {
        self.compute_with_scorer(0.0, cover, candidates, Box::new(weighted_entropy), true)
    }
}
