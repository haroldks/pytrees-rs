pub mod helpers;

use crate::algorithms::common::heuristics::helpers::{
    entropy, gini_index, information_gain, weighted_entropy,
};
use crate::algorithms::common::utils::deduce_sibling_error_with_buffer;
use crate::cover::Cover;
use crate::globals::item;

pub trait Heuristic {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>);

    fn compute_with_scorer<F>(
        &self,
        cover: &mut Cover,
        candidates: &mut Vec<usize>,
        scorer: F,
        lower_is_better: bool,
    ) where
        F: Fn(&[usize], &[usize], &[usize]) -> f64,
    {
        if candidates.is_empty() {
            return;
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

                let score = scorer(&root_distribution, &left_distribution, &right_distribution);

                (attr, score)
            })
            .collect();

        if lower_is_better {
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        candidates.clear();
        candidates.extend(scores.into_iter().map(|(attr, _)| attr));
    }
}

#[derive(Default)]
pub struct NoHeuristic;

impl Heuristic for NoHeuristic {
    fn compute(&self, _cover: &mut Cover, _candidates: &mut Vec<usize>) {}
}

#[derive(Default)]
pub struct GiniIndex;

impl Heuristic for GiniIndex {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) {
        self.compute_with_scorer(cover, candidates, gini_index, true)
    }
}

#[derive(Default)]
pub struct InformationGain;

impl Heuristic for InformationGain {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) {
        if candidates.is_empty() {
            return;
        }

        let parent_distribution = cover.labels_count();
        let parent_entropy = entropy(&parent_distribution);

        self.compute_with_scorer(
            cover,
            candidates,
            |root, left, right| information_gain(parent_entropy, root, left, right),
            false,
        )
    }
}

#[derive(Default)]
pub struct WeightedEntropy;

impl Heuristic for WeightedEntropy {
    fn compute(&self, cover: &mut Cover, candidates: &mut Vec<usize>) {
        self.compute_with_scorer(cover, candidates, weighted_entropy, true)
    }
}
