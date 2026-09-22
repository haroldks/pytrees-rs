//! The tree learners: optimal ([`optimal`]: DL8.5 and depth-2 solvers) and
//! greedy with lookahead ([`greedy`]: LGDT).

use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::find_valid_split_attributes;
use crate::cover::Cover;
use crate::tree::Tree;

pub mod common;
pub mod greedy;
pub mod optimal;

/// A tree learner.
pub trait TreeSearchAlgorithm {
    /// Learns a tree on the instances of `cover`.
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError>;

    /// The tree learned by the last `fit`.
    fn tree(&self) -> &Tree;

    /// Training error of the learned tree.
    fn error(&self) -> f64 {
        self.tree().root_error()
    }

    /// Features that can split the current node. See
    /// [`find_valid_split_attributes`].
    #[inline]
    fn get_candidates(
        &self,
        cover: &mut Cover,
        min_sup: usize,
        provided_candidates: Option<&[usize]>,
        previous: Option<usize>,
    ) -> Vec<usize> {
        find_valid_split_attributes(cover, min_sup, provided_candidates, previous)
    }
}
