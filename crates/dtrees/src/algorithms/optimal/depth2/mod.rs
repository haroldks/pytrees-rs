//! Specialised solvers for trees of depth one and two.
//!
//! Instead of recursing, they count the classes of every pair of features
//! once, in a matrix, and deduce every leaf of every depth-2 tree from it.
//! DL8.5 uses them for the last two levels, and LGDT uses them at each step.
//! The idea comes from MurTree (Demirović et al., JMLR 2022).

use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::find_valid_split_attributes;
use crate::cover::Cover;
use crate::tree::Tree;

mod config;
mod error_minimizer;
mod info_gain_maximizer;

pub use error_minimizer::ErrorMinimizer;
pub use info_gain_maximizer::InfoGainMaximizer;

/// A solver for trees of depth at most 2.
pub trait OptimalDepth2Tree {
    /// Finds the best tree of the given depth (1 or 2) on the instances of
    /// `cover`, using only `provided_candidates` when given.
    fn fit(
        &self,
        min_sup: usize,
        depth: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError> {
        match depth {
            1 => self.find_optimal_depth_one_tree(min_sup, cover, provided_candidates),
            2 => self.find_optimal_depth_two_tree(min_sup, cover, provided_candidates),
            x => Err(FitError::InvalidDepth(x)),
        }
    }

    /// The best tree with a single test.
    fn find_optimal_depth_one_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError>;

    /// The best tree with at most two levels of tests.
    fn find_optimal_depth_two_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError>;

    /// `(error, prediction)` of a leaf with the given class counts.
    fn error(&self, distribution: &[usize]) -> (f64, f64);

    /// Features that split the instances of `cover` with at least `min_sup`
    /// instances on each side.
    #[inline]
    fn get_candidates(
        &self,
        cover: &mut Cover,
        min_sup: usize,
        provided_candidates: Option<&[usize]>,
    ) -> Vec<usize> {
        find_valid_split_attributes(cover, min_sup, provided_candidates, None)
    }
}
