use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::find_valid_split_attributes;
use crate::cover::Cover;
use crate::tree::Tree;

mod config;
mod error_minimizer;
mod info_gain_maximizer;

pub use info_gain_maximizer::InfoGainMaximizer;
pub use error_minimizer::ErrorMinimizer;

pub trait OptimalDepth2Tree {
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

    fn find_optimal_depth_one_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError>;

    fn find_optimal_depth_two_tree(
        &self,
        min_sup: usize,
        cover: &mut Cover,
        provided_candidates: Option<&[usize]>,
    ) -> Result<Tree, FitError>;
    
    
    fn error(&self, distribution: &[usize]) -> (f64, f64);

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
