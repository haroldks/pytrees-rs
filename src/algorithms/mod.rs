use crate::algorithms::common::types::FitError;
use crate::algorithms::common::utils::find_valid_split_attributes;
use crate::cover::Cover;
use crate::tree::Tree;

pub mod common;
pub mod greedy;
pub mod optimal;

pub trait TreeSearchAlgorithm {
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError>;

    fn tree(&self) -> &Tree;

    fn error(&self) -> f64 {
        self.tree().root_error()
    }

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
