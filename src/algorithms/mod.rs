use crate::algorithms::common::types::FitError;
use crate::cover::Cover;
use crate::tree::Tree;

mod common;
pub mod optimal;
mod greedy;

pub trait TreeSearchAlgorithm {
    
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError>;
    
    fn tree(&self) -> &Tree;
    
}
