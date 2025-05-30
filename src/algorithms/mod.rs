use crate::cover::Cover;
use crate::tree::Tree;

mod common;
mod optimal;

pub trait TreeSearchAlgorithm {
    
    fn fit(&mut self, cover: &mut Cover);
    
    fn tree(&self) -> &Tree;
    
}