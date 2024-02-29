mod murtree;
mod odt_info_gain;

use crate::globals::item;
use crate::structures::Structure;
use crate::tree::Tree;

use crate::searches::SearchStrategy;
pub use murtree::Murtree;
pub use odt_info_gain::InfoGainDT;

pub enum GenericDepth2 {
    Murtree(Murtree),
    InfoGainOdt(InfoGainDT),
}

impl GenericDepth2 {
    pub fn new(strategy: SearchStrategy) -> Self {
        match strategy {
            SearchStrategy::LessGreedyMurtree => GenericDepth2::Murtree(Murtree::default()),
            SearchStrategy::LessGreedyInfoGain => GenericDepth2::InfoGainOdt(InfoGainDT::default()),
            SearchStrategy::DiscrepancySearch | SearchStrategy::None_ => {
                panic!("Strategy not available for LGDT")
            }
        }
    }

    pub fn fit<S: Structure>(&mut self, min_sup: usize, depth: usize, structure: &mut S) -> Tree {
        match self {
            GenericDepth2::Murtree(ref mut learner) => learner.fit(min_sup, depth, structure),
            GenericDepth2::InfoGainOdt(ref mut learner) => learner.fit(min_sup, depth, structure),
        }
    }
}

const MAX_ERROR: f64 = <f64>::INFINITY;
pub trait Depth2Algorithm {
    fn fit<S: Structure>(&self, min_sup: usize, depth: usize, structure: &mut S) -> Tree;

    fn generate_candidates_list<S: Structure>(
        &self,
        structure: &mut S,
        min_sup: usize,
    ) -> Vec<usize> {
        let num_attributes = structure.num_attributes();
        let mut candidates = Vec::with_capacity(num_attributes);
        for i in 0..num_attributes {
            if structure.temp_push(item(i, 0)) >= min_sup
                && structure.temp_push(item(i, 1)) >= min_sup
            {
                candidates.push(i);
            }
        }
        candidates
    }

    fn build_depth_two_matrix<S: Structure>(
        &self,
        structure: &mut S,
        candidates: &Vec<usize>,
    ) -> Vec<Vec<Vec<usize>>> {
        let size = candidates.len();
        let mut matrix = vec![vec![vec![]; size]; size];
        for i in 0..size {
            structure.push(item(candidates[i], 1));
            let val = structure.labels_support();
            matrix[i][i] = val.to_vec();

            for second in i + 1..size {
                structure.push(item(candidates[second], 1));
                let val = structure.labels_support();
                matrix[i][second] = val.to_vec();
                matrix[second][i] = val.to_vec();
                structure.backtrack();
            }
            structure.backtrack();
        }
        matrix
    }
}

// TODO : Move this in a utils files in search
