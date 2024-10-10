use crate::structures::Structure;

pub trait Rule {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) -> bool;
    fn weaken(&mut self, rate: f64);
}

pub struct TopK {
    k: usize,
}

impl Rule for TopK {
    fn compute(&self, _structure: &mut dyn Structure, candidates: &mut Vec<usize>) -> bool {
        if candidates.len() > self.k {
            candidates.truncate(self.k);
            return true;
        }
        false
    }

    fn weaken(&mut self, rate: f64) {
        self.k += rate as usize;
    }
}

pub struct Purity {
    threshold: f64,
}

impl Rule for Purity {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) -> bool {
        todo!()
    }

    fn weaken(&mut self, rate: f64) {
        todo!()
    }
}
