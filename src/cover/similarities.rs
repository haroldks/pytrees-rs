use crate::cover::reversible_cover::{Difference, ShallowBitset, SparseBitset};

#[derive(Debug)]
pub struct SimilarityCover {
    covers: [Option<ShallowBitset>; 2],
    errors: [f64; 2],
}

impl Default for SimilarityCover {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityCover {
    pub fn new() -> Self {
        Self {
            covers: [None, None],
            errors: [f64::INFINITY; 2],
        }
    }

    pub fn update(&mut self, cover: &SparseBitset, error: f64) {
        let shallow_cover: ShallowBitset = cover.into();

        match (self.covers[0].as_ref(), self.covers[1].as_ref()) {
            (None, _) => {
                self.covers[0] = Some(shallow_cover);
                self.errors[0] = error;
                return;
            }
            (_, None) => {
                self.covers[1] = Some(shallow_cover);
                self.errors[1] = error;
                return;
            }
            _ => {}
        }

        let differences: Vec<Difference> = self
            .covers
            .iter()
            .map(|sim_cover| sim_cover.as_ref().map(|c| cover - c).unwrap_or_default())
            .collect();

        let min_idx = differences
            .iter()
            .enumerate()
            .min_by_key(|&(_, diff)| diff)
            .map(|(idx, _)| idx);
        if let Some(idx) = min_idx {
            self.covers[idx] = Some(shallow_cover);
        }
    }

    pub fn compute_similarity(&self, cover: &SparseBitset) -> f64 {
        self.covers
            .iter()
            .enumerate()
            .filter_map(|(i, cover_opt)| {
                cover_opt.as_ref().map(|sim_cover| {
                    let diff = cover - sim_cover;
                    self.errors[i] - diff.out_count as f64
                })
            })
            .fold(0.0, f64::max)
    }
}
