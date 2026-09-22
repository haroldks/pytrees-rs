//! The similarity lower bound of DL8.5.

use crate::cover::reversible_cover::{Difference, ShallowBitset, SparseBitset};

/// Two solved sets of instances and their errors, used to bound the error of
/// a new set: if a solved set had error `e` and `k` of its instances are
/// missing from the new set, the new set's error is at least `e - k`.
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
    /// No reference set yet.
    pub fn new() -> Self {
        Self {
            covers: [None, None],
            errors: [f64::INFINITY; 2],
        }
    }

    /// Records a solved set of instances and its error, filling an empty slot
    /// or replacing the closest reference set.
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
            self.errors[idx] = error;
        }
    }

    /// The best lower bound the reference sets give for `cover`, at least 0.
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

#[cfg(test)]
mod tests {
    use super::SimilarityCover;
    use crate::bitsets::{BitCollection, Bitset, BitsetInit};
    use crate::cover::reversible_cover::SparseBitset;

    /// The rows of `0..8` in `rows`.
    fn cover(rows: &[usize]) -> SparseBitset {
        let mut feature = Bitset::new(BitsetInit::Empty(8));
        for &row in rows {
            feature.set(row);
        }
        let mut cover = SparseBitset::new(8);
        cover.intersect_with(&feature, false);
        cover
    }

    #[test]
    fn a_replaced_reference_set_takes_its_new_error() {
        let mut similarity = SimilarityCover::new();
        similarity.update(&cover(&[0, 1, 2, 3]), 3.0);
        similarity.update(&cover(&[4, 5, 6, 7]), 3.0);
        // Closest to the first set, so it replaces it.
        similarity.update(&cover(&[0, 1, 2]), 1.0);

        // The bound for {0, 1, 2} must come from its own error, 1, not from
        // the error of the set it replaced.
        assert_eq!(similarity.compute_similarity(&cover(&[0, 1, 2])), 1.0);
    }
}
