//! A reversible sparse bitset of instances.

use crate::bitsets::Bitset;
use search_trail::{
    ReversibleU64, ReversibleUsize, SaveAndRestore, StateManager, U64Manager, UsizeManager,
};
use std::cmp::Ordering;
use std::ops::Sub;

/// A set of instances that supports intersection and constant-time undo.
///
/// Words that become zero are moved out of the active part of
/// `non_zero_words`, so later operations only visit non-empty words.
pub struct SparseBitset {
    words: Vec<ReversibleU64>,
    non_zero_words: Vec<usize>,
    nb_non_zero: ReversibleUsize,

    state_manager: StateManager,
}

/// A non-reversible snapshot of a [`SparseBitset`]: every word, with the
/// inactive ones as 0.
#[derive(Debug)]
pub struct ShallowBitset {
    words: Vec<u64>,
}

/// Sizes of the two set differences between a [`SparseBitset`] and a
/// [`ShallowBitset`].
#[derive(Default)]
pub struct Difference {
    pub(crate) in_count: usize,
    pub(crate) out_count: usize,
}

impl SparseBitset {
    /// The full set `0..n`.
    pub fn new(n: usize) -> Self {
        let mut state_manager = StateManager::default();

        let nb_words = n.div_ceil(64);
        let mut words = Vec::with_capacity(nb_words);
        for _ in 0..nb_words {
            words.push(state_manager.manage_u64(u64::MAX));
        }

        let mask = if n % 64 == 0 {
            u64::MAX
        } else {
            (1u64 << (n % 64)) - 1
        };
        if let Some(last) = words.last_mut() {
            state_manager.set_u64(*last, mask);
        }
        let non_zero_words = (0..nb_words).collect();
        let nb_non_zero = state_manager.manage_usize(nb_words);

        state_manager.save_state();

        Self {
            words,
            non_zero_words,
            nb_non_zero,
            state_manager,
        }
    }

    /// Number of elements.
    pub fn count(&self) -> usize {
        let mut count = 0;
        let nb_non_zero = self.state_manager.get_usize(self.nb_non_zero);
        for i in (0..nb_non_zero).rev() {
            count += self
                .state_manager
                .get_u64(self.words[self.non_zero_words[i]])
                .count_ones();
        }
        count as usize
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.state_manager.get_usize(self.nb_non_zero) == 0
    }

    /// Intersects with `other` (with its complement when `invert`), saving the
    /// current state for [`Self::restore`]. Returns the new size.
    pub fn intersect_with(&mut self, other: &Bitset, invert: bool) -> usize {
        self.state_manager.save_state();

        let mut size = self.state_manager.get_usize(self.nb_non_zero);
        let mut count = 0;
        for i in (0..size).rev() {
            let idx = self.non_zero_words[i];
            let intersect = self.state_manager.get_u64(self.words[idx])
                & if invert { !other[idx] } else { other[idx] };
            if intersect == 0 {
                size -= 1;
                self.non_zero_words[i] = self.non_zero_words[size];
                self.non_zero_words[size] = idx;
            } else {
                self.state_manager.set_u64(self.words[idx], intersect);
                count += intersect.count_ones();
            }
        }
        self.state_manager.set_usize(self.nb_non_zero, size);

        count as usize
    }

    /// Size of the intersection with `other` (or its complement), without
    /// changing the set.
    pub fn count_intersect_with(&self, other: &Bitset, invert: bool) -> usize {
        let size = self.state_manager.get_usize(self.nb_non_zero);
        let mut count = 0;
        for i in (0..size).rev() {
            let idx = self.non_zero_words[i];
            let intersect = self.state_manager.get_u64(self.words[idx])
                & if invert { !other[idx] } else { other[idx] };
            if intersect != 0 {
                count += intersect.count_ones();
            }
        }
        count as usize
    }

    /// Size of the intersection with each of `others`.
    pub fn count_intersect_with_many(&self, others: &[Bitset]) -> Vec<usize> {
        let mut counts = vec![0; others.len()];
        let size = self.state_manager.get_usize(self.nb_non_zero);
        for i in (0..size).rev() {
            let idx = self.non_zero_words[i];
            let word = self.state_manager.get_u64(self.words[idx]);
            for (bid, other) in others.iter().enumerate() {
                counts[bid] += (word & other[idx]).count_ones() as usize
            }
        }
        counts
    }

    /// The elements of the set.
    pub fn to_vec(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let nb_non_zero = self.state_manager.get_usize(self.nb_non_zero);

        for i in 0..nb_non_zero {
            let word_idx = self.non_zero_words[i];
            let mut word = self.state_manager.get_u64(self.words[word_idx]);
            let base_idx = word_idx * 64;

            while word != 0 {
                let bit_pos = word.trailing_zeros() as usize;
                result.push(base_idx + bit_pos);
                word &= word - 1;
            }
        }
        result
    }

    /// Undoes the last intersection.
    #[inline]
    pub fn restore(&mut self) {
        self.state_manager.restore_state();
    }

    /// Every word of the set, with the inactive ones as 0.
    ///
    /// A word that becomes empty is moved out of the active part of
    /// `non_zero_words` without being overwritten, so its stored value is
    /// stale and must not be read.
    fn active_words(&self) -> Vec<u64> {
        let mut words = vec![0; self.words.len()];
        let nb_non_zero = self.state_manager.get_usize(self.nb_non_zero);
        for &idx in &self.non_zero_words[..nb_non_zero] {
            words[idx] = self.state_manager.get_u64(self.words[idx]);
        }
        words
    }
}

impl From<&SparseBitset> for ShallowBitset {
    fn from(val: &SparseBitset) -> Self {
        ShallowBitset {
            words: val.active_words(),
        }
    }
}

/// `in_count` counts the elements of `self` missing from `rhs`, and
/// `out_count` the elements of `rhs` missing from `self`.
impl Sub<&ShallowBitset> for &SparseBitset {
    type Output = Difference;
    fn sub(self, rhs: &ShallowBitset) -> Self::Output {
        let words = self.active_words();
        let in_count = words
            .iter()
            .zip(&rhs.words)
            .map(|(&own, &other)| (own & !other).count_ones() as usize)
            .sum();
        let out_count = words
            .iter()
            .zip(&rhs.words)
            .map(|(&own, &other)| (other & !own).count_ones() as usize)
            .sum();
        Difference {
            in_count,
            out_count,
        }
    }
}

impl Sub<ShallowBitset> for SparseBitset {
    type Output = Difference;

    fn sub(self, rhs: ShallowBitset) -> Self::Output {
        &self - &rhs
    }
}
impl Sub<&ShallowBitset> for SparseBitset {
    type Output = Difference;

    fn sub(self, rhs: &ShallowBitset) -> Self::Output {
        &self - rhs
    }
}

impl Sub<ShallowBitset> for &SparseBitset {
    type Output = Difference;

    fn sub(self, rhs: ShallowBitset) -> Self::Output {
        self - &rhs
    }
}

impl Ord for Difference {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.in_count + self.out_count).cmp(&(other.in_count + other.out_count))
    }
}

impl PartialOrd for Difference {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Difference {
    fn eq(&self, other: &Self) -> bool {
        (self.in_count + self.out_count) == (other.in_count + other.out_count)
    }
}

impl Eq for Difference {}

#[cfg(test)]
mod sparse_test {
    use crate::bitsets::{BitCollection, Bitset, BitsetInit};
    use crate::cover::reversible_cover::{ShallowBitset, SparseBitset};
    use search_trail::UsizeManager;

    #[test]
    fn create() {
        let mut cover = SparseBitset::new(10);

        let mut feature = Bitset::new(BitsetInit::Empty(10));

        feature.set(9);

        println!("{:?}", cover.to_vec());
        assert_eq!(cover.count(), 10);

        cover.intersect_with(&feature, false);
        println!("{:?}", cover.to_vec());
        println!("xx {:?}", cover.state_manager.get_usize(cover.nb_non_zero));

        let shallow_cover: ShallowBitset = (&cover).into();
        println!("Shaloow {:?}", shallow_cover);
        cover.restore();
        println!("{:?}", cover.to_vec());

        let _ = &cover - shallow_cover;
    }

    #[test]
    fn a_word_emptied_by_an_intersection_counts_as_empty() {
        // Rows 64..128 only: the intersection empties word 0, which keeps its
        // old value in storage but is no longer active.
        let mut feature = Bitset::new(BitsetInit::Empty(128));
        for row in 64..128 {
            feature.set(row);
        }
        let mut cover = SparseBitset::new(128);
        let full: ShallowBitset = (&cover).into();
        cover.intersect_with(&feature, false);

        let difference = &cover - &full;
        assert_eq!(difference.in_count, 0);
        assert_eq!(difference.out_count, 64, "rows 0..64 are only in `full`");

        let half: ShallowBitset = (&cover).into();
        cover.restore();
        let difference = &cover - &half;
        assert_eq!(difference.in_count, 64, "rows 0..64 are only in `cover`");
        assert_eq!(difference.out_count, 0);
    }
}
