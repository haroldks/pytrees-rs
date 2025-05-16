use crate::bitsets::Bitset;
use search_trail::{
    ReversibleU64, ReversibleUsize, SaveAndRestore, StateManager, U64Manager, UsizeManager,
};
use std::cmp::Ordering;
use std::ops::Sub;

pub struct SparseBitset {
    nb_words: usize,
    words: Vec<ReversibleU64>,
    non_zero_words: Vec<usize>,
    nb_non_zero: ReversibleUsize,

    state_manager: StateManager,
    mask: u64,
}

#[derive(Debug)]
pub struct ShallowBitset {
    words: Vec<u64>,
    non_zero_words: Vec<usize>,
    nb_non_zero: usize,
}

#[derive(Default)]
pub struct Difference {
    pub(crate) in_count: usize,
    pub(crate) out_count: usize,
}

impl SparseBitset {
    pub fn new(n: usize) -> Self {
        let mut state_manager = StateManager::default();

        let nb_words = (n + 63) / 64;
        let mut words = vec![state_manager.manage_u64(u64::MAX); nb_words];
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
            nb_words,
            words,
            non_zero_words,
            nb_non_zero,
            state_manager,
            mask,
        }
    }

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

    pub fn is_empty(&self) -> bool {
        self.state_manager.get_usize(self.nb_non_zero) == 0
    }

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

    pub fn to_vec(&self) -> Vec<usize> {
        // TODO
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

    #[inline]
    pub fn restore(&mut self) {
        self.state_manager.restore_state();
    }
}

// Helpers for more readability
impl Into<ShallowBitset> for &SparseBitset {
    fn into(self) -> ShallowBitset {
        let mut words = Vec::with_capacity(self.words.len());
        let non_zero_words = self.non_zero_words.clone();

        let nb_non_zero = self.state_manager.get_usize(self.nb_non_zero);
        for i in 0..self.words.len() {
            words.push(self.state_manager.get_u64(self.words[i]));
        }

        ShallowBitset {
            words,
            non_zero_words,
            nb_non_zero,
        }
    }
}

impl Sub<&ShallowBitset> for &SparseBitset {
    type Output = Difference;
    fn sub(self, rhs: &ShallowBitset) -> Self::Output {
        let in_count: usize = (0..self.state_manager.get_usize(self.nb_non_zero))
            .map(|i| {
                let idx = self.non_zero_words[i];
                let self_word = self.state_manager.get_u64(self.words[idx]);
                (self_word & !rhs.words[idx]).count_ones() as usize
            })
            .sum();

        let out_count: usize = (0..rhs.nb_non_zero)
            .map(|i| {
                let idx = rhs.non_zero_words[i];
                let self_size = self.state_manager.get_usize(self.nb_non_zero);
                let self_word = if i < self_size {
                    // TODO : Not clear
                    self.state_manager.get_u64(self.words[idx])
                } else {
                    0
                };
                (rhs.words[idx] & !self_word).count_ones() as usize
            })
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

        // feature.set(5);
        feature.set(9);
        // feature.set(3);
        // feature.set(54);
        // feature.set(32);

        println!("{:?}", cover.to_vec());
        assert_eq!(cover.count(), 10);

        cover.intersect_with(&feature, false);
        println!("{:?}", cover.to_vec());
        println!("xx {:?}", cover.state_manager.get_usize(cover.nb_non_zero));

        let shallow_cover: ShallowBitset = (&cover).into();
        println!("Shaloow {:?}", shallow_cover);
        cover.restore();
        println!("{:?}", cover.to_vec());

        let diff = &cover - shallow_cover;

        // let shallow : ShallowBitset = (&cover).into();
        // println!("shallow : {:?}", shallow)
    }
}
