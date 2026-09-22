//! Fixed-capacity bitsets.

use std::ops::Index;

/// Initial content of a [`Bitset`] of the given capacity.
pub enum BitsetInit {
    /// No element set.
    Empty(usize),
    /// Every element set.
    Full(usize),
}

/// A fixed-capacity set of integers stored as 64-bit words.
#[derive(Debug, Clone)]
pub struct Bitset {
    capacity: usize,
    words: Vec<u64>,
}

/// Operations on a bitset.
pub trait BitCollection {
    /// A bitset of the given capacity and content.
    fn new(init: BitsetInit) -> Self;

    /// Number of elements set.
    fn count(&self) -> usize;

    /// Whether `index` is set.
    fn test(&self, index: usize) -> bool;

    /// Sets `index`.
    fn set(&mut self, index: usize);

    /// Clears `index`.
    fn unset(&mut self, index: usize);

    /// Whether no element is set.
    fn is_empty(&self) -> bool;

    /// Clears every element.
    fn clear(&mut self);

    /// Largest number of elements.
    fn capacity(&self) -> usize;

    /// Changes the capacity.
    fn resize(&mut self, capacity: usize);

    /// Keeps only the elements also in `other`.
    fn intersect_with(&mut self, other: &Bitset);

    /// Adds the elements of `other`.
    fn union_with(&mut self, other: &Bitset);

    /// Size of the intersection with `other`.
    fn count_intersect_with(&self, other: &Bitset) -> usize;

    /// Size of the intersection with each of `others`.
    fn count_interest_with_many(&self, others: &[Bitset]) -> Vec<usize>;
}

impl BitCollection for Bitset {
    fn new(init: BitsetInit) -> Self {
        match init {
            BitsetInit::Empty(n) => {
                let word_count = n.div_ceil(64);
                Self {
                    capacity: n,
                    words: vec![0u64; word_count],
                }
            }
            BitsetInit::Full(n) => {
                let word_count = n.div_ceil(64);
                let mut words = vec![u64::MAX; word_count];
                if n > 0 && n % 64 != 0 {
                    if let Some(last) = words.last_mut() {
                        *last = (1u64 << (n % 64)) - 1;
                    }
                }
                Self { capacity: n, words }
            }
        }
    }

    fn count(&self) -> usize {
        self.words
            .iter()
            .map(|&word| word.count_ones() as usize)
            .sum()
    }

    fn test(&self, index: usize) -> bool {
        debug_assert!(index < self.capacity, "Index out of bounds");
        (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }

    fn set(&mut self, index: usize) {
        debug_assert!(index < self.capacity, "Index out of bounds");
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    fn unset(&mut self, index: usize) {
        debug_assert!(index < self.capacity, "Index out of bounds");
        self.words[index / 64] &= !(1u64 << (index % 64));
    }

    fn is_empty(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    fn clear(&mut self) {
        self.words.fill(0);
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resize(&mut self, capacity: usize) {
        let new_words = capacity.div_ceil(64);
        match new_words.cmp(&self.words.len()) {
            std::cmp::Ordering::Greater => self.words.resize(new_words, 0),
            std::cmp::Ordering::Less => self.words.truncate(new_words),
            std::cmp::Ordering::Equal => {}
        }
        self.capacity = 64 * self.words.len();
    }

    fn intersect_with(&mut self, other: &Bitset) {
        debug_assert_eq!(
            self.capacity, other.capacity,
            "Bitsets must have the same capacity"
        );
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word &= other_word;
        }
    }

    fn union_with(&mut self, other: &Bitset) {
        debug_assert_eq!(
            self.capacity, other.capacity,
            "Bitsets must have the same capacity"
        );
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word |= other_word;
        }
    }

    fn count_intersect_with(&self, other: &Bitset) -> usize {
        debug_assert_eq!(
            self.capacity, other.capacity,
            "Bitsets must have the same capacity"
        );
        let mut count = 0;
        for (word, other_word) in self.words.iter().zip(&other.words) {
            count += (*word & *other_word).count_ones() as usize;
        }
        count
    }

    fn count_interest_with_many(&self, others: &[Bitset]) -> Vec<usize> {
        others
            .iter()
            .map(|other| {
                self.words
                    .iter()
                    .zip(&other.words)
                    .map(|(&a, &b)| (a & b).count_ones() as usize)
                    .sum()
            })
            .collect()
    }
}

impl Index<usize> for Bitset {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(
            index < self.words.len(),
            "Index out for number of words bounds"
        );
        &self.words[index]
    }
}
