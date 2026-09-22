//! Fixed-capacity bitsets over instance ids.
//!
//! Every node of the search is identified by the set of training instances
//! that reach it, so these are hashed and compared constantly. Both the hash
//! and the population count are memoised -- see [`Bitset::get_hash`] and
//! [`Bitset::saved_count`] -- because recomputing either means another pass
//! over `n / 64` words.

use std::hash::{Hash, Hasher};
use std::ops::Index;

pub enum BitsetInit {
    Empty(usize),
    Full(usize),
}

#[derive(Debug, Clone)]
pub struct Bitset {
    capacity: usize,
    count: usize,
    hash: Option<u64>, // None = not computed yet
    words: Vec<u64>,
}

pub trait BitCollection {
    fn new(init: BitsetInit) -> Self;

    fn count(&self) -> usize;

    fn contains(&self, index: usize) -> bool;

    fn set(&mut self, index: usize);

    fn intersect_with(&self, other: &Bitset, invert: bool) -> Bitset;
}

impl BitCollection for Bitset {
    fn new(init: BitsetInit) -> Self {
        match init {
            BitsetInit::Empty(n) => {
                let word_count = n.div_ceil(64);
                Self {
                    capacity: n,
                    count: 0,
                    hash: None,
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
                Self {
                    capacity: n,
                    count: n,
                    hash: None,
                    words,
                }
            }
        }
    }

    fn count(&self) -> usize {
        self.words
            .iter()
            .map(|&word| word.count_ones() as usize)
            .sum()
    }

    fn contains(&self, index: usize) -> bool {
        debug_assert!(index < self.capacity, "Index out of bounds");
        (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }

    fn set(&mut self, index: usize) {
        debug_assert!(index < self.capacity, "Index out of bounds");
        self.words[index / 64] |= 1u64 << (index % 64);
        self.hash = None; // Invalidate hash
    }

    fn intersect_with(&self, other: &Bitset, invert: bool) -> Bitset {
        debug_assert_eq!(
            self.capacity, other.capacity,
            "Bitsets must have the same capacity"
        );
        let mut out = Bitset::new(BitsetInit::Empty(self.capacity));
        for (i, (&word, &other_word)) in self.words.iter().zip(&other.words).enumerate() {
            out.words[i] = word & if invert { !other_word } else { other_word };
        }
        out
    }
}

impl Bitset {
    /// Check if hash has been computed
    pub fn is_hash_set(&self) -> bool {
        self.hash.is_some()
    }

    /// Get the cached hash, computing it if necessary
    pub fn get_hash(&mut self) -> u64 {
        match self.hash {
            Some(hash) => hash,
            None => self.compute_hash(),
        }
    }

    /// Compute and cache the hash
    pub fn compute_hash(&mut self) -> u64 {
        let mut h = self.words.len() as u64;
        for &item in self.words.iter() {
            h ^= item
                .wrapping_add(0x9e3779b9)
                .wrapping_add(h << 6)
                .wrapping_add(h >> 2);
        }
        self.hash = Some(h);
        h
    }

    pub fn save_count(&mut self) {
        self.count = self.count();
    }

    /// The population count recorded by the last [`Self::save_count`].
    ///
    /// `count()` re-runs a full popcount over every word; on the cache's hot
    /// path that is one pass over `n / 64` words per lookup, for a number that
    /// was already computed when the bitset was built.
    pub fn saved_count(&self) -> usize {
        debug_assert_eq!(
            self.count,
            self.count(),
            "saved_count is stale: save_count was not called after the last mutation"
        );
        self.count
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

impl Hash for Bitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(h) = self.hash {
            h.hash(state);
        } else {
            let mut h = self.words.len() as u64;
            for &item in self.words.iter() {
                h ^= item
                    .wrapping_add(0x9e3779b9)
                    .wrapping_add(h << 6)
                    .wrapping_add(h >> 2);
            }
            h.hash(state);
        }
    }
}

impl PartialEq for Bitset {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(h1), Some(h2)) = (self.hash, other.hash) {
            if h1 != h2 {
                return false;
            }
        }

        self.count == other.count && self.words == other.words
    }
}

impl Eq for Bitset {}
