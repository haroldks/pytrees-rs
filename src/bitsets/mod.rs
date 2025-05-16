use std::ops::Index;

pub enum BitsetInit {
    Empty(usize),
    Full(usize),
}

#[derive(Debug, Clone)]
pub struct Bitset {
    capacity: usize,
    words: Vec<u64>,
}

pub trait BitCollection {
    fn new(init: BitsetInit) -> Self;

    fn count(&self) -> usize;

    fn test(&self, index: usize) -> bool;

    fn set(&mut self, index: usize);

    fn unset(&mut self, index: usize);

    fn is_empty(&self) -> bool;

    fn clear(&mut self);

    fn capacity(&self) -> usize;

    fn resize(&mut self, capacity: usize);

    fn intersect_with(&mut self, other: &Bitset);

    fn union_with(&mut self, other: &Bitset);

    fn count_intersect_with(&self, other: &Bitset) -> usize;

    fn count_interest_with_many(&self, others: &[Bitset]) -> Vec<usize>;
}

impl BitCollection for Bitset {
    fn new(init: BitsetInit) -> Self {
        match init {
            BitsetInit::Empty(n) => {
                let word_count = (n + 63) / 64;
                Self {
                    capacity: n,
                    words: vec![0u64; word_count],
                }
            }
            BitsetInit::Full(n) => {
                let word_count = (n + 63) / 64;
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
        let new_words = (capacity + 63) / 64;
        if new_words > self.words.len() {
            self.words.resize(new_words, 0);
        } else if new_words < self.words.len() {
            self.words.truncate(new_words);
        }
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
