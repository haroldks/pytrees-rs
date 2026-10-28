use std::collections::BTreeSet;

/// Position of an entry in the cache.
pub struct CacheKey(pub usize);

/// The set of items on the path from the root to the current node, kept
/// sorted.
#[derive(Default)]
pub struct SearchPath {
    inner: BTreeSet<usize>,
}

/// Result of a cache insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Index {
    /// A new (or never evaluated) entry at this position.
    New(usize),
    /// An entry evaluated before, at this position.
    Existing(usize),
}

impl SearchPath {
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    pub fn push(&mut self, value: usize) {
        self.inner.insert(value);
    }

    pub fn remove(&mut self, value: &usize) {
        self.inner.remove(value);
    }

    /// The items of the path, sorted.
    pub fn to_sorted_vec(&self) -> Vec<usize> {
        self.inner.iter().copied().collect()
    }
}

impl Index {
    pub fn new_at(position: usize) -> Self {
        Index::New(position)
    }

    pub fn existing(position: usize) -> Self {
        Index::Existing(position)
    }

    /// Whether the entry has not been evaluated yet.
    pub fn is_new(&self) -> bool {
        matches!(self, Index::New(_))
    }

    pub fn position(&self) -> usize {
        match self {
            Index::New(pos) | Index::Existing(pos) => *pos,
        }
    }

    /// The key of the entry.
    pub fn to_cache_key(&self) -> CacheKey {
        CacheKey(self.position())
    }
}
