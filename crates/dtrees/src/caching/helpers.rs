use std::collections::BTreeSet;

/// How to find an entry: by its position, or by its itemset.
pub enum CacheKey {
    /// Position in the cache.
    Index(usize),
    /// Sorted itemset of the path to the entry.
    Path(Vec<usize>),
}

/// The set of items on the path from the root to the current node, kept
/// sorted.
#[derive(Default)]
pub struct SearchPath {
    inner: BTreeSet<usize>,
}

/// Result of a cache insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Index {
    /// A new entry whose position is not known.
    NewUnknown,
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

    /// The path as a cache key.
    pub fn to_key(&self) -> CacheKey {
        CacheKey::Path(self.inner.iter().copied().collect())
    }

    /// The items of the path, sorted.
    pub fn to_sorted_vec(&self) -> Vec<usize> {
        self.inner.iter().copied().collect()
    }
}

impl CacheKey {
    pub fn from_index(index: usize) -> CacheKey {
        CacheKey::Index(index)
    }

    pub fn from_path(path: &SearchPath) -> CacheKey {
        path.to_key()
    }
}

impl Index {
    pub fn new_unknown() -> Self {
        Index::NewUnknown
    }

    pub fn new_at(position: usize) -> Self {
        Index::New(position)
    }

    pub fn existing(position: usize) -> Self {
        Index::Existing(position)
    }

    /// Whether the entry has not been evaluated yet.
    pub fn is_new(&self) -> bool {
        matches!(self, Index::New(_) | Index::NewUnknown)
    }

    pub fn position(&self) -> Option<usize> {
        match self {
            Index::New(pos) | Index::Existing(pos) => Some(*pos),
            Index::NewUnknown => None,
        }
    }

    pub fn has_position(&self) -> bool {
        !matches!(self, Index::NewUnknown)
    }

    /// A key for the entry: its position when known, `fallback_path`
    /// otherwise.
    pub fn to_cache_key(&self, fallback_path: &SearchPath) -> CacheKey {
        match self {
            Index::New(pos) | Index::Existing(pos) => CacheKey::Index(*pos),
            Index::NewUnknown => fallback_path.to_key(),
        }
    }
}
