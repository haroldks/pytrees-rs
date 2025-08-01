use std::collections::BTreeSet;

pub enum CacheKey {
    Index(usize),
    Path(Vec<usize>),
}

#[derive(Default)]
pub struct SearchPath {
    inner: BTreeSet<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Index {
    NewUnknown,
    New(usize),
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

    pub fn to_key(&self) -> CacheKey {
        CacheKey::Path(self.inner.iter().copied().collect()) // Create a copy of the
    }

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

    pub fn to_cache_key(&self, fallback_path: &SearchPath) -> CacheKey {
        match self {
            Index::New(pos) | Index::Existing(pos) => CacheKey::Index(*pos),
            Index::NewUnknown => fallback_path.to_key(),
        }
    }
}
