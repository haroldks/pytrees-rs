use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct Key {
    internal: Vec<usize>,
    pub hash: usize,
    pub depth: usize,
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            false
        } else {
            (self.depth == other.depth)
                && (self.internal.len() == other.internal.len())
                && self.internal.iter().eq(other.internal.iter())
        }
    }
}

impl Eq for Key {}

impl Key {
    pub fn new(internal: &[usize], depth: usize) -> Self {
        let hash = Self::hash_function(internal);
        Self {
            internal: internal.to_vec(),
            hash,
            depth,
        }
    }

    fn hash_function(itemset: &[usize]) -> usize {
        let mut h = itemset.len();
        for item in itemset.iter() {
            h ^= item
                .wrapping_add(0x9e3779b9)
                .wrapping_add(h << 6)
                .wrapping_add(h >> 2);
        }
        h
    }
}
