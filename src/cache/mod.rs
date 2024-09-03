pub mod trie;

pub use trie::Trie;

use std::collections::BTreeSet;

pub const MAX_ERROR: f64 = <f64>::INFINITY;
pub trait Caching {
    // Will return the root node index as an Option and the Root cache entry so that the error can be taken at a point>

    fn init(&mut self) -> Option<usize>;

    fn get_root_infos(&self) -> Option<&CacheEntry>;

    fn set_root_infos(&mut self) -> Option<&mut CacheEntry>;

    // Check if there is a node inside the cache for the current itemset
    fn get(&mut self, itemset: &BTreeSet<usize>, index: Option<usize>) -> Option<&mut CacheEntry>;

    fn find(&self, itemset: &BTreeSet<usize>) -> Option<&CacheEntry>;

    // Get mutable entry of a node

    // Insert node inside the cache and returns if it is new or not
    fn insert(&mut self, itemset: &BTreeSet<usize>) -> (bool, Option<usize>);

    fn size(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn print(&self);
}

#[derive(Copy, Clone, Debug)]
pub struct CacheEntry {
    pub item: usize,
    pub test: usize,
    pub discrepancy: usize,
    pub error: f64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub leaf_error: f64,
    pub target: f64,
    pub is_optimal: bool,
    pub is_leaf: bool,
}

impl CacheEntry {
    pub fn new(item: usize) -> Self {
        Self {
            item,
            test: <usize>::MAX,
            discrepancy: 0,
            error: MAX_ERROR,
            upper_bound: MAX_ERROR,
            lower_bound: 0.0,
            leaf_error: MAX_ERROR,
            target: 0.0,
            is_optimal: false,
            is_leaf: false,
        }
    }

    pub fn to_leaf(&mut self) {
        self.is_leaf = true;
        self.error = self.leaf_error;
    }
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            item: <usize>::MAX,
            test: <usize>::MAX,
            discrepancy: 0,
            error: MAX_ERROR,
            upper_bound: MAX_ERROR,
            lower_bound: 0.0,
            leaf_error: MAX_ERROR,
            target: 0.0,
            is_optimal: false,
            is_leaf: false,
        }
    }
}
