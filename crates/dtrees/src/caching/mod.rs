//! The cache of subproblems used by DL8.5.
//!
//! A subproblem is identified by the itemset of tests leading to it, sorted,
//! so that paths testing the same features in a different order share one
//! entry.

use crate::caching::entry::CacheEntryUpdater;

mod entry;
mod helpers;
mod trie;
pub use entry::CacheEntry;
pub use helpers::{CacheKey, Index, SearchPath};
pub use trie::Trie;

/// A cache of subproblems keyed by sorted itemsets.
pub trait Caching {
    /// Clears the cache and creates the root entry.
    fn init(&mut self) -> Index;

    /// Index of the root entry.
    fn root_index(&mut self) -> Index;

    /// The root entry.
    fn root(&self) -> Option<&CacheEntry>;

    /// Finds the entry of the sorted itemset `key`, creating it if needed.
    /// The index is `New` for an entry that has not been evaluated yet.
    fn insert(&mut self, key: &[usize]) -> Index;

    /// The entry for `key`.
    fn node(&self, key: &CacheKey) -> Option<&CacheEntry>;

    /// Whether the cache has an entry for `key`.
    fn contains(&self, key: &CacheKey) -> bool;

    /// An updater for the root entry.
    fn update_root(&mut self) -> Option<CacheEntryUpdater<'_>>;

    /// An updater for the entry of `key`.
    fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater<'_>>;

    /// Number of entries.
    fn size(&self) -> usize;

    /// Whether the cache has no entry.
    fn is_empty(&self) -> bool;
}
