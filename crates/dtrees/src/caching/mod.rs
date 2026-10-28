//! The cache of subproblems used by DL8.5.
//!
//! [`Trie`] identifies a subproblem by the sorted itemset of its path.
//! [`CoverMap`] identifies it by its depth and instances.

use crate::caching::entry::CacheEntryUpdater;
use crate::cover::Cover;

mod cover_map;
mod entry;
mod helpers;
mod trie;
pub use cover_map::CoverMap;
pub use entry::{CacheEntry, EXACT};
pub use helpers::{CacheKey, Index, SearchPath};
pub use trie::Trie;

/// A cache of subproblems. Lookups take the sorted itemset of the path and
/// `cover`, positioned on the instances of that path.
pub trait Caching {
    /// Clears the cache and creates the root entry.
    fn init(&mut self) -> Index;

    /// Index of the root entry.
    fn root_index(&mut self) -> Index;

    /// The root entry.
    fn root(&self) -> Option<&CacheEntry>;

    /// Finds the entry of the subproblem, creating it if needed. The index is
    /// `New` for an entry that has not been evaluated yet.
    fn insert(&mut self, path: &[usize], cover: &Cover) -> Index;

    /// The entry of the subproblem, if cached.
    fn find(&self, path: &[usize], cover: &Cover) -> Option<&CacheEntry>;

    /// Whether lookups use `cover`.
    fn keyed_by_cover(&self) -> bool;

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
