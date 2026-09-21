use crate::caching::entry::CacheEntryUpdater;

mod entry;
mod helpers;
// The key of the hashmap cache in map.rs, which is not implemented yet.
#[allow(dead_code)]
mod key;
mod map;
mod trie;
pub use entry::CacheEntry;
pub use helpers::{CacheKey, Index, SearchPath};
pub use trie::Trie;

pub trait Caching {
    fn init(&mut self) -> Index;

    fn root_index(&mut self) -> Index;

    fn root(&self) -> Option<&CacheEntry>;

    fn insert(&mut self, key: &[usize]) -> Index;

    fn node(&self, key: &CacheKey) -> Option<&CacheEntry>;

    fn contains(&self, key: &CacheKey) -> bool;

    fn update_root(&mut self) -> Option<CacheEntryUpdater<'_>>;

    fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater<'_>>;

    fn size(&self) -> usize;

    fn is_empty(&self) -> bool;
}
