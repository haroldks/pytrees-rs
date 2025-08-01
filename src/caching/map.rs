// use std::collections::HashMap;
// use crate::caching::entry::{CacheEntry, CacheEntryUpdater};
// use crate::caching::key::Key;
// use crate::caching::{Caching, CacheKey, Index};
//
// /// A HashMap-based cache implementation that uses the custom Key struct
// /// for efficient lookups and insertions with precomputed hashing
// pub struct TransactionsMap {
//     inner: HashMap<Key, CacheEntry>,
// }
//
//
//
// impl Caching for TransactionsMap {
//     fn init(&mut self) -> Index {
//         if !self.inner.is_empty() {
//             self.inner.clear();
//         }
//         self.inner.insert(Key::new(&[], 0), CacheEntry::default());
//         Index::NewUnknown
//     }
//
//     fn root(&self) -> Option<&CacheEntry> {
//         self.inner.get(&Key::new(&[], 0))
//     }
//
//     fn insert(&mut self, key_path: &[usize]) -> Index {
//         // Use depth 0 for entries added through the Caching trait
//         let key = Key::new(key_path, 0);
//         let is_new = !self.inner.contains_key(&key);
//
//         if is_new {
//             let entry = CacheEntry::new(0); // Default item value
//             self.inner.insert(key, entry);
//             Index::Existing(key.hash)
//         }
//     }
//
//     fn node(&self, key: CacheKey) -> Option<&CacheEntry> {
//         match key {
//             CacheKey::Index(idx) => {
//                 // For index lookups, we need to scan the HashMap
//                 // This is inefficient but necessary for the Caching trait
//                 for (k, v) in &self.inner {
//                     if k.hash == idx {
//                         return Some(v);
//                     }
//                 }
//                 None
//             },
//             CacheKey::Path(path) => {
//                 // For path lookups, use our efficient Key implementation
//                 let key = Key::new(&path, 0);
//                 self.inner.get(&key)
//             }
//         }
//     }
//
//     fn contains(&self, key: CacheKey) -> bool {
//         match key {
//             CacheKey::Index(idx) => {
//                 for k in self.inner.keys() {
//                     if k.hash == idx {
//                         return true;
//                     }
//                 }
//                 false
//             },
//             CacheKey::Path(path) => {
//                 let key = Key::new(&path, 0);
//                 self.inner.contains_key(&key)
//             }
//         }
//     }
//
//     fn update_root(&mut self) -> Option<CacheEntryUpdater> {
//         if let Some(root) = self.root_entry.as_mut() {
//             Some(CacheEntryUpdater::new(root))
//         } else {
//             None
//         }
//     }
//
//     fn update_node(&mut self, key: CacheKey) -> Option<CacheEntryUpdater> {
//         match key {
//             CacheKey::Index(idx) => {
//                 // Find by hash
//                 for (k, v) in &mut self.inner {
//                     if k.hash == idx {
//                         return Some(CacheEntryUpdater::new(v));
//                     }
//                 }
//                 None
//             },
//             CacheKey::Path(path) => {
//                 let key = Key::new(&path, 0);
//                 self.inner.get_mut(&key).map(CacheEntryUpdater::new)
//             }
//         }
//     }
//
//     fn size(&self) -> usize {
//         self.inner.len() + if self.root_entry.is_some() { 1 } else { 0 }
//     }
//
//     fn is_empty(&self) -> bool {
//         self.inner.is_empty() && self.root_entry.is_none()
//     }
//
//     fn print(&self) {
//         println!("TransactionsMap contents:");
//         println!("Root entry: {:?}", self.root_entry);
//         println!("Number of entries: {}", self.inner.len());
//         for (k, v) in &self.inner {
//             println!("Key hash: {}, depth: {}, Entry: {:?}", k.hash, k.depth, v);
//         }
//     }
// }
//
//
//
//
//
// impl TransactionsMap {
//     /// Create a new empty TransactionsMap
//     pub fn new() -> Self {
//         Self {
//             inner: HashMap::new(),
//             root_entry: None,
//         }
//     }
//
//     /// Create with a specified capacity
//     pub fn with_capacity(capacity: usize) -> Self {
//         Self {
//             inner: HashMap::with_capacity(capacity),
//             root_entry: None,
//         }
//     }
//
//     /// Get a cache entry for the given path and depth
//     pub fn get(&self, path: &[usize], depth: usize) -> Option<&CacheEntry> {
//         let key = Key::new(path, depth);
//         self.inner.get(&key)
//     }
//
//     /// Get a mutable cache entry for the given path and depth
//     pub fn get_mut(&mut self, path: &[usize], depth: usize) -> Option<&mut CacheEntry> {
//         let key = Key::new(path, depth);
//         self.inner.get_mut(&key)
//     }
//
//     /// Insert a new entry for the given path and depth
//     /// Returns a reference to the inserted entry
//     pub fn insert(&mut self, path: &[usize], depth: usize, entry: CacheEntry) -> &mut CacheEntry {
//         let key = Key::new(path, depth);
//         self.inner.insert(key, entry);
//         // Safe to unwrap as we just inserted it
//         self.inner.get_mut(&Key::new(path, depth)).unwrap()
//     }
//
//     /// Get or create an entry for the given path and depth
//     /// If the entry doesn't exist, it creates one with the specified item
//     pub fn get_or_create(&mut self, path: &[usize], depth: usize, item: usize) -> &mut CacheEntry {
//         let key = Key::new(path, depth);
//
//         if !self.inner.contains_key(&key) {
//             let entry = CacheEntry::new(item);
//             self.inner.insert(key, entry);
//         }
//
//         // Safe to unwrap as it now exists
//         self.inner.get_mut(&Key::new(path, depth)).unwrap()
//     }
//
//     /// Insert or update a cache entry
//     /// Returns (is_new, &mut entry) where is_new indicates if the entry was newly created
//     pub fn insert_or_get(&mut self, path: &[usize], depth: usize, item: usize) -> (bool, &mut CacheEntry) {
//         let key = Key::new(path, depth);
//         let is_new = !self.inner.contains_key(&key);
//
//         if is_new {
//             let entry = CacheEntry::new(item);
//             self.inner.insert(key, entry);
//         }
//
//         (is_new, self.inner.get_mut(&Key::new(path, depth)).unwrap())
//     }
//
//     /// Check if an entry exists for the given path and depth
//     pub fn contains_path(&self, path: &[usize], depth: usize) -> bool {
//         let key = Key::new(path, depth);
//         self.inner.contains_key(&key)
//     }
//
//     /// Remove an entry for the given path and depth
//     /// Returns the removed entry if it existed
//     pub fn remove(&mut self, path: &[usize], depth: usize) -> Option<CacheEntry> {
//         let key = Key::new(path, depth);
//         self.inner.remove(&key)
//     }
//
//     /// Get the number of entries in the cache
//     pub fn len(&self) -> usize {
//         self.inner.len()
//     }
//
//     /// Check if the cache is empty
//     pub fn is_empty(&self) -> bool {
//         self.inner.is_empty()
//     }
//
//     /// Clear all entries from the cache
//     pub fn clear(&mut self) {
//         self.inner.clear();
//         self.root_entry = None;
//     }
//
//     /// Update an existing entry with a closure
//     pub fn update<F>(&mut self, path: &[usize], depth: usize, f: F) -> Option<&mut CacheEntry>
//     where
//         F: FnOnce(&mut CacheEntry),
//     {
//         let key = Key::new(path, depth);
//         if let Some(entry) = self.inner.get_mut(&key) {
//             f(entry);
//             Some(entry)
//         } else {
//             None
//         }
//     }
// }
