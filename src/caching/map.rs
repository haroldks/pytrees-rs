use crate::bitsets::{BitCollection, Bitset};
use crate::caching::entry::CacheEntryUpdater;
use crate::caching::{CacheEntry, CacheKey, Index};
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub struct MapHash {
    depth: usize,
    samples: usize,
    arena: Vec<CacheEntry>,
    map: Vec<Vec<FxHashMap<Bitset, usize>>>,
    root_index: usize,
}

impl Default for MapHash {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl MapHash {
    pub fn new(depth: usize, samples: usize) -> Self {
        Self {
            depth,
            samples,
            arena: Vec::new(),
            map: vec![vec![FxHashMap::default(); samples + 1]; depth + 1],
            root_index: 0,
        }
    }

    pub fn root_index(&self) -> Index {
        Index::new_at(self.root_index)
    }

    pub fn root(&self) -> Option<&CacheEntry> {
        self.arena.get(self.root_index)
    }

    pub fn insert(&mut self, bitset: &Bitset, depth: usize) -> Index {
        //println!("Depth {:?}", depth);
        let cache = &mut self.map[depth][bitset.count()];
        if cache.contains_key(&bitset) {
            Index::existing(*cache.get(&bitset).unwrap())
        } else {
            let index = self.arena.len();
            self.arena.push(CacheEntry::default());
            cache.insert(bitset.clone(), index);
            Index::new_at(index)
        }
    }

    pub fn node(&self, key: &CacheKey) -> Option<&CacheEntry> {
        match key {
            CacheKey::Index(index) => self.arena.get(*index),
            _ => panic!("Should not come here."),
        }
    }

    pub fn find(&self, key: &Bitset, depth: usize) -> Option<&CacheEntry> {
        let cache = &self.map[depth][key.count()];
        let index = *cache.get(key).unwrap();
        self.arena.get(index)
    }

    pub fn update_root(&mut self) -> Option<CacheEntryUpdater> {
        self.arena
            .get_mut(self.root_index)
            .map(|node| CacheEntryUpdater::new(node))
    }

    pub fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater> {
        match key {
            CacheKey::Index(index) => self
                .arena
                .get_mut(*index)
                .map(|node| CacheEntryUpdater::new(node)),

            _ => panic!("Should not come here."),
        }
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.node(key).is_some()
    }

    pub fn init(&mut self) -> usize {
        debug_assert!(self.arena.len() == 0, "Cache must me empty to init");
        self.arena.push(CacheEntry::default());
        self.root_index
    }

    pub(crate) fn size(&self) -> usize {
        self.arena.len()
    }

    fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub(crate) fn print(&self) {
        println!("{:#?}", self.arena)
    }
}
