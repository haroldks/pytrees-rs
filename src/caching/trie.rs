use crate::caching::entry::{CacheEntry, CacheEntryUpdater};
use crate::caching::helpers::{CacheKey, Index};
use crate::caching::Caching;
use std::collections::HashMap;

#[derive(Debug)]
struct TrieNode {
    index: usize,
    entry: CacheEntry,
    children: HashMap<usize, usize>,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            index: usize::MAX,
            children: HashMap::new(),
            entry: CacheEntry::default(),
        }
    }
}

impl TrieNode {
    pub fn new(item: usize) -> Self {
        Self {
            index: usize::MAX,
            children: HashMap::new(),
            entry: CacheEntry::new(item),
        }
    }
}

#[derive(Default)]
pub struct Trie {
    arena: Vec<TrieNode>,
    root_index: usize,
}

impl Caching for Trie {
    fn init(&mut self) -> Index {
        if !self.arena.is_empty() {
            self.arena.clear();
        }
        let index = self.add_default_root();
        Index::new_at(index)
    }

    fn root_index(&mut self) -> Index {
        Index::new_at(self.get_root_index())
    }

    fn root(&self) -> Option<&CacheEntry> {
        self.get_node(self.get_root_index()).map(|node| &node.entry)
    }

    fn insert(&mut self, path: &[usize]) -> Index {
        let mut current_index = self.root_index;
        let mut is_new = false;
        for &item in path {
            if let Some(&child_index) = self
                .get_node(current_index)
                .and_then(|node| node.children.get(&item))
            {
                current_index = child_index;
            } else {
                is_new = true;
                current_index = self.create_child(current_index, item);
            }
        }

        let leaf_is_infinite = self
            .get_node(current_index)
            .map_or(true, |node| !node.entry.has_finite_leaf_error());

        if is_new || leaf_is_infinite {
            return Index::New(current_index);
        }
        Index::Existing(current_index)
    }

    fn node(&self, key: &CacheKey) -> Option<&CacheEntry> {
        match key {
            CacheKey::Index(index) => self.get_node(*index).map(|node| &node.entry),
            CacheKey::Path(path) => self.find(path),
        }
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.node(key).is_some()
    }

    fn update_root(&mut self) -> Option<CacheEntryUpdater> {
        self.get_node_mut(self.root_index)
            .map(|node| CacheEntryUpdater::new(&mut node.entry))
    }

    fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater> {
        match key {
            CacheKey::Index(index) => self
                .get_node_mut(*index)
                .map(|node| CacheEntryUpdater::new(&mut node.entry)),
            CacheKey::Path(path) => self.find_mut(path).map(CacheEntryUpdater::new),
        }
    }

    fn size(&self) -> usize {
        self.arena.len()
    }

    fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    fn print(&self) {
        println!("{:#?}", self.arena)
    }
}

impl Trie {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            root_index: 0,
        }
    }

    fn add_node(&mut self, parent_index: usize, mut node: TrieNode) -> usize {
        let new_index = self.arena.len();
        node.index = new_index;
        let item = node.entry.item();
        self.arena.push(node);
        if new_index != 0 {
            if let Some(parent) = self.arena.get_mut(parent_index) {
                parent.children.insert(item, new_index);
            }
        }

        new_index
    }

    fn find(&self, itemset: &[usize]) -> Option<&CacheEntry> {
        let mut current_index = self.root_index;
        for &item in itemset {
            let current_node = self.get_node(current_index)?;
            current_index = *current_node.children.get(&item)?;
        }
        self.get_node(current_index).map(|node| &node.entry)
    }

    fn find_mut(&mut self, itemset: &[usize]) -> Option<&mut CacheEntry> {
        let mut current_index = self.root_index;
        for &item in itemset {
            let current_node = self.get_node(current_index)?;
            current_index = *current_node.children.get(&item)?;
        }
        self.get_node_mut(current_index).map(|node| &mut node.entry)
    }

    fn add_root(&mut self, root: TrieNode) -> usize {
        self.root_index = self.add_node(0, root);
        self.root_index
    }

    fn add_default_root(&mut self) -> usize {
        self.root_index = self.add_root(TrieNode::default());
        self.root_index
    }

    #[inline]
    fn get_root_index(&self) -> usize {
        self.root_index
    }

    #[inline]
    fn get_node(&self, index: usize) -> Option<&TrieNode> {
        self.arena.get(index)
    }

    #[inline]
    fn get_node_mut(&mut self, index: usize) -> Option<&mut TrieNode> {
        self.arena.get_mut(index)
    }

    fn create_child(&mut self, parent: usize, item: usize) -> usize {
        let node = TrieNode::new(item);
        self.add_node(parent, node)
    }
}

#[cfg(test)]
mod trie_test {
    use crate::caching::helpers::{CacheKey, Index};
    use crate::caching::trie::{Trie, TrieNode};
    use crate::caching::Caching;

    #[test]
    fn test_cache_init() {
        let mut cache = Trie::new();
        assert_eq!(cache.is_empty(), true);

        let root_data = TrieNode::default();
        cache.add_root(root_data);
        cache.add_node(0, TrieNode::default());
        println!("Cache 0 {:?}", cache.arena[0].children);

        assert_eq!(cache.is_empty(), false);

        let mut itemset = Vec::new();
        itemset.push(0);
        itemset.push(1);
        itemset.push(3);

        let idx = cache.insert(&itemset);

        match idx {
            Index::NewUnknown => {}
            Index::New(pos) | Index::Existing(pos) => {}
        }

        itemset.remove(1);

        // let mut_infos = cache.contains(Index(idx().unwrap()));
        // println!("Should be none = {:#?}", mut_infos);
        //
        // itemset.insert(1, 1);
        // let infos = cache.node(Index(idx.get_position().unwrap()));
        //
        // println!("Should have 33.0 as ub: {:#?}", infos);
    }
}
