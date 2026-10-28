use crate::caching::entry::{CacheEntry, CacheEntryUpdater};
use crate::caching::helpers::{CacheKey, Index};
use crate::caching::Caching;
use crate::cover::Cover;
use rustc_hash::FxHashMap;

#[derive(Debug)]
struct TrieNode {
    index: usize,
    entry: CacheEntry,
    children: FxHashMap<usize, usize>,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            index: usize::MAX,
            children: FxHashMap::default(),
            entry: CacheEntry::default(),
        }
    }
}

impl TrieNode {
    pub fn new(item: usize) -> Self {
        Self {
            index: usize::MAX,
            children: FxHashMap::default(),
            entry: CacheEntry::new(item),
        }
    }
}

/// A [`Caching`] implementation as a trie of items stored in an arena.
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

    fn insert(&mut self, path: &[usize], _cover: &Cover) -> Index {
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

    fn find(&self, path: &[usize], _cover: &Cover) -> Option<&CacheEntry> {
        let mut current_index = self.root_index;
        for item in path {
            current_index = *self.get_node(current_index)?.children.get(item)?;
        }
        self.get_node(current_index).map(|node| &node.entry)
    }

    fn keyed_by_cover(&self) -> bool {
        false
    }

    fn node(&self, key: &CacheKey) -> Option<&CacheEntry> {
        self.get_node(key.0).map(|node| &node.entry)
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.node(key).is_some()
    }

    fn update_root(&mut self) -> Option<CacheEntryUpdater<'_>> {
        self.get_node_mut(self.root_index)
            .map(|node| CacheEntryUpdater::new(&mut node.entry))
    }

    fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater<'_>> {
        self.get_node_mut(key.0)
            .map(|node| CacheEntryUpdater::new(&mut node.entry))
    }

    fn size(&self) -> usize {
        self.arena.len()
    }

    fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }
}

impl Trie {
    /// An empty trie.
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
    use crate::bitsets::{BitCollection, Bitset, BitsetInit};
    use crate::caching::trie::{Trie, TrieNode};
    use crate::caching::Caching;
    use crate::cover::Cover;

    #[test]
    fn test_cache_init() {
        let mut cache = Trie::new();
        assert!(cache.is_empty());

        let root_data = TrieNode::default();
        cache.add_root(root_data);
        cache.add_node(0, TrieNode::default());
        println!("Cache 0 {:?}", cache.arena[0].children);

        assert!(!cache.is_empty());

        let mut itemset = Vec::new();
        itemset.push(0);
        itemset.push(1);
        itemset.push(3);

        let full = Bitset::new(BitsetInit::Full(4));
        let cover = Cover::new(vec![full.clone(); 2], vec![full], 4);
        assert!(cache.insert(&itemset, &cover).is_new());
        assert!(cache.find(&itemset, &cover).is_some());

        itemset.remove(1);
    }
}
