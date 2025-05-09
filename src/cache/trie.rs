use crate::cache::{CacheEntry, Caching};
use std::collections::BTreeSet;
use std::slice::Iter;

#[derive(Debug)]
struct TrieNode {
    index: usize,
    children: Vec<usize>,
    infos: CacheEntry,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            index: <usize>::MAX,
            children: vec![],
            infos: CacheEntry::default(),
        }
    }
}

impl TrieNode {
    pub fn new(item: usize) -> Self {
        Self {
            index: <usize>::MAX,
            children: vec![],
            infos: CacheEntry::new(item),
        }
    }
}

pub struct Trie {
    elements: Vec<TrieNode>,
}

impl Caching for Trie {
    fn init(&mut self) -> Option<usize> {
        let root = TrieNode::default();
        Some(self.add_root(root))
    }

    fn get_root_infos(&self) -> Option<&CacheEntry> {
        self.get_node(self.get_root_index()).map(|node| &node.infos)
    }

    fn set_root_infos(&mut self) -> Option<&mut CacheEntry> {
        self.get_node_mut(self.get_root_index())
            .map(|node| &mut node.infos)
    }

    // Check if there is a node inside the cache for the current itemset and return a mutable ref
    fn get(&mut self, itemset: &BTreeSet<usize>, index: Option<usize>) -> Option<&mut CacheEntry> {
        // If index is given and exists go for it
        if let Some(idx) = index {
            return self.get_node_mut(idx).map(|node| &mut node.infos);
        }

        // We moving using Itemset
        let mut index = self.get_root_index();
        for item in itemset.iter() {
            let mut children = self.children(index);
            if let Some(child) = children.find(|&&c| {
                self.get_node(c)
                    .map_or(false, |node| node.infos.item == *item)
            }) {
                index = *child
            } else {
                return None;
            }
        }
        self.get_node_mut(index).map(|node| &mut node.infos)
    }

    fn find(&self, itemset: &BTreeSet<usize>) -> Option<&CacheEntry> {
        let mut index = self.get_root_index();
        for item in itemset.iter() {
            let mut children = self.children(index);
            if let Some(child) = children.find(|&&c| {
                self.get_node(c)
                    .map_or(false, |node| node.infos.item == *item)
            }) {
                index = *child;
            } else {
                return None;
            }
        }
        self.get_node(index).map(|node| &node.infos)
    }

    // Insert Optimal node inside the cache
    fn insert(&mut self, itemset: &BTreeSet<usize>) -> (bool, Option<usize>) {
        let mut index = self.get_root_index();
        let mut is_new = false;

        for item in itemset.iter() {
            let mut children = self.children(index);
            if let Some(child) = children.find(|&&c| {
                self.get_node(c)
                    .map_or(false, |node| node.infos.item == *item)
            }) {
                index = *child;
            } else {
                is_new = true;
                index = self.create_cache_entry(index, *item);
            }
        }
        let leaf_is_infinite = self
            .get_node(index)
            .map_or(false, |node| node.infos.leaf_error.is_infinite());
        (leaf_is_infinite || is_new, Some(index))
    }

    fn size(&self) -> usize {
        self.elements.len()
    }

    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    fn print(&self) {
        println!("{:#?}", self.elements)
    }
}

// ! Add default implementation

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    pub fn new() -> Self {
        Self { elements: vec![] }
    }

    fn add_node(&mut self, parent: usize, mut node: TrieNode) -> usize {
        node.index = self.elements.len();
        self.elements.push(node);
        let position = self.elements.len() - 1;
        if position == 0 {
            return position;
        }
        self.add_child(parent, position);
        position
    }

    fn add_child(&mut self, parent: usize, index: usize) {
        self.elements[parent].children.push(index);
    }

    fn children(&self, index: usize) -> Iter<usize> {
        self.elements[index].children.iter()
    }

    fn add_root(&mut self, root: TrieNode) -> usize {
        self.add_node(0, root)
    }

    fn get_root_index(&self) -> usize {
        0
    }

    fn get_node(&self, index: usize) -> Option<&TrieNode> {
        self.elements.get(index)
    }

    fn get_node_mut(&mut self, index: usize) -> Option<&mut TrieNode> {
        self.elements.get_mut(index)
    }

    fn create_cache_entry(&mut self, parent: usize, item: usize) -> usize {
        let node = TrieNode::new(item);
        self.add_node(parent, node)
    }
}

#[cfg(test)]
mod trie_test {
    use crate::cache::trie::{Trie, TrieNode};
    use crate::cache::{CacheEntry, Caching};
    use std::collections::BTreeSet;

    #[test]
    fn test_cache_init() {
        let mut cache = Trie::new();
        assert_eq!(cache.is_empty(), true);

        let root_data = TrieNode::default();
        cache.add_root(root_data);

        assert_eq!(cache.is_empty(), false);

        let mut itemset = BTreeSet::new();
        itemset.insert(0);
        itemset.insert(1);
        itemset.insert(3);

        let mut infos = cache.insert(&itemset);
        println!("(is_new, index) : {:?}", infos);

        if let Some(index) = infos.1 {
            let mut node = cache.get(&itemset, Some(index));
            println!("Node infos = {:#?}", node);
            if let Some(ref mut inf) = node {
                inf.upper_bound = 33.0;
                inf.is_optimal = true;
            }
        }

        itemset.remove(&1);

        let mut_infos = cache.get(&itemset, None);
        println!("Should be none = {:#?}", mut_infos);

        itemset.insert(1);
        let infos = cache.get(&itemset, None);

        println!("Should have 33.0 as ub: {:#?}", infos);
    }
}
