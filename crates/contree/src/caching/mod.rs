//! The cache of solved subproblems.
//!
//! A subproblem is "the best tree of depth `d` over this set of instances", so
//! it is keyed by `(remaining depth, the set)`. The set is a [`Bitset`] over
//! instance ids, and its population count is pulled out as a second array
//! index so a lookup never hashes a bitset against entries that could not
//! possibly match.
//!
//! Entries live in one arena and refer to each other by index, which is what
//! lets the solution tree be rebuilt afterwards by walking from the root.

use crate::bitsets::Bitset;
use crate::tree::Tree;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// One solved (or partly solved) subproblem.
// TODO: use private fields and updater
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct Entry {
    /// The feature this node tests. `usize::MAX` means "leaf"; the tree spells
    /// that `feature: None`, and `shared::build_solution_tree` is where the
    /// two conventions meet.
    pub feature: usize,
    /// The threshold, or `INFINITY` for a leaf.
    pub split: f64,
    /// Misclassifications of the best subtree found so far -- an *upper* bound
    /// on this subproblem's optimum, and exactly equal to it only when
    /// `is_optimal`.
    pub error: usize,
    /// The majority class, used when this entry is a leaf.
    pub label: usize,

    /// A proven *lower* bound on this subproblem's optimum.
    ///
    /// Equal to `error` when the search exhausted the space, and equal to the
    /// bound it was cut off at otherwise -- all that was established then is
    /// "nothing in here beats the bound". Anything reasoning about what a
    /// subtree *cannot* do must read this and never `error`.
    pub lower_bound: usize,
    /// Whether `error` describes a tree that was actually found, rather than a
    /// subproblem the bound cut short before one was.
    pub is_valid: bool,

    pub is_leaf: bool,
    /// Whether `error` is proved optimal for this subproblem rather than just
    /// the best seen so far. Only an entry with this set may be reused from
    /// the cache in place of searching again.
    pub is_optimal: bool,

    /// Distance from the root, not remaining depth.
    pub depth: usize,
    /// Arena indices of the two children. `0` means "no child": index 0 is the
    /// root, which is nobody's child.
    pub left: usize,
    pub right: usize,
    /// Set when the depth-2 solver produced a whole subtree for this node
    /// instead of cache entries; indexes `Cache::trees`.
    pub tree_idx: Option<usize>,
    /// What the anytime search's last search of this subproblem ran under.
    /// Only `ConTreeLds` writes it; see [`SearchedUnder`].
    pub searched_under: Option<SearchedUnder>,
}

/// The budget and bound a subproblem was last searched under, and whether the
/// budget cut that search short.
///
/// A result obtained under a budget is at least as good as anything a smaller
/// budget could find, so a revisit under a budget and bound no larger than
/// these can reuse it. And a search the budget did *not* cut short proved its
/// lower bound, which settles any later revisit whose bound is no higher.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct SearchedUnder {
    /// Discrepancy still available below this node.
    pub discrepancy: usize,
    /// Best-ranked splits each node could try; `usize::MAX` when the split
    /// budget did not apply.
    pub split_budget: usize,
    /// The parent's budget: no tree at or above it was of any use.
    pub upper_bound: usize,
    /// The discrepancy or split budget cut the search short.
    pub truncated: bool,
}

impl SearchedUnder {
    /// Whether this search's result stands in for a search under `other`.
    ///
    /// It must have had at least `other`'s discrepancy and split budget. The
    /// bound matters only when this search came back empty-handed
    /// (`found_a_tree` false): pruning against a bound discards only what
    /// could not beat it, so a tree found under one bound is the best the
    /// budget allows under any looser bound too.
    pub fn covers(&self, other: &SearchedUnder, found_a_tree: bool) -> bool {
        other.discrepancy <= self.discrepancy
            && other.split_budget <= self.split_budget
            && (found_a_tree || other.upper_bound <= self.upper_bound)
    }
}

impl Entry {
    /// Records what the search actually proved about this subproblem.
    ///
    /// Ported from upstream ConTree's `Tree::finalize_lower_bound` (commit
    /// 61ebd49, "bug fixes upper bound pruning"). When the best tree found is
    /// worse than the bound the search ran under, nothing was established
    /// except that the bound cannot be beaten here: the bound becomes the
    /// lower bound, and the entry stops counting as a usable solution.
    pub fn finalize_lower_bound(&mut self, upper_bound: usize) {
        if self.error > upper_bound {
            self.lower_bound = upper_bound;
            self.is_valid = false;
        } else {
            self.lower_bound = self.error;
            self.is_valid = true;
        }
    }

    /// Marks the entry as an exact, reusable answer.
    pub fn mark_exact(&mut self) {
        self.lower_bound = self.error;
        self.is_valid = true;
        self.is_optimal = true;
    }
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            feature: usize::MAX,
            split: f64::INFINITY,
            error: usize::MAX,
            label: usize::MAX,
            lower_bound: 0,
            is_valid: false,
            is_leaf: false,
            is_optimal: false,
            depth: 0,
            left: 0,
            right: 0,
            tree_idx: None,
            searched_under: None,
        }
    }
}
#[derive(Debug)]
pub struct Cache {
    arena: Vec<Entry>,
    trees: Vec<Tree>,
    map: Vec<Vec<FxHashMap<Bitset, usize>>>,
    root_index: usize,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Cache {
    /// Subproblems are keyed by `(depth, population count, bitset)`. The first
    /// two are array indices, so lookups never hash a whole bitset against
    /// entries that could not possibly match.
    pub fn new(depth: usize, _num_samples: usize) -> Self {
        // The rows start empty and grow to the largest count each depth
        // actually sees. Allocating `depth * num_samples` maps up front is
        // 10^6 empty hash maps -- tens of megabytes and the time to construct
        // them -- before the search has looked at a single split.
        Self {
            arena: Vec::new(),
            trees: vec![],
            map: vec![Vec::new(); depth + 1],
            root_index: 0,
        }
    }

    /// The bucket for one `(depth, count)` pair, grown on demand.
    fn bucket(&mut self, depth: usize, count: usize) -> &mut FxHashMap<Bitset, usize> {
        let row = &mut self.map[depth];
        if row.len() <= count {
            row.resize_with(count + 1, FxHashMap::default);
        }
        &mut row[count]
    }

    pub fn root(&self) -> Option<&Entry> {
        self.arena.get(self.root_index)
    }

    pub fn root_index(&self) -> usize {
        self.root_index
    }

    pub fn init(&mut self) -> usize {
        debug_assert!(self.arena.is_empty(), "Cache must me empty to init");
        self.arena.push(Entry::default());
        self.root_index
    }

    /// Looks the subproblem up, inserting a fresh entry if it is new.
    ///
    /// Returns `(is_new, index into the arena)`.
    pub fn insert(&mut self, bitset: &Bitset, depth: usize) -> (bool, usize) {
        #[cfg(feature = "profiling")]
        coz::scope!("insert in cache");
        let count = bitset.saved_count();

        // One hash on a hit, which is the case that dominates; the old code
        // did `contains_key` and then `get`, hashing the bitset twice.
        if let Some(&index) = self.bucket(depth, count).get(bitset) {
            return (false, index);
        }

        let index = self.arena.len();
        self.arena.push(Entry::default());
        self.bucket(depth, count).insert(bitset.clone(), index);
        (true, index)
    }

    pub fn get(&self, index: usize) -> Option<&Entry> {
        self.arena.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Entry> {
        self.arena.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn insert_tree(&mut self, tree: Tree) -> usize {
        let len = self.trees.len();
        self.trees.push(tree);
        len
    }

    pub fn get_tree(&self, tree_idx: usize) -> Option<&Tree> {
        self.trees.get(tree_idx)
    }

    pub fn get_children(&self, index: usize) -> [usize; 2] {
        assert!(index < self.arena.len());
        [self.arena[index].left, self.arena[index].right]
    }
}
