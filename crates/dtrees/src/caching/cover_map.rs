use crate::caching::entry::{CacheEntry, CacheEntryUpdater};
use crate::caching::helpers::{CacheKey, Index};
use crate::caching::Caching;
use crate::cover::Cover;
use rustc_hash::FxHashMap;

/// The subproblems of one depth and one number of instances, by instances.
type Bucket = FxHashMap<Box<[u64]>, usize>;

/// A [`Caching`] implementation keyed by the depth and the instances of a
/// subproblem: paths reaching the same instances at the same depth share one
/// entry.
///
/// Entries live in an arena, found by depth, then number of instances, then
/// instances. Each entry keeps its instances, one word per 64 rows, so the
/// memory grows with the size of the dataset.
#[derive(Default)]
pub struct CoverMap {
    arena: Vec<CacheEntry>,
    map: Vec<Vec<Bucket>>,
    /// Words of the instances being inserted, reused between insertions.
    buffer: Vec<u64>,
    /// Number of instances of the dataset being fitted.
    num_samples: Option<usize>,
}

impl Caching for CoverMap {
    fn init(&mut self) -> Index {
        self.arena.clear();
        self.map.clear();
        self.num_samples = None;
        self.arena.push(CacheEntry::default());
        Index::new_at(0)
    }

    fn root_index(&mut self) -> Index {
        Index::new_at(0)
    }

    fn root(&self) -> Option<&CacheEntry> {
        self.arena.first()
    }

    fn insert(&mut self, path: &[usize], cover: &Cover) -> Index {
        let num_samples = *self.num_samples.get_or_insert(cover.num_samples);
        debug_assert_eq!(num_samples, cover.num_samples, "one dataset per fit");
        cover.write_words(&mut self.buffer);
        let bucket = bucket(&mut self.map, path.len(), cover.count());
        let index = match bucket.get(self.buffer.as_slice()) {
            Some(&index) => index,
            None => {
                let index = self.arena.len();
                bucket.insert(self.buffer.as_slice().into(), index);
                self.arena.push(CacheEntry::default());
                index
            }
        };
        if self.arena[index].has_finite_leaf_error() {
            Index::existing(index)
        } else {
            Index::new_at(index)
        }
    }

    fn find(&self, path: &[usize], cover: &Cover) -> Option<&CacheEntry> {
        if path.is_empty() {
            debug_assert_eq!(
                cover.count(),
                cover.num_samples,
                "the root has every instance"
            );
            return self.root();
        }
        let mut words = Vec::new();
        cover.write_words(&mut words);
        let index = self
            .map
            .get(path.len())?
            .get(cover.count())?
            .get(words.as_slice())?;
        self.arena.get(*index)
    }

    fn keyed_by_cover(&self) -> bool {
        true
    }

    fn node(&self, key: &CacheKey) -> Option<&CacheEntry> {
        self.arena.get(key.0)
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.node(key).is_some()
    }

    fn update_root(&mut self) -> Option<CacheEntryUpdater<'_>> {
        self.arena.first_mut().map(CacheEntryUpdater::new)
    }

    fn update_node(&mut self, key: &CacheKey) -> Option<CacheEntryUpdater<'_>> {
        self.arena.get_mut(key.0).map(CacheEntryUpdater::new)
    }

    fn size(&self) -> usize {
        self.arena.len()
    }

    fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }
}

/// The map of the subproblems at `depth` with `count` instances, created if
/// needed.
fn bucket(map: &mut Vec<Vec<Bucket>>, depth: usize, count: usize) -> &mut Bucket {
    if map.len() <= depth {
        map.resize_with(depth + 1, Vec::new);
    }
    let row = &mut map[depth];
    if row.len() <= count {
        row.resize_with(count + 1, FxHashMap::default);
    }
    &mut row[count]
}

#[cfg(test)]
mod cover_map_test {
    use crate::bitsets::{BitCollection, Bitset, BitsetInit};
    use crate::caching::cover_map::CoverMap;
    use crate::caching::Caching;
    use crate::cover::Cover;
    use crate::reader::data_reader::DataReader;
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::hash::BuildHasher;
    use std::path::Path;

    #[test]
    fn paths_reaching_the_same_instances_share_an_entry() {
        // Features 0 and 1 are both 1 on instances 0 and 1, so items 1 and 3
        // reach the same instances.
        let mut feature = Bitset::new(BitsetInit::Empty(4));
        feature.set(0);
        feature.set(1);
        let mut cover = Cover::new(
            vec![feature.clone(), feature],
            vec![Bitset::new(BitsetInit::Full(4))],
            4,
        );
        let mut cache = CoverMap::default();
        cache.init();

        cover.branch_on(1);
        let first = cache.insert(&[1], &cover);
        cover.backtrack();
        cover.branch_on(3);
        let second = cache.insert(&[3], &cover);
        assert!(first.is_new() && second.is_new());
        assert_eq!(first.position(), second.position());
        assert!(cache.find(&[1], &cover).is_some());

        // The same instances one level deeper are another subproblem.
        cover.branch_on(1);
        let deeper = cache.insert(&[1, 3], &cover);
        assert_ne!(deeper.position(), first.position());
        assert_eq!(cache.size(), 3);
    }

    /// Adds the key of every subproblem up to `depth` below the current one,
    /// testing features after `first_item`, to its `(depth, count)` bucket.
    fn collect_keys(
        cover: &mut Cover,
        first_item: usize,
        depth: usize,
        keys: &mut FxHashMap<(usize, usize), FxHashSet<Box<[u64]>>>,
    ) {
        let mut words = Vec::new();
        cover.write_words(&mut words);
        keys.entry((depth, cover.count()))
            .or_default()
            .insert(words.into());
        if depth == 0 {
            return;
        }
        for item in first_item..2 * cover.num_attributes {
            if cover.count_if_branch_on(item) > 0 {
                cover.branch_on(item);
                collect_keys(cover, (item / 2 + 1) * 2, depth - 1, keys);
                cover.backtrack();
            }
        }
    }

    #[test]
    fn distinct_subproblems_have_distinct_hashes() {
        let mut cover = DataReader::default()
            .read_file(Path::new("test_data/german-credit.txt"))
            .expect("the dataset reads");
        let mut keys = FxHashMap::default();
        collect_keys(&mut cover, 0, 3, &mut keys);

        let hasher = *FxHashMap::<Box<[u64]>, usize>::default().hasher();
        let mut total = 0;
        for bucket in keys.values() {
            let hashes: FxHashSet<u64> = bucket.iter().map(|key| hasher.hash_one(key)).collect();
            assert_eq!(hashes.len(), bucket.len());
            total += bucket.len();
        }
        assert_eq!(total, 582_030);
    }
}
