//! The instances reaching the current node of the search.

use crate::bitsets::Bitset;
use crate::cover::reversible_cover::{ShallowBitset, SparseBitset};
use crate::globals::{attribute, item_type};

pub mod reversible_cover;
pub mod similarities;

/// A binary dataset and the set of instances reaching the current node.
///
/// Features and classes are stored as bitsets over the instances. Branching
/// on an item intersects the current set with a feature's bitset (or its
/// complement), and backtracking restores the previous set in constant time
/// thanks to a reversible sparse bitset.
///
/// An item encodes a feature and a branch: `2 * feature` is the branch where
/// the feature is 0 (left) and `2 * feature + 1` the branch where it is 1
/// (right). See [`crate::globals`].
pub struct Cover {
    /// Number of binary features.
    pub num_attributes: usize,
    /// Number of classes.
    pub num_labels: usize,
    /// Number of instances in the dataset.
    pub num_samples: usize,
    attributes: Vec<Bitset>,
    labels: Vec<Bitset>,
    cover: SparseBitset,
    branch: Vec<usize>,
}

impl Cover {
    /// A cover over all instances, from one bitset per feature (instances
    /// where it is 1) and one per class.
    pub fn new(attributes: Vec<Bitset>, labels: Vec<Bitset>, num_samples: usize) -> Self {
        Self {
            num_attributes: attributes.len(),
            num_labels: labels.len(),
            num_samples,
            attributes,
            labels,
            cover: SparseBitset::new(num_samples),
            branch: vec![],
        }
    }

    /// Number of instances in the current node.
    pub fn count(&self) -> usize {
        self.cover.count()
    }

    /// Number of instances of each class in the current node.
    pub fn labels_count(&self) -> Vec<usize> {
        self.cover.count_intersect_with_many(&self.labels)
    }

    /// [`Self::labels_count`] writing into `buffer`.
    pub fn labels_count_with_buffer(&self, buffer: &mut Vec<usize>) {
        buffer.clear();
        buffer.extend_from_slice(&self.cover.count_intersect_with_many(&self.labels));
    }

    /// Moves to the child reached by `item` and returns its number of
    /// instances.
    pub fn branch_on(&mut self, item: usize) -> usize {
        self.branch.push(item);
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .intersect_with(&self.attributes[attribute], invert)
    }

    /// Number of instances the child reached by `item` would have.
    pub fn count_if_branch_on(&self, item: usize) -> usize {
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .count_intersect_with(&self.attributes[attribute], invert)
    }

    /// Returns to the parent of the current node.
    ///
    /// # Panics
    /// At the root.
    pub fn backtrack(&mut self) {
        assert_ne!(self.branch.len(), 0, "No backtrack when at root");
        self.branch.pop();
        self.cover.restore();
    }

    /// Ids of the instances in the current node.
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.cover.to_vec()
    }

    /// A plain copy of the current set of instances.
    pub fn shallow_cover(&self) -> ShallowBitset {
        let cover = &self.cover;
        cover.into()
    }

    /// The reversible set of instances.
    pub fn sparse(&self) -> &SparseBitset {
        &self.cover
    }

    /// Items branched on from the root to the current node.
    pub fn path(&self) -> &[usize] {
        &self.branch
    }
}
