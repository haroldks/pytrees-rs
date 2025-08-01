use crate::bitsets::Bitset;
use crate::cover::reversible_cover::{ShallowBitset, SparseBitset};
use crate::globals::{attribute, item_type};

pub mod reversible_cover;
pub mod similarities;

pub struct Cover {
    pub num_attributes: usize,
    pub num_labels: usize,
    attributes: Vec<Bitset>,
    labels: Vec<Bitset>,
    cover: SparseBitset,
    branch: Vec<usize>,
}

impl Cover {
    pub fn new(attributes: Vec<Bitset>, labels: Vec<Bitset>, size: usize) -> Self {
        Self {
            num_attributes: attributes.len(),
            num_labels: labels.len(),
            attributes,
            labels,
            cover: SparseBitset::new(size),
            branch: vec![],
        }
    }

    pub fn count(&self) -> usize {
        self.cover.count()
    }

    pub fn labels_count(&self) -> Vec<usize> {
        self.cover.count_intersect_with_many(&self.labels)
    }

    // TODO : Implementation to use when using a placeholder for hot activities
    pub fn labels_count_with_buffer(&self, buffer: &mut Vec<usize>) {
        buffer.clear();
        buffer.extend_from_slice(&self.cover.count_intersect_with_many(&self.labels));
    }

    pub fn branch_on(&mut self, item: usize) -> usize {
        self.branch.push(item);
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .intersect_with(&self.attributes[attribute], invert)
    }

    pub fn count_if_branch_on(&self, item: usize) -> usize {
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .count_intersect_with(&self.attributes[attribute], invert)
    }

    pub fn backtrack(&mut self) {
        assert_ne!(self.branch.len(), 0, "No backtrack when at root");
        self.branch.pop();
        self.cover.restore();
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.cover.to_vec()
    }

    pub fn shallow_cover(&self) -> ShallowBitset {
        let cover = &self.cover;
        cover.into()
    }

    pub fn sparse(&self) -> &SparseBitset {
        &self.cover
    }

    pub fn path(&self) -> &[usize] {
        &self.branch
    }
}
