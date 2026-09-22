use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

mod dataset;
pub mod view;

pub use dataset::{Dataset, DatasetError};

/// One observation of one feature: the instance id, its value and its label.
#[derive(Copy, Clone, Debug)]
pub struct DataPoint {
    tid: usize,
    value: f64,
    unique_value_idx: usize,
    label: f64,
}

impl DataPoint {
    /// An observation of instance `tid`, not yet indexed.
    pub fn new(tid: usize, value: f64, label: f64) -> Self {
        Self {
            tid,
            value,
            unique_value_idx: usize::MAX,
            label,
        }
    }

    pub fn tid(&self) -> usize {
        self.tid
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn label(&self) -> f64 {
        self.label
    }

    /// Sets the index of the value among the column's distinct values.
    pub fn set_unique_value_id(&mut self, value: usize) {
        self.unique_value_idx = value
    }

    /// The index of the value among the column's distinct values.
    pub fn unique_value_id(&self) -> usize {
        self.unique_value_idx
    }
}

impl PartialEq for DataPoint {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for DataPoint {}

impl PartialOrd for DataPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DataPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.total_cmp(&other.value)
    }
}

/// One column of the dataset: every observation of a single feature, kept
/// sorted by value so that split candidates are consecutive positions.
#[derive(Debug, Default)]
pub struct Feature {
    elements: Vec<DataPoint>,
}

impl Feature {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sort(&mut self) {
        self.elements.sort()
    }

    pub fn insert(&mut self, data_point: DataPoint) {
        self.elements.push(data_point);
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl Index<usize> for Feature {
    type Output = DataPoint;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl IndexMut<usize> for Feature {
    fn index_mut(&mut self, index: usize) -> &mut DataPoint {
        &mut self.elements[index]
    }
}
