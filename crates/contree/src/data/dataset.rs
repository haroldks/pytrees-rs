use crate::data::{DataPoint, Feature};
use std::ops::{Index, IndexMut};

#[derive(Debug, Default)]
pub struct Dataset {
    features: Vec<Feature>,
    num_labels: usize,
    // The search compares unique-value indices within sorted columns. Skipping
    // either step does not fail loudly -- it makes every fit return a single
    // leaf -- so the dataset remembers whether it has been prepared and `fit`
    // refuses an unprepared one.
    sorted: bool,
    indexed: bool,
}

impl Dataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether both preparation steps have run. See [`SearchError::UnpreparedDataset`].
    ///
    /// [`SearchError::UnpreparedDataset`]: crate::common::SearchError::UnpreparedDataset
    pub fn is_prepared(&self) -> bool {
        self.sorted && self.indexed
    }

    /// Number of instances. Zero for an empty dataset.
    pub fn count(&self) -> usize {
        self.features.first().map_or(0, Feature::len)
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
    pub fn insert(&mut self, data_point: DataPoint, feature_index: usize) {
        if data_point.tid() == 0 {
            self.features.push(Feature::new())
        }
        self.features[feature_index].insert(data_point)
    }

    pub fn set_num_label(&mut self, value: usize) {
        self.num_labels = value;
    }

    pub fn sort_features(&mut self) {
        for column in self.features.iter_mut() {
            column.sort();
        }
        self.sorted = true;
    }

    /// Assigns each observation the index of its value among the column's
    /// distinct values. Independent of the order the column happens to be in,
    /// so it may run before or after `sort_features`. The search compares these indices, so this must run
    /// before any fit; skipping it makes every fit return a single leaf.
    pub fn compute_unique_feature_values(&mut self) {
        let size = self.count();
        let mut idx = vec![0; size];
        for column in &mut self.features {
            (0..size).for_each(|i| idx[i] = i);
            idx.sort_unstable_by(|&idx1, &idx2| column[idx1].cmp(&column[idx2]));

            let mut cur_unique = 0;
            let mut prev: Option<f64> = None;
            for &index in &idx {
                let el = &mut column[index];
                if let Some(prev_val) = prev {
                    if (el.value - prev_val).abs() >= f64::EPSILON {
                        // TODO : The use a larger epsilon and not the absolute value
                        cur_unique += 1;
                    }
                }

                el.unique_value_idx = cur_unique;
                prev = Some(el.value);
            }
        }
        self.indexed = true;
    }
}

/// Why a dataset could not be built from memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatasetError {
    Empty,
    /// `values.len()` is not `labels.len() * n_features`.
    ShapeMismatch {
        values: usize,
        rows: usize,
        n_features: usize,
    },
    /// A feature value that is NaN or infinite.
    NonFiniteValue {
        row: usize,
        feature: usize,
    },
    /// A label outside what can be a dense `0..k` encoding.
    InvalidLabel {
        row: usize,
        label: usize,
        rows: usize,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::Empty => write!(f, "the dataset has no rows"),
            DatasetError::ShapeMismatch {
                values,
                rows,
                n_features,
            } => write!(
                f,
                "{values} values do not form {rows} rows of {n_features} features"
            ),
            DatasetError::NonFiniteValue { row, feature } => {
                write!(f, "row {row}, feature {feature}: values must be finite")
            }
            DatasetError::InvalidLabel { row, label, rows } => write!(
                f,
                "row {row}: label {label} in a dataset of {rows} rows; labels must be a dense \
                 encoding starting at 0"
            ),
        }
    }
}

impl std::error::Error for DatasetError {}

impl Dataset {
    /// Builds a dataset from a row-major value buffer and its labels.
    ///
    /// This is the in-memory counterpart of
    /// [`DataReader::read_file`](crate::reader::data_reader::DataReader::read_file),
    /// and takes numpy's C-order `(n_rows, n_features)` layout directly.
    ///
    /// It also performs the two steps that were previously undocumented
    /// obligations on the caller -- sorting each column and computing the
    /// unique-value indices. Skipping the second one does not fail loudly: it
    /// makes every fit return a single leaf.
    pub fn from_rows(
        values: &[f64],
        labels: &[usize],
        n_features: usize,
    ) -> Result<Self, DatasetError> {
        let rows = labels.len();
        if rows == 0 || n_features == 0 {
            return Err(DatasetError::Empty);
        }
        if values.len() != rows * n_features {
            return Err(DatasetError::ShapeMismatch {
                values: values.len(),
                rows,
                n_features,
            });
        }

        let mut max_label = 0;
        for (row, &label) in labels.iter().enumerate() {
            if label >= rows {
                return Err(DatasetError::InvalidLabel { row, label, rows });
            }
            max_label = max_label.max(label);
        }

        let mut dataset = Dataset::new();
        for (row, (chunk, &label)) in values.chunks_exact(n_features).zip(labels).enumerate() {
            for (feature, &value) in chunk.iter().enumerate() {
                if !value.is_finite() {
                    return Err(DatasetError::NonFiniteValue { row, feature });
                }
                dataset.insert(DataPoint::new(row, value, label as f64), feature);
            }
        }

        dataset.set_num_label(max_label + 1);
        dataset.sort_features();
        dataset.compute_unique_feature_values();
        Ok(dataset)
    }
}

impl Index<usize> for Dataset {
    type Output = Feature;

    fn index(&self, index: usize) -> &Self::Output {
        &self.features[index]
    }
}

impl IndexMut<usize> for Dataset {
    fn index_mut(&mut self, index: usize) -> &mut Feature {
        &mut self.features[index]
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a Feature;
    type IntoIter = std::slice::Iter<'a, Feature>;

    fn into_iter(self) -> Self::IntoIter {
        self.features.iter()
    }
}
