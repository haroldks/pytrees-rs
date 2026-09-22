use crate::bitsets::{BitCollection, Bitset, BitsetInit};
use crate::data::{Dataset, Feature};

/// Per-feature Gini scores used to order features and split candidates when
/// the heuristic ordering is enabled.
#[derive(Clone, Debug)]
pub struct HeuristicValues {
    /// For each feature, indices into its possible splits, best Gini first.
    gini_per_split: Vec<Vec<usize>>,
    /// `(best Gini, feature index)` for each feature; sorted best first by
    /// [`Self::sort_by_gini`].
    best_gini_per_feature: Vec<(f64, usize)>,
}

impl HeuristicValues {
    /// Neutral scores (Gini 1.0) for `num_features` features, in index order.
    pub fn new(num_features: usize) -> Self {
        Self {
            gini_per_split: vec![Vec::new(); num_features],
            best_gini_per_feature: (0..num_features).map(|i| (1.0, i)).collect(),
        }
    }

    /// Stores the split order and best Gini of `feature`. `split_indices` must
    /// already be sorted best first.
    pub fn set_feature_ginis(&mut self, feature: usize, split_indices: Vec<usize>, best_gini: f64) {
        if feature < self.gini_per_split.len() {
            self.gini_per_split[feature] = split_indices;
            self.best_gini_per_feature[feature] = (best_gini, feature);
        }
    }

    /// Sorts features by their best Gini, lowest first.
    pub fn sort_by_gini(&mut self) {
        self.best_gini_per_feature
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
}

/// A subset of the dataset: the instances that reach one node of the tree.
///
/// Each feature column is kept sorted by value, so candidate thresholds are
/// the positions where the value changes. Splitting a view partitions every
/// column in one linear pass, preserving the order.
pub struct DataView<'a> {
    /// The full dataset the view refers into.
    pub dataset: &'a Dataset,
    /// Number of instances in the full dataset.
    pub total_instances: usize,
    /// For each feature, positions into `dataset[f]` of the instances in the
    /// view, in increasing value order.
    pub feature_columns: Vec<Vec<usize>>,
    /// For each feature, positions in `feature_columns[f]` where the value
    /// changes: the candidate thresholds.
    pub possible_split_indices: Vec<Vec<usize>>,
    /// Number of instances of each class in the view.
    pub label_freq: Vec<usize>,
    /// Whether features and splits are ordered by Gini.
    pub sort_by_heuristic: bool,
    /// Gini ordering of features and splits, when `sort_by_heuristic` is set.
    pub heuristic_values: HeuristicValues,
    /// The instances in the view; the cache key of the subproblem.
    pub bitset: Bitset,
}

impl<'a> DataView<'a> {
    /// The view of the whole dataset. The dataset's features must be sorted.
    pub fn root(dataset: &'a Dataset, sort_by_heuristic: bool) -> Self {
        let total_instances = dataset.count();
        let mut label_freq = vec![0; dataset.num_labels()];
        let mut feature_columns = Vec::new();
        let mut possible_split_indices = Vec::new();

        for (feature_idx, feature) in dataset.into_iter().enumerate() {
            let feature_len = feature.len();

            let mut splits = Vec::new();
            let mut last_unique_index: Option<usize> = None;

            for pos in 0..feature_len {
                let el = &feature[pos];

                if feature_idx == 0 {
                    label_freq[el.label as usize] += 1;
                }

                if let Some(last) = last_unique_index {
                    if el.unique_value_idx != last {
                        splits.push(pos);
                    }
                }
                last_unique_index = Some(el.unique_value_idx);
            }
            possible_split_indices.push(splits);
            feature_columns.push((0..feature_len).collect::<Vec<usize>>());
        }

        let mut heuristic_values = HeuristicValues::new(dataset.num_features());

        if sort_by_heuristic {
            for feature_idx in 0..dataset.num_features() {
                let feature = &dataset[feature_idx];
                let idx = &feature_columns[feature_idx];
                let (ordered_index, gini) = Self::compute_gini_for_all_splits(
                    feature,
                    idx,
                    &possible_split_indices[feature_idx],
                    &label_freq,
                    dataset.num_labels(),
                );

                heuristic_values.set_feature_ginis(feature_idx, ordered_index, gini);
            }
            heuristic_values.sort_by_gini();
        }

        let mut bitset = Bitset::new(BitsetInit::Full(total_instances));
        bitset.save_count();

        Self {
            dataset,
            total_instances,
            feature_columns,
            possible_split_indices,
            label_freq,
            sort_by_heuristic,
            heuristic_values,
            bitset,
        }
    }

    /// Number of instances in the view.
    pub fn get_dataset_size(&self) -> usize {
        self.feature_columns[0].len()
    }

    /// Number of features.
    pub fn get_feature_number(&self) -> usize {
        self.dataset.num_features()
    }

    /// The full, sorted column of feature `f`.
    pub fn get_sorted_feature(&self, f: usize) -> &Feature {
        &self.dataset[f]
    }

    /// Positions into [`Self::get_sorted_feature`] of the instances in the
    /// view, in value order.
    pub fn get_feature_indices(&self, f: usize) -> &[usize] {
        &self.feature_columns[f]
    }

    /// Number of instances of each class in the view.
    pub fn get_labels_freqs(&self) -> &[usize] {
        &self.label_freq
    }

    /// Number of classes in the dataset.
    pub fn get_num_labels(&self) -> usize {
        self.dataset.num_labels()
    }

    /// Candidate thresholds of feature `f`, as positions in the sorted column.
    pub fn get_possible_split_indices(&self, f: usize) -> &[usize] {
        &self.possible_split_indices[f]
    }

    /// Largest number of candidate thresholds of any feature.
    pub fn get_max_splits(&self) -> usize {
        self.possible_split_indices
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap_or(0)
    }

    /// Indices into the candidate thresholds of `feature`, best Gini first.
    /// Empty unless the heuristic ordering is enabled.
    pub fn ordered_possible_splits(&self, feature: usize) -> &[usize] {
        &self.heuristic_values.gini_per_split[feature]
    }

    /// `(best Gini, feature)` pairs, best first when the heuristic ordering is
    /// enabled and in feature order otherwise.
    pub fn features_best_score(&self) -> &[(f64, usize)] {
        &self.heuristic_values.best_gini_per_feature
    }

    #[inline]
    fn compute_split_indices_for(col: &Feature, idxs: &[usize]) -> Vec<usize> {
        if idxs.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(idxs.len() / 10);
        let mut last = None::<usize>;

        for (i, &pos) in idxs.iter().enumerate() {
            let cur = col[pos].unique_value_idx;
            if i > 0 && Some(cur) != last {
                out.push(i);
            }
            last = Some(cur);
        }

        out.shrink_to_fit();
        out
    }

    /// Fills the class histograms of both sides of a split of `feature_index`
    /// at `split_point`. Both slices must start zeroed.
    pub fn initialize_split_parameters(
        &self,
        feature_index: usize,
        split_point: usize,
        left_freq: &mut [usize],
        right_freq: &mut [usize],
    ) {
        let num_labels = self.dataset.num_labels();

        let total_size = self.get_dataset_size();
        let feature_ids = self.get_feature_indices(feature_index);
        let feature = self.get_sorted_feature(feature_index);

        // Count the smaller side and derive the other by subtraction.
        if 2 * split_point < total_size {
            for i in 0..split_point {
                let data_point = &feature[feature_ids[i]];
                left_freq[data_point.label as usize] += 1;
            }
            for label in 0..num_labels {
                right_freq[label] = self.label_freq[label] - left_freq[label];
            }
        } else {
            for i in split_point..total_size {
                let data_point = &feature[feature_ids[i]];
                right_freq[data_point.label as usize] += 1;
            }
            for label in 0..num_labels {
                left_freq[label] = self.label_freq[label] - right_freq[label];
            }
        }
    }

    /// Computes the weighted Gini of every candidate threshold of a feature.
    ///
    /// Returns the indices of the candidates sorted best first, and the best
    /// Gini value.
    fn compute_gini_for_all_splits(
        feature: &Feature,
        idx: &[usize],
        possible_splits: &[usize],
        label_freq: &[usize],
        num_labels: usize,
    ) -> (Vec<usize>, f64) {
        if feature.is_empty() || possible_splits.is_empty() {
            return (Vec::new(), 1.0);
        }

        let mut gini_values = Vec::with_capacity(possible_splits.len());
        let mut left_label_freq = vec![0usize; num_labels];
        let mut right_label_freq = label_freq.to_vec();
        let mut best_gini = 1.0;

        let mut last_pos = 0;

        for (split_idx, &split_pos) in possible_splits.iter().enumerate() {
            // Move the instances before this threshold to the left side.
            for &data_idx in &idx[last_pos..split_pos] {
                let label = feature[data_idx].label as usize;
                right_label_freq[label] -= 1;
                left_label_freq[label] += 1;
            }

            let left_count = split_pos;
            let right_count = idx.len() - left_count;

            let gini = Self::compute_gini_from_frequencies(
                &left_label_freq,
                &right_label_freq,
                left_count,
                right_count,
            );

            if gini < best_gini {
                best_gini = gini;
            }

            gini_values.push((split_idx, gini));
            last_pos = split_pos;
        }

        gini_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_indices: Vec<usize> = gini_values.into_iter().map(|(idx, _)| idx).collect();

        (sorted_indices, best_gini)
    }

    /// Weighted Gini impurity of a split, from the class histograms of its
    /// two sides.
    #[inline]
    fn compute_gini_from_frequencies(
        left_freq: &[usize],
        right_freq: &[usize],
        left_count: usize,
        right_count: usize,
    ) -> f64 {
        let mut left_gini = 1.0;
        let mut right_gini = 1.0;

        if left_count > 0 {
            for &freq in left_freq {
                let prob = freq as f64 / left_count as f64;
                left_gini -= prob * prob;
            }
        }

        if right_count > 0 {
            for &freq in right_freq {
                let prob = freq as f64 / right_count as f64;
                right_gini -= prob * prob;
            }
        }

        let total = left_count + right_count;
        if total > 0 {
            (left_gini * left_count as f64 + right_gini * right_count as f64) / total as f64
        } else {
            1.0
        }
    }

    /// Splits the view on feature `sf` at position `split_point` of its sorted
    /// column. The left view holds the instances before the position.
    pub fn split(&self, sf: usize, split_point: usize) -> (Self, Self) {
        #[cfg(feature = "profiling")]
        coz::scope!("Split view");
        let col = &self.dataset[sf];
        let sf_idxs = &self.feature_columns[sf];

        let (sf_left_idxs, sf_right_idxs) = sf_idxs.split_at(split_point);

        let mut left_bitset = Bitset::new(BitsetInit::Empty(self.total_instances));
        let mut left_label_freq = vec![0; self.dataset.num_labels()];

        for &pos in sf_left_idxs {
            let el = &col[pos];
            left_bitset.set(el.tid);
            left_label_freq[el.label as usize] += 1;
        }
        left_bitset.save_count();

        let mut right_label_freq = vec![0; self.dataset.num_labels()];
        for label in 0..self.dataset.num_labels() {
            right_label_freq[label] = self.label_freq[label] - left_label_freq[label];
        }

        let num_features = self.get_feature_number();
        let left_size_estimate = split_point;
        let right_size_estimate = self.get_dataset_size() - split_point;

        let mut left_pfi: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(left_size_estimate))
            .collect();
        let mut right_pfi: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(right_size_estimate))
            .collect();

        let mut left_split_indices: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(left_size_estimate / 10))
            .collect();
        let mut right_split_indices: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(right_size_estimate / 10))
            .collect();

        let mut left_heuristics = HeuristicValues::new(num_features);
        let mut right_heuristics = HeuristicValues::new(num_features);

        // Scratch buffers for the Gini pass, reused across features. They are
        // empty when the heuristic ordering is off.
        let num_labels = self.dataset.num_labels();
        let heuristic_width = if self.sort_by_heuristic {
            num_labels
        } else {
            0
        };
        let mut left_label_running = vec![0usize; heuristic_width];
        let mut right_label_running = vec![0usize; heuristic_width];
        let mut left_label_remaining = vec![0usize; heuristic_width];
        let mut right_label_remaining = vec![0usize; heuristic_width];
        let mut left_gini_values: Vec<(usize, f64)> = Vec::new();
        let mut right_gini_values: Vec<(usize, f64)> = Vec::new();

        for f in 0..num_features {
            if f == sf {
                // The split feature is already partitioned by `split_at`.
                left_pfi[f] = sf_left_idxs.to_vec();
                right_pfi[f] = sf_right_idxs.to_vec();

                left_split_indices[f] =
                    Self::compute_split_indices_for(&self.dataset[f], sf_left_idxs);
                right_split_indices[f] =
                    Self::compute_split_indices_for(&self.dataset[f], sf_right_idxs);

                if self.sort_by_heuristic {
                    let (ordered_splits, best_gini) = Self::compute_gini_for_all_splits(
                        &self.dataset[f],
                        &left_pfi[f],
                        &left_split_indices[f],
                        &left_label_freq,
                        self.dataset.num_labels(),
                    );
                    left_heuristics.set_feature_ginis(f, ordered_splits, best_gini);

                    let (ordered_splits, best_gini) = Self::compute_gini_for_all_splits(
                        &self.dataset[f],
                        &right_pfi[f],
                        &right_split_indices[f],
                        &right_label_freq,
                        self.dataset.num_labels(),
                    );
                    right_heuristics.set_feature_ginis(f, ordered_splits, best_gini);
                }
                continue;
            }

            let fcol = &self.dataset[f];
            let parent_idxs = &self.feature_columns[f];

            let mut left_last_unique: Option<usize> = None;
            let mut right_last_unique: Option<usize> = None;
            let mut left_counter = 0;
            let mut right_counter = 0;

            let mut left_best_gini = 1.0;
            let mut right_best_gini = 1.0;

            if self.sort_by_heuristic {
                left_gini_values.clear();
                right_gini_values.clear();
                left_label_running.fill(0);
                right_label_running.fill(0);
                left_label_remaining.copy_from_slice(&left_label_freq);
                right_label_remaining.copy_from_slice(&right_label_freq);
            }

            for &pos in parent_idxs {
                let el = &fcol[pos];
                let row = el.tid;
                let label = el.label as usize;

                if left_bitset.contains(row) {
                    left_pfi[f].push(pos);

                    if let Some(last) = left_last_unique {
                        if el.unique_value_idx != last {
                            left_split_indices[f].push(left_counter);

                            if self.sort_by_heuristic {
                                let gini = Self::compute_gini_from_frequencies(
                                    &left_label_running,
                                    &left_label_remaining,
                                    left_counter,
                                    left_size_estimate - left_counter,
                                );
                                if gini < left_best_gini {
                                    left_best_gini = gini;
                                }
                                left_gini_values.push((left_split_indices[f].len() - 1, gini));
                            }
                        }
                    }
                    left_last_unique = Some(el.unique_value_idx);
                    left_counter += 1;

                    if self.sort_by_heuristic {
                        left_label_remaining[label] -= 1;
                        left_label_running[label] += 1;
                    }
                } else {
                    right_pfi[f].push(pos);

                    if let Some(last) = right_last_unique {
                        if el.unique_value_idx != last {
                            right_split_indices[f].push(right_counter);

                            if self.sort_by_heuristic {
                                let gini = Self::compute_gini_from_frequencies(
                                    &right_label_running,
                                    &right_label_remaining,
                                    right_counter,
                                    right_size_estimate - right_counter,
                                );
                                if gini < right_best_gini {
                                    right_best_gini = gini;
                                }
                                right_gini_values.push((right_split_indices[f].len() - 1, gini));
                            }
                        }
                    }
                    right_last_unique = Some(el.unique_value_idx);
                    right_counter += 1;

                    if self.sort_by_heuristic {
                        right_label_remaining[label] -= 1;
                        right_label_running[label] += 1;
                    }
                }
            }

            if self.sort_by_heuristic {
                left_gini_values
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                right_gini_values
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let left_sorted_indices: Vec<usize> =
                    left_gini_values.iter().map(|&(idx, _)| idx).collect();
                let right_sorted_indices: Vec<usize> =
                    right_gini_values.iter().map(|&(idx, _)| idx).collect();

                left_heuristics.set_feature_ginis(f, left_sorted_indices, left_best_gini);
                right_heuristics.set_feature_ginis(f, right_sorted_indices, right_best_gini);
            }

            // No `shrink_to_fit`: a view only lives while its node is expanded.
        }

        let mut right_bitset = self.bitset.intersect_with(&left_bitset, true);
        right_bitset.save_count();

        if self.sort_by_heuristic {
            left_heuristics.sort_by_gini();
            right_heuristics.sort_by_gini();
        }

        let left = Self {
            dataset: self.dataset,
            total_instances: self.total_instances,
            feature_columns: left_pfi,
            possible_split_indices: left_split_indices,
            label_freq: left_label_freq,
            sort_by_heuristic: self.sort_by_heuristic,
            heuristic_values: left_heuristics,
            bitset: left_bitset,
        };

        let right = Self {
            dataset: self.dataset,
            total_instances: self.total_instances,
            feature_columns: right_pfi,
            possible_split_indices: right_split_indices,
            label_freq: right_label_freq,
            sort_by_heuristic: self.sort_by_heuristic,
            heuristic_values: right_heuristics,
            bitset: right_bitset,
        };

        (left, right)
    }

    /// Number of instances in the view.
    pub fn len(&self) -> usize {
        debug_assert!(
            self.bitset.count() == self.feature_columns[0].len(),
            "Mismatched size in bitset and feature columns"
        );
        self.feature_columns[0].len()
    }

    /// Whether the view has no instances.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod data_view_tests {
    use crate::bitsets::BitCollection;
    use crate::data::view::DataView;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;
    use crate::tests::fixture;

    #[test]
    fn root_view_covers_the_whole_dataset() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("avila_1k.txt"))?;
        dataset.sort_features();
        dataset.compute_unique_feature_values();

        let view = DataView::root(&dataset, true);

        assert_eq!(view.len(), dataset.count());
        assert_eq!(view.bitset.count(), dataset.count());
        assert_eq!(view.get_feature_number(), dataset.num_features());
        assert_eq!(
            view.get_labels_freqs().iter().sum::<usize>(),
            dataset.count(),
            "the root label histogram must account for every instance"
        );

        // Split candidates are positions strictly inside the column, and are
        // reported in increasing order.
        for f in 0..dataset.num_features() {
            let splits = view.get_possible_split_indices(f);
            assert!(splits.windows(2).all(|w| w[0] < w[1]));
            assert!(splits.iter().all(|&i| i > 0 && i < dataset.count()));
        }

        Ok(())
    }

    #[test]
    fn split_partitions_the_view_without_loss() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let mut dataset = reader.read_file(&fixture("avila_1k.txt"))?;
        dataset.sort_features();
        dataset.compute_unique_feature_values();

        let view = DataView::root(&dataset, false);
        let feature = 4;
        let split = view.get_possible_split_indices(feature)[3];
        let (left, right) = view.split(feature, split);

        assert_eq!(left.len() + right.len(), view.len());
        assert_eq!(left.bitset.count(), left.len());
        assert_eq!(right.bitset.count(), right.len());
        for label in 0..dataset.num_labels() {
            assert_eq!(
                left.get_labels_freqs()[label] + right.get_labels_freqs()[label],
                view.get_labels_freqs()[label]
            );
        }

        Ok(())
    }
}
