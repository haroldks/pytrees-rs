pub mod schedule;
pub mod search;

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use crate::bitsets::{BitCollection, Bitset, BitsetInit};
use crate::cover::Cover;

#[derive(Clone, Copy, Debug)]
pub enum ImpurityMetric {
    Gini,
    Entropy,
}

// Wrapper for f64 that implements Ord for use in BinaryHeap.
#[derive(PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// Entry in the per-feature max-heap: (gain, threshold, lo, split_idx, hi).
struct HeapEntry {
    gain: OrdF64,
    threshold: f64,
    lo: usize,
    split_idx: usize,
    hi: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain.cmp(&other.gain)
    }
}

fn impurity(counts: &[usize], metric: ImpurityMetric) -> f64 {
    let n: usize = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    match metric {
        ImpurityMetric::Gini => {
            let mut sum = 0.0;
            for &c in counts {
                let p = c as f64 / n_f;
                sum += p * p;
            }
            1.0 - sum
        }
        ImpurityMetric::Entropy => {
            let mut h = 0.0;
            for &c in counts {
                if c > 0 {
                    let p = c as f64 / n_f;
                    h -= p * p.ln();
                }
            }
            h
        }
    }
}

// Find the best threshold (and its split index) within x[lo..hi] (exclusive hi).
// Returns None if no gain-positive split exists.
// split_idx is the last index belonging to the left partition (x[split_idx] <= threshold).
fn best_split(
    x: &[f64],
    y: &[usize],
    n_labels: usize,
    lo: usize,
    hi: usize, // exclusive
    metric: ImpurityMetric,
) -> Option<HeapEntry> {
    let n = hi - lo;
    if n < 2 {
        return None;
    }

    let mut total_counts = vec![0usize; n_labels];
    for i in lo..hi {
        total_counts[y[i]] += 1;
    }
    let total_imp = impurity(&total_counts, metric);

    let mut left_counts = vec![0usize; n_labels];
    let mut right_counts = total_counts.clone();

    let mut best_gain = 0.0;
    let mut best_threshold = 0.0;
    let mut best_split_idx = lo;
    let mut found = false;

    for i in lo..hi - 1 {
        left_counts[y[i]] += 1;
        right_counts[y[i]] -= 1;

        if x[i] == x[i + 1] {
            continue;
        }

        let n_left = i - lo + 1;
        let n_right = n - n_left;

        let weighted = (n_left as f64 / n as f64) * impurity(&left_counts, metric)
            + (n_right as f64 / n as f64) * impurity(&right_counts, metric);
        let gain = total_imp - weighted;

        if gain > best_gain {
            best_gain = gain;
            best_threshold = (x[i] + x[i + 1]) / 2.0;
            best_split_idx = i;
            found = true;
        }
    }

    if !found {
        return None;
    }

    Some(HeapEntry {
        gain: OrdF64(best_gain),
        threshold: best_threshold,
        lo,
        split_idx: best_split_idx,
        hi: hi - 1, // convert to inclusive
    })
}

// Compute the ranked candidate list for a single feature.
// Returns (threshold, gain) pairs in descending gain order (best first).
fn compute_candidates(
    x: &[f64],
    y: &[usize],
    n_labels: usize,
    metric: ImpurityMetric,
) -> std::collections::VecDeque<(f64, f64)> {
    let n = x.len();
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut result = std::collections::VecDeque::new();

    if let Some(entry) = best_split(x, y, n_labels, 0, n, metric) {
        heap.push(entry);
    }

    while let Some(entry) = heap.pop() {
        result.push_back((entry.threshold, entry.gain.0));

        // Left sub-range [lo, split_idx] (inclusive, length split_idx - lo + 1)
        let left_lo = entry.lo;
        let left_hi = entry.split_idx + 1; // exclusive
        if let Some(e) = best_split(x, y, n_labels, left_lo, left_hi, metric) {
            heap.push(e);
        }

        // Right sub-range [split_idx+1, hi] (inclusive)
        let right_lo = entry.split_idx + 1;
        let right_hi = entry.hi + 1; // exclusive
        if right_lo < right_hi {
            if let Some(e) = best_split(x, y, n_labels, right_lo, right_hi, metric) {
                heap.push(e);
            }
        }
    }

    result
}

// Global merge (Algorithm 1 from the MID paper).
// Selects all thresholds in global gain order from the per-feature ranked lists.
fn global_merge(
    mut candidates_per_feature: Vec<std::collections::VecDeque<(f64, f64)>>,
) -> Vec<(usize, f64)> {
    let mut thresholds: Vec<(usize, f64)> = Vec::new();

    loop {
        let mut best_idx = usize::MAX;
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_threshold = 0.0;
        let mut all_empty = true;

        for (i, list) in candidates_per_feature.iter().enumerate() {
            if let Some(&(threshold, gain)) = list.front() {
                all_empty = false;
                if gain > best_gain {
                    best_gain = gain;
                    best_idx = i;
                    best_threshold = threshold;
                }
            }
        }

        if all_empty {
            break;
        }

        thresholds.push((best_idx, best_threshold));
        candidates_per_feature[best_idx].pop_front();
    }

    thresholds
}

pub struct MIDDiscretizer {
    // Globally ranked (feature_idx, threshold) pairs, best-gain first.
    thresholds: Vec<(usize, f64)>,
    // Stored features and labels for cover construction.
    features: Vec<Vec<f64>>,
    labels: Vec<usize>,
    n_labels: usize,
    n_samples: usize,
}

impl MIDDiscretizer {
    /// Fit the discretizer: sort per-feature candidates and merge globally.
    ///
    /// `features` is a slice of column vectors (one per feature).
    /// `labels` is a flat slice of class indices (one per sample).
    pub fn fit(features: &[Vec<f64>], labels: &[usize], metric: ImpurityMetric) -> Self {
        let n_samples = labels.len();

        // Remap arbitrary label values to compact 0-based indices so that
        // label values can be used directly as bitset/array indices.
        // e.g. {0, 7, 10} → {0→0, 7→1, 10→2}
        let unique: HashSet<usize> = labels.iter().copied().collect();
        let mut sorted_unique: Vec<usize> = unique.into_iter().collect();
        sorted_unique.sort_unstable();
        let n_labels = sorted_unique.len();
        let label_map: HashMap<usize, usize> = sorted_unique
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        let remapped: Vec<usize> = labels.iter().map(|&l| label_map[&l]).collect();

        let mut candidates_per_feature = Vec::with_capacity(features.len());

        for col in features {
            // Sort by feature value, keeping label alignment.
            let mut paired: Vec<(f64, usize)> = col
                .iter()
                .copied()
                .zip(remapped.iter().copied())
                .collect();
            paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            let x_sorted: Vec<f64> = paired.iter().map(|p| p.0).collect();
            let y_sorted: Vec<usize> = paired.iter().map(|p| p.1).collect();

            let cands = compute_candidates(&x_sorted, &y_sorted, n_labels, metric);
            candidates_per_feature.push(cands);
        }

        let thresholds = global_merge(candidates_per_feature);

        Self {
            thresholds,
            features: features.to_vec(),
            labels: remapped,
            n_labels,
            n_samples,
        }
    }

    /// Return the globally ranked threshold list (feature_idx, threshold).
    pub fn thresholds(&self) -> &[(usize, f64)] {
        &self.thresholds
    }

    /// Total number of computed thresholds.
    pub fn len(&self) -> usize {
        self.thresholds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.thresholds.is_empty()
    }

    /// Build a binary `Cover` using the top-`n` thresholds.
    ///
    /// For threshold `(feat_idx, t)`, sample `i` gets bit 1 iff
    /// `features[feat_idx][i] <= t`.
    pub fn get_cover(&self, n: usize) -> Cover {
        let n_use = n.min(self.thresholds.len());

        let mut attributes = Vec::with_capacity(n_use);
        for &(feat_idx, threshold) in &self.thresholds[..n_use] {
            let mut bs = Bitset::new(BitsetInit::Empty(self.n_samples));
            for (sample_idx, &val) in self.features[feat_idx].iter().enumerate() {
                if val <= threshold {
                    bs.set(sample_idx);
                }
            }
            attributes.push(bs);
        }

        let mut label_bitsets = vec![Bitset::new(BitsetInit::Empty(self.n_samples)); self.n_labels];
        for (sample_idx, &label) in self.labels.iter().enumerate() {
            label_bitsets[label].set(sample_idx);
        }

        Cover::new(attributes, label_bitsets, self.n_samples)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::mid::schedule::ThresholdSchedule;
    use crate::mid::search::{DL85Config, MIDSearch};
    use crate::reader::ContinuousDataReader;
    use super::*;

    fn toy_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        // 2 features, 6 samples, 2 classes.
        // Feature 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        // Feature 1: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        // Labels:    [0,    0,    0,    1,    1,    1]
        let f0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f1 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let y = vec![0, 0, 0, 1, 1, 1];
        (vec![f0, f1], y)
    }

    #[test]
    fn fit_produces_thresholds() {
        let (features, labels) = toy_data();
        let mid = MIDDiscretizer::fit(&features, &labels, ImpurityMetric::Gini);
        assert!(!mid.is_empty(), "should have at least one threshold");
    }

    #[test]
    fn nested_set_property() {
        let (features, labels) = toy_data();
        let mid = MIDDiscretizer::fit(&features, &labels, ImpurityMetric::Gini);
        let all = mid.thresholds().to_vec();
        // Thresholds for N must be a prefix of thresholds for N+1.
        for k in 1..all.len() {
            assert_eq!(
                &all[..k],
                &mid.thresholds()[..k],
                "nested-set property violated at k={}",
                k
            );
        }
    }

    #[test]
    fn get_cover_dimensions() {
        let (features, labels) = toy_data();
        let mid = MIDDiscretizer::fit(&features, &labels, ImpurityMetric::Gini);
        let n = 3.min(mid.len());
        if n == 0 {
            return;
        }
        let cover = mid.get_cover(n);
        assert_eq!(cover.num_attributes, n);
        assert_eq!(cover.count(), labels.len());
    }

    #[test]
    fn anneal() -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new("test_data/anneal.txt");
        let data = ContinuousDataReader::default().read_file(&path)?;

        println!("{:?}",data.features.len());
        println!("{:?}",data.labels);

        let mid  = MIDDiscretizer::fit(&data.features, &data.labels, ImpurityMetric::Entropy);

        println!("{:?}", mid.thresholds());

        let schedule = ThresholdSchedule::new(1, 1, 100);
        let config = DL85Config { max_depth: 4, min_sup: 1, max_error: f64::INFINITY };


        let mut algo = MIDSearch::new(&data, ImpurityMetric::Entropy, config, schedule, 30.0 );

        while algo.partial_fit() {
                println!("iteration {} — best error: {}", algo.iteration, algo.error());
        }
        println!("Runtime: {:.3}s", algo.elapsed());


        Ok(())

    }
 }
