//! Differential test against a brute-force optimum.
//!
//! An over-eager pruning rule or a cache entry reused outside the bound it
//! was computed under makes the search return a suboptimal tree that is still
//! self-consistent. Only an independent optimum reveals it, so this test
//! enumerates every tree of the given depth on small random instances, with
//! no pruning and no cache, and compares it with every search configuration.

use std::path::PathBuf;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{PointSelector, ScheduleKind};
use contree::data::Dataset;
use contree::reader::data_reader::DataReader;

/// A small deterministic generator. Seeded explicitly so a failure is
/// reproducible from the reported case number alone.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Instance {
    /// Row-major values.
    values: Vec<f64>,
    labels: Vec<usize>,
    n_rows: usize,
    n_features: usize,
    n_labels: usize,
}

impl Instance {
    fn random(rng: &mut Lcg, n_rows: usize, n_features: usize, n_labels: usize) -> Self {
        // Few distinct values per column on purpose: ties are where a split
        // enumeration and a search are most likely to disagree.
        let values = (0..n_rows * n_features)
            .map(|_| rng.below(6) as f64)
            .collect();
        let labels = (0..n_rows).map(|_| rng.below(n_labels)).collect();
        Self {
            values,
            labels,
            n_rows,
            n_features,
            n_labels,
        }
    }

    fn dataset(&self) -> Dataset {
        Dataset::from_rows(&self.values, &self.labels, self.n_features).expect("valid instance")
    }

    fn value(&self, row: usize, feature: usize) -> f64 {
        self.values[row * self.n_features + feature]
    }
}

/// Misclassifications of the best single leaf over `rows`.
fn leaf_error(instance: &Instance, rows: &[usize]) -> usize {
    let mut counts = vec![0usize; instance.n_labels];
    for &row in rows {
        counts[instance.labels[row]] += 1;
    }
    rows.len() - counts.iter().copied().max().unwrap_or(0)
}

/// The optimum over every tree of at most `depth` levels, by enumeration.
///
/// Candidate thresholds are the midpoints between consecutive distinct values
/// of a column, which is the same candidate set the search uses.
fn brute_force(instance: &Instance, rows: &[usize], depth: usize, min_sup: usize) -> usize {
    let best_leaf = leaf_error(instance, rows);
    if depth == 0 || best_leaf == 0 || rows.len() < 2 * min_sup {
        return best_leaf;
    }

    let mut best = best_leaf;
    for feature in 0..instance.n_features {
        let mut distinct: Vec<f64> = rows.iter().map(|&r| instance.value(r, feature)).collect();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();

        for pair in distinct.windows(2) {
            let threshold = (pair[0] + pair[1]) / 2.0;
            let (left, right): (Vec<usize>, Vec<usize>) = rows
                .iter()
                .partition(|&&r| instance.value(r, feature) <= threshold);

            if left.len() < min_sup || right.len() < min_sup {
                continue;
            }
            let error = brute_force(instance, &left, depth - 1, min_sup)
                + brute_force(instance, &right, depth - 1, min_sup);
            best = best.min(error);
            if best == 0 {
                return 0;
            }
        }
    }
    best
}

fn search(dataset: &Dataset, depth: usize, min_sup: usize, fast_d2: bool) -> usize {
    let mut solver = ConTree::new(
        min_sup,
        depth,
        60.0,
        usize::MAX,
        PointSelector::Mid,
        0,
        false,
        fast_d2,
    );
    solver.fit(dataset).expect("fit").error()
}

/// Every other configuration that has to reach the same optimum: the
/// exhaustive search with the `first` selector, and the anytime search with
/// both selectors under every budget schedule. `first` goes through its own
/// Gini-priority split loop, which the plain search does not exercise.
fn other_searches(
    dataset: &Dataset,
    depth: usize,
    min_sup: usize,
    fast_d2: bool,
) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut first = ConTree::new(
        min_sup,
        depth,
        60.0,
        usize::MAX,
        PointSelector::First,
        0,
        true,
        fast_d2,
    );
    out.push((
        "ConTree first".to_string(),
        first.fit(dataset).expect("fit").error(),
    ));
    for schedule in ScheduleKind::ALL {
        for selector in [PointSelector::Mid, PointSelector::First] {
            let mut lds =
                ConTreeLds::new(min_sup, depth, 60.0, usize::MAX, selector, 0, true, fast_d2)
                    .with_schedule(schedule);
            let label = format!("LDS {selector} {schedule}");
            out.push((label, lds.fit(dataset).expect("fit").error()));
        }
    }
    out
}

#[test]
fn the_search_finds_the_optimum() {
    let mut rng = Lcg(0x5eed);
    let mut wrong = Vec::new();

    for case in 0..60 {
        let n_rows = 12 + rng.below(13);
        let n_features = 2 + rng.below(3);
        let n_labels = 2 + rng.below(2);
        let instance = Instance::random(&mut rng, n_rows, n_features, n_labels);
        let dataset = instance.dataset();
        let rows: Vec<usize> = (0..n_rows).collect();

        for depth in 1..=3 {
            for min_sup in [1, 2] {
                if 2 * min_sup > n_rows {
                    continue;
                }
                let expected = brute_force(&instance, &rows, depth, min_sup);
                for fast_d2 in [false, true] {
                    let got = search(&dataset, depth, min_sup, fast_d2);
                    for (label, other) in other_searches(&dataset, depth, min_sup, fast_d2) {
                        if other != expected {
                            wrong.push(format!(
                                "case {case}: n={n_rows} d={n_features} k={n_labels} depth={depth} \
                                 min_sup={min_sup} fast_d2={fast_d2} {label}: says {other}, the \
                                 optimum is {expected}"
                            ));
                        }
                    }
                    if got != expected {
                        wrong.push(format!(
                            "case {case}: n={n_rows} d={n_features} k={n_labels} depth={depth} \
                             min_sup={min_sup} fast_d2={fast_d2}: search says {got}, the optimum \
                             is {expected}"
                        ));
                    }
                }
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "the search disagrees with the brute-force optimum in {} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn the_search_finds_the_brute_force_optimum_on_a_real_fixture() {
    let dataset = DataReader::default()
        .read_file(&fixture("small.txt"))
        .expect("readable");

    let instance = instance_from(&dataset);
    let rows: Vec<usize> = (0..instance.n_rows).collect();

    for depth in 1..=3 {
        let expected = brute_force(&instance, &rows, depth, 1);
        for fast_d2 in [false, true] {
            assert_eq!(
                search(&dataset, depth, 1, fast_d2),
                expected,
                "small.txt depth={depth} fast_d2={fast_d2}"
            );
        }
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn instance_from(dataset: &Dataset) -> Instance {
    let n_rows = dataset.count();
    let n_features = dataset.num_features();
    let mut values = vec![0.0; n_rows * n_features];
    let mut labels = vec![0usize; n_rows];
    for feature in 0..n_features {
        for i in 0..n_rows {
            let point = &dataset[feature][i];
            values[point.tid() * n_features + feature] = point.value();
            labels[point.tid()] = point.label() as usize;
        }
    }
    Instance {
        values,
        labels,
        n_rows,
        n_features,
        n_labels: dataset.num_labels(),
    }
}
