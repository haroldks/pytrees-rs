//! Checks DL8.5 against a brute-force oracle on many small random datasets.
//!
//! After every pass, every cache entry is checked against the true optimum of
//! its subproblem: its lower bound must not exceed it, its best error must
//! not be below it, and what the search trusts (an entry it reuses as solved,
//! a lower bound for dynamic branching) must hold. The tree of each pass must
//! have the reported error and be no worse than the previous one, and the
//! last pass must find the optimum.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::{Heuristic, InformationGain};
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy,
};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, Monotonic, Rule, TopkRule};
use dtrees_rs::algorithms::optimal::Reason;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::bitsets::{BitCollection, Bitset, BitsetInit};
use dtrees_rs::caching::{CacheKey, Caching, Trie};
use dtrees_rs::cover::Cover;
use dtrees_rs::tree::Tree;

const EPS: f64 = 1e-9;

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> usize {
        (self.next() % n) as usize
    }
}

struct Dataset {
    rows: Vec<Vec<bool>>,
    labels: Vec<usize>,
    num_labels: usize,
}

impl Dataset {
    /// Labels follow a random function of the features, with some noise.
    /// Large datasets have 20 to 60 rows and 4 to 7 features, the others 6 to
    /// 24 rows and 3 to 5.
    fn random(seed: u64, large: bool) -> Self {
        let mut rng = Lcg(seed);
        let (num_rows, num_features) = match large {
            true => (20 + rng.below(41), 4 + rng.below(4)),
            false => (6 + rng.below(19), 3 + rng.below(3)),
        };
        let num_labels = 2 + rng.below(2);
        let key = [
            rng.below(num_features as u64),
            rng.below(num_features as u64),
        ];
        let mut rows = Vec::with_capacity(num_rows);
        let mut labels = Vec::with_capacity(num_rows);
        for _ in 0..num_rows {
            let row: Vec<bool> = (0..num_features).map(|_| rng.below(2) == 1).collect();
            let label = match rng.below(4) {
                0 => rng.below(num_labels as u64),
                _ => (row[key[0]] as usize + 2 * row[key[1]] as usize) % num_labels,
            };
            rows.push(row);
            labels.push(label);
        }
        Self {
            rows,
            labels,
            num_labels,
        }
    }

    fn num_features(&self) -> usize {
        self.rows[0].len()
    }

    fn all_rows(&self) -> u64 {
        u64::MAX >> (64 - self.rows.len())
    }

    fn cover(&self) -> Cover {
        let n = self.rows.len();
        let mut attributes = vec![Bitset::new(BitsetInit::Empty(n)); self.num_features()];
        let mut labels = vec![Bitset::new(BitsetInit::Empty(n)); self.num_labels];
        for (row, (values, &label)) in self.rows.iter().zip(&self.labels).enumerate() {
            for (feature, &value) in values.iter().enumerate() {
                if value {
                    attributes[feature].set(row);
                }
            }
            labels[label].set(row);
        }
        Cover::new(attributes, labels, n)
    }

    /// Rows where `feature` has the value of `branch` (0 or 1).
    fn branch_mask(&self, feature: usize, branch: usize) -> u64 {
        let mut mask = 0;
        for (i, row) in self.rows.iter().enumerate() {
            if row[feature] as usize == branch {
                mask |= 1 << i;
            }
        }
        mask
    }
}

/// Exhaustive search of the best tree of a subproblem.
struct Oracle<'a> {
    data: &'a Dataset,
    min_sup: usize,
    memo: HashMap<(u64, usize), f64>,
}

impl<'a> Oracle<'a> {
    fn new(data: &'a Dataset, min_sup: usize) -> Self {
        Self {
            data,
            min_sup,
            memo: HashMap::new(),
        }
    }

    fn leaf_error(&self, mask: u64) -> f64 {
        let mut counts = vec![0usize; self.data.num_labels];
        for (i, &label) in self.data.labels.iter().enumerate() {
            if mask >> i & 1 == 1 {
                counts[label] += 1;
            }
        }
        let total: usize = counts.iter().sum();
        (total - counts.iter().max().copied().unwrap_or(0)) as f64
    }

    /// Best error of a tree of depth at most `depth` on the rows of `mask`.
    fn optimum(&mut self, mask: u64, depth: usize) -> f64 {
        if let Some(&value) = self.memo.get(&(mask, depth)) {
            return value;
        }
        let mut best = self.leaf_error(mask);
        if depth > 0 {
            for feature in 0..self.data.num_features() {
                let left = mask & self.data.branch_mask(feature, 0);
                let right = mask & self.data.branch_mask(feature, 1);
                if (left.count_ones() as usize) < self.min_sup
                    || (right.count_ones() as usize) < self.min_sup
                {
                    continue;
                }
                let value = self.optimum(left, depth - 1) + self.optimum(right, depth - 1);
                if value < best {
                    best = value;
                }
            }
        }
        self.memo.insert((mask, depth), best);
        best
    }
}

#[derive(Clone, Copy, Debug)]
enum Relaxation {
    None,
    Discrepancy,
    Topk,
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    depth: usize,
    min_sup: usize,
    depth2: bool,
    dynamic: bool,
    similarity: bool,
    relaxation: Relaxation,
}

/// The kinds of failure found in one run, each with its first example.
#[derive(Default)]
struct Failures(BTreeMap<&'static str, String>);

impl Failures {
    fn check(&mut self, ok: bool, kind: &'static str, example: impl FnOnce() -> String) {
        if !ok {
            self.0.entry(kind).or_insert_with(example);
        }
    }
}

/// Checks the cache entries of every subproblem reachable within the depth.
fn check_cache<C: Caching + ?Sized>(
    cache: &C,
    oracle: &mut Oracle<'_>,
    setup: &Setup,
    context: &str,
    failures: &mut Failures,
) {
    let all_rows = oracle.data.all_rows();
    let mut itemset = Vec::new();
    walk(
        cache,
        oracle,
        setup,
        context,
        0,
        all_rows,
        &mut itemset,
        failures,
    );
}

#[allow(clippy::too_many_arguments)]
fn walk<C: Caching + ?Sized>(
    cache: &C,
    oracle: &mut Oracle<'_>,
    setup: &Setup,
    context: &str,
    first_feature: usize,
    mask: u64,
    itemset: &mut Vec<usize>,
    failures: &mut Failures,
) {
    if let Some(entry) = cache.node(&CacheKey::Path(itemset.clone())) {
        if entry.has_finite_leaf_error() {
            let remaining = setup.depth - itemset.len();
            let optimum = oracle.optimum(mask, remaining);
            let leaf_error = oracle.leaf_error(mask);
            let at = || format!("{context} itemset {itemset:?} optimum {optimum} entry {entry:?}");
            failures.check(
                (entry.leaf_error() - leaf_error).abs() < EPS,
                "wrong leaf error",
                at,
            );
            failures.check(
                entry.lower_bound() <= optimum + EPS,
                "lower bound above the optimum",
                at,
            );
            failures.check(
                !entry.error().is_finite() || entry.error() >= optimum - EPS,
                "best error below the optimum",
                at,
            );
            failures.check(
                !entry.is_solved() || (entry.error() - optimum).abs() < EPS,
                "reused as solved but not optimal",
                at,
            );
            failures.check(
                !setup.dynamic || entry.known_lower_bound() <= optimum + EPS,
                "dynamic branching bound above the optimum",
                at,
            );
            failures.check(
                !(entry.is_leaf() && entry.error().is_finite())
                    || (entry.error() - entry.leaf_error()).abs() < EPS,
                "leaf whose error is not its leaf error",
                at,
            );
        }
    }
    if itemset.len() == setup.depth {
        return;
    }
    for feature in first_feature..oracle.data.num_features() {
        for branch in 0..2 {
            let child = mask & oracle.data.branch_mask(feature, branch);
            if child == 0 {
                continue;
            }
            itemset.push(2 * feature + branch);
            walk(
                cache,
                oracle,
                setup,
                context,
                feature + 1,
                child,
                itemset,
                failures,
            );
            itemset.pop();
        }
    }
}

/// Error of `tree` on the rows of the dataset.
fn evaluate(tree: &Tree, data: &Dataset) -> Option<f64> {
    if tree.is_empty() {
        return None;
    }
    let mut errors = 0.0;
    for (row, &label) in data.rows.iter().zip(&data.labels) {
        let mut index = tree.get_root_index();
        while let Some(feature) = tree.node_test(index) {
            let (left, right) = tree.node_children(index);
            index = if row[feature] { right } else { left };
        }
        if tree.node_output(index)? as usize != label {
            errors += 1.0;
        }
    }
    Some(errors)
}

fn run(seed: u64, large: bool, setup: Setup) -> Failures {
    let mut failures = Failures::default();
    let data = Dataset::random(seed, large);
    let mut cover = data.cover();
    let mut oracle = Oracle::new(&data, setup.min_sup);
    let optimum = oracle.optimum(data.all_rows(), setup.depth);

    let error_fn = Box::<NativeError>::default();
    let heuristic: Box<dyn Heuristic> = Box::<InformationGain>::default();
    let mut builder = DL85Builder::default()
        .max_depth(setup.depth)
        .min_support(setup.min_sup)
        .always_sort(true)
        .specialization(match setup.depth2 {
            true => OptimalDepth2Policy::Enabled,
            false => OptimalDepth2Policy::Disabled,
        })
        .branching_strategy(match setup.dynamic {
            true => BranchingPolicy::Dynamic,
            false => BranchingPolicy::Default,
        })
        .lower_bound_strategy(match setup.similarity {
            true => LowerBoundPolicy::Similarity,
            false => LowerBoundPolicy::Disabled,
        });
    let rule: Option<Box<dyn Rule>> = match setup.relaxation {
        Relaxation::None => None,
        Relaxation::Discrepancy => Some(Box::new(DiscrepancyRule::new(
            usize::MAX,
            Box::new(Monotonic::new(1)),
        ))),
        Relaxation::Topk => Some(Box::new(TopkRule::new(
            data.num_features(),
            Box::new(Monotonic::new(1)),
        ))),
    };
    if let Some(rule) = rule {
        builder = builder.add_search_rule(rule);
    }
    let mut algo = builder
        .cache(Box::<Trie>::default())
        .heuristic(heuristic)
        .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
        .error_function(error_fn)
        .build()
        .expect("the search builds");

    let mut previous = f64::INFINITY;
    for pass in 1..=1000 {
        let result = algo.partial_fit(&mut cover);
        let context = format!("seed {seed} large {large} {setup:?} pass {pass}");
        let reported = algo.statistics().tree_error;
        let tree = evaluate(algo.tree(), &data);
        let objective = tree.unwrap_or(f64::INFINITY);
        failures.check(tree.is_some(), "no valid tree", || {
            format!("{context}\n{}", algo.tree())
        });
        failures.check(
            (objective - reported).abs() < EPS || objective == reported,
            "tree does not score the reported error",
            || format!("{context}: tree {objective}, reported {reported}"),
        );
        failures.check(
            objective <= previous + EPS,
            "tree worse than the previous pass",
            || format!("{context}: tree {objective}, before {previous}"),
        );
        previous = previous.min(objective);
        check_cache(algo.cache(), &mut oracle, &setup, &context, &mut failures);
        if result.reason != Reason::RuleReason {
            failures.check(
                (objective - optimum).abs() < EPS,
                "final tree not optimal",
                || format!("{context}: tree {objective}, optimum {optimum}"),
            );
            return failures;
        }
    }
    failures.check(false, "search did not finish", || {
        format!("seed {seed} large {large} {setup:?}")
    });
    failures
}

fn setups() -> Vec<Setup> {
    let mut all = vec![];
    for depth in 1..=4 {
        for min_sup in 1..=3 {
            for depth2 in [false, true] {
                for dynamic in [false, true] {
                    for similarity in [false, true] {
                        for relaxation in
                            [Relaxation::None, Relaxation::Discrepancy, Relaxation::Topk]
                        {
                            all.push(Setup {
                                depth,
                                min_sup,
                                depth2,
                                dynamic,
                                similarity,
                                relaxation,
                            });
                        }
                    }
                }
            }
        }
    }
    all
}

/// Runs every setup on every seed and reports, for each kind of failure, the
/// number of runs it affects.
fn run_all(seeds: Range<u64>, large: bool) {
    let mut kinds: BTreeMap<&'static str, (usize, String)> = BTreeMap::new();
    // Runs per kind of failure and per option of the setup.
    let mut breakdown: BTreeMap<(&'static str, String), usize> = BTreeMap::new();
    let mut runs = 0;
    let mut failed = 0;
    for seed in seeds {
        for setup in setups() {
            runs += 1;
            let failures = run(seed, large, setup);
            failed += !failures.0.is_empty() as usize;
            for (kind, example) in failures.0 {
                kinds.entry(kind).or_insert((0, example)).0 += 1;
                for option in [
                    format!("depth={}", setup.depth),
                    format!("min_sup={}", setup.min_sup),
                    format!("depth2={}", setup.depth2),
                    format!("dynamic={}", setup.dynamic),
                    format!("similarity={}", setup.similarity),
                    format!("relaxation={:?}", setup.relaxation),
                    format!(
                        "config=d2:{}/dyn:{}/sim:{}/{:?}",
                        setup.depth2 as u8,
                        setup.dynamic as u8,
                        setup.similarity as u8,
                        setup.relaxation
                    ),
                ] {
                    *breakdown.entry((kind, option)).or_default() += 1;
                }
            }
        }
    }
    let mut report = format!("{failed} of {runs} runs failed\n");
    for (kind, (count, example)) in &kinds {
        report += &format!("\n[{count}x] {kind}\n  e.g. {example}\n  by option:");
        for ((_, option), n) in breakdown.iter().filter(|((k, _), _)| k == kind) {
            report += &format!(" {option}:{n}");
        }
        report += "\n";
    }
    println!("{report}");
    assert!(kinds.is_empty(), "{report}");
}

#[test]
fn search_matches_the_oracle() {
    run_all(0..40, false);
}

/// More datasets; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn search_matches_the_oracle_on_more_datasets() {
    run_all(40..400, false);
}

/// Larger datasets; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn search_matches_the_oracle_on_larger_datasets() {
    run_all(0..300, true);
}
