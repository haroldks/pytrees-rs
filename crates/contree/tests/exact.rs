//! Differential test against a brute-force optimum.
//!
//! An over-eager pruning rule or a cache entry reused outside the bound it
//! was computed under makes the search return a suboptimal tree that is still
//! self-consistent. Only an independent optimum reveals it, so this test
//! enumerates every tree of the given depth on small random instances, with
//! no pruning and no cache, and compares it with every search configuration.

mod common;

use std::path::PathBuf;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{
    Cost, FitOutcome, Leaf, LeafLoss, Loss, LossInput, PointSelector, ScheduleKind, SearchError,
    SearchStatus, Skipping,
};
use contree::data::Dataset;
use contree::reader::data_reader::DataReader;

use common::{deep_check, instance_from, misclassified, Instance, Lcg, Node};

/// The class counts of `rows`.
fn class_counts(instance: &Instance, rows: &[usize]) -> Vec<usize> {
    let mut counts = vec![0; instance.n_labels];
    for &row in rows {
        counts[instance.labels[row]] += 1;
    }
    counts
}

/// Instances misclassified by the best single leaf over `rows`.
fn leaf_error(instance: &Instance, rows: &[usize]) -> f64 {
    misclassification(&class_counts(instance, rows)).0
}

/// Instances of the minority classes, and the majority class (ties to the
/// highest index, like the built-in loss).
fn misclassification(counts: &[usize]) -> (f64, usize) {
    let mut label = 0;
    for (index, &count) in counts.iter().enumerate() {
        if count >= counts[label] {
            label = index;
        }
    }
    let error: usize = (0..counts.len())
        .filter(|&l| l != label)
        .map(|l| counts[l])
        .sum();
    (error as f64, label)
}

/// A loss that breaks the skipping assumption: moving an instance into a
/// leaf can raise its cost by more than 1.
fn squared_misclassification(counts: &[usize]) -> (f64, usize) {
    let (error, label) = misclassification(counts);
    (error * error, label)
}

/// The optimum over every tree of at most `depth` levels, by enumeration.
///
/// Candidate thresholds are the midpoints between consecutive distinct values
/// of a column, which is the same candidate set the search uses.
fn brute_force(instance: &Instance, rows: &[usize], depth: usize, min_sup: usize) -> f64 {
    brute_force_with(instance, rows, depth, min_sup, &|rows| {
        leaf_error(instance, rows)
    })
}

/// `brute_force` for any leaf cost. Stopping at a zero cost holds for any
/// non-negative cost.
fn brute_force_with(
    instance: &Instance,
    rows: &[usize],
    depth: usize,
    min_sup: usize,
    leaf: &dyn Fn(&[usize]) -> f64,
) -> f64 {
    let best_leaf = leaf(rows);
    if depth == 0 || best_leaf == 0.0 || rows.len() < 2 * min_sup {
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
            let error = brute_force_with(instance, &left, depth - 1, min_sup, leaf)
                + brute_force_with(instance, &right, depth - 1, min_sup, leaf);
            best = best.min(error);
            if best == 0.0 {
                return 0.0;
            }
        }
    }
    best
}

fn search(dataset: &Dataset, depth: usize, min_sup: usize, fast_d2: bool) -> f64 {
    fit_mid(dataset, depth, min_sup, fast_d2).error()
}

/// The plain search: `mid` selector, no Gini ordering.
fn fit_mid(dataset: &Dataset, depth: usize, min_sup: usize, fast_d2: bool) -> FitOutcome {
    let mut solver = ConTree::new(
        min_sup,
        depth,
        60.0,
        f64::INFINITY,
        PointSelector::Mid,
        0.0,
        false,
        fast_d2,
    );
    solver.fit(dataset).expect("fit")
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
) -> Vec<(String, FitOutcome)> {
    let mut out = Vec::new();
    let mut first = ConTree::new(
        min_sup,
        depth,
        60.0,
        f64::INFINITY,
        PointSelector::First,
        0.0,
        true,
        fast_d2,
    );
    out.push((
        "ConTree first".to_string(),
        first.fit(dataset).expect("fit"),
    ));
    for schedule in ScheduleKind::ALL {
        for selector in [PointSelector::Mid, PointSelector::First] {
            let mut lds = ConTreeLds::new(
                min_sup,
                depth,
                60.0,
                f64::INFINITY,
                selector,
                0.0,
                true,
                fast_d2,
            )
            .with_schedule(schedule);
            let label = format!("LDS {selector} {schedule}");
            out.push((label, lds.fit(dataset).expect("fit")));
        }
    }
    out
}

/// Why `outcome` is not the optimum `expected`, if it is not. Its tree is
/// checked too (see `deep_check`): its cost, recomputed from the rows, must
/// also be the optimum. `cost(rows, label)` is the cost of a leaf.
fn not_optimal(
    instance: &Instance,
    depth: usize,
    min_sup: usize,
    outcome: &FitOutcome,
    expected: f64,
    cost: &dyn Fn(&[usize], usize) -> f64,
) -> Option<String> {
    // The search and the enumeration add the same leaf costs in different
    // orders, so they agree up to rounding.
    let tolerance = 1e-9 * (1.0 + expected.abs());
    if (outcome.error() - expected).abs() > tolerance {
        return Some(format!(
            "says {}, the optimum is {expected}",
            outcome.error()
        ));
    }
    let Some(tree) = Node::from_tree(&outcome.tree) else {
        return Some("returned no tree".to_string());
    };
    match deep_check(&tree, instance, depth, min_sup, cost) {
        Err(why) => Some(format!("its tree: {why}")),
        Ok(rescored) if (rescored - expected).abs() > tolerance => Some(format!(
            "its tree costs {rescored} on the rows, the optimum is {expected}"
        )),
        Ok(_) => None,
    }
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
                let cost = |rows: &[usize], label| misclassified(&instance, rows, label);
                for fast_d2 in [false, true] {
                    let mut results = other_searches(&dataset, depth, min_sup, fast_d2);
                    results.push((
                        "ConTree mid".to_string(),
                        fit_mid(&dataset, depth, min_sup, fast_d2),
                    ));
                    for (label, outcome) in results {
                        if let Some(why) =
                            not_optimal(&instance, depth, min_sup, &outcome, expected, &cost)
                        {
                            wrong.push(format!(
                                "case {case}: n={n_rows} d={n_features} k={n_labels} depth={depth} \
                                 min_sup={min_sup} fast_d2={fast_d2} {label}: {why}"
                            ));
                        }
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

/// Found by a random search against the brute force. In one side of a
/// depth-2 split the right part becomes pure while the left part is still
/// below the minimum support; the depth-2 solver used to stop scanning the
/// feature there, missing a later threshold that meets the support.
#[test]
fn a_pure_side_below_the_minimum_support_does_not_stop_the_scan() {
    let values = vec![
        3.0, 0.0, 1.0, 1.0, 5.0, 2.0, 3.0, 3.0, 5.0, 4.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 4.0, 5.0,
        2.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0,
    ];
    let labels = vec![1, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 2, 0, 2, 2];
    let instance = Instance {
        values,
        labels,
        n_rows: 15,
        n_features: 2,
        n_labels: 3,
    };
    let rows: Vec<usize> = (0..instance.n_rows).collect();
    let expected = brute_force(&instance, &rows, 2, 3);
    assert_eq!(expected, 4.0);
    for fast_d2 in [false, true] {
        assert_eq!(
            search(&instance.dataset(), 2, 3, fast_d2),
            expected,
            "fast_d2={fast_d2}"
        );
    }
}

/// A loss written by a caller, as a closure.
struct FnLoss<F>(F);

impl<F: Fn(Leaf<'_>) -> (Cost, usize) + Send + Sync> LeafLoss for FnLoss<F> {
    fn cost(&self, leaf: Leaf<'_>) -> (Cost, usize) {
        (self.0)(leaf)
    }
}

/// A loss on class counts.
fn counts_loss(function: fn(&[usize]) -> (f64, usize)) -> Loss {
    Loss::custom(
        Box::new(FnLoss(move |leaf: Leaf<'_>| match leaf {
            Leaf::ClassCounts(counts) => function(counts),
            Leaf::Indices(_) => unreachable!("asked for class counts"),
        })),
        LossInput::ClassCounts,
    )
}

/// The same loss on instance ids, recomputing the class counts itself.
fn indices_loss(instance: &Instance, function: fn(&[usize]) -> (f64, usize)) -> Loss {
    let labels = instance.labels.clone();
    let n_labels = instance.n_labels;
    Loss::custom(
        Box::new(FnLoss(move |leaf: Leaf<'_>| match leaf {
            Leaf::Indices(rows) => {
                let mut counts = vec![0; n_labels];
                for &row in rows {
                    counts[labels[row]] += 1;
                }
                function(&counts)
            }
            Leaf::ClassCounts(_) => unreachable!("asked for indices"),
        })),
        LossInput::Indices,
    )
}

/// Every search configuration, each with a fresh loss and the given
/// skipping setting.
fn all_searches(
    dataset: &Dataset,
    depth: usize,
    min_sup: usize,
    fast_d2: bool,
    loss: &dyn Fn() -> Loss,
    skipping: Skipping,
) -> Vec<(String, Result<FitOutcome, SearchError>)> {
    let mut out = Vec::new();
    for selector in [PointSelector::Mid, PointSelector::First] {
        let mut solver = ConTree::new(
            min_sup,
            depth,
            60.0,
            f64::INFINITY,
            selector,
            0.0,
            true,
            fast_d2,
        )
        .with_loss(loss())
        .with_skipping(skipping);
        out.push((format!("ConTree {selector}"), solver.fit(dataset)));
    }
    for schedule in ScheduleKind::ALL {
        for selector in [PointSelector::Mid, PointSelector::First] {
            let mut lds = ConTreeLds::new(
                min_sup,
                depth,
                60.0,
                f64::INFINITY,
                selector,
                0.0,
                true,
                fast_d2,
            )
            .with_schedule(schedule)
            .with_loss(loss())
            .with_skipping(skipping);
            out.push((format!("LDS {selector} {schedule}"), lds.fit(dataset)));
        }
    }
    out
}

/// Checks one search result: `(instance, depth, min_sup, outcome)`, and why
/// it is wrong, if it is.
type Check<'a> =
    dyn Fn(&Instance, usize, usize, &Result<FitOutcome, SearchError>) -> Option<String> + 'a;

/// Runs every search configuration on random instances, checks each result
/// with `check`, and returns the failures.
fn on_random_instances(
    seed: u64,
    cases: usize,
    loss: &dyn Fn(&Instance) -> Box<dyn Fn() -> Loss>,
    skipping: Skipping,
    check: &Check<'_>,
) -> Vec<String> {
    let mut rng = Lcg(seed);
    let mut wrong = Vec::new();
    for case in 0..cases {
        let n_rows = 12 + rng.below(9);
        let n_features = 2 + rng.below(2);
        let n_labels = 2 + rng.below(2);
        let instance = Instance::random(&mut rng, n_rows, n_features, n_labels);
        let dataset = instance.dataset();
        let make_loss = loss(&instance);
        for depth in 1..=3 {
            for min_sup in [1, 2] {
                for fast_d2 in [false, true] {
                    for (label, outcome) in
                        all_searches(&dataset, depth, min_sup, fast_d2, &*make_loss, skipping)
                    {
                        if let Some(why) = check(&instance, depth, min_sup, &outcome) {
                            wrong.push(format!(
                                "case {case}: depth={depth} min_sup={min_sup} fast_d2={fast_d2} \
                                 {label}: {why}"
                            ));
                        }
                    }
                }
            }
        }
    }
    wrong
}

/// Agreement with the brute force for the leaf cost `cost(rows, label)`,
/// checked on the tree as well (see `not_optimal`).
fn matches_optimum(
    instance: &Instance,
    depth: usize,
    min_sup: usize,
    outcome: &Result<FitOutcome, SearchError>,
    cost: &dyn Fn(&[usize], usize) -> f64,
) -> Option<String> {
    let rows: Vec<usize> = (0..instance.n_rows).collect();
    let leaf = |rows: &[usize]| {
        (0..instance.n_labels)
            .map(|label| cost(rows, label))
            .fold(f64::INFINITY, f64::min)
    };
    let expected = brute_force_with(instance, &rows, depth, min_sup, &leaf);
    match outcome {
        Ok(outcome) => not_optimal(instance, depth, min_sup, outcome, expected, cost),
        Err(err) => Some(format!("failed: {err}")),
    }
}

/// Misclassification written as a custom loss, on class counts or on
/// instance ids, reaches the built-in optimum, with skipping off (the
/// default for a custom loss) and forced on. Forced on, it never breaks the
/// assumption skipping makes.
#[test]
fn misclassification_as_a_custom_loss_finds_the_optimum() {
    for input in [LossInput::ClassCounts, LossInput::Indices] {
        for skipping in [Skipping::Auto, Skipping::On] {
            let loss = |instance: &Instance| -> Box<dyn Fn() -> Loss> {
                match input {
                    LossInput::ClassCounts => Box::new(|| counts_loss(misclassification)),
                    LossInput::Indices => {
                        let instance = Instance {
                            values: instance.values.clone(),
                            labels: instance.labels.clone(),
                            ..*instance
                        };
                        Box::new(move || indices_loss(&instance, misclassification))
                    }
                }
            };
            let check = |instance: &Instance,
                         depth: usize,
                         min_sup: usize,
                         outcome: &Result<FitOutcome, SearchError>| {
                if let Ok(outcome) = outcome {
                    if outcome.statistics.skipping_violations > 0 {
                        return Some(format!(
                            "{} skipping violations for misclassification",
                            outcome.statistics.skipping_violations
                        ));
                    }
                    if outcome.status != SearchStatus::Optimal {
                        return Some(format!("status {}", outcome.status));
                    }
                }
                matches_optimum(instance, depth, min_sup, outcome, &|rows, label| {
                    misclassified(instance, rows, label)
                })
            };
            let wrong = on_random_instances(0x1055, 8, &loss, skipping, &check);
            assert!(
                wrong.is_empty(),
                "{input:?} {skipping:?}: {} case(s):\n{}",
                wrong.len(),
                wrong.join("\n")
            );
        }
    }
}

/// With skipping off, any non-negative loss reaches its optimum: here one
/// that breaks the skipping assumption.
#[test]
fn a_custom_loss_finds_its_optimum_without_skipping() {
    let loss = |_: &Instance| -> Box<dyn Fn() -> Loss> {
        Box::new(|| counts_loss(squared_misclassification))
    };
    let wrong = on_random_instances(0x5a1e, 8, &loss, Skipping::Auto, &|instance, d, m, o| {
        matches_optimum(instance, d, m, o, &|rows, label| {
            misclassified(instance, rows, label).powi(2)
        })
    });
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Skipping on and off reach the same optimum with the built-in loss.
#[test]
fn skipping_on_and_off_agree() {
    for skipping in [Skipping::Auto, Skipping::Off] {
        let loss = |_: &Instance| -> Box<dyn Fn() -> Loss> { Box::new(Loss::default) };
        let wrong = on_random_instances(0x0ff, 8, &loss, skipping, &|instance, d, m, o| {
            matches_optimum(instance, d, m, o, &|rows, label| {
                misclassified(instance, rows, label)
            })
        });
        assert!(
            wrong.is_empty(),
            "{skipping:?}: {} case(s):\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
}

/// Forcing skipping for a loss that breaks its assumption is caught by the
/// depth-2 solver: the moves are counted and the tree is not called optimal.
#[test]
fn forcing_skipping_on_a_loss_that_breaks_it_is_reported() {
    let dataset = DataReader::default()
        .read_file(&fixture("iris.txt"))
        .expect("readable");
    let mut solver = ConTree::new(
        1,
        2,
        60.0,
        f64::INFINITY,
        PointSelector::Mid,
        0.0,
        true,
        true,
    )
    .with_loss(counts_loss(squared_misclassification))
    .with_skipping(Skipping::On);
    let outcome = solver.fit(&dataset).expect("fit");
    assert!(outcome.statistics.skipping_violations > 0);
    assert_eq!(outcome.status, SearchStatus::NotProven);
}

#[test]
fn a_negative_cost_is_an_error() {
    let dataset = DataReader::default()
        .read_file(&fixture("small.txt"))
        .expect("readable");
    for fast_d2 in [false, true] {
        let mut solver = ConTree::new(
            1,
            2,
            60.0,
            f64::INFINITY,
            PointSelector::Mid,
            0.0,
            true,
            fast_d2,
        )
        .with_loss(counts_loss(|counts| (-1.0, counts.len() - 1)));
        assert_eq!(
            solver.fit(&dataset).map(|o| o.error()),
            Err(SearchError::InvalidLeafCost { cost: -1.0 })
        );
    }
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
