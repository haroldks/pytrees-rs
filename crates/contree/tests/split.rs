//! SPLIT in the anytime search, against an enumeration.
//!
//! The enumeration computes the SPLIT value independently: every threshold is
//! tried down to the lookahead, and each node below it is completed by its
//! own greedy tree, grown with the same rule as the search's (lowest weighted
//! Gini, first feature and threshold on a tie, kept only when it beats the
//! leaf). With the interval pruner off, the search must reach that value
//! exactly; with it on, never below it. A lookahead covering the depth, or one
//! that deepens until it does, must reach the brute-force optimum.

mod common;

use contree::algorithms::ConTreeLds;
use contree::common::{Deepening, FitOutcome, PointSelector, ScheduleKind, SearchStatus};
use contree::data::view::DataView;
use contree::data::Dataset;

use common::{deep_check, misclassified, Instance, Lcg, Node};

fn class_counts(instance: &Instance, rows: &[usize]) -> Vec<usize> {
    let mut counts = vec![0; instance.n_labels];
    for &row in rows {
        counts[instance.labels[row]] += 1;
    }
    counts
}

/// Rows not of the majority class.
fn leaf_error(instance: &Instance, rows: &[usize]) -> f64 {
    let counts = class_counts(instance, rows);
    (rows.len() - counts.iter().max().copied().unwrap_or(0)) as f64
}

/// The weighted Gini impurity of a split, computed as the search does.
fn gini(left: &[usize], right: &[usize], n_left: usize, n_right: usize) -> f64 {
    let side = |counts: &[usize], n: usize| {
        let mut impurity = 1.0;
        if n > 0 {
            for &count in counts {
                let p = count as f64 / n as f64;
                impurity -= p * p;
            }
        }
        impurity
    };
    (side(left, n_left) * n_left as f64 + side(right, n_right) * n_right as f64)
        / (n_left + n_right) as f64
}

/// The `(left, right)` rows of every threshold of `feature` over `rows`, in
/// increasing order of threshold, between consecutive distinct values.
fn thresholds(
    instance: &Instance,
    rows: &[usize],
    feature: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut distinct: Vec<f64> = rows.iter().map(|&r| instance.value(r, feature)).collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    distinct
        .windows(2)
        .map(|pair| {
            let threshold = (pair[0] + pair[1]) / 2.0;
            rows.iter()
                .partition(|&&r| instance.value(r, feature) <= threshold)
        })
        .collect()
}

/// The cost of the greedy tree of at most `depth` levels over `rows`.
fn greedy(instance: &Instance, rows: &[usize], depth: usize, min_sup: usize) -> f64 {
    let leaf = leaf_error(instance, rows);
    if depth == 0 || leaf == 0.0 {
        return leaf;
    }
    let mut best: Option<(f64, Vec<usize>, Vec<usize>)> = None;
    for feature in 0..instance.n_features {
        for (left, right) in thresholds(instance, rows, feature) {
            if left.len() < min_sup || right.len() < min_sup {
                continue;
            }
            let impurity = gini(
                &class_counts(instance, &left),
                &class_counts(instance, &right),
                left.len(),
                right.len(),
            );
            if best
                .as_ref()
                .map_or(true, |(lowest, _, _)| impurity < *lowest)
            {
                best = Some((impurity, left, right));
            }
        }
    }
    let Some((_, left, right)) = best else {
        return leaf;
    };
    let split =
        greedy(instance, &left, depth - 1, min_sup) + greedy(instance, &right, depth - 1, min_sup);
    split.min(leaf)
}

/// The SPLIT value: the best tree whose top `lookahead` levels are free and
/// whose nodes below them are greedy trees. With the depth-2 solver, a node
/// with 2 levels left is solved exactly whatever the lookahead.
fn split_value(
    instance: &Instance,
    rows: &[usize],
    depth: usize,
    lookahead: usize,
    min_sup: usize,
    fast_d2: bool,
) -> f64 {
    if fast_d2 && depth == 2 {
        return split_value(instance, rows, 2, 2, min_sup, false);
    }
    if lookahead == 0 {
        return greedy(instance, rows, depth, min_sup);
    }
    let leaf = leaf_error(instance, rows);
    if depth == 0 || leaf == 0.0 || rows.len() < 2 * min_sup {
        return leaf;
    }
    let mut best = leaf;
    for feature in 0..instance.n_features {
        for (left, right) in thresholds(instance, rows, feature) {
            if left.len() < min_sup || right.len() < min_sup {
                continue;
            }
            let cost = split_value(instance, &left, depth - 1, lookahead - 1, min_sup, fast_d2)
                + split_value(instance, &right, depth - 1, lookahead - 1, min_sup, fast_d2);
            best = best.min(cost);
        }
    }
    best
}

/// The anytime search with a SPLIT lookahead.
#[allow(clippy::too_many_arguments)]
fn solver(
    depth: usize,
    min_sup: usize,
    selector: PointSelector,
    fast_d2: bool,
    lookahead: usize,
    deepening: Deepening,
    lds: bool,
    prune: bool,
) -> ConTreeLds {
    ConTreeLds::new(
        min_sup,
        depth,
        60.0,
        f64::INFINITY,
        selector,
        0.0,
        true,
        fast_d2,
    )
    .with_lookahead(lookahead, deepening)
    .with_lds(lds)
    .with_lookahead_pruning(prune)
}

/// Why the tree of `outcome` does not cost what the search reports, if it
/// does not.
fn tree_mismatch(
    instance: &Instance,
    depth: usize,
    min_sup: usize,
    outcome: &FitOutcome,
) -> Option<String> {
    let cost = |rows: &[usize], label| misclassified(instance, rows, label);
    let Some(tree) = Node::from_tree(&outcome.tree) else {
        return Some("no tree".to_string());
    };
    match deep_check(&tree, instance, depth, min_sup, &cost) {
        Err(why) => Some(format!("its tree: {why}")),
        Ok(rescored) if rescored != outcome.error() => Some(format!(
            "its tree costs {rescored}, the search says {}",
            outcome.error()
        )),
        Ok(_) => None,
    }
}

/// Random instances, as `(case, instance, dataset)`.
fn instances(seed: u64, count: usize) -> Vec<(usize, Instance, Dataset)> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|case| {
            let n_rows = 12 + rng.below(13);
            let n_features = 2 + rng.below(3);
            let n_labels = 2 + rng.below(2);
            let instance = Instance::random(&mut rng, n_rows, n_features, n_labels);
            let dataset = instance.dataset();
            (case, instance, dataset)
        })
        .collect()
}

const SELECTORS: [PointSelector; 2] = [PointSelector::Mid, PointSelector::First];

#[test]
fn a_fixed_lookahead_without_pruning_reaches_the_split_value() {
    let mut wrong = Vec::new();
    for (case, instance, dataset) in instances(0x5917, 40) {
        let rows: Vec<usize> = (0..instance.n_rows).collect();
        // At depth 4 only the lookaheads that leave greedy trees under the
        // depth-2 solver, to keep the enumeration small.
        for (depth, lookaheads) in [(1, 0..=1), (2, 0..=2), (3, 0..=3), (4, 0..=1)] {
            for lookahead in lookaheads {
                for min_sup in [1, 2] {
                    for fast_d2 in [false, true] {
                        let expected =
                            split_value(&instance, &rows, depth, lookahead, min_sup, fast_d2);
                        for selector in SELECTORS {
                            let label = format!(
                                "case {case} depth={depth} lookahead={lookahead} \
                                 min_sup={min_sup} {selector} fast_d2={fast_d2}"
                            );
                            let outcome = solver(
                                depth,
                                min_sup,
                                selector,
                                fast_d2,
                                lookahead,
                                Deepening::Fixed,
                                false,
                                false,
                            )
                            .fit(&dataset)
                            .expect("fit");
                            if outcome.error() != expected {
                                wrong.push(format!(
                                    "{label}: says {}, SPLIT gives {expected}",
                                    outcome.error()
                                ));
                            }
                            if let Some(why) = tree_mismatch(&instance, depth, min_sup, &outcome) {
                                wrong.push(format!("{label}: {why}"));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn with_pruning_a_fixed_lookahead_never_beats_the_split_value() {
    let mut wrong = Vec::new();
    for (case, instance, dataset) in instances(0x9e3, 40) {
        let rows: Vec<usize> = (0..instance.n_rows).collect();
        for depth in 1..=3 {
            for lookahead in 0..=depth {
                let expected = split_value(&instance, &rows, depth, lookahead, 1, true);
                for selector in SELECTORS {
                    for lds in [false, true] {
                        let label = format!(
                            "case {case} depth={depth} lookahead={lookahead} {selector} lds={lds}"
                        );
                        let outcome = solver(
                            depth,
                            1,
                            selector,
                            true,
                            lookahead,
                            Deepening::Fixed,
                            lds,
                            true,
                        )
                        .fit(&dataset)
                        .expect("fit");
                        if outcome.error() < expected {
                            wrong.push(format!(
                                "{label}: says {}, below the SPLIT value {expected}",
                                outcome.error()
                            ));
                        }
                        if lookahead == depth && outcome.error() != expected {
                            wrong.push(format!(
                                "{label}: full lookahead says {}, the optimum is {expected}",
                                outcome.error()
                            ));
                        }
                        if let Some(why) = tree_mismatch(&instance, depth, 1, &outcome) {
                            wrong.push(format!("{label}: {why}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_deepening_lookahead_ends_on_the_optimum() {
    let mut wrong = Vec::new();
    for (case, instance, dataset) in instances(0xa11, 30) {
        let rows: Vec<usize> = (0..instance.n_rows).collect();
        for depth in 1..=3 {
            for min_sup in [1, 2] {
                let expected = split_value(&instance, &rows, depth, depth, min_sup, true);
                for start in 0..depth {
                    for (deepening, lds) in [
                        (Deepening::EveryPass, false),
                        (Deepening::EveryPass, true),
                        (Deepening::Synced, true),
                    ] {
                        for selector in SELECTORS {
                            for schedule in ScheduleKind::ALL {
                                let label = format!(
                                    "case {case} depth={depth} min_sup={min_sup} start={start} \
                                     {deepening:?} lds={lds} {selector} {schedule}"
                                );
                                let outcome = solver(
                                    depth, min_sup, selector, true, start, deepening, lds, true,
                                )
                                .with_schedule(schedule)
                                .fit(&dataset)
                                .expect("fit");
                                if outcome.error() != expected
                                    || outcome.status != SearchStatus::Optimal
                                {
                                    wrong.push(format!(
                                        "{label}: says {} ({}), the optimum is {expected}",
                                        outcome.error(),
                                        outcome.status
                                    ));
                                }
                                if let Some(why) =
                                    tree_mismatch(&instance, depth, min_sup, &outcome)
                                {
                                    wrong.push(format!("{label}: {why}"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The error of every pass, checking each pass's tree on the way.
fn pass_errors(
    mut solver: ConTreeLds,
    instance: &Instance,
    dataset: &Dataset,
    depth: usize,
    wrong: &mut Vec<String>,
    label: &str,
) -> Vec<f64> {
    let view = DataView::root(dataset, true);
    let cost = |rows: &[usize], l| misclassified(instance, rows, l);
    let mut errors = Vec::new();
    loop {
        let done = solver.partial_fit(&view);
        let error = solver.statistics().error;
        let tree = solver.get_solution_tree();
        match Node::from_tree(&tree).map(|node| deep_check(&node, instance, depth, 1, &cost)) {
            Some(Ok(rescored)) if rescored == error => {}
            Some(Ok(rescored)) => wrong.push(format!(
                "{label} pass {}: tree costs {rescored}, the search says {error}",
                errors.len()
            )),
            Some(Err(why)) => wrong.push(format!("{label} pass {}: {why}", errors.len())),
            None => wrong.push(format!("{label} pass {}: no tree", errors.len())),
        }
        if errors.last().is_some_and(|&last| error > last) {
            wrong.push(format!(
                "{label} pass {}: error went up to {error}",
                errors.len()
            ));
        }
        errors.push(error);
        if done {
            return errors;
        }
    }
}

#[test]
fn every_pass_reports_the_cost_of_its_tree() {
    let mut wrong = Vec::new();
    for (case, instance, dataset) in instances(0xbee, 20) {
        for depth in 2..=3 {
            for (deepening, lds) in [
                (Deepening::EveryPass, false),
                (Deepening::EveryPass, true),
                (Deepening::Synced, true),
            ] {
                for selector in SELECTORS {
                    let label =
                        format!("case {case} depth={depth} {deepening:?} lds={lds} {selector}");
                    let solver = solver(depth, 1, selector, true, 0, deepening, lds, true);
                    pass_errors(solver, &instance, &dataset, depth, &mut wrong, &label);
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_lookahead_covering_the_depth_changes_nothing() {
    let mut wrong = Vec::new();
    for (case, instance, dataset) in instances(0xc0de, 20) {
        for depth in 2..=3 {
            for selector in SELECTORS {
                for schedule in ScheduleKind::ALL {
                    let label = format!("case {case} depth={depth} {selector} {schedule}");
                    let plain =
                        ConTreeLds::new(1, depth, 60.0, f64::INFINITY, selector, 0.0, true, true)
                            .with_schedule(schedule);
                    let full = solver(
                        depth,
                        1,
                        selector,
                        true,
                        depth,
                        Deepening::Fixed,
                        true,
                        true,
                    )
                    .with_schedule(schedule);
                    let a = pass_errors(plain, &instance, &dataset, depth, &mut wrong, &label);
                    let b = pass_errors(full, &instance, &dataset, depth, &mut wrong, &label);
                    if a != b {
                        wrong.push(format!(
                            "{label}: passes {a:?} without lookahead, {b:?} with"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
