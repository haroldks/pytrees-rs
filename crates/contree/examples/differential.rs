//! Measures how far each solver is from the true optimum.
//!
//! Enumerates every tree of a given depth on small random instances -- no
//! pruning, no cache, no bounds -- and compares. `tests/exact.rs` asserts on
//! the same sweep; this prints the breakdown so a change can be measured
//! rather than just pass or fail.
//!
//!     cargo run --release -p contree-rs --example differential

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{PointSelector, ScheduleKind};
use contree::data::Dataset;

/// Deterministic, so a failure is reproducible from the case number alone.
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

fn leaf_error(labels: &[usize], rows: &[usize], n_labels: usize) -> usize {
    let mut counts = vec![0usize; n_labels];
    for &row in rows {
        counts[labels[row]] += 1;
    }
    rows.len() - counts.iter().copied().max().unwrap_or(0)
}

fn brute_force(
    values: &[f64],
    labels: &[usize],
    n_features: usize,
    n_labels: usize,
    rows: &[usize],
    depth: usize,
    min_sup: usize,
) -> usize {
    let best_leaf = leaf_error(labels, rows, n_labels);
    if depth == 0 || best_leaf == 0 || rows.len() < 2 * min_sup {
        return best_leaf;
    }
    let mut best = best_leaf;
    for feature in 0..n_features {
        let mut distinct: Vec<f64> = rows
            .iter()
            .map(|&r| values[r * n_features + feature])
            .collect();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        for pair in distinct.windows(2) {
            let threshold = (pair[0] + pair[1]) / 2.0;
            let (left, right): (Vec<usize>, Vec<usize>) = rows
                .iter()
                .partition(|&&r| values[r * n_features + feature] < threshold);
            if left.len() < min_sup || right.len() < min_sup {
                continue;
            }
            let error = brute_force(
                values,
                labels,
                n_features,
                n_labels,
                &left,
                depth - 1,
                min_sup,
            ) + brute_force(
                values,
                labels,
                n_features,
                n_labels,
                &right,
                depth - 1,
                min_sup,
            );
            best = best.min(error);
            if best == 0 {
                return 0;
            }
        }
    }
    best
}

#[derive(Default)]
struct Tally {
    worse: usize,
    impossible: usize,
    by_depth: [usize; 4],
    cases: Vec<String>,
}

impl Tally {
    fn record(&mut self, got: usize, optimum: usize, depth: usize, what: &str) {
        match got.cmp(&optimum) {
            std::cmp::Ordering::Greater => {
                self.worse += 1;
                self.by_depth[depth] += 1;
                self.cases
                    .push(format!("{what}: {got} vs optimum {optimum}"));
            }
            std::cmp::Ordering::Less => {
                self.impossible += 1;
                self.by_depth[depth] += 1;
                self.cases
                    .push(format!("{what}: {got} vs optimum {optimum}  IMPOSSIBLE"));
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    fn report(&self, name: &str, total: usize) {
        println!(
            "  {name:<28} {:>3} / {total} off the optimum   \
             (depth1={} depth2={} depth3={}){}",
            self.worse + self.impossible,
            self.by_depth[1],
            self.by_depth[2],
            self.by_depth[3],
            if self.impossible > 0 {
                format!("   [{} report an impossible error!]", self.impossible)
            } else {
                String::new()
            }
        );
        for case in &self.cases {
            println!("      {case}");
        }
    }
}

fn main() {
    let mut rng = Lcg(0x5eed);
    let (mut exhaustive, mut specialized, mut anytime, mut exhaustive_first) = (
        Tally::default(),
        Tally::default(),
        Tally::default(),
        Tally::default(),
    );
    // every schedule x [mid, first]
    let mut schedules: [[Tally; 2]; ScheduleKind::ALL.len()] = Default::default();
    let (mut runs, mut lds_runs) = (0usize, 0usize);

    for case in 0..60 {
        let n_rows = 12 + rng.below(13);
        let n_features = 2 + rng.below(3);
        let n_labels = 2 + rng.below(2);
        let values: Vec<f64> = (0..n_rows * n_features)
            .map(|_| rng.below(6) as f64)
            .collect();
        let labels: Vec<usize> = (0..n_rows).map(|_| rng.below(n_labels)).collect();

        let dataset = Dataset::from_rows(&values, &labels, n_features).expect("valid instance");
        // `CONTREE_DUMP=<dir>` writes each instance in upstream ConTree's own
        // input format, so the same cases can be run through its binary.
        if let Some(dir) = std::env::var_os("CONTREE_DUMP") {
            let mut out = String::new();
            for row in 0..n_rows {
                out.push_str(&labels[row].to_string());
                for f in 0..n_features {
                    out.push(' ');
                    out.push_str(&values[row * n_features + f].to_string());
                }
                out.push('\n');
            }
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("case{case}.txt")), out).unwrap();
        }
        let rows: Vec<usize> = (0..n_rows).collect();

        for depth in 1..=3 {
            for min_sup in [1usize, 2] {
                if 2 * min_sup > n_rows {
                    continue;
                }
                let optimum = brute_force(
                    &values, &labels, n_features, n_labels, &rows, depth, min_sup,
                );

                for fast_d2 in [false, true] {
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
                    let got = solver.fit(&dataset).expect("fit").error();
                    let what = format!(
                        "case {case} n={n_rows} d={n_features} k={n_labels}                          depth={depth} min_sup={min_sup} fast_d2={fast_d2}"
                    );
                    if fast_d2 {
                        &mut specialized
                    } else {
                        &mut exhaustive
                    }
                    .record(got, optimum, depth, &what);
                    runs += 1;

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
                    let got = first.fit(&dataset).expect("fit").error();
                    exhaustive_first.record(got, optimum, depth, &what);

                    // The anytime search widens its budget until it is
                    // exhausted, so run to completion it should land on the
                    // same answer as the exhaustive one.
                    let mut lds = ConTreeLds::new(
                        min_sup,
                        depth,
                        60.0,
                        usize::MAX,
                        PointSelector::Mid,
                        0,
                        false,
                        fast_d2,
                    );
                    let got = lds.fit(&dataset).expect("fit").error();
                    anytime.record(got, optimum, depth, &what);
                    lds_runs += 1;

                    // Every schedule has to converge to the same optimum, for
                    // either selector: `first` is where the split budget applies.
                    for (si, schedule) in ScheduleKind::ALL.into_iter().enumerate() {
                        for (pi, selector) in [PointSelector::Mid, PointSelector::First]
                            .into_iter()
                            .enumerate()
                        {
                            let mut lds = ConTreeLds::new(
                                min_sup,
                                depth,
                                60.0,
                                usize::MAX,
                                selector,
                                0,
                                true,
                                fast_d2,
                            )
                            .with_schedule(schedule);
                            let got = lds.fit(&dataset).expect("fit").error();
                            schedules[si][pi].record(got, optimum, depth, &what);
                        }
                    }
                }
            }
        }
    }

    let per_solver = runs / 2;
    println!("Differential against brute-force enumeration:\n");
    exhaustive.report("ConTree", per_solver);
    specialized.report("ConTree --fast-d2", per_solver);
    exhaustive_first.report("ConTree first+Gini", runs);
    anytime.report("ConTreeLds (both)", lds_runs);
    for (si, schedule) in ScheduleKind::ALL.iter().enumerate() {
        for (pi, selector) in ["mid", "first"].iter().enumerate() {
            schedules[si][pi].report(&format!("LDS+Gini {schedule} {selector}"), lds_runs);
        }
    }
}
