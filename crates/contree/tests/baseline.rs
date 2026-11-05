//! The frozen baseline (`tests/baseline/expected/`), checked from `cargo test`.
//!
//! For each configuration of `tests/baseline/matrix.txt` it runs the search
//! as the `baseline` example does and checks that the error, the tree and the
//! work counters are those recorded. It also checks the recorded tree itself,
//! and the new one, against the data (see `common::deep_check`): a tree can
//! be recorded and reproduced exactly and still be wrong.
//!
//! The datasets are not in the repository, so the test is ignored by default:
//!
//!     CONTREE_DATASETS=<dir> cargo test -p contree-rs --test baseline -- --ignored
//!
//! It runs the `#smoke` subset; `BASELINE_FULL=1` runs the whole matrix
//! (about ten minutes in a release build).

mod common;

use std::path::{Path, PathBuf};

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{FitOutcome, PointSelector};
use contree::reader::data_reader::DataReader;

use common::{deep_check, instance_from, misclassified, Node};

/// One line of `matrix.txt`: `<dataset> <depth> <flags...>   [#tags]`.
struct Entry {
    name: String,
    dataset: String,
    depth: usize,
    fast_d2: bool,
    heuristic: bool,
    lds: bool,
}

fn matrix(full: bool) -> Vec<Entry> {
    let text = std::fs::read_to_string(baseline_dir().join("matrix.txt")).expect("matrix.txt");
    let mut entries = Vec::new();
    for line in text.lines() {
        let (line, tags) = line.split_once('#').unwrap_or((line, ""));
        let words: Vec<&str> = line.split_whitespace().collect();
        let [dataset, depth, flags @ ..] = words.as_slice() else {
            continue;
        };
        if !full && !tags.contains("smoke") {
            continue;
        }
        // Same file name as capture.sh: the flags without their dashes.
        let slug: String =
            flags
                .iter()
                .map(|f| f.trim_start_matches('-'))
                .fold(String::new(), |mut acc, f| {
                    acc.push('-');
                    acc.push_str(f);
                    acc
                });
        entries.push(Entry {
            name: format!("{dataset}-d{depth}{slug}"),
            dataset: dataset.to_string(),
            depth: depth.parse().expect("a depth"),
            fast_d2: flags.contains(&"--fast-d2"),
            heuristic: flags.contains(&"--sort-by-heuristic"),
            lds: flags.contains(&"--use-lds"),
        });
    }
    entries
}

fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/baseline")
}

fn datasets_dir() -> PathBuf {
    std::env::var_os("CONTREE_DATASETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets"))
}

/// The search the `baseline` example runs for this entry.
fn run(entry: &Entry, path: &Path) -> (contree::data::Dataset, FitOutcome) {
    let mut dataset = DataReader::default()
        .read_file(path)
        .expect("readable dataset");
    dataset.sort_features();
    let outcome = if entry.lds {
        ConTreeLds::new(
            1,
            entry.depth,
            900.0,
            f64::INFINITY,
            PointSelector::Mid,
            0.0,
            entry.heuristic,
            entry.fast_d2,
        )
        .fit(&dataset)
    } else {
        ConTree::new(
            1,
            entry.depth,
            900.0,
            f64::INFINITY,
            PointSelector::Mid,
            0.0,
            entry.heuristic,
            entry.fast_d2,
        )
        .fit(&dataset)
    };
    (dataset, outcome.expect("fit"))
}

#[test]
#[ignore = "needs the datasets: set CONTREE_DATASETS and pass --ignored"]
fn the_baseline_trees_are_reproduced_and_correct() {
    let full = std::env::var_os("BASELINE_FULL").is_some();
    let datasets = datasets_dir();
    assert!(
        datasets.is_dir(),
        "no datasets at {}; set CONTREE_DATASETS",
        datasets.display()
    );

    let mut wrong = Vec::new();
    let entries = matrix(full);
    for entry in &entries {
        let record_path = baseline_dir()
            .join("expected")
            .join(format!("{}.json", entry.name));
        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record_path).expect("a record"))
                .expect("valid JSON");
        let support = record["support"].as_u64().expect("support") as usize;
        let (dataset, outcome) = run(entry, &datasets.join(format!("{}.txt", entry.dataset)));
        let instance = instance_from(&dataset);
        let cost = |rows: &[usize], label| misclassified(&instance, rows, label);
        let name = &entry.name;

        // The recorded tree, on its own: its cost recomputed from the data
        // must be the error recorded next to it.
        let recorded = Node::from_json(&record["tree"]);
        let recorded_error = record["error"].as_f64().expect("error");
        match deep_check(&recorded, &instance, entry.depth, support, &cost) {
            Err(why) => wrong.push(format!("{name}: the recorded tree: {why}")),
            Ok(cost) if cost != recorded_error => wrong.push(format!(
                "{name}: the recorded tree costs {cost}, its record says {recorded_error}"
            )),
            Ok(_) => {}
        }

        // The new tree, on its own.
        let Some(tree) = Node::from_tree(&outcome.tree) else {
            wrong.push(format!("{name}: no tree"));
            continue;
        };
        match deep_check(&tree, &instance, entry.depth, support, &cost) {
            Err(why) => wrong.push(format!("{name}: the new tree: {why}")),
            Ok(cost) if cost != outcome.error() => wrong.push(format!(
                "{name}: the new tree costs {cost}, the search says {}",
                outcome.error()
            )),
            Ok(_) => {}
        }

        // The new run against the record.
        if outcome.error() != recorded_error {
            wrong.push(format!(
                "{name}: error {} instead of {recorded_error}",
                outcome.error()
            ));
        }
        if let Some(difference) = tree.first_difference(&recorded, 0.0, "root ") {
            wrong.push(format!("{name}: new vs recorded tree: {difference}"));
        }
        let counters = &record["counters"];
        let stats = &outcome.statistics;
        for (key, value) in [
            ("cache_size", stats.cache_size),
            ("cache_hits", stats.cache_hits),
            ("general_solver_call", stats.general_solver_call),
            ("specialized_solver_call", stats.specialized_solver_call),
            ("num_samples", stats.num_samples),
            ("num_features", stats.num_features),
        ] {
            if counters[key].as_u64() != Some(value as u64) {
                wrong.push(format!(
                    "{name}: {key} {value} instead of {}",
                    counters[key]
                ));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "{} problem(s) over {} configurations:\n{}",
        wrong.len(),
        entries.len(),
        wrong.join("\n")
    );
}
