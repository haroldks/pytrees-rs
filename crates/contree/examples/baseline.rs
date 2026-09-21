//! Baseline capture harness.
//!
//! Runs a single (dataset, depth, flags) configuration and writes a JSON record
//! of the resulting tree and the deterministic search counters. Timing is
//! deliberately excluded: it is not reproducible and must not gate a refactor.
//!
//! This exists so that the production-hardening refactor can be verified as
//! behaviour-preserving. Run it at the pre-refactor commit to freeze the
//! expected outputs, then re-run and diff after each change.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::PointSelector;
use contree::data::view::DataView;
use contree::reader::data_reader::DataReader;
use contree::tree::Tree;

struct Args {
    input: PathBuf,
    output: PathBuf,
    depth: usize,
    support: usize,
    max_time: f64,
    fast_d2: bool,
    heuristic: bool,
    lds: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut depth = None;
    let mut support = 1usize;
    let mut max_time = 900.0f64;
    let (mut fast_d2, mut heuristic, mut lds) = (false, false, false);

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--input" | "-i" => input = Some(PathBuf::from(value()?)),
            "--output" | "-o" => output = Some(PathBuf::from(value()?)),
            "--depth" | "-d" => depth = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--support" | "-s" => support = value()?.parse().map_err(|e| format!("{e}"))?,
            "--time-limit" => max_time = value()?.parse().map_err(|e| format!("{e}"))?,
            "--fast-d2" => fast_d2 = true,
            "--sort-by-heuristic" => heuristic = true,
            "--use-lds" => lds = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }

    Ok(Args {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        depth: depth.ok_or("--depth is required")?,
        support,
        max_time,
        fast_d2,
        heuristic,
        lds,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("baseline: {err}");
            return ExitCode::FAILURE;
        }
    };

    let reader = DataReader::default();
    let mut dataset = match reader.read_file(&args.input) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("baseline: reading {}: {err}", args.input.display());
            return ExitCode::FAILURE;
        }
    };
    dataset.sort_features();

    // `error` is the search's own count; `tree` is what it claims to have found.
    // Recording both is the point: they are supposed to agree, and nothing in
    // the crate has ever checked that they do.
    let (error, tree, counters) = if args.lds {
        let mut solver: ConTreeLds = ConTreeLds::new(
            args.support,
            args.depth,
            args.max_time,
            usize::MAX,
            PointSelector::Mid,
            0,
            args.heuristic,
            args.fast_d2,
        );
        let view = DataView::root(&dataset, args.heuristic);
        while !solver.partial_fit(&view) {}
        let stats = *solver.statistics();
        let tree = solver.get_solution_tree();
        (stats.error, tree, counters_of(&stats))
    } else {
        let mut solver: ConTree = ConTree::new(
            args.support,
            args.depth,
            args.max_time,
            usize::MAX,
            PointSelector::Mid,
            0,
            args.heuristic,
            args.fast_d2,
        );
        let outcome = match solver.fit(&dataset) {
            Ok(outcome) => outcome,
            Err(err) => {
                eprintln!("baseline: {err}");
                return ExitCode::FAILURE;
            }
        };
        (
            outcome.statistics.error,
            outcome.tree,
            counters_of(&outcome.statistics),
        )
    };

    let record = serde_json::json!({
        "dataset": args.input.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
        "depth": args.depth,
        "support": args.support,
        "flags": {
            "fast_d2": args.fast_d2,
            "sort_by_heuristic": args.heuristic,
            "use_lds": args.lds,
        },
        "error": error,
        "counters": counters,
        "tree": tree_shape(&tree),
    });

    if let Some(parent) = args.output.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("baseline: creating {}: {err}", parent.display());
            return ExitCode::FAILURE;
        }
    }
    match fs::write(&args.output, format!("{record:#}\n")) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("baseline: writing {}: {err}", args.output.display());
            ExitCode::FAILURE
        }
    }
}

fn counters_of(stats: &contree::common::Statistics) -> serde_json::Value {
    serde_json::json!({
        "cache_size": stats.cache_size,
        "cache_hits": stats.cache_hits,
        "general_solver_call": stats.general_solver_call,
        "specialized_solver_call": stats.specialized_solver_call,
        "num_samples": stats.num_samples,
        "num_features": stats.num_features,
    })
}

/// Normalize the tree to a nested form that survives a change of arena layout.
///
/// The arena indices are an implementation detail and shift whenever node
/// insertion order changes; the *shape* — the feature/threshold at each internal
/// node and the label at each leaf — is the thing that must not change.
fn tree_shape(tree: &Tree) -> serde_json::Value {
    fn walk(tree: &Tree, index: usize, depth: usize) -> serde_json::Value {
        // Guard against the arena's ambiguous "0 means no child" convention and
        // against cycles, which a malformed tree could otherwise turn into a hang.
        if depth > 64 {
            return serde_json::json!("<depth-limit>");
        }
        let Some(node) = tree.get_node(index) else {
            return serde_json::json!(null);
        };
        let (left, right) = (node.left, node.right);
        let feature = node.value.feature;

        // Today a leaf is spelled three different ways depending on which solver
        // produced it: no children, `feature: None`, or the `usize::MAX` sentinel
        // written by the cache path. Collapse all three here so the baseline is
        // stable across the leaf-encoding unification.
        let is_leaf = (left == 0 && right == 0) || feature.is_none() || feature == Some(usize::MAX);

        if is_leaf {
            serde_json::json!({ "leaf": node.value.label, "error": node.value.error })
        } else {
            serde_json::json!({
                "feature": feature,
                "split": node.value.split,
                "error": node.value.error,
                "left": walk(tree, left, depth + 1),
                "right": walk(tree, right, depth + 1),
            })
        }
    }

    if tree.is_empty() {
        return serde_json::json!(null);
    }
    walk(tree, tree.get_root_index(), 0)
}
