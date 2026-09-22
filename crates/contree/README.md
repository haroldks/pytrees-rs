# contree

Optimal decision trees on **continuous** features, in Rust.

The crate provides two searches:

- **`ConTree`**, an exact branch-and-bound search. Candidate thresholds lie
  between consecutive values of each feature, subproblems are cached by the
  set of instances they cover, depth-2 subtrees have a dedicated solver, and
  whole intervals of thresholds are pruned at once. This is the algorithm of
  Brită, van der Linden and Demirović (AAAI 2025).
- **`ConTreeLds`**, an anytime version of the same search. It runs in passes
  of growing limited discrepancy budget, so a good tree is available after a
  fraction of the time the exact search needs, and the last pass still
  proves optimality. See *Anytime Optimal Decision Tree Learning with
  Continuous Features* (Kiossou, Schaus and Nijssen, ECML PKDD 2026,
  [arXiv:2601.14765](https://arxiv.org/abs/2601.14765)).

The Python package `pytrees` wraps both as `ConTreeClassifier`. Its command
line front end is the `con-tree` binary in `crates/contree-cli`.

## Using the library

```rust
use contree::algorithms::ConTree;
use contree::common::{PointSelector, SearchConfig, SearchStatus};
use contree::data::Dataset;

// Row-major values and dense labels 0..k, as in a numpy (n, d) array.
let dataset = Dataset::from_rows(&values, &labels, n_features)?;

let config = SearchConfig::new(
    1,                 // min_sup: minimum instances per leaf
    3,                 // max_depth
    600.0,             // time limit in seconds
    0,                 // max_gap: 0 for an exact search
    usize::MAX,        // initial upper bound on the error
    false,             // order features and thresholds by Gini
    true,              // use the depth-2 solver
    PointSelector::Mid,
);
let outcome = ConTree::with_config(config).fit(&dataset)?;

if outcome.status == SearchStatus::Optimal {
    println!("optimal tree, {} training errors", outcome.error());
}
let predictions = outcome.tree.predict(&test_values, n_features)?;
```

`fit` returns the tree, the search statistics, and why the search stopped.
Only `SearchStatus::Optimal` means the tree is proven best for the given
depth and minimum support; `TimeLimit` and `BudgetExhausted` mean the search
ran out of time or budget first.

For the anytime search, build a `ConTreeLds` the same way. Its
`with_schedule` method chooses how the budget grows between passes
(`ScheduleKind::Diagonal` or `ScheduleKind::Square`), and `trajectory()`
returns every improvement of the tree as `(seconds, error)`.

A split sends an instance left when `x[feature] <= threshold`, as in
scikit-learn.

## Reading datasets

`DataReader` reads text files with one instance per line, whitespace
separated, and the label in the first column:

```text
0 5.1 3.5 1.4 0.2
1 7.0 3.2 4.7 1.4
```

Labels must be non-negative integers. `with_format`, `with_label_column`,
`with_headers` and `with_comment_char` adapt the reader to other layouts.
The returned dataset is ready to fit.

## Command line

```bash
cargo run --release -p contree-cli -- \
    --input data.txt --depth 3 --sort-by-heuristic \
    --print-tree --print-stats --result-dir .
```

`--use-lds` switches to the anytime search, and `--budget-schedule` picks its
schedule. `con-tree --help` lists every option.

## Testing

```bash
cargo test -p contree-rs
```

Beyond unit tests, two integration tests carry most of the weight:

- `tests/exact.rs` compares every search configuration with a brute-force
  optimum on small random instances.
- `tests/predict.rs` checks that each returned tree classifies its training
  data with exactly the error the search reports.

`tests/baseline/` holds a regression baseline over larger datasets (not
included in the repository): `capture.sh` runs the configurations of
`matrix.txt`, and `compare.py` reports changed errors, trees and counters.
`bench/anytime/` contains the scripts used to measure anytime behaviour.

The optional `profiling` feature adds probes for the
[`coz`](https://github.com/plasma-umass/coz) causal profiler.
