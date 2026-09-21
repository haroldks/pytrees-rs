# contree

Optimal decision trees over **continuous** features, by branch and bound.

The search is DL8.5-style: a depth-bounded branch and bound over candidate split
points, with a bitset-keyed cache of solved subproblems, a specialization for
depth-2 subtrees, interval pruning over the candidates of a feature, and an
anytime limited-discrepancy variant.

```
crates/contree/                                 the library      (crate `contree`)
crates/contree/tests/baseline/                  behavioural regression harness
crates/contree/bench/anytime/                   anytime benchmark
crates/contree-cli/                             the `con-tree` binary
crates/contree-py/                              the PyO3 extension module
crates/contree-py/python/pytrees_continuous/    the scikit-learn estimator
```

Paths below are relative to the repository root. The benchmark instances are
not tracked; the commands expect them in `datasets/`, or wherever
`CONTREE_DATASETS` points.

## Build

```bash
cargo build --release -p contree-rs -p contree-cli
```

The library has one optional feature, `profiling`, which turns on the
[`coz`](https://github.com/plasma-umass/coz) probes in the hot path. It is off
by default.

## Command line

```bash
cargo run --release -p contree-cli -- \
    --input datasets/avila.txt \
    --depth 3 \
    --support 1 \
    --fast-d2 \
    --sort-by-heuristic \
    --print-tree --print-stats \
    --result-dir .
```

`--use-lds` switches to the anytime search, which reports improving trees as its
discrepancy and split budgets widen instead of running to the optimum in one go.

## Library

```rust
use contree::algorithms::ConTree;
use contree::common::{PointSelector, SearchStatus};
use contree::data::Dataset;

// Row-major values plus labels: the layout a numpy `(n, d)` array already has.
let dataset = Dataset::from_rows(&values, &labels, n_features)?;

let mut solver = ConTree::new(
    /* min_sup */ 1,
    /* max_depth */ 3,
    /* max_time */ 600.0,
    /* max_error */ usize::MAX,
    PointSelector::Mid,
    /* max_gap */ 0,
    /* sort_by_heuristic */ false,
    /* fast_d2 */ true,
);

let outcome = solver.fit(&dataset)?;
assert_eq!(outcome.status, SearchStatus::Optimal);

let predictions = outcome.tree.predict(&test_values, n_features)?;
```

`fit` returns a `FitOutcome`: the tree, the search counters, and **why the
search stopped**. Only `SearchStatus::Optimal` means the tree is proven best for
the given depth and support; `TimeLimit` and `BudgetExhausted` mean the search
ran out of something first.

## Python

```bash
pip install maturin
maturin develop --release -m crates/contree-py/Cargo.toml   # or: maturin build
```

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from pytrees_continuous import ConTreeClassifier

X, y = load_iris(return_X_y=True)

clf = ConTreeClassifier(max_depth=3, min_sup=5, fast_d2=True).fit(X, y)
clf.status_        # "optimal" -- anything else means the search ran out of something
clf.train_error_   # misclassifications on the training set
clf.tree_          # children_left / children_right / feature / threshold / value

cross_val_score(clf, X, y, cv=5)
```

`ConTreeClassifier` passes `sklearn.utils.estimator_checks.check_estimator`, so
it clones, pickles, and drops into `Pipeline` and `GridSearchCV` unchanged. `y`
may be anything `np.unique` accepts -- strings, non-contiguous integers -- and
`classes_` maps back to it; the Rust core keeps its dense `0..k` contract.

`predict` runs the whole batch in Rust, and `fit` releases the GIL for the
duration of the search.

Because the search is exact it can take a long time, and the anytime variant
exists to make that bearable:

```python
clf = ConTreeClassifier(max_depth=4, fast_d2=True)
clf.fit_anytime(X, y, callback=lambda error, seconds, status:
                print(f"{seconds:6.2f}s  error={error}  ({status})"))
```

## Two conventions that are easy to get wrong

**The label is column 0.** `DataReader` reads whitespace-separated text with the
label first and the features after it, and labels must be a dense integer
encoding `0..k`. `with_label_column` moves it.

**A split routes left when `x[feature] < threshold`,** and right otherwise. This
is the rule the search itself partitions by. `crates/contree/tests/baseline/check_predictions.py`
and `crates/contree/tests/predict.rs` both pin it: every tree in the baseline
must classify its training set with exactly the error the search reported.

## Tests

```bash
cargo test -p contree-rs -p contree-cli -p contree-py   # Rust
pytest crates/contree-py/python/tests                   # Python, after `maturin develop`
```

Four of them carry most of the weight:

- `tests/predict.rs` — the tree a search returns must reproduce the error the
  search reports, on every instance. The search counts its error while
  exploring; the tree is rebuilt from the cache afterwards, and nothing used to
  check that the two agree.
- `tests/exact.rs` — a differential test against a brute-force optimum on small
  random instances. It asserts the search never reports an error *below* what
  any tree of that shape can achieve, and pins how often it misses the optimum
  so the gap can shrink but not grow.
- `python/tests/test_sklearn_api.py` — runs `check_estimator` in full, plus
  the contracts it does not cover: the reported error must match the
  predictions, labels must survive a round trip, the anytime callback must see
  monotonically improving trees.
- `tests/baseline/` — a 52-configuration behavioural baseline over the real
  datasets. `capture.sh --smoke` runs a 19-configuration subset in ~12 seconds
  for iterating; the full matrix takes about six minutes.

```bash
cargo build --release -p contree-rs --examples
crates/contree/tests/baseline/capture.sh --smoke /tmp/check
python crates/contree/tests/baseline/compare.py /tmp/check
```

## Known gap

The upper bound a cache entry was proved under (`Entry::ub`) is recorded and
never read, so an entry established only as "no better than UB" is later reused
as if it were exact, and the same value is written into the interval pruner.
Both can prune away a tree that should have been kept. `tests/exact.rs`
demonstrates it and bounds how often it happens; fixing it is a change to the
search's pruning semantics rather than a cleanup.
