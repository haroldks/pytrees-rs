# Status — development paused 2026-09-21

Where the project stands, what changed since `master`, and where to pick it
up. Open work is itemised in [`TODO.md`](TODO.md); how to reproduce the
anytime numbers is in [`bench/anytime/README.md`](bench/anytime/README.md).

## In one paragraph

The crate is now a Cargo workspace with a library (`contree`), a CLI
(`con-tree`) and Python bindings, now part of `pytrees`, whose
`ConTreeClassifier` passes scikit-learn's `check_estimator`. Every solver is
exact against brute-force enumeration: 0 misses on 1,440 cases for the
exhaustive search and 720 per anytime variant, across both split selectors and
every budget schedule. Several bugs were fixed: some from upstream ConTree,
some from the port, some new. The anytime search (LDS) clearly beats ConTree
at depth 5. At depth 4, ConTree's home ground, LDS is level with it. The
exhaustive search is still 2.4–5× slower than upstream.

## Layout

Paths are relative to the repository root. The CI workflows and the
experiment notebooks stayed in the contree-rs repository.

| Path | What |
|---|---|
| `crates/contree` | the library: `ConTree` (exhaustive), `ConTreeLds` (anytime), cache, pruning, depth-2 solver |
| `crates/contree-cli` | the `con-tree` binary |
| `crates/pytrees-py/src/contree.rs`, `python/pytrees/supervised/contree.py` | Python bindings (in `pytrees._native`) and the scikit-learn wrapper |
| `crates/contree/tests/baseline` | behavioural baseline: `capture.sh [--smoke]`, `compare.py`, `check_predictions.py` |
| `crates/contree/tests/exact.rs` | brute-force exactness test, all solvers |
| `crates/contree/examples` | `differential` (miss counts), `anytime`, `lds_passes`, `baseline` |
| `crates/contree/bench/anytime` | anytime sweep and primal-gap analysis |

## What changed since `master`

**Production cleanup.**
- Workspace split, dead modules deleted, clippy and rustfmt clean, MSRV 1.77.
- `Cargo.lock` is committed, and the bulk datasets are no longer tracked.
- Input validation: labels, ragged rows, empty fields, headers, NaN/inf and
  empty files are rejected with errors instead of panics.
- `fit` returns a `Result` with a `SearchStatus` (optimal, time limit, …).
- There is a `predict` path, and a single leaf encoding.
- Allocation-free hot paths (16–18 % off the big exhaustive runs).

**Python.** `ConTreeClassifier(max_depth=3, min_sup=1, max_error=None,
max_time=600.0, max_gap=0, split_selection="mid", sort_by_heuristic=False,
fast_d2=True, use_lds=False, random_state=None, budget_schedule="diagonal")`.
- Supports `fit`, `fit_anytime(callback)`, `predict`, `decision_path`,
  `tree_json_` and pickle.
- Passes the 55 `check_estimator` checks, and 31 pytest tests pass.

**Correctness fixes.** Each is a separate commit with its evidence.
- A1: the anytime search marked a budget-truncated tree as optimal
  (a shadowed `stopped` flag).
- A3: an overflowing "unexplored subtree" sentinel.
- A4–A6: label encoding, the leaf encoding, and `fast_d2`.
- Upstream 0fe6b4d: a neighbourhood-pruning off-by-one.
- Upstream 61ebd49: an upper bound was being treated as an exact answer. Fixed
  with lower bounds, `finalize_lower_bound`, and a cache that stores only
  exact entries.
- Interval pruning is unsound with `min_sup > 1`, so it is switched off there.
- The `first` selector pruned splits it had not ruled out (e82057a). This was
  in the paper's code too.

**Speed.**
- `--fast-d2` (the depth-2 specialisation) is the default; `--no-fast-d2`
  turns it off.
- Subproblems are pruned against the parent's budget, as upstream does. This
  was a large speedup.

**Anytime search (LDS).**
- The budget schedule is pluggable (`BudgetSchedule` trait, `--budget-schedule`,
  `budget_schedule=`). The schedules are `diagonal` (default) and `square`.
  `geometric` was tried and dropped.
- The first budget no longer runs three times.
- Depth-2 subproblems are solved exactly once and reused.
- The cache is budget-aware: it reuses a result searched under at least the
  current budget.

## Anytime results

Average primal gap, 60 s, lower is better. ConTree here is upstream,
logged at every root improvement.

| | depth 5, 16 datasets |
|---|---|
| ConTree-Gini, 55c4349 (paper's version) | 76.5 % |
| ConTree-Gini, 61ebd49 (current upstream) | 78.7 % |
| LDS first+Gini, diagonal | 29.4 % |
| LDS mid+Gini, diagonal | 32.6 % |

At depth 4 (6 datasets), `mid`+Gini is level with ConTree at 60 s and ahead of
it before that. `first`+Gini wins only at 5 s, and its weakness is a stall:
on bank d3 it sits at 31 errors (optimum 23) for the whole minute.

The paper's ConTree baseline printed once per completed root feature rather
than at every improvement, which overstated ConTree's gap. The depth-4 claim
needs re-running with fair logging. Depth 5 holds either way.

## Where to pick up

In rough priority order (details in `TODO.md`):

1. **Adaptive budget schedule.** Grow only the dimension that cut the last
   pass. This targets `first`'s stall without `geometric`'s blow-up.
   `PassReport` already carries the signal.
2. **Bound the cost of a `mid` pass.** A single pass can run 26 s with
   nothing to report.
3. **Re-run the paper's anytime comparison** with fair ConTree logging at
   depths 4–8, and check whether e82057a affects any published `first` result.
4. **Close the speed gap with upstream.** Profile first. Also check the cost
   of the bound fixes without `--fast-d2`: the results are the same, but it
   does several times the work.
5. **Consolidation step 2:** unify the two solvers behind a search policy.
6. **Docs:** the README does not yet cover `--no-fast-d2`, `--budget-schedule`
   or the `fast_d2=True` default.

## Baseline

`tests/baseline/expected/` was re-frozen at the merge. A full capture against
the old records showed 0 error changes, 13 ties broken differently and 25
counter changes. Two entries without `--fast-d2` no longer finish in 2 minutes
and were dropped, which leaves 50.

## Checking a change

```sh
cargo test -p contree-rs --release              # includes exact.rs
cargo run --release -p contree-rs --example differential   # miss counts, all solvers
cargo build --release -p contree-rs --examples \
  && crates/contree/tests/baseline/capture.sh --smoke /tmp/b \
  && python3 crates/contree/tests/baseline/compare.py /tmp/b   # ~15 s; full matrix before merging
maturin develop --release && pytest python/tests
```
