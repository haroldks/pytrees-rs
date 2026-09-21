# TODO

Open work, roughly in priority order. Each item says why it is open and where
the evidence is.

## Anytime search (LDS)

- **Adaptive budget growth.** Fixed schedules trade one failure for another:
  unit steps (`diagonal`, `square`) stall when the good split is ranked low —
  bank d3 with `first` sits at 31 (optimum 23) for 60 s — while a geometric
  schedule (split budget doubling each pass; tried in 3affacb, removed since)
  reaches it in 3 s but loses on average, because doubling the split budget
  makes passes grow multiplicatively with depth (bidding d5 ends at 15 errors
  against 4). Grow only the dimension that actually cut the last pass, and
  faster only when passes stop improving. `PassReport` already carries
  `improved`, `cut_by_discrepancy` and `cut_by_split`; this is a new
  `BudgetSchedule`, no solver change.
- **Bound the cost of a `mid` pass.** After its first pass `mid` ignores the
  split budget, so a single pass can run 26 s with nothing to report (wilt d4).
  Option: restrict each node's interval search to its top `s + 1` splits by
  Gini, like `first`. `mid` is the variant level with ConTree at depth 4, so
  measure against it carefully.
- **Incremental re-search.** The budget-aware cache (30d75af) reuses results
  whose budget covers a revisit, but under nested budgets a revisit always has
  a *larger* budget, so truncated subproblems are searched again from scratch
  (with a warm incumbent). Skipping the features/splits an earlier pass
  already exhausted is the remaining cross-pass saving.
- **Discrepancy accounting.** Feature rank counts as discrepancy and
  accumulates down the tree; split rank is a separate uniform cap. Options:
  one unified budget (taking the i-th split spends i), or depth-weighted
  discrepancies (Walsh's DDS).
- **Old budget strategies.** `Lexicographic`, `GeometricBoth`,
  `FeaturePriority` from 8958439 (removed in bfd53ac), as extra `ScheduleKind`s.
- **Pruning with `min_sup > 1`.** Interval pruning is switched off entirely
  there (`IntervalsPruner::is_sound`) because its monotonicity argument does
  not survive a support constraint. A support-aware version of the rules
  would restore the speed.

## Evaluation (paper)

- Re-run the anytime comparison on the full suite with ConTree logged at every
  root improvement (upstream's `PRINT_INTERMEDIARY_TIME_SOLUTIONS`), not once
  per completed root feature as in `results_27122025/contree_heuristic_*`.
  Done so far on 6 datasets at depth 4 and all 16 at depth 5: LDS wins
  decisively at depth 5 either way; at depth 4 `mid`+Gini is level with
  ConTree at 60 s and ahead before. Depths 6–8 not yet re-run.
- `first` runs before e82057a could report a suboptimal tree as optimal once
  the budget reached full width (pruning-range bug). Worth checking whether
  any `first_heuristic_yes` result in the paper was affected.
- How to reproduce these numbers: `bench/anytime/README.md`.

## Exhaustive search (ConTree)

- **Consolidation: unify the two solvers.** `ConTree` and `ConTreeLds`
  duplicate most of the node expansion, and every fix so far had to be made
  in both; one search with a policy (exhaustive or budgeted) would remove that.
- **Speed gap with upstream.** 2.4–5× slower with `--fast-d2` after 0be232f:
  1.3–2.6× more solver calls, 1.5–2× slower per call. `reduce_node_budget` is
  not ported; the per-call cost needs a profiler run before any guess.
- **Lower-bound reuse.** A cached entry whose proven lower bound is ≥ the
  current upper bound can return immediately (the DL8.5 rule). The anytime
  search does this now; the exhaustive search does not.
- **`DataView::split` copies the whole index structure per node.** A
  reversible/trail-based cover is the fix; get `cargo bench` numbers first.

## Housekeeping

- **Cost of the bound fixes without `--fast-d2`.** Same results, several
  times the work: page d2 20k -> 208k solver calls, `page 2 --use-lds` 16 s
  -> over 2 min (dropped from the baseline with `wilt 2`). Confirm which
  commit (5eaf77e or 0be232f) and whether the lower-bound reuse below
  recovers it. The cost column in `tests/baseline/matrix.txt` predates this.
- README: `--no-fast-d2`, `--budget-schedule`, `fast_d2=True` default in
  Python, the `first` selector fix.
- `tests/predict.rs` takes ~125 s (`min_sup` 10 and 50 run without interval
  pruning); trim it.
- Python: a `plot_tree`-compatible adapter or text export for `tree_`.
- Code TODOs left by the original author: `dataset.rs:79` (epsilon),
  `contree_lds.rs` / `continuous_tree.rs` ("larger and smaller tree
  comparison"), `caching/mod.rs:18` (private fields).
