# Roadmap

Open work on the contree crate, roughly in order of priority.

## Anytime search

- **Adaptive budget schedule.** The fixed schedules grow both budgets one step
  at a time, which can stall for a long time when the best split is ranked
  low by the heuristic. A schedule that grows only the budget that actually
  truncated the last pass, and grows faster when passes stop improving,
  should avoid this. `PassReport` already reports which budget cut a pass.
- **Bound the cost of a `mid` pass.** After the first pass, the `mid`
  selector ignores the split budget, so a single pass can run for a long
  time without reporting anything. One option is to restrict each node to
  its best `s + 1` thresholds by Gini, as `first` does.
- **Incremental re-search.** A pass searches truncated subproblems again from
  scratch (with a warm incumbent). Skipping the features and thresholds an
  earlier pass already exhausted would save work across passes.
- **Discrepancy accounting.** Feature rank counts as discrepancy and adds up
  along the path, while split rank is a separate cap. A single budget, or
  depth-weighted discrepancies, are worth trying.
- **Pruning with a minimum support above 1.** Interval pruning is disabled in
  that case, because its monotonicity argument does not hold under a support
  constraint. A support-aware version of the rules would restore the speed.

## Exact search

- **One search with two policies.** `ConTree` and `ConTreeLds` duplicate most
  of the node expansion; a single search with an exhaustive or budgeted
  policy would remove that.
- **Speed.** The exact search is still slower than the reference C++
  implementation. Profile before optimising.
- **Lower-bound reuse.** A cached entry whose proven lower bound reaches the
  current upper bound can be answered immediately. The anytime search does
  this; the exact search does not yet.
- **Cheaper splits.** `DataView::split` copies the index structure of every
  feature at each node. A reversible, trail-based structure would avoid it.

## Housekeeping

- `tests/predict.rs` takes about two minutes in debug builds (the runs with a
  minimum support of 10 and 50 have no interval pruning); trim it.
- Python: a text export of `tree_` compatible with scikit-learn's
  `plot_tree`.
