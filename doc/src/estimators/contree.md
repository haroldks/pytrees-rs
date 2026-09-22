# ConTreeClassifier

`ConTreeClassifier` finds the decision tree of at most `max_depth` levels with
the fewest training errors, directly on continuous features: no binarisation
is needed. Candidate thresholds lie halfway between consecutive distinct
values of each feature, and a row goes left when `x[feature] <= threshold`.

The exact search is ConTree, a depth-first branch-and-bound. Its key idea is
that moving a threshold only moves rows from one child to the other, so the
errors already computed for some thresholds bound the errors of their
neighbours, and whole intervals of thresholds can be skipped at once.

```python
from pytrees import ConTreeClassifier

clf = ConTreeClassifier(max_depth=3, min_sup=5, max_time=60).fit(X, y)
clf.status_        # "optimal", "time_limit", ...
```

## Anytime search

The exact search completes the left subtree of a split before looking at the
right one, so when the time limit is short it can end with a poor tree. With
`use_lds=True` (or `fit_anytime`), the search runs in passes of growing
*limited discrepancy* budget instead: the first pass only follows the features
and thresholds ranked best by the Gini index, and each pass allows more
deviations from that ranking. A good tree is available after the first passes,
and a pass that completes without being cut proves the tree optimal.

```python
clf = ConTreeClassifier(max_depth=5, sort_by_heuristic=True, split_selection="first")
clf.fit_anytime(X, y, callback=lambda error, seconds, status: print(seconds, error))
```

Set `sort_by_heuristic=True` with the anytime search, since the discrepancy
budget follows the Gini ranking.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `3` | Maximum depth of the tree. The cost grows steeply with it. |
| `min_sup` | `1` | Minimum number of training rows in each leaf. Interval pruning is only used when this is 1, so larger values make the search slower. |
| `max_error` | `None` | Stop once a tree with at most this many errors is found. `None` searches for the optimum. |
| `max_time` | `600.0` | Seconds before the search stops with the best tree found so far. |
| `max_gap` | `0` | Accept a tree within this many errors of the optimum. Larger values prune more and finish sooner. |
| `split_selection` | `"mid"` | Which candidate threshold of an interval is evaluated next: `"mid"` bisects the interval, `"first"` tries thresholds one at a time (in Gini order with `sort_by_heuristic`), `"random"` picks one at random. |
| `sort_by_heuristic` | `False` | Try features and thresholds in Gini order. Usually finds a good tree sooner, which prunes more. |
| `fast_d2` | `True` | Solve depth-2 subtrees with a specialised solver. It is exact and much faster than the general search. |
| `use_lds` | `False` | Use the anytime search in `fit`. |
| `budget_schedule` | `"diagonal"` | How the anytime search widens its budgets between passes. It has two budgets, the discrepancy over features and the number of thresholds tried per feature. `"diagonal"` grows their sum by one per pass; `"square"` completes every budget pair up to `(k, k)` before moving to `k + 1`. |
| `random_state` | `None` | Seed for `split_selection="random"`. |

## Fitted attributes

| Attribute | Description |
|---|---|
| `classes_`, `n_classes_` | The class labels and their number. |
| `n_features_in_` | Number of features seen in `fit`. |
| `tree_` | The [fitted tree](tree.md). |
| `train_error_` | Training misclassifications of the tree. |
| `status_` | `"optimal"`, `"time_limit"`, `"budget_exhausted"` (the anytime search had no larger budget to try) or `"error_bound_reached"` (`max_gap` was reached). Only `"optimal"` means the tree is proven best. |
| `statistics_` | `duration`, `cache_size`, `cache_hits`, `general_solver_calls`, `specialized_solver_calls`, `n_samples`, `n_features`, `error`. |

## References

- C. E. Briţa, J. G. M. van der Linden and E. Demirović. *Optimal
  Classification Trees for Continuous Feature Data Using Dynamic Programming
  with Branch-and-Bound.* AAAI 2025. (The exact search.)
- H. Kiossou, P. Schaus and S. Nijssen. *Anytime Optimal Decision Tree
  Learning with Continuous Features.* ECML PKDD 2026. (The anytime search.)
