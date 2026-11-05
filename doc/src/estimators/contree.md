# ConTreeClassifier

`ConTreeClassifier` finds the decision tree of at most `max_depth` levels with
the fewest training errors, directly on continuous features: no binarisation
is needed. The error can be replaced by any cost you define for a leaf
([`error_function`](#custom-error-functions)). Candidate thresholds lie halfway between consecutive distinct
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
| `max_error` | `None` | Stop once a tree with at most this error is found. `None` searches for the optimum. |
| `max_time` | `600.0` | Seconds before the search stops with the best tree found so far. |
| `max_gap` | `0` | Accept a tree within this error of the optimum. Larger values prune more and finish sooner. |
| `split_selection` | `"mid"` | Which candidate threshold of an interval is evaluated next: `"mid"` bisects the interval, `"first"` tries thresholds one at a time (in Gini order with `sort_by_heuristic`), `"random"` picks one at random. |
| `sort_by_heuristic` | `False` | Try features and thresholds in Gini order. Usually finds a good tree sooner, which prunes more. |
| `fast_d2` | `True` | Solve depth-2 subtrees with a specialised solver. It is exact and much faster than the general search. Not used with `error_function_input="indices"`. |
| `use_lds` | `False` | Use the anytime search in `fit`. |
| `budget_schedule` | `"diagonal"` | How the anytime search widens its budgets between passes. It has two budgets, the discrepancy over features and the number of thresholds tried per feature. `"diagonal"` grows their sum by one per pass; `"square"` completes every budget pair up to `(k, k)` before moving to `k + 1`. |
| `random_state` | `None` | Seed for `split_selection="random"`. |
| `error_function` | `None` | A custom cost for the leaves, see [below](#custom-error-functions). |
| `error_function_input` | `"class_counts"` | What `error_function` receives: `"class_counts"` (the number of rows of each class in the leaf) or `"indices"` (the rows in the leaf). |
| `threshold_skipping` | `"auto"` | Whether the search skips thresholds it can prove useless: `"auto"`, `True` or `False`. See [below](#threshold-skipping). |

## Fitted attributes

| Attribute | Description |
|---|---|
| `classes_`, `n_classes_` | The class labels and their number. |
| `n_features_in_` | Number of features seen in `fit`. |
| `tree_` | The [fitted tree](tree.md). |
| `train_error_` | Training error of the tree, a float: its misclassifications, or the sum of its leaf costs with an `error_function`. |
| `status_` | `"optimal"`, `"time_limit"`, `"budget_exhausted"` (the anytime search had no larger budget to try), `"error_bound_reached"` (`max_gap` was reached) or `"not_proven"` (threshold skipping was forced and the `error_function` broke its assumption). Only `"optimal"` means the tree is proven best. |
| `statistics_` | `duration`, `cache_size`, `cache_hits`, `general_solver_calls`, `specialized_solver_calls`, `n_samples`, `n_features`, `error`, `skipping_violations`. |

## Custom error functions

As in [DL8.5](dl85.md#custom-error-functions), the search finds the tree that
minimises the sum of the costs of its leaves, for any cost you define for a
leaf. Pass it as `error_function`, a function that receives the content of a
leaf as a list and returns `(error, prediction)`: a non-negative number, and
the index of the predicted class in `classes_`.

What the function receives depends on `error_function_input`:

- `"class_counts"` (the default): the number of rows of each class in the
  leaf, in the order of `classes_`, as ints.
- `"indices"`: the indices of the rows of the leaf in `X`. Wrap the list with
  `np.asarray` to index arrays, as in `y[np.asarray(rows)]`.

Misclassifying class 2 can cost ten times more, for instance:

```python
import numpy as np
from pytrees import ConTreeClassifier

costs = np.array([1.0, 1.0, 10.0])   # cost of misclassifying a row of each class

def cost_sensitive(class_counts):
    counts = np.asarray(class_counts)
    per_prediction = [(costs * counts).sum() - costs[k] * counts[k] for k in range(len(counts))]
    best = int(np.argmin(per_prediction))
    return per_prediction[best], best

clf = ConTreeClassifier(max_depth=3, error_function=cost_sensitive).fit(X, y)
```

A few things change with a custom error:

- Threshold skipping is off by default (see below), so the search is slower.
- With `"indices"`, the depth-2 solver is not used, because it works from
  class counts.
- The function is called from Rust for every leaf the search evaluates, often
  thousands of times, so keep it cheap. An exception raised inside it stops
  the search and is raised again by `fit`. A negative or NaN error, or a
  prediction that is not the index of a class, is a `ValueError`.
- `train_error_` is the cost your function measured.

## Threshold skipping

Most of ConTree's speed comes from skipping thresholds without evaluating
them: interval pruning in the general search, and skipping in the depth-2
solver's scan. Both rest on one assumption, which the built-in error meets:
moving `n` rows into a leaf never lowers its error and raises it by at most
`n`, and moving them out never raises it and lowers it by at most `n`.

`threshold_skipping` chooses whether the search relies on it:

- `"auto"` (the default): on for the built-in error, off for an
  `error_function`, which may not meet the assumption.
- `True`: on, also for an `error_function`. Use it when you know your function
  meets the assumption, as misclassification does. `fit` warns with
  `ThresholdSkippingWarning`. The depth-2 solver then checks the assumption on
  the leaves it evaluates; if your function breaks it, `fit` warns again,
  `statistics_["skipping_violations"]` counts the cases, and `status_` is
  `"not_proven"`. The check catches a function that breaks the assumption; it
  cannot prove one correct.
- `False`: off. Slower, and never changes the result for the built-in error.

With `min_sup > 1`, interval pruning is never used, whatever the setting: its
argument does not hold when a leaf must keep a minimum number of rows. `True`
then forces only the depth-2 solver's skipping, which does not depend on
`min_sup`.


## References

- C. E. Briţa, J. G. M. van der Linden and E. Demirović. *Optimal
  Classification Trees for Continuous Feature Data Using Dynamic Programming
  with Branch-and-Bound.* AAAI 2025. (The exact search.)
- H. Kiossou, P. Schaus and S. Nijssen. *Anytime Optimal Decision Tree
  Learning with Continuous Features.* ECML PKDD 2026. (The anytime search.)
