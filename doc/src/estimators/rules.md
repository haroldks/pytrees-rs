# Anytime search rules

The rules in `pytrees.rules` turn DL8.5 into an anytime search. Each rule
restricts a pass of the search to part of the search space, usually the part
the heuristic considers most promising. When a pass ends and a rule has cut
something, the search restarts with every rule relaxed (its budget widened),
reusing everything it cached. When a pass runs without any cut, the tree is
optimal.

This gives a good tree after the first, cheap passes, and the optimal tree if
the search has time to finish. It is the framework of CA-DL8.5 (a complete
anytime beam search), which generalises LDS-DL8.5 and Top-k-DL8.5.

Rules are passed to `DL85Classifier` or `DL85Cluster` by keyword, and several
can be combined:

```python
from pytrees import DL85Classifier
from pytrees.rules import DiscrepancyRule, TopKRule

clf = DL85Classifier(
    max_depth=5,
    heuristic="information_gain",   # the rules follow the heuristic's ranking
    discrepancy=DiscrepancyRule(),
    max_time=120,
)
```

Rules work with the feature ranking, so set a `heuristic`: with
`heuristic="none"` the ranking is just the column order. When a rule is given,
`similarity_lb` and `dynamic_branching` are turned off.

## How budgets grow

Several rules have a budget that grows at each restart. `step_strategy` and
`base` say how:

| `step_strategy` | Budgets |
|---|---|
| `"monotonic"` | `0, base, 2·base, 3·base, …` |
| `"exponential"` | `1, base, base², base³, …` |
| `"luby"` | running sums of the Luby sequence `1, 1, 2, 1, 1, 2, 4, …`, scaled by `base` |

Monotonic growth gives the finest control; exponential and Luby growth reach
large budgets in fewer passes.

## DiscrepancyRule

Limited discrepancy search (LDS-DL8.5). At each node, the features are ranked
by the heuristic, and choosing the feature of rank `i` costs `i`
discrepancies. A pass only explores trees whose total cost, summed along the
path from the root, is within the budget. The first pass therefore follows the
heuristic exactly, like a greedy learner, and each later pass allows more
deviations.

| Parameter | Default | Description |
|---|---|---|
| `initial_value` | `0` | Budget of the first pass. |
| `limit` | `None` | Largest budget; `None` grows it until the search is complete. |
| `step_strategy`, `base` | `"monotonic"`, `1` | How the budget grows. |

## TopKRule

Top-k search. A pass with budget `k` only branches on the `k + 1` best-ranked
features at each node, and `k` grows at each restart. Unlike the discrepancy
budget, it applies to every node separately rather than to a whole path.

| Parameter | Default | Description |
|---|---|---|
| `initial_value` | `0` | `k` on the first pass. |
| `limit` | `None` | Largest `k`; `None` grows it to every feature. |
| `step_strategy`, `base` | `"monotonic"`, `1` | How `k` grows. |

## GainRule

Choosing a feature other than the best-ranked one loses the difference
between their heuristic scores. A pass skips the paths whose accumulated loss
exceeds a gap, and the gap widens at each restart. Where the discrepancy rule
counts ranks, this rule measures how much worse the alternatives actually are.

| Parameter | Default | Description |
|---|---|---|
| `min_gain` | `0.0` | Gap of the first pass. |
| `epsilon` | `1e-4` | Step by which the gap widens. |
| `limit` | `6.0` | Largest gap; usually about the maximum depth. |
| `step_strategy`, `base` | `"monotonic"`, `1` | How the gap grows. |

## PurityRule

A pass does not split nodes whose share of correctly classified rows is at
least a threshold; the threshold rises at each restart until every node may be
split.

| Parameter | Default | Description |
|---|---|---|
| `min_purity` | `0.0` | Threshold of the first pass. |
| `epsilon` | `1e-4` | Amount the threshold rises by at each restart. |

## RestartRule

Ends each pass after `limit` seconds and starts a new one, keeping the cache.
Combined with a heuristic ordering, this periodically sends the search back
to the most promising part of the space.

| Parameter | Default | Description |
|---|---|---|
| `limit` | `1.0` | Seconds per pass. |

## References

- H. Kiossou, P. Schaus, S. Nijssen and V. R. Houndji. *Time Constrained DL8.5
  Using Limited Discrepancy Search.* ECML PKDD 2022.
- H. Kiossou and P. Schaus. *A Generic Complete Anytime Beam Search for
  Optimal Decision Tree.* IDA 2026.
