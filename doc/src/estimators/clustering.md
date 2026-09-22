# DL85Cluster

`DL85Cluster` clusters rows with a decision tree: the tree splits the rows on
binary features, and each leaf is a cluster. DL8.5 searches for the tree whose
clusters are tightest, by default the one that minimises the sum of the
Euclidean distances from each row to the centroid of its cluster. The result
describes each cluster by a short rule, the path to its leaf.

```python
import numpy as np
from pytrees import DL85Cluster

X = np.random.default_rng(0).integers(0, 2, size=(200, 8))
model = DL85Cluster(max_depth=2, min_sup=10).fit(X)
model.labels_          # the cluster of each row
model.n_clusters_      # at most 2 ** max_depth
print(model.to_dot())
```

The tree must split on binary features, but the distances can be measured in
other data: pass `X_error` to `fit`, with one row per row of `X`. This lets you
describe clusters of continuous measurements with interpretable binary
attributes:

```python
model = DL85Cluster(max_depth=3, min_sup=20).fit(X_binary, X_error=X_measurements)
```

## Parameters

`min_sup`, `max_depth`, `max_error`, `max_time`, `always_sort`, `heuristic`,
`similarity_lb`, `dynamic_branching` and the [search rules](rules.md) work as
for [DL85Classifier](dl85.md); here `min_sup` is the minimum cluster size.

| Parameter | Default | Description |
|---|---|---|
| `error_function` | `None` | `error_function(indices) -> (error, value)`, called with the row indices of a candidate cluster. Replaces the distance to the centroid. `value` is ignored. |

## Fitted attributes

| Attribute | Description |
|---|---|
| `labels_` | Cluster of each training row, from 0 to `n_clusters_ - 1`. |
| `n_clusters_` | Number of clusters (leaves). |
| `tree_` | The [fitted tree](tree.md); each leaf's `value` is its cluster number. |
| `train_error_` | Error of the clustering, as the error function measures it. |
| `status_` | `"optimal"` or `"time_limit"`. |

`predict(X)` assigns new rows to clusters, and `fit_predict(X)` does both
steps at once.
