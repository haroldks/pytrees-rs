# The fitted tree

Every estimator stores its tree in `tree_`, an instance of
`pytrees.tree.Tree`. It uses the layout of scikit-learn's own trees: one entry
per node in flat NumPy arrays, node 0 being the root.

| Array | Description |
|---|---|
| `children_left`, `children_right` | Indices of the children of each node, `-1` at a leaf. |
| `feature` | Feature tested by each node, `-1` at a leaf. |
| `threshold` | Threshold of each test, `NaN` at a leaf. A row goes left when `x[feature] <= threshold`. On binary features the threshold is 0.5, so 0 goes left. |
| `value` | What each leaf predicts, `-1` at an internal node: an index into the estimator's `classes_`, or a cluster number. |
| `error` | Error of the subtree under each node. |

`node_count`, `n_leaves` and `max_depth` give the size of the tree.

## Methods

The estimators expose these directly, after checking their input:

- `apply(X)`: the index of the leaf each row reaches.
- `decision_path(X)`: a sparse matrix of shape `(n_samples, node_count)`
  marking the nodes each row passes through.
- `to_dot(feature_names=None, class_names=None)`: the tree in Graphviz DOT
  format. Feature names default to the column names when `fit` received a
  pandas DataFrame, and class names to `classes_`.

```python
clf.apply(X[:3])                  # leaf index of each row
clf.decision_path(X[:3]).toarray()
open("tree.dot", "w").write(clf.to_dot(feature_names=names))
```

Render the DOT output with the `dot` program (`dot -Tpng tree.dot -o tree.png`)
or the `graphviz` Python package.
