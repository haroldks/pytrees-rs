# Estimators

All estimators live in the `pytrees` package and follow the scikit-learn
conventions: parameters are set in the constructor, `fit` returns the
estimator, fitted attributes end with an underscore, and the estimators can be
cloned, pickled and used inside `Pipeline`, `GridSearchCV` or
`cross_val_score`.

| Estimator | Task | Features | Search |
|---|---|---|---|
| [`DL85Classifier`](dl85.md) | classification | binary | optimal, optionally anytime |
| [`LGDTClassifier`](lgdt.md) | classification | binary | greedy with a depth-2 lookahead |
| [`ConTreeClassifier`](contree.md) | classification | continuous | optimal, optionally anytime |
| [`DL85Cluster`](clustering.md) | clustering | binary | optimal |

Some behaviour is shared by all of them:

- **Labels.** `y` can hold any labels `np.unique` accepts (integers, strings,
  and so on). `classes_` lists them, and `predict` returns them.
- **Binary features.** DL8.5, LGDT and DL85Cluster only accept 0 and 1 in `X`
  and raise a `ValueError` otherwise, both in `fit` and in `predict`. Use a
  `Binarizer`, a `KBinsDiscretizer` with one-hot encoding, or your own
  thresholds to prepare the data.
- **The fitted tree.** `tree_` is a [`pytrees.tree.Tree`](tree.md) with the
  same layout as scikit-learn's. `apply`, `decision_path` and `to_dot` work on
  it for every estimator.
- **Search results.** `train_error_` is the training error of the tree,
  `status_` says why the search stopped, and `statistics_` holds counters such
  as the search time (`duration`, in seconds) and the size of the cache.
