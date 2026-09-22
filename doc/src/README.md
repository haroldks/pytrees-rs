# pytrees-rs

pytrees-rs learns anytime optimal decision trees. It is written mostly in
Rust and comes with a Python wrapper that follows the scikit-learn API, so the
estimators work with pipelines, grid search and cross-validation.

A greedy learner such as CART picks, at each node, the split that looks best
right now, and never revisits it. That is fast, but the tree it builds can be
much worse than the best tree of the same size. The learners here look
further ahead:

| Estimator | Features | What it learns |
|---|---|---|
| [`DL85Classifier`](estimators/dl85.md) | binary | The optimal tree of a given depth (DL8.5) |
| [`LGDTClassifier`](estimators/lgdt.md) | binary | A tree grown top-down whose tests are chosen with a depth-2 lookahead (LGDT) |
| [`ConTreeClassifier`](estimators/contree.md) | continuous | The optimal tree of a given depth (ConTree) |
| [`DL85Cluster`](estimators/clustering.md) | binary | A clustering whose clusters are the leaves of an optimal tree |

An optimal tree is the tree with the fewest training errors among all trees
of at most `max_depth` levels with at least `min_sup` training rows per leaf.
Finding it is NP-hard, and on large datasets or deep trees the search may not
finish. Two things make that manageable:

- Every search has a time limit (`max_time`) and returns the best tree found
  so far, with `status_` telling you whether it was proven optimal.
- The anytime searches find a good tree early and keep improving it. For
  DL8.5 they are configured with the [search rules](estimators/rules.md); for
  ConTree with `use_lds=True`. Both estimators have a `fit_anytime` method
  that reports each improvement as it happens.

Small optimal trees are often as accurate as much larger greedy ones, and they
are easy to read: a depth-3 tree is at most seven tests.

## Where to go next

- [Installation](installation.md) and the [quick start](quickstart.md).
- The [estimators](estimators/README.md) chapter documents every parameter.
- The [command line tools](cli.md) and the [Rust crates](rust.md) give access
  to the same algorithms without Python.
- [Publications](publications.md) lists the papers behind each algorithm.
