# Quick start

## An optimal tree on continuous data

`ConTreeClassifier` works directly on numeric features:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pytrees import ConTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = ConTreeClassifier(max_depth=2, min_sup=5, max_time=60)
clf.fit(X_train, y_train)

print(clf.status_)       # "optimal" if the search finished
print(clf.train_error_)  # number of misclassified training rows
print(clf.score(X_test, y_test))
```

`status_` is `"optimal"` when the search proved that no tree of that depth
makes fewer training errors. If the time limit is reached first, it is
`"time_limit"` and the tree is the best one found.

## An optimal tree on binary data

`DL85Classifier` and `LGDTClassifier` need features that are 0 or 1. Put a
discretiser in front of them:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from pytrees import DL85Classifier, LGDTClassifier

binarize = KBinsDiscretizer(n_bins=4, encode="onehot-dense")

optimal = make_pipeline(binarize, DL85Classifier(max_depth=3, min_sup=5, max_time=60))
optimal.fit(X_train, y_train)

lookahead = make_pipeline(binarize, LGDTClassifier(max_depth=6, min_sup=5))
lookahead.fit(X_train, y_train)

print(optimal.score(X_test, y_test), lookahead.score(X_test, y_test))
```

DL8.5 finds the optimal tree, which gets expensive beyond depth 4 or 5 on
datasets with many features. LGDT is not optimal but scales to deep trees.

## Watching an anytime search

On a hard problem, the anytime searches give you a good tree early.
`fit_anytime` calls a function each time the tree improves:

```python
from pytrees import ConTreeClassifier

def report(error, seconds, status):
    print(f"{seconds:7.2f}s  error={error}  {status}")

clf = ConTreeClassifier(max_depth=4, sort_by_heuristic=True, max_time=60)
clf.fit_anytime(X_train, y_train, callback=report)
```

For DL8.5, pass one of the [search rules](estimators/rules.md), for example
`DL85Classifier(discrepancy=DiscrepancyRule())`, and call `fit_anytime` the
same way.

## Looking at the tree

Every estimator stores its tree in `tree_`, and `to_dot` exports it to
Graphviz:

```python
dot = clf.to_dot(feature_names=load_breast_cancer().feature_names)

import graphviz            # pip install graphviz
graphviz.Source(dot).render("tree", format="png")
```

`apply(X)` returns the leaf each row reaches, and `decision_path(X)` the nodes
it passes through. See [The fitted tree](estimators/tree.md).

## Using scikit-learn tools

The estimators can be cloned, pickled and tuned like any other:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    ConTreeClassifier(max_time=30),
    {"max_depth": [1, 2, 3], "min_sup": [1, 5, 20]},
    cv=5,
)
search.fit(X_train, y_train)
print(search.best_params_)
```
