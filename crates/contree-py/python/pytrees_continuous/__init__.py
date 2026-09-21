"""Optimal decision trees over continuous features.

The estimator is a drop-in scikit-learn classifier:

    >>> from sklearn.datasets import load_iris
    >>> from pytrees_continuous import ConTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ConTreeClassifier(max_depth=3, min_sup=5).fit(X, y)
    >>> clf.score(X, y)  # doctest: +SKIP
    0.98

Unlike a greedy tree, `fit` searches for the *optimal* tree of the given depth.
That can take a long time, so `status_` reports whether the search finished:
only ``"optimal"`` means the tree is proven best.
"""

from pytrees_continuous._native import native_version
from pytrees_continuous.sklearn import ConTreeClassifier

__all__ = ["ConTreeClassifier", "native_version"]
