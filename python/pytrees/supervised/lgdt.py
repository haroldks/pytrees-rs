import numpy as np

from ..base import DecisionTree, validate_binary_classification
from pytrees._native.greedy import lgdt
from sklearn.base import BaseEstimator, ClassifierMixin


class LGDTClassifier(BaseEstimator, ClassifierMixin, DecisionTree):
    """
    Less Greedy Decision Tree (LGDT) Classifier for fast approximate solutions.

    LGDTClassifier implements greedy decision tree construction algorithms that
    provide fast approximate solutions by making locally optimal choices with at each node
    with a lookahead of 2.
    This approach trades global optimality for significantly improved
    computational efficiency, making it suitable for large datasets.


    Parameters
    ----------
    min_sup : int, default=1
        Minimum support (number of samples) required for a node to be split.
        Higher values lead to simpler trees and prevent overfitting while
        also improving computational efficiency.

    max_depth : int, default=2
        Maximum depth of the decision tree. Controls tree complexity and
        prevents overfitting. Smaller depths result in faster construction.

    criterion : {"error", "information_gain"}, default="error"
        What the depth-2 lookahead at each step optimises: the
        misclassification error, or the information gain.

    Attributes
    ----------
    results : SearchOutput
        Detailed results from the last fit operation including tree structure,
        error metrics, and construction statistics.

    Examples
    --------
    Basic usage for fast tree construction:

    >>> from pytrees import LGDTClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> clf = LGDTClassifier(max_depth=5, min_sup=10)
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    >>> print(f"Accuracy: {clf.accuracy_}")

    Using different search strategies:

    >>> clf = LGDTClassifier(max_depth=4, criterion="information_gain")
    >>> clf.fit(X, y)
    >>>
    """

    def __init__(
        self,
        min_sup=1,
        max_depth=2,
        criterion="error",
    ):
        # Stored verbatim: see DL85Classifier.__init__.
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        """Build the tree greedily and return ``self``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary features: every value must be 0 or 1.
        y : array-like of shape (n_samples,)
            Class labels, of any type ``np.unique`` accepts.
        """
        X, self.classes_, encoded = validate_binary_classification(self, X, y)
        self.results = lgdt(
            X,
            encoded.astype(np.float64),
            self.criterion,
            self.min_sup,
            self.max_depth,
        )
        self.refresh_stats()
        return self

    def predict(self, X):
        """Classify each row of ``X``."""
        # The base class checks that the model is fitted before anything
        # reads classes_.
        encoded = np.asarray(super().predict(X), dtype=np.intp)
        return self.classes_.take(encoded)
