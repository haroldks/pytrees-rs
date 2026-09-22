import numpy as np

from ..base import TreeClassifier, validate_binary_classification
from pytrees._native.dtrees import RawLGDT
from sklearn.base import BaseEstimator, ClassifierMixin


class LGDTClassifier(ClassifierMixin, TreeClassifier, BaseEstimator):
    """Less greedy decision tree classifier over binary features (LGDT).

    Grows the tree top-down like CART, but chooses each test by solving the
    best depth-2 tree at the node and keeping its root. This two-level
    lookahead is much less myopic than a single greedy split, and still fast
    enough for deep trees. Every feature must be 0 or 1.

    Parameters
    ----------
    min_sup : int, default=1
        Minimum number of training rows in each leaf.

    max_depth : int, default=2
        Maximum depth of the tree.

    criterion : {"error", "information_gain"}, default="error"
        What the depth-2 lookahead at each step optimises: the
        misclassification error, or the information gain.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels seen during ``fit``.
    n_classes_ : int
    n_features_in_ : int
    tree_ : pytrees.tree.Tree or None
        The fitted tree, in scikit-learn's layout; ``None`` if the search
        found no tree.
    train_error_ : float
        Training error of the tree, as the error function measures it.
    statistics_ : dict
        The error, duration and sizes of the search.

    References
    ----------
    H. Kiossou, P. Schaus, S. Nijssen and G. Aglin. Efficient Lookahead
    Decision Trees. IDA 2024.

    Examples
    --------

    >>> from pytrees import LGDTClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> X = (X > 0).astype(int)  # LGDT needs binary features
    >>> clf = LGDTClassifier(max_depth=5, min_sup=10).fit(X, y)
    >>> accuracy = clf.score(X, y)

    A lookahead that maximises information gain:

    >>> clf = LGDTClassifier(max_depth=4, criterion="information_gain")
    >>> clf = clf.fit(X, y)
    """

    _binary_features = True

    def __init__(
        self,
        min_sup=1,
        max_depth=2,
        criterion="error",
    ):
        # Stored as given; validated in fit.
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
        native = RawLGDT(
            criterion=self.criterion, min_sup=self.min_sup, max_depth=self.max_depth
        )
        native.fit(X, encoded.astype(np.int64))
        self._store_dtrees_result(native)
        self.n_classes_ = len(self.classes_)
        return self
