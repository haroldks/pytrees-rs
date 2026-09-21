import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data

from .._dl85 import dl85_search
from ..base import DecisionTree, check_binary


class DL85Cluster(ClusterMixin, DecisionTree, BaseEstimator):
    """Clustering with an optimal decision tree.

    The tree splits the rows on binary features, and each leaf is a cluster.
    DL8.5 searches for the tree whose clusters are tightest: by default, the
    one that minimises the sum of each row's Euclidean distance to the
    centroid of its cluster. The result reads as rules, one path per cluster.

    Parameters
    ----------
    min_sup : int, default=1
        Minimum number of rows in each cluster.
    max_depth : int, default=1
        Maximum depth of the tree; at most ``2 ** max_depth`` clusters.
    max_error : float or None, default=None
        Stop as soon as a tree with at most this error is found. ``None``
        searches for the optimum.
    max_time : float, default=600.0
        Seconds before the search stops with the best tree found so far.
    always_sort : bool, default=True
    heuristic : {"none", "gini", "information_gain", "weighted_entropy"}, default="none"
    similarity_lb, dynamic_branching : bool, default=True
    discrepancy, gain, topk, restart, purity : optional
        Rules from ``pytrees.rules``, as for ``DL85Classifier``.
    error_function : callable, optional
        ``error_function(indices) -> (error, value)``, called with the row
        indices of a candidate cluster. Replaces the default distance to the
        centroid; ``value`` is stored in ``tree_["value"]``.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        The cluster of each training row.
    n_clusters_ : int
    n_features_in_ : int
    tree_ : dict of ndarray or None
        The tree as flat arrays; see ``pytrees.base.DecisionTree``.
    train_error_ : float
        The error of the clustering, as the error function measures it.
    status_ : str
        ``"optimal"``, or ``"time_limit"``.
    statistics_ : dict

    Examples
    --------
    >>> import numpy as np
    >>> from pytrees import DL85Cluster
    >>> X = np.random.default_rng(0).integers(0, 2, size=(100, 6))
    >>> labels = DL85Cluster(max_depth=2, min_sup=5).fit_predict(X)
    """

    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=None,
        max_time=600.0,
        always_sort=True,
        heuristic="none",
        similarity_lb=True,
        dynamic_branching=True,
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ):
        # Stored verbatim: see DL85Classifier.__init__.
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.always_sort = always_sort
        self.heuristic = heuristic
        self.similarity_lb = similarity_lb
        self.dynamic_branching = dynamic_branching
        self.discrepancy = discrepancy
        self.gain = gain
        self.topk = topk
        self.restart = restart
        self.purity = purity
        self.error_function = error_function

    def fit(self, X, y=None, X_error=None):
        """Search for the tightest clustering and return ``self``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary features the tree splits on: every value must be 0 or 1.
        y : ignored
        X_error : array-like of shape (n_samples, n_other_features), optional
            The data the default error measures distances in, when it should
            not be ``X`` itself: continuous measurements of the same rows,
            for instance. Ignored when ``error_function`` is given.
        """
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=True)
        check_binary(self, X)
        error_function = self.error_function
        if error_function is None:
            points = X if X_error is None else check_array(X_error, dtype=np.float64)
            if points.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X_error has {points.shape[0]} rows, but X has {X.shape[0]}"
                )
            error_function = _DistanceToCentroid(points)

        native = dl85_search(
            self,
            # The depth-2 solver only works with class counts, and the error
            # of a cluster depends on which rows are in it.
            fast_d2=False,
            error_function_input="indices",
            error_function=error_function,
        )
        native.fit(X, None)
        self._set_tree(native.stats)
        self.status_ = native.status
        if self.tree_ is not None:
            self.n_clusters_ = int((self.tree_["children_left"] == -1).sum())
            self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """The cluster of each row of ``X``, numbered 0 to ``n_clusters_ - 1``."""
        leaves = self._leaves(X)
        # Clusters are numbered in the order their leaves appear in tree_.
        return np.searchsorted(
            np.flatnonzero(self.tree_["children_left"] == -1), leaves
        )

    def _leaf_label(self, node):
        cluster = np.searchsorted(
            np.flatnonzero(self.tree_["children_left"] == -1), node
        )
        return f"{{cluster|{cluster}}}"


class _DistanceToCentroid:
    """The default error: the sum of the distances to the cluster's centroid,
    measured in ``points``. Its value, 0, is unused: a cluster's label is its
    number, not a value computed from its rows."""

    def __init__(self, points):
        self.points = points

    def __call__(self, indices):
        if not indices:
            return 0.0, 0.0
        rows = self.points[indices]
        return float(np.linalg.norm(rows - rows.mean(axis=0), axis=1).sum()), 0.0
