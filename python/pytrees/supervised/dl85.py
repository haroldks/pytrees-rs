import numpy as np
from ..base import TreeClassifier, validate_binary_classification
from sklearn.base import BaseEstimator, ClassifierMixin
from .._dl85 import dl85_search


class DL85Classifier(ClassifierMixin, TreeClassifier, BaseEstimator):
    """Optimal decision tree classifier over binary features (DL8.5).

    Searches for the tree of at most ``max_depth`` levels with the fewest
    training errors, by dynamic programming with branch-and-bound and a cache
    of subproblems. Every feature must be 0 or 1; binarise continuous data
    first, or use ``ConTreeClassifier``.

    The search can also be anytime. The rules in ``pytrees.rules`` restrict
    each pass of the search, and the search restarts with relaxed rules until
    a pass completes; ``fit_anytime`` reports each improvement.

    Parameters
    ----------
    min_sup : int, default=1
        Minimum number of training rows in each leaf.

    max_depth : int, default=1
        Maximum depth of the tree. Depth 1 gives a decision stump.

    max_error : float or None, default=None
        Stop as soon as a tree with at most this error is found. ``None``
        searches for the optimum.

    max_time : float, default=600.0
        Seconds before the search stops with the best tree found so far.

    always_sort : bool, default=True
        Sort the features by ``heuristic`` at every node, rather than only at
        the root.

    heuristic : {"none", "gini", "information_gain", "weighted_entropy"}, default="none"
        Order in which the features are tried at each node.

    fast_d2 : bool, default=True
        Solve depth-2 subtrees with the specialised exact solver.

    similarity_lb : bool, default=True
        Bound the error of a node from similar nodes already solved, to prune
        earlier. Turned off when any rule is given.

    dynamic_branching : bool, default=True
        Search first the branch with the higher known lower bound. Turned off
        when any rule is given.

    error_function_input : {"class_counts", "indices"}, default="class_counts"
        What ``error_function`` receives at each node: the count of each class,
        or the indices of the rows in the node. ``"indices"`` needs an
        ``error_function``.

    discrepancy : pytrees.rules.DiscrepancyRule, optional
        Limited discrepancy search.

    gain : pytrees.rules.GainRule, optional
        Explore only paths close to the heuristic's choices.

    topk : pytrees.rules.TopKRule, optional
        Explore only the most promising features at each node.

    restart : pytrees.rules.RestartRule, optional
        Restart the search periodically with relaxed rules.

    purity : pytrees.rules.PurityRule, optional
        Stop splitting nodes that are pure enough.

    error_function : callable, optional
        ``error_function(data) -> (error, prediction)``, called at each leaf
        with what ``error_function_input`` selects. Replaces the built-in
        misclassification error.

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
        Search counters: cache size and hits, restarts, duration.
    status_ : str
        ``"optimal"``, or ``"time_limit"`` if the search stopped at
        ``max_time`` with the best tree found so far.

    References
    ----------
    G. Aglin, S. Nijssen and P. Schaus. Learning Optimal Decision Trees Using
    Caching Branch-and-Bound Search. AAAI 2020.

    H. Kiossou, P. Schaus, S. Nijssen and V. R. Houndji. Time Constrained
    DL8.5 Using Limited Discrepancy Search. ECML PKDD 2022.

    H. Kiossou and P. Schaus. A Generic Complete Anytime Beam Search for
    Optimal Decision Tree. IDA 2026.

    Examples
    --------
    >>> from pytrees import DL85Classifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> X = (X > 0).astype(int)  # DL8.5 needs binary features
    >>> clf = DL85Classifier(max_depth=3, min_sup=5).fit(X, y)
    >>> clf.status_
    'optimal'
    >>> accuracy = clf.score(X, y)

    An anytime search that first explores the trees preferred by the
    heuristic:

    >>> from pytrees.rules import DiscrepancyRule
    >>> clf = DL85Classifier(
    ...     max_depth=4,
    ...     min_sup=10,
    ...     heuristic="information_gain",
    ...     discrepancy=DiscrepancyRule(),
    ... )
    >>> clf = clf.fit(X, y)
    """

    _binary_features = True

    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=None,
        max_time=600.0,
        always_sort=True,
        heuristic="none",
        fast_d2=True,
        similarity_lb=True,
        dynamic_branching=True,
        error_function_input="class_counts",
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ):
        # As scikit-learn requires, parameters are stored as given and only
        # validated in fit.
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.always_sort = always_sort
        self.heuristic = heuristic
        self.fast_d2 = fast_d2
        self.similarity_lb = similarity_lb
        self.dynamic_branching = dynamic_branching
        self.error_function_input = error_function_input
        self.discrepancy = discrepancy
        self.gain = gain
        self.topk = topk
        self.restart = restart
        self.purity = purity
        self.error_function = error_function

    def fit(self, X, y):
        """Search for the optimal tree and return ``self``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary features: every value must be 0 or 1.
        y : array-like of shape (n_samples,)
            Class labels, of any type ``np.unique`` accepts.
        """
        X, self.classes_, encoded = validate_binary_classification(self, X, y)
        native = self._native_search()
        native.fit(X, encoded.astype(np.int64))
        self._store(native)
        return self

    def fit_anytime(self, X, y, callback):
        """Fit pass by pass, calling ``callback`` when a pass improves the tree.

        The passes differ only when rules bound the search (``discrepancy``,
        ``gain``, ``topk``, ``restart``): each pass relaxes them until one
        runs unbounded. ``callback(error, seconds, status)`` receives the
        training error, the time spent so far, and ``"budget_exhausted"``
        while a rule still bounds the search, then ``"optimal"`` or
        ``"time_limit"``. Returns ``self``.
        """
        X, self.classes_, encoded = validate_binary_classification(self, X, y)
        native = self._native_search()
        native.fit_anytime(X, encoded.astype(np.int64), callback)
        self._store(native)
        return self

    def _native_search(self):
        if self.error_function_input == "indices" and self.error_function is None:
            raise ValueError(
                'error_function_input="indices" needs an error_function: the '
                "built-in error only understands class counts"
            )
        return dl85_search(
            self,
            fast_d2=self.fast_d2,
            error_function_input=self.error_function_input,
            error_function=self.error_function,
        )

    def _store(self, native):
        self._store_dtrees_result(native)
        self.n_classes_ = len(self.classes_)
        self.status_ = native.status
