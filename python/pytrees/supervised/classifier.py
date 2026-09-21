import json
import numpy as np
from ..base import DecisionTree, validate_binary_classification
from sklearn.base import BaseEstimator, ClassifierMixin
from pytrees._native.odt import PyDL85


class DL85Classifier(BaseEstimator, ClassifierMixin, DecisionTree):
    """
    Optimal Decision Tree Classifier using the DL8.5 algorithm.

    DL85Classifier implements the DL8.5 algorithm for constructing globally optimal
    decision trees.
    The algorithm uses dynamic programming with advanced caching and pruning
    techniques to efficiently explore the exponential search space of possible
    decision trees.

    Parameters
    ----------
    min_sup : int, default=1
        Minimum support (number of samples) required for a node to be split.
        Higher values lead to simpler trees and prevent overfitting.

    max_depth : int, default=1
        Maximum depth of the decision tree. Controls tree complexity and
        prevents overfitting. Depth 1 creates decision stumps.

    max_error : float or None, default=None
        Stop as soon as a tree with at most this error is found. ``None``
        searches for the optimum.

    max_time : float, default=600.0
        Maximum time limit in seconds for the search. Prevents infinite
        computation on difficult instances.

    always_sort : bool, default=True
        Whether to always sort features based on heuristic at each node

    heuristic : {"none", "gini", "information_gain", "weighted_entropy"}, default="none"
        Order in which the features are tried at each node.

    fast_d2 : bool, default=True
        Solve depth-2 subtrees with the specialised exact solver.

    similarity_lb : bool, default=True
        Use similarity lower bounds to prune branches early. Turned off when
        any rule is given.

    dynamic_branching : bool, default=True
        Choose the branch to explore first dynamically. Turned off when any
        rule is given.

    error_function_input : {"class_counts", "indices"}, default="class_counts"
        What ``error_function`` receives at each node: the count of each class,
        or the indices of the rows in the node. ``"indices"`` needs an
        ``error_function``.

    discrepancy : pytrees.rules.DiscrepancyRule, optional
        Limited discrepancy search.

    gain : pytrees.rules.GainRule, optional
        Skip splits with too little information gain.

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
    config : dict
        Complete algorithm configuration as parsed from JSON.

    results : SearchOutput
        Detailed results from the last fit operation including tree,
        error, and search statistics.

    Examples
    --------
    Basic usage with default parameters:

    >>> from pytrees import DL85Classifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> clf = DL85Classifier(max_depth=3, min_sup=5)
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    >>> print(f"Accuracy: {clf.accuracy_}")

    Advanced usage with rules and heuristics:

    >>> from pytrees.rules import GainRule, PurityRule
    >>> clf = DL85Classifier(
    ...     max_depth=4,
    ...     min_sup=10,
    ...     heuristic="information_gain",
    ...     gain=GainRule(min_gain=0.01),
    ...     purity=PurityRule(min_purity=0.9)
    ... )
    >>> clf.fit(X, y)
    """

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
        # Stored verbatim, with no validation: get_params, clone and
        # GridSearchCV all read a parameter back exactly as it was passed.
        # The search itself is only built in fit.
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
        native.fit(X, encoded.astype(np.float64))
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
        native.fit_anytime(X, encoded.astype(np.float64), callback)
        self._store(native)
        return self

    def predict(self, X):
        """Classify each row of ``X``."""
        # The base class checks that the model is fitted before anything
        # reads classes_.
        encoded = np.asarray(super().predict(X), dtype=np.intp)
        return self.classes_.take(encoded)

    def _native_search(self):
        if self.error_function_input == "indices" and self.error_function is None:
            raise ValueError(
                'error_function_input="indices" needs an error_function: the '
                "built-in error only understands class counts"
            )
        rules = (self.discrepancy, self.gain, self.topk, self.restart, self.purity)
        # The rules relax the search pass by pass, which the similarity bounds
        # and dynamic branching do not take into account.
        bounded = any(rule is not None for rule in rules)
        return PyDL85(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_error=self.max_error,
            time_limit=self.max_time,
            always_sort=self.always_sort,
            heuristic=self.heuristic,
            fast_d2=self.fast_d2,
            similarity_lb=self.similarity_lb and not bounded,
            dynamic_branching=self.dynamic_branching and not bounded,
            error_function_input=self.error_function_input,
            discrepancy=self.discrepancy,
            gain=self.gain,
            topk=self.topk,
            restart=self.restart,
            purity=self.purity,
            error_function=self.error_function,
        )

    def _store(self, native):
        self.results = native.stats
        self.config = json.loads(native.config)
        self.refresh_stats()
