import json
import numpy as np
from .. import DecisionTree, SearchFailedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
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
        """
        Initialize a DL85Classifier with specified parameters.

        Sets up the underlying PyDL85 Rust implementation with the provided
        configuration and initializes the Python wrapper state.
        """
        super().__init__()
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

        self.results = None

        # Disable certain optimizations when rules are used
        if any(
            rule is not None
            for rule in [
                self.discrepancy,
                self.gain,
                self.topk,
                self.restart,
                self.purity,
            ]
        ):
            self.similarity_lb = False
            self.dynamic_branching = False

        # Initialize the underlying Rust implementation
        self.__obj = PyDL85(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_error=self.max_error,
            time_limit=self.max_time,
            always_sort=self.always_sort,
            heuristic=self.heuristic,
            fast_d2=self.fast_d2,
            similarity_lb=self.similarity_lb,
            dynamic_branching=self.dynamic_branching,
            error_function_input=self.error_function_input,
            discrepancy=self.discrepancy,
            gain=self.gain,
            topk=self.topk,
            restart=self.restart,
            purity=self.purity,
            error_function=self.error_function,
        )
        self.config = json.loads(self.__obj.config)

    def fit(self, X, y=None):
        """
        Fit the DL85 optimal decision tree classifier.

        This method trains the optimal decision tree using the DL8.5 algorithm,
        which guarantees finding the globally optimal tree within the specified
        constraints.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Features should preferably be binary (0/1) for
            best performance, though the algorithm can handle continuous
            features through preprocessing.

        y : array-like of shape (n_samples,), optional
            Target values (class labels). If None, assumes unsupervised
            learning mode or error function specified

        Returns
        -------
        self : DL85Classifier
            Returns self for method chaining.

        Raises
        ------
        SearchFailedError
            If the algorithm fails to find a solution within the given
            constraints (time limit, memory, etc.).

        Examples
        --------
        >>> clf = DL85Classifier(max_depth=3, min_sup=5)
        >>> clf.fit(X_train, y_train)
        >>> print(f"Training accuracy: {clf.accuracy_}")
        >>> print(f"Tree error: {clf.tree_error_}")

        Notes
        -----
        - The fitting process may take significant time for large datasets
        - Use max_time parameter to limit computation time
        - Monitor the statistics attribute for detailed search information
        """
        target_is_need = True if y is not None else False

        if target_is_need:  # supervised learning
            # Check that X and y have correct shape and raise ValueError if not
            y = y.astype(np.float64)
            X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        else:  # unsupervised learning
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype="float64")

        try:
            self.__obj.fit(X, y)
            self.results = self.__obj.stats
            self.refresh_stats()
        except Exception:
            self.is_fitted_ = False
            raise SearchFailedError

    def load_data(self, X, y):
        """
        Load training data without immediately fitting.

        This method allows for data loading followed by incremental fitting
        using partial_fit(), which can be useful for implementing custom
        training loops.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.

        y : array-like of shape (n_samples,)
            Training data labels.

        Examples
        --------
        >>> clf = DL85Classifier(max_depth=3)
        >>> clf.load_data(X_train, y_train)
        >>> clf.partial_fit()  # Perform the actual training
        """
        self.__obj.load_data(X, y)

    def partial_fit(self):
        """
        Perform incremental fitting on pre-loaded data.

        This method continues or starts the optimization process on data
        that was previously loaded using load_data(). Useful for implementing
        or custom training procedures.

        Examples
        --------
        >>> clf = DL85Classifier(max_depth=3)
        >>> clf.load_data(X_train, y_train)
        >>> clf.partial_fit()
        >>> print(f"Current best error: {clf.results.error}")
        """
        self.__obj.partial_fit()
