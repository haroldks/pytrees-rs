import json
from .. import DecisionTree, SearchFailedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.odt import PyDL85
from pytreesrs.enums import *


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

    max_error : float, default=inf
        Maximum acceptable error for early termination. The algorithm stops
        when a tree with error <= max_error is found.

    max_time : float, default=600.0
        Maximum time limit in seconds for the search. Prevents infinite
        computation on difficult instances.

    always_sort : bool, default=True
        Whether to always sort features based on heuristic at each node

    node_data_type : ExposedNodeDataType, default=ClassSupports
        Type of data used to compute the error.

    depth2_policy : ExposedDepth2Policy, default=Enabled
        Depth-2 specialization policy for improved performance on depth-2 trees.

    lower_bound_policy : ExposedLowerBoundPolicy, default=Similarity
        Strategy for computing lower bounds during search. Helps prune
        unpromising branches early.

    branching_policy : ExposedBranchingPolicy, default=Dynamic
        Branching strategy for tree construction. Affects search order.

    heuristic : ExposedHeuristic, default=Disabled
        Heuristic function for guiding the search.

    discrepancy : ExposedDiscrepancyRule, optional
        Limited discrepancy search rule for controlling exploration.

    gain : ExposedGainRule, optional
        Information gain-based stopping rule for pruning.

    topk : ExposedTopKRule, optional
        Top-K search limitation rule.

    restart : ExposedRestartRule, optional
        Time-based restart rule for anytime behavior.

    purity : ExposedPurityRule, optional
        Node purity-based stopping rule.

    error_function : callable, optional
        Custom Python error function for specialized loss functions.

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

    >>> from pytreesrs.odt.rules import ExposedGainRule, ExposedPurityRule
    >>> from pytreesrs.enums import ExposedHeuristic
    >>> clf = DL85Classifier(
    ...     max_depth=4,
    ...     min_sup=10,
    ...     heuristic=ExposedHeuristic.InformationGain,
    ...     gain=ExposedGainRule(min_gain=0.01),
    ...     purity=ExposedPurityRule(min_purity=0.9)
    ... )
    >>> clf.fit(X, y)
    """

    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=float("inf"),
        max_time=600.0,
        always_sort=True,
        node_data_type=ExposedNodeDataType.ClassSupports,
        depth2_policy=ExposedDepth2Policy.Enabled,
        lower_bound_policy=ExposedLowerBoundPolicy.Similarity,
        branching_policy=ExposedBranchingPolicy.Dynamic,
        heuristic=ExposedHeuristic.Disabled,
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
        self.node_data_type = node_data_type
        self.depth2_policy = depth2_policy
        self.lower_bound_policy = lower_bound_policy
        self.branching_policy = branching_policy
        self.heuristic = heuristic
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
            self.lower_bound_policy = ExposedLowerBoundPolicy.Disabled
            self.branching_policy = ExposedBranchingPolicy.Default

        # Initialize the underlying Rust implementation
        self.__obj = PyDL85(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_error=self.max_error,
            time_limit=self.max_time,
            always_sort=self.always_sort,
            heuristic=self.heuristic,
            depth2_policy=self.depth2_policy,
            lower_bound=self.lower_bound_policy,
            branching_policy=self.branching_policy,
            data_type=self.node_data_type,
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
            X, y = check_X_y(X, y, dtype="float64", y_numeric=True)
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
