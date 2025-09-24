from .. import DecisionTree, SearchFailedError
from pytreesrs.enums import ExposedSearchStrategy
from pytreesrs.greedy import lgdt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y


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

    search_strategy : ExposedSearchStrategy, default=LGDTErrorMinimizer
        The search strategy to use for tree construction. Available options:

        - LGDTErrorMinimizer: Focuses on minimizing classification error
        - LGDTInfoGainMaximizer: Focuses on maximizing information gain

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

    >>> from pytreesrs.enums import ExposedSearchStrategy
    >>>
    >>> # Error minimizer strategy search for quality
    >>> clf = LGDTClassifier(
    ...     max_depth=4,
    ...     search_strategy=ExposedSearchStrategy.LGDTInfoGainMaximizer
    ... )
    >>> clf.fit(X, y)
    >>>
    """

    def __init__(
        self,
        min_sup=1,
        max_depth=2,
        search_strategy=ExposedSearchStrategy.LGDTErrorMinimizer,
    ):
        """
        Initialize an LGDTClassifier with specified parameters.

        Parameters
        ----------
        min_sup : int, default=1
            Minimum support required for splits.
        max_depth : int, default=2
            Maximum tree depth.
        search_strategy : ExposedSearchStrategy, default=LGDTErrorMinimizer
            Search strategy for tree construction.
        """
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.search_strategy = search_strategy

    def fit(self, X, y):
        """
        Fit the LGDT classifier using greedy tree construction.

        This method constructs a decision tree using the specified greedy
        algorithm, which makes locally optimal choices at each step to
        build the tree efficiently.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can handle both binary and continuous features,
            though binary features may provide better performance.

        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : LGDTClassifier
            Returns self for method chaining.

        Raises
        ------
        SearchFailedError
            If the greedy algorithm fails during tree construction due to
            invalid data, memory issues, or other computational problems.

        Examples
        --------
        Basic fitting:

        >>> clf = LGDTClassifier(max_depth=3, min_sup=5)
        >>> clf.fit(X_train, y_train)
        >>> print(f"Tree constructed in {clf.results.duration:.3f} seconds")
        >>> print(f"Training error: {clf.tree_error_}")

        """
        X, y = check_X_y(X, y, dtype="float64", y_numeric=True)

        try:
            self.results = lgdt(
                X,
                y,
                self.search_strategy,
                self.min_sup,
                self.max_depth,
            )
            self.refresh_stats()
        except Exception as e:
            raise SearchFailedError
