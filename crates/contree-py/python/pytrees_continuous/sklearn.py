"""A scikit-learn estimator over the Rust search."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from pytrees_continuous._native import RawConTree

__all__ = ["ConTreeClassifier"]


class ConTreeClassifier(ClassifierMixin, BaseEstimator):
    """An optimal decision tree classifier for continuous features.

    Where a greedy learner picks the locally best split at each node, this
    searches for the tree of the given depth with the fewest training errors.
    The search is exact, so it can be slow; ``max_time`` bounds it and
    ``status_`` says whether it finished.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum number of levels of tests. Cost grows steeply with it.
    min_sup : int, default=1
        Minimum number of training instances in each leaf.
    max_error : int or None, default=None
        Stop once a tree with at most this many errors is found. ``None``
        searches for the optimum.
    max_time : float, default=600.0
        Seconds before the search gives up and returns the best tree so far.
        When it does, ``status_`` is ``"time_limit"``.
    max_gap : int, default=0
        Accept a tree within this many errors of the optimum. Larger values
        prune more and finish sooner.
    split_selection : {"mid", "first", "random"}, default="mid"
        Where inside a candidate interval the threshold is placed.
    sort_by_heuristic : bool, default=False
        Try features and splits in Gini order. Usually finds a good incumbent
        sooner, which prunes more.
    fast_d2 : bool, default=True
        Use the specialized solver for depth-2 subtrees. Exact, and much
        faster; set it to False only to run the general search all the way
        down, which becomes impractical from depth 2 upward.
    use_lds : bool, default=False
        Use the anytime limited-discrepancy search, which widens its budget
        over successive passes rather than running to the optimum in one go.
    random_state : int or None, default=None
        Seeds the split-point generator. Only ``split_selection="random"``
        draws from it; every other setting is deterministic already.
    budget_schedule : {"diagonal", "square"}, default="diagonal"
        How the anytime search (``use_lds=True`` or ``fit_anytime``) widens
        its budget from one pass to the next.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels seen during ``fit``, in the order the internal
        encoding uses.
    n_features_in_ : int
    n_classes_ : int
    tree_ : dict of ndarray
        ``children_left``, ``children_right``, ``feature``, ``threshold``,
        ``value`` and ``error``, one entry per node. ``children_left == -1``
        marks a leaf.
    train_error_ : int
        Training misclassifications of the returned tree.
    status_ : str
        ``"optimal"``, ``"time_limit"``, ``"budget_exhausted"`` or
        ``"error_bound_reached"``. Only the first means the tree is proven
        best for the given depth and support.
    statistics_ : dict
        Search counters: cache size and hits, solver calls, duration.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from pytrees_continuous import ConTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ConTreeClassifier(max_depth=2).fit(X, y)
    >>> clf.status_
    'optimal'
    """

    def __init__(
        self,
        max_depth=3,
        min_sup=1,
        max_error=None,
        max_time=600.0,
        max_gap=0,
        split_selection="mid",
        sort_by_heuristic=False,
        fast_d2=True,
        use_lds=False,
        random_state=None,
        budget_schedule="diagonal",
    ):
        # Stored verbatim under their own names, with no validation and no
        # name mangling: `get_params`, `clone` and `GridSearchCV` all depend on
        # being able to read a parameter back exactly as it was passed.
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_error = max_error
        self.max_time = max_time
        self.max_gap = max_gap
        self.split_selection = split_selection
        self.sort_by_heuristic = sort_by_heuristic
        self.fast_d2 = fast_d2
        self.use_lds = use_lds
        self.random_state = random_state
        self.budget_schedule = budget_schedule

    def fit(self, X, y):
        """Fit the tree and return ``self``."""
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_all_finite=True)
        check_classification_targets(y)

        # The Rust core works with a dense 0..k-1 encoding; keeping the
        # original labels here is what lets arbitrary ones -- strings,
        # non-contiguous integers -- work at all.
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        native = RawConTree(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_time=self.max_time,
            max_error=self.max_error,
            max_gap=self.max_gap,
            split_selection=self.split_selection,
            sort_by_heuristic=self.sort_by_heuristic,
            fast_d2=self.fast_d2,
            use_lds=self.use_lds,
            random_state=self.random_state,
            budget_schedule=self.budget_schedule,
        )
        native.fit(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(encoded, dtype=np.int64),
        )

        self._native = native
        self.tree_ = native.tree_arrays()
        self.train_error_ = native.error
        self.status_ = native.status
        self.statistics_ = native.statistics
        return self

    def fit_anytime(self, X, y, callback=None):
        """Fit with the anytime search, reporting improvements as they happen.

        ``callback(error, seconds, status)`` is invoked each time a pass finds
        a better tree, so a caller can watch a long search converge, stop
        early, or plot the anytime profile. This is what the LDS solver is for;
        plain ``fit`` only ever hands back the final answer.

        Returns ``self``.
        """
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_all_finite=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        native = RawConTree(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_time=self.max_time,
            max_error=self.max_error,
            max_gap=self.max_gap,
            split_selection=self.split_selection,
            sort_by_heuristic=self.sort_by_heuristic,
            fast_d2=self.fast_d2,
            use_lds=True,
            random_state=self.random_state,
            budget_schedule=self.budget_schedule,
        )
        native.fit_anytime(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(encoded, dtype=np.int64),
            callback,
        )

        self._native = native
        self.tree_ = native.tree_arrays()
        self.train_error_ = native.error
        self.status_ = native.status
        self.statistics_ = native.statistics
        return self

    def predict(self, X):
        """Classify each row of ``X``."""
        check_is_fitted(self)
        X = validate_data(
            self, X, dtype=np.float64, ensure_all_finite=True, reset=False
        )
        encoded = self._native.predict(np.ascontiguousarray(X, dtype=np.float64))
        return self.classes_.take(encoded)

    def decision_path(self, X):
        """The node indices each row of ``X`` visits, root first."""
        check_is_fitted(self)
        X = validate_data(
            self, X, dtype=np.float64, ensure_all_finite=True, reset=False
        )
        return self._native.decision_path(np.ascontiguousarray(X, dtype=np.float64))

    @property
    def tree_json_(self):
        """The fitted tree as JSON."""
        check_is_fitted(self)
        return self._native.tree_json

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
