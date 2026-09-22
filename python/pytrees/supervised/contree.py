"""ConTree: optimal decision trees over continuous features."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from pytrees._native.contree import RawConTree

from ..base import TreeClassifier
from ..tree import LEAF, Tree

__all__ = ["ConTreeClassifier"]


class ConTreeClassifier(ClassifierMixin, TreeClassifier, BaseEstimator):
    """Optimal decision tree classifier over continuous features (ConTree).

    Where a greedy learner picks the locally best split at each node, this
    searches for the tree of the given depth with the fewest training errors,
    with thresholds between observed values, without binarising the data. The
    search is exact, so it can be slow; ``max_time`` bounds it and ``status_``
    says whether it finished. The anytime variant (``use_lds=True`` or
    ``fit_anytime``) finds good trees much earlier.

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
        Which candidate threshold of an interval is evaluated next: the
        middle one (bisection), each one in turn from the left (in Gini order
        with ``sort_by_heuristic``), or a random one.
    sort_by_heuristic : bool, default=False
        Try features and splits in Gini order. Usually finds a good incumbent
        sooner, which prunes more.
    fast_d2 : bool, default=True
        Use the specialised solver for depth-2 subtrees. It is exact and much
        faster than the general search.
    use_lds : bool, default=False
        Use the anytime limited discrepancy search, which widens its budget
        over successive passes rather than running to the optimum in one go.
    random_state : int or None, default=None
        Seed for ``split_selection="random"``; the other settings are
        deterministic.
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
    tree_ : pytrees.tree.Tree
        The fitted tree, in scikit-learn's layout: a row goes left when
        ``x[feature] <= threshold``.
    train_error_ : int
        Training misclassifications of the returned tree.
    status_ : str
        ``"optimal"``, ``"time_limit"``, ``"budget_exhausted"`` or
        ``"error_bound_reached"``. Only the first means the tree is proven
        best for the given depth and support.
    statistics_ : dict
        Search counters: cache size and hits, solver calls, duration.

    References
    ----------
    C. E. Briţa, J. G. M. van der Linden and E. Demirović. Optimal
    Classification Trees for Continuous Feature Data Using Dynamic
    Programming with Branch-and-Bound. AAAI 2025.

    H. Kiossou, P. Schaus and S. Nijssen. Anytime Optimal Decision Tree
    Learning with Continuous Features. ECML PKDD 2026.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from pytrees import ConTreeClassifier
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
        # As scikit-learn requires, parameters are stored as given and only
        # validated in fit.
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
        X, encoded = self._validate(X, y)
        native = self._native_search(use_lds=self.use_lds)
        native.fit(X, encoded)
        self._store(native)
        return self

    def fit_anytime(self, X, y, callback=None):
        """Fit with the anytime search, reporting improvements as they happen.

        ``callback(error, seconds, status)`` is invoked each time a pass finds
        a better tree, so a caller can follow a long search or plot how the
        error decreases over time.

        Returns ``self``.
        """
        X, encoded = self._validate(X, y)
        native = self._native_search(use_lds=True)
        native.fit_anytime(X, encoded, callback)
        self._store(native)
        return self

    def _validate(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_all_finite=True)
        check_classification_targets(y)
        # The native search expects labels 0..k-1; classes_ maps them back to
        # the original labels.
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return np.ascontiguousarray(X), np.ascontiguousarray(encoded, dtype=np.int64)

    def _native_search(self, use_lds):
        return RawConTree(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_time=self.max_time,
            max_error=self.max_error,
            max_gap=self.max_gap,
            split_selection=self.split_selection,
            sort_by_heuristic=self.sort_by_heuristic,
            fast_d2=self.fast_d2,
            use_lds=use_lds,
            random_state=self.random_state,
            budget_schedule=self.budget_schedule,
        )

    def _store(self, native):
        # Keep only plain data, so the fitted estimator can be pickled.
        arrays = native.tree_arrays()
        self.tree_ = Tree(
            arrays["children_left"],
            arrays["children_right"],
            arrays["feature"],
            arrays["threshold"],
            np.where(arrays["children_left"] == LEAF, arrays["value"], LEAF),
            arrays["error"],
        )
        self.train_error_ = native.error
        self.status_ = native.status
        self.statistics_ = native.statistics

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
