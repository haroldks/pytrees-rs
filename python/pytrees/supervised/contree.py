"""ConTree: optimal decision trees over continuous features."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from pytrees._native.contree import RawConTree

from ..base import TreeClassifier
from ..exceptions import ThresholdSkippingWarning
from ..tree import LEAF, Tree

__all__ = ["ConTreeClassifier"]


class ConTreeClassifier(ClassifierMixin, TreeClassifier, BaseEstimator):
    """Optimal decision tree classifier over continuous features (ConTree).

    Where a greedy learner picks the locally best split at each node, this
    searches for the tree of the given depth with the fewest training errors,
    with thresholds between observed values, without binarising the data.
    The error can be replaced by any leaf cost (``error_function``). The
    search is exact, so it can be slow; ``max_time`` bounds it and ``status_``
    says whether it finished. The anytime variant (``use_lds=True`` or
    ``fit_anytime``) finds good trees much earlier.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum number of levels of tests. Cost grows steeply with it.
    min_sup : int, default=1
        Minimum number of training instances in each leaf.
    max_error : float or None, default=None
        Stop once a tree with at most this error is found. ``None`` searches
        for the optimum.
    max_time : float, default=600.0
        Seconds before the search gives up and returns the best tree so far.
        When it does, ``status_`` is ``"time_limit"``.
    max_gap : float, default=0
        Accept a tree within this error of the optimum. Larger values prune
        more and finish sooner.
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
    error_function : callable, optional
        ``error_function(data) -> (error, prediction)``, called for each leaf
        the search considers, with what ``error_function_input`` selects, as a
        list. Replaces the built-in misclassification error; the
        tree then minimises the sum of the leaf errors. ``error`` must be a
        non-negative number, and ``prediction`` the index of a class in
        ``classes_``. With ``"indices"``, the depth-2 solver is not used,
        since it works from class counts.
    error_function_input : {"class_counts", "indices"}, default="class_counts"
        What ``error_function`` receives: the number of rows of each class in
        the leaf, as ints, or the indices of the leaf's rows in ``X``, e.g. to read ``y[rows]`` (wrap
        the list with ``np.asarray`` first). ``"indices"`` needs an
        ``error_function``.
    threshold_skipping : {"auto", True, False}, default="auto"
        Whether the search skips candidate thresholds it can prove useless.
        Skipping assumes what the built-in error guarantees: moving ``n``
        rows into a leaf never lowers its error and raises it by at most
        ``n``. ``"auto"`` skips for the built-in error only. ``True``
        forces skipping for an ``error_function`` too, with a
        ``ThresholdSkippingWarning``; the depth-2 solver then checks the
        assumption on the leaves it evaluates and, if the error function
        breaks it, warns again and sets ``status_`` to ``"not_proven"``.
        With ``min_sup > 1`` the interval pruning is never used, and ``True``
        forces only the depth-2 solver's skipping. ``False`` turns skipping
        off, which is slower but never changes the result for the built-in
        error.

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
    train_error_ : float
        Training error of the returned tree: its misclassifications,
        or the sum of its leaf errors with an ``error_function``.
    status_ : str
        ``"optimal"``, ``"time_limit"``, ``"budget_exhausted"``,
        ``"error_bound_reached"`` or ``"not_proven"`` (threshold skipping was
        forced and the error function broke its assumption). Only the first
        means the tree is proven best for the given depth and support.
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
        error_function=None,
        error_function_input="class_counts",
        threshold_skipping="auto",
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
        self.error_function = error_function
        self.error_function_input = error_function_input
        self.threshold_skipping = threshold_skipping

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
        return (
            np.ascontiguousarray(X),
            np.ascontiguousarray(encoded, dtype=np.int64),
        )

    def _skipping(self):
        """``threshold_skipping`` as the native search spells it."""
        if (
            isinstance(self.threshold_skipping, str)
            and self.threshold_skipping == "auto"
        ):
            return "auto"
        if self.threshold_skipping is True:
            if self.error_function is not None:
                warnings.warn(
                    "threshold_skipping=True with an error_function: the tree is optimal "
                    "only if adding n rows to a leaf never lowers its error and raises "
                    "it by at most n",
                    ThresholdSkippingWarning,
                    stacklevel=4,
                )
            return "on"
        if self.threshold_skipping is False:
            return "off"
        raise ValueError(
            f"threshold_skipping must be 'auto', True or False, not {self.threshold_skipping!r}"
        )

    def _native_search(self, use_lds):
        if self.error_function_input == "indices" and self.error_function is None:
            raise ValueError(
                'error_function_input="indices" needs an error_function: the '
                "built-in error only understands class counts"
            )
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
            error_function=self.error_function,
            error_function_input=self.error_function_input,
            threshold_skipping=self._skipping(),
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
        violations = self.statistics_["skipping_violations"]
        if violations:
            warnings.warn(
                f"the error_function broke the threshold-skipping assumption on {violations} "
                "moves of rows the search checked; the tree may not be optimal "
                "(status_ is 'not_proven'). Use threshold_skipping='auto'.",
                ThresholdSkippingWarning,
                stacklevel=3,
            )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
