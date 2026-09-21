import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .exceptions import TreeNotFoundError
from .tree import LEAF, Tree


def validate_binary_classification(estimator, X, y):
    """Check ``X`` and ``y`` for a binary-feature classifier.

    Returns ``X`` as float64, the classes, and ``y`` encoded as indices into
    them: the Rust side needs labels 0..k-1, and this is what lets users pass
    any labels at all.
    """
    X, y = validate_data(estimator, X, y, dtype=np.float64, ensure_all_finite=True)
    check_classification_targets(y)
    check_binary(estimator, X)
    classes, encoded = np.unique(y, return_inverse=True)
    return X, classes, encoded


def check_binary(estimator, X):
    if not np.isin(X, (0.0, 1.0)).all():
        raise ValueError(
            f"{type(estimator).__name__} needs binary features: every value "
            "of X must be 0 or 1"
        )


def tree_from_native(native):
    """The ``Tree`` of a fitted dtrees search, or ``None`` if it found none.

    Its leaves hold what the search output, as floats on the Rust side: class
    indices for the classifiers.
    """
    arrays = native.tree_arrays()
    leaf = arrays["children_left"] == LEAF
    if leaf[0] and np.isnan(arrays["value"][0]):
        return None
    value = np.where(leaf, np.nan_to_num(arrays["value"]), LEAF)
    return Tree(
        arrays["children_left"],
        arrays["children_right"],
        arrays["feature"],
        arrays["threshold"],
        value,
        arrays["error"],
        binary_features=True,
    )


class DecisionTree:
    """What every pytrees estimator does with its fitted ``tree_``, a
    ``pytrees.tree.Tree``: find leaves, report paths, and draw it."""

    # Set by the estimators over binary features, which check it on predict.
    _binary_features = False
    # How to_dot names what a leaf holds.
    _value_label = "class"

    def _fitted_tree(self):
        check_is_fitted(self, "tree_")
        if self.tree_ is None:
            raise TreeNotFoundError(
                "the search found no tree, so there is nothing to predict with"
            )
        return self.tree_

    def _check_X(self, X):
        X = validate_data(
            self, X, dtype=np.float64, ensure_all_finite=True, reset=False
        )
        if self._binary_features:
            check_binary(self, X)
        return X

    def _value_names(self):
        """The names of the leaf values, for to_dot; ``None`` shows them as is."""
        return None

    def _store_dtrees_result(self, native):
        """Keep the tree, error and statistics of a fitted RawDL85 or RawLGDT."""
        self.tree_ = tree_from_native(native)
        self.train_error_ = native.error
        self.statistics_ = native.statistics

    def apply(self, X):
        """The index in ``tree_`` of the leaf each row of ``X`` reaches."""
        tree = self._fitted_tree()
        return tree.apply(self._check_X(X))

    def decision_path(self, X):
        """The nodes each row of ``X`` passes through, as a sparse indicator
        matrix of shape ``(n_samples, tree_.node_count)``."""
        tree = self._fitted_tree()
        return tree.decision_path(self._check_X(X))

    def to_dot(self, feature_names=None, class_names=None):
        """The fitted tree in Graphviz DOT format; see ``Tree.to_dot``.

        ``feature_names`` defaults to ``feature_names_in_`` when ``fit`` saw
        a DataFrame, and ``class_names`` to ``classes_``.
        """
        tree = self._fitted_tree()
        if feature_names is None:
            feature_names = getattr(self, "feature_names_in_", None)
        if class_names is None:
            class_names = self._value_names()
        return tree.to_dot(feature_names, class_names, self._value_label)


class TreeClassifier(DecisionTree):
    """A classifier over its ``tree_``: leaves hold indices into ``classes_``."""

    def predict(self, X):
        """Classify each row of ``X``."""
        tree = self._fitted_tree()
        return self.classes_.take(tree.value[tree.apply(self._check_X(X))])

    def _value_names(self):
        return self.classes_
