import json

import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .exceptions import TreeNotFoundError


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


class DecisionTree:
    """Behaviour shared by the estimators over binary features.

    A fitted estimator holds its tree in ``tree_``, as flat arrays in the same
    layout as ``ConTreeClassifier.tree_``:

    - ``children_left``, ``children_right``: child indices, ``-1`` at a leaf
    - ``feature``: the feature node ``i`` tests; a row goes left when it is 0
    - ``threshold``: 0.5 at every test, ``NaN`` at a leaf
    - ``value``: a leaf's prediction, ``NaN`` at a test
    - ``error``: the error of the subtree under each node

    ``tree_`` is ``None`` when the search found no tree, for instance when
    ``max_error`` is below the best error achievable.
    """

    def _set_tree(self, output):
        """Store the tree and statistics of a native search output."""
        tree = output.tree_arrays
        found = tree["children_left"][0] != -1 or not np.isnan(tree["value"][0])
        self.tree_ = tree if found else None
        self.train_error_ = output.error
        self.statistics_ = json.loads(output.statistics)

    def _leaf_values(self, X):
        """The value of the leaf each row of ``X`` reaches."""
        check_is_fitted(self, "tree_")
        if self.tree_ is None:
            raise TreeNotFoundError(
                "the search found no tree, so there is nothing to predict with"
            )
        X = validate_data(
            self, X, dtype=np.float64, ensure_all_finite=True, reset=False
        )
        check_binary(self, X)
        tree = self.tree_
        node = np.zeros(X.shape[0], dtype=np.intp)
        rows = np.arange(X.shape[0])
        # One step down per level, for every row that is not at a leaf yet.
        while True:
            internal = tree["children_left"][node] != -1
            if not internal.any():
                return tree["value"][node]
            at, where = node[internal], rows[internal]
            goes_left = X[where, tree["feature"][at]] < tree["threshold"][at]
            node[internal] = np.where(
                goes_left, tree["children_left"][at], tree["children_right"][at]
            )

    def _leaf_label(self, value):
        """A leaf's record field in ``to_dot``."""
        return f"{{value|{value:g}}}"

    def to_dot(self):
        """The fitted tree in Graphviz DOT format.

        Tests read ``feature|<index>``, leaves ``class|<label>`` (``value``
        for clustering) with their error. The edge labelled 0 is taken when
        the feature is 0.
        """
        check_is_fitted(self, "tree_")
        lines = ["digraph Tree {", "graph [ranksep=0];", "node [shape=record];"]
        tree = self.tree_
        if tree is not None:
            for node in range(len(tree["feature"])):
                error = f"{{error|{tree['error'][node]:g}}}"
                left = tree["children_left"][node]
                if left == -1:
                    leaf = self._leaf_label(tree["value"][node])
                    lines.append(f'{node} [label="{{{leaf}|{error}}}"];')
                else:
                    test = f"{{feature|{tree['feature'][node]}}}"
                    lines.append(f'{node} [label="{{{test}|{error}}}"];')
                    lines.append(f"{node} -> {left} [label=0];")
                    lines.append(f"{node} -> {tree['children_right'][node]} [label=1];")
        lines.append("}")
        return "\n".join(lines)


class TreeClassifier(DecisionTree):
    """What DL85Classifier and LGDTClassifier add: leaves hold class indices."""

    def predict(self, X):
        """Classify each row of ``X``."""
        # _leaf_values checks that the model is fitted, so it must run before
        # anything reads classes_.
        values = self._leaf_values(X)
        return self.classes_.take(values.astype(np.intp))

    def _leaf_label(self, value):
        return f"{{class|{self.classes_[int(value)]}}}"
