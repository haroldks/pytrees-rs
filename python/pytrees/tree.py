"""The fitted tree every pytrees estimator exposes as ``tree_``.

It uses scikit-learn's layout: one entry per node in flat arrays, node 0 the
root, and ``children_left[i] == -1`` for a leaf. Every estimator shares it,
so prediction, decision paths and drawing work the same way for all of them.

**A row goes left when ``x[feature] <= threshold``**, and right otherwise, as
in scikit-learn. On binary features the threshold is 0.5: 0 goes left.
"""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["Tree"]

LEAF = -1


class Tree:
    """A fitted binary decision tree.

    Attributes
    ----------
    children_left, children_right : ndarray of int64
        The children of each node, ``-1`` at a leaf.
    feature : ndarray of int64
        The feature each test reads, ``-1`` at a leaf.
    threshold : ndarray of float64
        The threshold of each test, ``NaN`` at a leaf.
    value : ndarray of int64
        What each leaf predicts, ``-1`` at a test: the index of a class in
        the estimator's ``classes_``, or a cluster number.
    error : ndarray of float64
        The error of the subtree under each node.
    binary_features : bool
        Whether the features are 0/1. Only changes how ``to_dot`` writes a
        test: ``x3 = 0`` instead of ``x3 <= 0.5``.
    """

    def __init__(
        self,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        error,
        binary_features=False,
    ):
        self.children_left = np.asarray(children_left, dtype=np.int64)
        self.children_right = np.asarray(children_right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.value = np.asarray(value, dtype=np.int64)
        self.error = np.asarray(error, dtype=np.float64)
        self.binary_features = binary_features

    @property
    def node_count(self):
        return len(self.children_left)

    @property
    def n_leaves(self):
        return int((self.children_left == LEAF).sum())

    @property
    def max_depth(self):
        """The number of tests on the longest path from the root."""
        deepest, stack = 0, [(0, 0)]
        while stack:
            node, depth = stack.pop()
            if self.children_left[node] == LEAF:
                deepest = max(deepest, depth)
            else:
                stack.append((self.children_left[node], depth + 1))
                stack.append((self.children_right[node], depth + 1))
        return deepest

    def _descend(self, X, on_step=None):
        """Move every row of ``X`` down one level at a time, all rows at once;
        return the leaf each reaches. ``on_step(rows, nodes)`` sees each node
        a row passes through, the root and the leaf included."""
        node = np.zeros(X.shape[0], dtype=np.int64)
        rows = np.arange(X.shape[0])
        if on_step is not None:
            # A copy: `node` changes in place as the rows move down.
            on_step(rows, node.copy())
        while True:
            internal = self.children_left[node] != LEAF
            if not internal.any():
                return node
            at, where = node[internal], rows[internal]
            goes_left = X[where, self.feature[at]] <= self.threshold[at]
            node[internal] = np.where(
                goes_left, self.children_left[at], self.children_right[at]
            )
            if on_step is not None:
                on_step(where, node[internal])

    def apply(self, X):
        """The index of the leaf each row of ``X`` reaches."""
        return self._descend(X)

    def decision_path(self, X):
        """The nodes each row of ``X`` passes through, as a sparse indicator
        matrix of shape ``(n_samples, node_count)``, as in scikit-learn."""
        rows, nodes = [], []

        def record(where, at):
            rows.append(where)
            nodes.append(at)

        self._descend(X, record)
        rows, nodes = np.concatenate(rows), np.concatenate(nodes)
        ones = np.ones(len(rows), dtype=np.int64)
        path = csr_matrix((ones, (rows, nodes)), shape=(X.shape[0], self.node_count))
        path.sort_indices()
        return path

    def to_dot(self, feature_names=None, value_names=None, value_label="class"):
        """The tree in Graphviz DOT format.

        Each test reads ``<feature> <= <threshold>`` (``<feature> = 0`` on
        binary features), and its left edge, ``true``, is taken when the test
        holds. Each leaf reads ``<value_label> = <name>`` and its error.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Defaults to ``x0``, ``x1``, ...
        value_names : sequence, optional
            The name of each leaf value: an estimator passes its ``classes_``.
            Defaults to the value itself.
        value_label : str, default="class"
        """

        def feature_name(index):
            return (
                str(feature_names[index]) if feature_names is not None else f"x{index}"
            )

        def value_name(value):
            return str(value_names[value]) if value_names is not None else str(value)

        lines = ["digraph Tree {", "node [shape=box];"]
        for node in range(self.node_count):
            error = f"error = {self.error[node]:g}"
            left = self.children_left[node]
            if left == LEAF:
                text = f"{value_label} = {value_name(self.value[node])}"
            else:
                name = feature_name(self.feature[node])
                if self.binary_features:
                    text = f"{name} = 0"
                else:
                    text = f"{name} <= {self.threshold[node]:g}"
            lines.append(f'{node} [label="{_escape(text)}\\n{error}"];')
            if left != LEAF:
                lines.append(f'{node} -> {left} [label="true"];')
                lines.append(f'{node} -> {self.children_right[node]} [label="false"];')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Tree(node_count={self.node_count}, n_leaves={self.n_leaves}, max_depth={self.max_depth})"


def _escape(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')
