"""The fitted tree every pytrees estimator exposes as ``tree_``.

It uses scikit-learn's layout: one entry per node in flat arrays, node 0 the
root, and ``children_left[i] == -1`` for a leaf. Every estimator shares it,
so prediction, decision paths and drawing work the same way for all of them.

**A row goes left when ``x[feature] <= threshold``**, and right otherwise, as
in scikit-learn. On binary features the threshold is 0.5: 0 goes left. The
rule lives in one place, ``pytrees._native.tree.apply``; everything else
here is derived from the leaf each row reaches.
"""

import numpy as np
from scipy.sparse import csr_matrix

from pytrees._native.tree import apply as _apply

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
        # Copies, so the tree owns its arrays whatever built them: arrays
        # backed by Rust memory cannot have their flags changed.
        self.children_left = np.array(children_left, dtype=np.int64)
        self.children_right = np.array(children_right, dtype=np.int64)
        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.value = np.array(value, dtype=np.int64)
        self.error = np.array(error, dtype=np.float64)
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

    def apply(self, X):
        """The index of the leaf each row of ``X`` reaches."""
        return _apply(
            self.children_left,
            self.children_right,
            self.feature,
            self.threshold,
            np.ascontiguousarray(X, dtype=np.float64),
        )

    def decision_path(self, X):
        """The nodes each row of ``X`` passes through, as a sparse indicator
        matrix of shape ``(n_samples, node_count)``, as in scikit-learn."""
        leaves = self.apply(X)
        n = len(leaves)
        # A leaf is reached by one path only, so a row's path is its leaf's.
        reached = csr_matrix(
            (np.ones(n, dtype=np.int64), (np.arange(n), leaves)),
            shape=(n, self.node_count),
        )
        path = reached @ self._paths_to_nodes()
        path.sort_indices()
        return path

    def _paths_to_nodes(self):
        """A ``(node_count, node_count)`` indicator: row ``i`` marks the
        nodes on the path from the root to node ``i``."""
        rows, columns = [], []
        stack = [(0, [0])]
        while stack:
            node, path = stack.pop()
            rows.extend([node] * len(path))
            columns.extend(path)
            if self.children_left[node] != LEAF:
                for child in (self.children_left[node], self.children_right[node]):
                    stack.append((child, path + [child]))
        ones = np.ones(len(rows), dtype=np.int64)
        return csr_matrix((ones, (rows, columns)), shape=(self.node_count,) * 2)

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
