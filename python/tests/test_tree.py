"""The Tree every estimator exposes as tree_, and what the estimators do
with it: apply, decision_path, predict and to_dot share one implementation
and one convention, x[feature] <= threshold goes left."""

import pickle

import numpy as np
import pytest

from sklearn.datasets import load_iris

from pytrees import ConTreeClassifier, DL85Classifier, DL85Cluster, LGDTClassifier
from pytrees.tree import Tree


@pytest.fixture(
    params=[
        lambda X, y: DL85Classifier(max_depth=3, min_sup=5).fit(X, y),
        lambda X, y: LGDTClassifier(max_depth=3, min_sup=5).fit(X, y),
        lambda X, y: DL85Cluster(max_depth=2, min_sup=40).fit(X[:300]),
        None,
    ],
    ids=["DL85Classifier", "LGDTClassifier", "DL85Cluster", "ConTreeClassifier"],
)
def fitted(request, anneal):
    if request.param is None:
        # ConTree works on continuous features.
        X, y = load_iris(return_X_y=True)
        return ConTreeClassifier(max_depth=3).fit(X, y), X
    X, y = anneal
    return request.param(X, y), X


def test_apply_lands_on_leaves(fitted):
    clf, X = fitted
    leaves = clf.apply(X)
    assert (clf.tree_.children_left[leaves] == -1).all()


def test_predict_reads_the_leaf_values(fitted):
    clf, X = fitted
    values = clf.tree_.value[clf.apply(X)]
    expected = clf.classes_.take(values) if hasattr(clf, "classes_") else values
    assert np.array_equal(clf.predict(X), expected)


def test_decision_path_runs_from_the_root_to_the_leaf(fitted):
    clf, X = fitted
    path = clf.decision_path(X)
    assert path.shape == (len(X), clf.tree_.node_count)
    assert (path[:, 0].toarray() == 1).all()
    leaves = clf.apply(X)
    assert (path[np.arange(len(X)), leaves] == 1).all()
    assert path.sum(axis=1).max() <= clf.tree_.max_depth + 1


def test_each_step_of_a_path_follows_the_split_rule(fitted):
    clf, X = fitted
    tree, path = clf.tree_, clf.decision_path(X[:50]).toarray().astype(bool)
    for x, visited in zip(X[:50], path):
        node = 0
        while tree.children_left[node] != -1:
            assert visited[node]
            node = (
                tree.children_left[node]
                if x[tree.feature[node]] <= tree.threshold[node]
                else tree.children_right[node]
            )
        assert visited[node]


def test_to_dot_draws_every_node_with_its_error(fitted):
    clf, _ = fitted
    dot = clf.to_dot()
    tree = clf.tree_
    assert dot.startswith("digraph Tree {") and dot.endswith("}")
    assert dot.count("error = ") == tree.node_count
    assert dot.count('[label="true"]') == tree.node_count - tree.n_leaves
    assert dot.count('[label="false"]') == tree.node_count - tree.n_leaves


def test_to_dot_uses_names_when_given(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=1).fit(X, np.where(y == 0, "no", "yes"))
    dot = clf.to_dot(feature_names=[f"f{i}" for i in range(X.shape[1])])
    feature = clf.tree_.feature[0]
    assert f"f{feature} = 0" in dot
    assert "class = no" in dot or "class = yes" in dot


def test_to_dot_names_clusters(anneal):
    X, _ = anneal
    dot = DL85Cluster(max_depth=1, min_sup=40).fit(X[:300]).to_dot()
    assert "cluster = 0" in dot and "cluster = 1" in dot


def test_a_tree_survives_pickling(fitted):
    clf, X = fitted
    revived = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(
        revived.decision_path(X).toarray(), clf.decision_path(X).toarray()
    )


def test_a_value_on_the_threshold_goes_left():
    tree = Tree(
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        feature=[0, -1, -1],
        threshold=[2.5, np.nan, np.nan],
        value=[-1, 0, 1],
        error=[1.0, 0.0, 0.0],
    )
    X = np.array([[2.5], [np.nextafter(2.5, 3)]])
    assert list(tree.apply(X)) == [1, 2]
    assert tree.max_depth == 1
    assert "x0 <= 2.5" in tree.to_dot()


def stump(**overrides):
    arrays = dict(
        children_left=np.array([1, -1, -1]),
        children_right=np.array([2, -1, -1]),
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, np.nan, np.nan]),
    )
    arrays.update(overrides)
    return arrays


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(feature=np.array([0, -1])), "same, non-zero length"),
        (dict(children_left=np.array([7, -1, -1])), "outside the tree"),
        (dict(feature=np.array([3, -1, -1])), "tests feature 3"),
        (
            # Node 1 sends a row with x0 = 0 back to itself.
            dict(
                children_left=np.array([1, 1, -1]),
                children_right=np.array([2, 2, -1]),
                feature=np.array([0, 0, -1]),
                threshold=np.array([0.5, 0.5, np.nan]),
            ),
            "cycle",
        ),
    ],
)
def test_a_malformed_tree_is_a_value_error(overrides, message):
    from pytrees._native.tree import apply

    arrays = stump(**overrides)
    with pytest.raises(ValueError, match=message):
        apply(
            arrays["children_left"].astype(np.int64),
            arrays["children_right"].astype(np.int64),
            arrays["feature"].astype(np.int64),
            arrays["threshold"].astype(np.float64),
            np.zeros((2, 1)),
        )
