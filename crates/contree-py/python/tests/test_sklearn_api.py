"""The estimator has to behave like a scikit-learn estimator, not merely look
like one: clone, grid search, pipelines and cross-validation all rely on
contracts that are easy to break by accident."""

import json
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from pytrees_continuous import ConTreeClassifier


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def test_it_passes_the_scikit_learn_estimator_checks():
    results = check_estimator(
        ConTreeClassifier(max_depth=2, fast_d2=True), on_fail=None
    )
    failures = [r for r in results if r["status"] != "passed"]
    assert not failures, "\n".join(
        f"{r['check_name']}: {r['exception']}" for r in failures
    )


def test_fit_returns_self(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True)
    assert clf.fit(X, y) is clf


def test_get_params_round_trips_through_clone():
    original = ConTreeClassifier(
        max_depth=4,
        min_sup=7,
        max_error=3,
        max_time=12.5,
        max_gap=2,
        split_selection="first",
        sort_by_heuristic=True,
        fast_d2=True,
        use_lds=True,
        budget_schedule="diagonal",
    )
    assert clone(original).get_params() == original.get_params()


def test_the_reported_error_matches_the_predictions(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    assert (clf.predict(X) != y).sum() == clf.train_error_


def test_a_finished_search_says_so(iris):
    X, y = iris
    assert ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y).status_ == "optimal"


def test_a_search_that_runs_out_of_time_does_not_claim_optimality():
    X, y = load_wine(return_X_y=True)
    clf = ConTreeClassifier(max_depth=5, max_time=1e-6).fit(X, y)
    assert clf.status_ != "optimal"


def test_deeper_trees_do_not_score_worse(iris):
    X, y = iris
    errors = [
        ConTreeClassifier(max_depth=d, fast_d2=True).fit(X, y).train_error_
        for d in (1, 2)
    ]
    assert errors == sorted(errors, reverse=True)


def test_non_integer_labels_work(iris):
    X, y = iris
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, names)

    assert list(clf.classes_) == ["setosa", "versicolor", "virginica"]
    assert set(clf.predict(X)) <= set(clf.classes_)
    assert (clf.predict(X) != names).sum() == clf.train_error_


def test_non_contiguous_integer_labels_work(iris):
    X, y = iris
    relabelled = np.array([10, 20, 30])[y]
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, relabelled)
    assert list(clf.classes_) == [10, 20, 30]
    assert (clf.predict(X) != relabelled).sum() == clf.train_error_


def test_it_survives_a_pickle_round_trip(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    revived = pickle.loads(pickle.dumps(clf))

    np.testing.assert_array_equal(revived.predict(X), clf.predict(X))
    assert revived.train_error_ == clf.train_error_
    assert revived.status_ == clf.status_


def test_it_works_in_a_pipeline_and_under_cross_validation(iris):
    X, y = iris
    pipeline = make_pipeline(
        StandardScaler(), ConTreeClassifier(max_depth=2, fast_d2=True)
    )
    assert cross_val_score(pipeline, X, y, cv=5).mean() > 0.8


def test_it_works_under_grid_search(iris):
    X, y = iris
    search = GridSearchCV(
        ConTreeClassifier(fast_d2=True),
        {"max_depth": [1, 2], "min_sup": [1, 5]},
        cv=3,
    ).fit(X, y)
    assert search.best_params_["max_depth"] in (1, 2)


def test_the_tree_arrays_describe_the_same_tree_predict_walks(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    tree = clf.tree_

    left, right = tree["children_left"], tree["children_right"]
    feature, threshold, value = tree["feature"], tree["threshold"], tree["value"]

    def walk(x):
        node = 0
        while left[node] != -1:
            node = left[node] if x[feature[node]] < threshold[node] else right[node]
        return value[node]

    walked = clf.classes_.take([walk(row) for row in X])
    np.testing.assert_array_equal(walked, clf.predict(X))


def test_leaves_are_marked_consistently(iris):
    X, y = iris
    tree = ConTreeClassifier(max_depth=3, fast_d2=True).fit(X, y).tree_
    for i in range(len(tree["children_left"])):
        is_leaf = tree["children_left"][i] == -1
        assert (tree["children_right"][i] == -1) == is_leaf
        assert (tree["feature"][i] == -1) == is_leaf
        assert np.isnan(tree["threshold"][i]) == is_leaf
        if is_leaf:
            assert tree["value"][i] >= 0


def test_decision_path_ends_at_the_predicted_leaf(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    paths = clf.decision_path(X)

    assert len(paths) == len(X)
    assert all(path[0] == 0 for path in paths)
    leaves = np.array([clf.tree_["value"][path[-1]] for path in paths])
    np.testing.assert_array_equal(clf.classes_.take(leaves), clf.predict(X))


def test_the_tree_serializes_to_json(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    assert json.loads(clf.tree_json_)


def test_the_anytime_search_is_usable_from_python(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, use_lds=True, fast_d2=True).fit(X, y)
    exhaustive = ConTreeClassifier(max_depth=2, fast_d2=True).fit(X, y)
    assert clf.train_error_ >= exhaustive.train_error_


def test_it_generalizes(iris):
    X, y = iris
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = ConTreeClassifier(max_depth=2, min_sup=5, fast_d2=True).fit(X_train, y_train)
    assert clf.score(X_test, y_test) > 0.8


def test_the_anytime_callback_sees_every_improvement():
    X, y = load_wine(return_X_y=True)
    seen = []

    clf = ConTreeClassifier(max_depth=2, fast_d2=True).fit_anytime(
        X,
        y,
        callback=lambda error, seconds, status: seen.append((error, seconds, status)),
    )

    assert seen, "the anytime search reported no improvement at all"
    errors = [error for error, _, _ in seen]
    assert errors == sorted(errors, reverse=True), "improvements must improve"
    assert errors[-1] == clf.train_error_
    assert all(seconds >= 0.0 for _, seconds, _ in seen)


def test_fit_anytime_returns_self_and_leaves_a_usable_estimator(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, fast_d2=True)
    assert clf.fit_anytime(X, y) is clf
    assert (clf.predict(X) != y).sum() == clf.train_error_


def test_a_callback_that_raises_propagates(iris):
    X, y = iris

    def boom(*_):
        raise KeyError("from the callback")

    with pytest.raises(KeyError, match="from the callback"):
        ConTreeClassifier(max_depth=2, fast_d2=True).fit_anytime(X, y, callback=boom)


def test_random_split_selection_is_reproducible_when_seeded(iris):
    X, y = iris
    trees = [
        ConTreeClassifier(
            max_depth=2, fast_d2=True, split_selection="random", random_state=7
        )
        .fit(X, y)
        .tree_json_
        for _ in range(2)
    ]
    assert trees[0] == trees[1]
