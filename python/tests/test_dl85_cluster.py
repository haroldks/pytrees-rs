"""DL85Cluster: optimal trees for clustering, scored by an error function."""

import math
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from pytrees import DL85Cluster


@pytest.fixture
def rows(anneal):
    X, _ = anneal
    return X[:200]


def test_fit_returns_the_estimator(rows):
    clf = DL85Cluster(max_depth=1, min_sup=10)
    assert clf.fit(rows) is clf


def test_the_default_error_function_gives_a_finite_error(rows):
    clf = DL85Cluster(max_depth=2, min_sup=10)
    clf.fit(rows)
    assert math.isfinite(clf.train_error_)


def test_a_custom_error_function_set_after_construction_is_used(rows):
    calls = []

    def error(tids):
        calls.append(len(tids))
        return float(len(tids)), 0.0

    clf = DL85Cluster(max_depth=1, min_sup=10)
    clf.set_params(error_function=error)
    clf.fit(rows)
    assert calls


def test_it_finds_the_tightest_clustering_of_depth_two(rows):
    clf = DL85Cluster(max_depth=2, min_sup=10).fit(rows)
    single_cluster = np.linalg.norm(rows - rows.mean(axis=0), axis=1).sum()
    assert clf.n_clusters_ == 4
    assert clf.train_error_ == pytest.approx(500.47, abs=0.01)
    assert clf.train_error_ < single_cluster


def test_labels_are_the_predicted_clusters_of_the_training_rows(rows):
    clf = DL85Cluster(max_depth=2, min_sup=10).fit(rows)
    assert np.array_equal(clf.labels_, clf.predict(rows))
    assert set(clf.labels_) == set(range(clf.n_clusters_))
    assert np.array_equal(
        DL85Cluster(max_depth=2, min_sup=10).fit_predict(rows), clf.labels_
    )


def test_the_error_is_measured_in_x_error_when_given(rows):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(rows.shape[0], 3))
    clf = DL85Cluster(max_depth=1, min_sup=10).fit(rows, X_error=points)
    total = sum(
        np.linalg.norm(
            points[clf.labels_ == k] - points[clf.labels_ == k].mean(axis=0), axis=1
        ).sum()
        for k in range(clf.n_clusters_)
    )
    assert clf.train_error_ == pytest.approx(total)


def test_x_error_must_have_one_row_per_row_of_x(rows):
    with pytest.raises(ValueError, match="rows"):
        DL85Cluster().fit(rows, X_error=np.zeros((3, 2)))


def test_it_can_be_cloned_and_pickled(rows):
    clf = clone(DL85Cluster(max_depth=2, min_sup=10)).fit(rows)
    revived = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(revived.predict(rows), clf.predict(rows))
