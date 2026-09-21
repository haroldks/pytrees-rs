"""DL85Cluster: optimal trees for clustering, scored by an error function.

Tests marked xfail are known bugs. They are strict, so fixing a bug makes
its test fail until the marker is removed.
"""

import math

import pytest

from pytrees import DL85Cluster


def known_bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


@pytest.fixture
def rows(anneal):
    X, _ = anneal
    return X[:200]


@known_bug("fit returns None")
def test_fit_returns_the_estimator(rows):
    clf = DL85Cluster(max_depth=1, min_sup=10)
    assert clf.fit(rows) is clf


@known_bug("the default error function is set after the native object is built")
def test_the_default_error_function_gives_a_finite_error(rows):
    clf = DL85Cluster(max_depth=2, min_sup=10)
    clf.fit(rows)
    assert math.isfinite(clf.results.error)


@known_bug("error_function is only read in __init__")
def test_a_custom_error_function_set_after_construction_is_used(rows):
    calls = []

    def error(tids):
        calls.append(len(tids))
        return float(len(tids)), 0.0

    clf = DL85Cluster(max_depth=1, min_sup=10)
    clf.set_params(error_function=error)
    clf.fit(rows)
    assert calls
