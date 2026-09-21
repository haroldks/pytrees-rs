"""LGDTClassifier: the greedy search on binary features.

Tests marked xfail are known bugs. They are strict, so fixing a bug makes
its test fail until the marker is removed.
"""

import numpy as np
import pytest
from sklearn.base import clone

from pytrees import LGDTClassifier


def known_bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def errors(clf, X, y):
    return int((np.asarray(clf.predict(X)) != y).sum())


def test_its_predictions_reproduce_the_error_the_search_reports(anneal):
    X, y = anneal
    clf = LGDTClassifier(max_depth=3, min_sup=1)
    clf.fit(X, y)
    assert errors(clf, X, y) == clf.results.error


def test_its_result_on_anneal_does_not_drift(anneal):
    X, y = anneal
    clf = LGDTClassifier(max_depth=3, min_sup=1)
    clf.fit(X, y)
    assert errors(clf, X, y) == 119


def test_fit_returns_the_estimator(anneal):
    X, y = anneal
    clf = LGDTClassifier(max_depth=1)
    assert clf.fit(X, y) is clf


def test_it_can_be_cloned():
    clone(LGDTClassifier(max_depth=2))


def test_labels_do_not_have_to_start_at_zero(anneal):
    X, y = anneal
    clf = LGDTClassifier(max_depth=2)
    clf.fit(X, y + 1)
    assert set(np.unique(clf.predict(X))) <= {1, 2}


def test_an_invalid_support_is_a_value_error_that_says_why(anneal):
    X, y = anneal
    with pytest.raises(ValueError, match="min_sup"):
        LGDTClassifier(min_sup=0).fit(X, y)


@known_bug("at depth 2 or less, the Rust search prints the tree on stdout")
def test_fit_prints_nothing(anneal, capfd):
    X, y = anneal
    LGDTClassifier(max_depth=2).fit(X, y)
    assert capfd.readouterr().out == ""
