"""SPLIT in ConTreeClassifier: a lookahead with a greedy completion below it."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_wine

from pytrees import BudgetScheduleWarning, ConTreeClassifier

DEPTH = 3


@pytest.fixture(scope="module")
def wine():
    return load_wine(return_X_y=True)


@pytest.fixture(scope="module")
def optimum(wine):
    X, y = wine
    return ConTreeClassifier(max_depth=DEPTH).fit(X, y).train_error_


def misclassified(clf, X, y):
    return int(np.sum(clf.predict(X) != y))


def test_plain_split_is_a_heuristic(wine, optimum):
    X, y = wine
    clf = ConTreeClassifier(max_depth=DEPTH, lookahead=0).fit(X, y)
    assert clf.status_ == "budget_exhausted"
    assert clf.train_error_ >= optimum
    assert clf.train_error_ == misclassified(clf, X, y)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"use_lds": True},
        {"use_lds": True, "lookahead_sync": True},
        {"lookahead_pruning": False},
        {"split_selection": "first"},
    ],
)
def test_anytime_split_ends_on_the_optimum(wine, optimum, params):
    X, y = wine
    clf = ConTreeClassifier(
        max_depth=DEPTH, lookahead=0, anytime_lookahead=True, **params
    ).fit(X, y)
    assert clf.status_ == "optimal"
    assert clf.train_error_ == optimum
    assert clf.train_error_ == misclassified(clf, X, y)


def test_a_lookahead_the_depth_2_solver_covers_is_exact(wine, optimum):
    X, y = wine
    clf = ConTreeClassifier(max_depth=DEPTH, lookahead=DEPTH - 2).fit(X, y)
    assert clf.status_ == "optimal"
    assert clf.train_error_ == optimum


def test_fit_anytime_reports_split_improvements(wine, optimum):
    X, y = wine
    seen = []
    clf = ConTreeClassifier(
        max_depth=DEPTH, lookahead=0, anytime_lookahead=True
    ).fit_anytime(X, y, callback=lambda error, seconds, status: seen.append(error))
    assert seen, "no improvement was reported"
    assert seen == sorted(seen, reverse=True)
    assert clf.status_ == "optimal"
    assert clf.train_error_ == optimum


def test_anytime_lookahead_needs_a_lookahead(wine):
    X, y = wine
    with pytest.raises(ValueError, match="needs a lookahead"):
        ConTreeClassifier(anytime_lookahead=True).fit(X, y)


def test_the_schedule_warning_only_comes_with_the_lds_budgets(wine):
    X, y = wine
    params = dict(
        max_depth=DEPTH,
        lookahead=0,
        anytime_lookahead=True,
        split_selection="mid",
        budget_schedule="diagonal",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", BudgetScheduleWarning)
        ConTreeClassifier(**params).fit(X, y)
        ConTreeClassifier(**params).fit_anytime(X, y)
    with pytest.warns(BudgetScheduleWarning):
        ConTreeClassifier(use_lds=True, **params).fit(X, y)
