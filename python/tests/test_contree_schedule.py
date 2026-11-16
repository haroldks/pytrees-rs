"""The anytime search's budget schedule follows the split selection."""

import warnings

import pytest
from sklearn.datasets import load_iris

from pytrees import BudgetScheduleWarning, ConTreeClassifier


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def fit_quietly(iris, **params):
    X, y = iris
    with warnings.catch_warnings():
        warnings.simplefilter("error", BudgetScheduleWarning)
        return ConTreeClassifier(max_depth=2, **params).fit(X, y)


@pytest.mark.parametrize("split_selection", ["mid", "first", "random"])
def test_the_default_schedule_suits_every_split_selection(iris, split_selection):
    fit_quietly(iris, use_lds=True, split_selection=split_selection, random_state=0)


@pytest.mark.parametrize("split_selection", ["mid", "random"])
def test_diagonal_without_first_warns(iris, split_selection):
    X, y = iris
    with pytest.warns(BudgetScheduleWarning, match="only the `first`"):
        ConTreeClassifier(
            max_depth=2,
            use_lds=True,
            split_selection=split_selection,
            budget_schedule="diagonal",
            random_state=0,
        ).fit(X, y)


def test_diagonal_with_first_does_not_warn(iris):
    fit_quietly(iris, use_lds=True, split_selection="first", budget_schedule="diagonal")


def test_the_exhaustive_search_ignores_the_schedule(iris):
    fit_quietly(iris, split_selection="mid", budget_schedule="diagonal")


def test_fit_anytime_warns_too(iris):
    X, y = iris
    with pytest.warns(BudgetScheduleWarning):
        ConTreeClassifier(max_depth=2, budget_schedule="diagonal").fit_anytime(X, y)


def test_every_schedule_reaches_the_optimum(iris):
    X, y = iris
    exact = ConTreeClassifier(max_depth=2).fit(X, y).train_error_
    for split_selection in ["mid", "first"]:
        for budget_schedule in [None, "square"]:
            clf = fit_quietly(
                iris,
                use_lds=True,
                split_selection=split_selection,
                budget_schedule=budget_schedule,
            )
            assert clf.train_error_ == exact
            assert clf.status_ == "optimal"
