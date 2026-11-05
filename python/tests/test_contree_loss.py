"""ConTree with custom error functions, from Python.

The Rust tests check the search itself against a brute force; these check
what the Python layer adds: argument handling, warnings, errors raised from
the caller's function, and that the reported error is the error of the tree
returned.
"""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris

from pytrees import ConTreeClassifier, ThresholdSkippingWarning


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def misclassification(counts):
    """Misclassification on class counts, ties to the highest class."""
    counts = np.asarray(counts)
    label = len(counts) - 1 - int(np.argmax(counts[::-1]))
    return float(counts.sum() - counts[label]), label


def squared_misclassification(counts):
    """Breaks the skipping assumption: one row can raise the cost by more than 1."""
    error, label = misclassification(counts)
    return error * error, label


def cost_of(clf, X, y, row_cost=None):
    """What the rows `clf` misclassifies cost, each `row_cost` (default 1)."""
    row_cost = np.ones(len(y)) if row_cost is None else np.asarray(row_cost)
    return float(((clf.predict(X) != y) * row_cost).sum())


# --- error_function ----------------------------------------------------------


@pytest.mark.parametrize("fast_d2", [False, True])
def test_misclassification_as_an_error_function_is_the_built_in_error(iris, fast_d2):
    X, y = iris
    built_in = ConTreeClassifier(max_depth=3, fast_d2=fast_d2).fit(X, y)
    custom = ConTreeClassifier(
        max_depth=3, fast_d2=fast_d2, error_function=misclassification
    ).fit(X, y)
    assert custom.status_ == "optimal"
    assert custom.train_error_ == built_in.train_error_
    assert cost_of(custom, X, y) == custom.train_error_


def test_an_indices_error_function_can_read_the_rows(iris):
    X, y = iris
    received = []

    def error(rows):
        received.append(rows)
        rows = np.asarray(rows, dtype=int)
        return misclassification(np.bincount(y[rows], minlength=3))

    custom = ConTreeClassifier(
        max_depth=3, error_function=error, error_function_input="indices"
    ).fit(X, y)
    built_in = ConTreeClassifier(max_depth=3).fit(X, y)
    assert custom.train_error_ == built_in.train_error_
    assert all(isinstance(rows, list) for rows in received)
    assert all(0 <= row < len(y) for rows in received for row in rows)


def test_a_class_counts_error_function_receives_a_list_of_ints(iris):
    X, y = iris
    received = []

    def error(counts):
        received.append(counts)
        return misclassification(counts)

    ConTreeClassifier(max_depth=2, error_function=error).fit(X, y)
    assert received
    assert all(isinstance(counts, list) and len(counts) == 3 for counts in received)
    assert all(isinstance(count, int) for counts in received for count in counts)
    assert max(sum(counts) for counts in received) == len(y)


@pytest.mark.parametrize("use_lds", [False, True])
@pytest.mark.parametrize("fast_d2", [False, True])
def test_a_custom_cost_is_minimised(iris, use_lds, fast_d2):
    X, y = iris
    # Misclassifying class 2 costs ten times more.
    cost = np.array([1.0, 1.0, 10.0])

    def error(counts):
        counts = np.asarray(counts)
        errors = [float((counts * cost).sum() - counts[k] * cost[k]) for k in range(3)]
        label = int(np.argmin(errors))
        return errors[label], label

    clf = ConTreeClassifier(
        max_depth=2, use_lds=use_lds, fast_d2=fast_d2, error_function=error
    ).fit(X, y)
    assert clf.status_ == "optimal"
    assert clf.train_error_ == pytest.approx(cost_of(clf, X, y, cost[y]), rel=1e-12)


def test_an_exception_in_the_error_function_reaches_the_caller(iris):
    X, y = iris

    def failing(counts):
        raise KeyError("from the error function")

    with pytest.raises(KeyError, match="from the error function"):
        ConTreeClassifier(max_depth=2, error_function=failing).fit(X, y)


def test_a_prediction_that_is_not_a_class_is_a_value_error(iris):
    X, y = iris
    with pytest.raises(ValueError, match="not the index of one of the 3 classes"):
        ConTreeClassifier(max_depth=2, error_function=lambda counts: (0.0, 7)).fit(X, y)


def test_a_negative_cost_is_a_value_error(iris):
    X, y = iris
    with pytest.raises(ValueError, match="costs must be non-negative"):
        ConTreeClassifier(max_depth=2, error_function=lambda counts: (-1.0, 0)).fit(
            X, y
        )


def test_indices_without_an_error_function_is_a_value_error(iris):
    X, y = iris
    with pytest.raises(ValueError, match="needs an error_function"):
        ConTreeClassifier(error_function_input="indices").fit(X, y)


# --- threshold_skipping ------------------------------------------------------


def test_forcing_skipping_with_an_error_function_warns(iris):
    X, y = iris
    with pytest.warns(ThresholdSkippingWarning, match="optimal only if"):
        clf = ConTreeClassifier(
            max_depth=3, error_function=misclassification, threshold_skipping=True
        ).fit(X, y)
    # Misclassification meets the assumption: no second warning, still optimal.
    assert clf.status_ == "optimal"
    assert clf.statistics_["skipping_violations"] == 0
    assert clf.train_error_ == ConTreeClassifier(max_depth=3).fit(X, y).train_error_


def test_a_loss_that_breaks_forced_skipping_is_reported(iris):
    X, y = iris
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = ConTreeClassifier(
            max_depth=2,
            error_function=squared_misclassification,
            threshold_skipping=True,
        ).fit(X, y)
    messages = [
        str(w.message) for w in caught if w.category is ThresholdSkippingWarning
    ]
    assert any("broke the threshold-skipping assumption" in m for m in messages)
    assert clf.status_ == "not_proven"
    assert clf.statistics_["skipping_violations"] > 0


def test_forcing_skipping_is_allowed_with_a_minimum_support(iris):
    X, y = iris
    clf = ConTreeClassifier(max_depth=2, min_sup=5, threshold_skipping=True).fit(X, y)
    assert clf.status_ == "optimal"
    assert (
        clf.train_error_
        == ConTreeClassifier(max_depth=2, min_sup=5).fit(X, y).train_error_
    )


def test_the_built_in_error_does_not_warn_when_skipping_is_forced(iris):
    X, y = iris
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThresholdSkippingWarning)
        ConTreeClassifier(max_depth=2, threshold_skipping=True).fit(X, y)


@pytest.mark.parametrize("value", ["on", 1, None])
def test_an_unknown_threshold_skipping_is_a_value_error(iris, value):
    X, y = iris
    with pytest.raises(ValueError, match="threshold_skipping"):
        ConTreeClassifier(max_depth=2, threshold_skipping=value).fit(X, y)
