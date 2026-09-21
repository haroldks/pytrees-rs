"""Bad input has to reach the caller as a Python exception with something
useful in it, not as a `pyo3_runtime.PanicException`."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pytrees import ConTreeClassifier
from pytrees._native.contree import RawConTree


def test_an_unknown_split_selection_is_a_value_error():
    with pytest.raises(ValueError, match="mid, first, random"):
        ConTreeClassifier(split_selection="middle").fit(np.zeros((4, 2)), [0, 1, 0, 1])


def test_a_support_larger_than_the_data_is_a_value_error():
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="min_sup"):
        ConTreeClassifier(min_sup=9).fit(X, y)


def test_predicting_before_fitting_is_a_not_fitted_error():
    with pytest.raises(NotFittedError):
        ConTreeClassifier().predict(np.zeros((2, 2)))


def test_the_native_object_says_so_too():
    with pytest.raises(RuntimeError, match="not been fitted"):
        RawConTree().predict(np.zeros((2, 2)))


def test_predicting_with_the_wrong_width_is_a_value_error():
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.array([0, 1] * 5)
    clf = ConTreeClassifier(max_depth=1).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((2, 5)))


def test_nan_features_are_rejected():
    X = np.zeros((4, 2))
    X[1, 1] = np.nan
    with pytest.raises(ValueError):
        ConTreeClassifier().fit(X, [0, 1, 0, 1])


def test_a_negative_label_reaches_the_native_layer_as_a_value_error():
    native = RawConTree(max_depth=1)
    X = np.ascontiguousarray(np.random.default_rng(0).normal(size=(4, 2)))
    with pytest.raises(ValueError, match="negative"):
        native.fit(X, np.array([-1, 0, 1, 0], dtype=np.int64))


def test_mismatched_lengths_are_a_value_error():
    native = RawConTree(max_depth=1)
    X = np.ascontiguousarray(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="rows but y has"):
        native.fit(X, np.array([0, 1], dtype=np.int64))


def test_an_unknown_budget_schedule_is_a_value_error():
    with pytest.raises(ValueError, match="budget schedule"):
        ConTreeClassifier(use_lds=True, budget_schedule="sideways").fit(
            np.zeros((4, 2)), [0, 1, 0, 1]
        )
