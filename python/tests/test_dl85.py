"""DL85Classifier: the optimal search on binary features.

Tests marked xfail are known bugs. They are strict, so fixing a bug makes
its test fail until the marker is removed.
"""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from pytrees import DL85Classifier, LGDTClassifier
from pytrees.rules import GainRule


def known_bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def errors(clf, X, y):
    return int((np.asarray(clf.predict(X)) != y).sum())


# --- What already works --------------------------------------------------


@pytest.mark.parametrize("depth, expected", [(2, 137), (3, 112)])
def test_it_finds_the_same_optimum_as_the_rust_suite(anneal, depth, expected):
    X, y = anneal
    clf = DL85Classifier(max_depth=depth, min_sup=1)
    clf.fit(X, y)
    assert errors(clf, X, y) == expected


def test_its_predictions_reproduce_the_error_the_search_reports(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=3, min_sup=5)
    clf.fit(X, y)
    assert errors(clf, X, y) == clf.results.error


def test_it_is_never_worse_than_the_greedy_tree_of_the_same_depth(anneal):
    X, y = anneal
    optimal = DL85Classifier(max_depth=3, min_sup=1)
    greedy = LGDTClassifier(max_depth=3, min_sup=1)
    optimal.fit(X, y)
    greedy.fit(X, y)
    assert errors(optimal, X, y) <= errors(greedy, X, y)


# --- scikit-learn contracts ----------------------------------------------


@known_bug("fit returns None")
def test_fit_returns_the_estimator(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=1)
    assert clf.fit(X, y) is clf


def test_it_can_be_cloned():
    clone(DL85Classifier(max_depth=2))


@known_bug("the native object is built in __init__, so set_params is ignored")
def test_set_params_changes_the_search(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=1, min_sup=1)
    clf.set_params(max_depth=3)
    clf.fit(X, y)
    assert errors(clf, X, y) == 112


@known_bug("__init__ overwrites the policies when a rule is given")
def test_init_keeps_the_parameters_as_given():
    clf = DL85Classifier(gain=GainRule(), similarity_lb=True)
    assert clf.get_params()["similarity_lb"] is True


def test_it_works_under_cross_validation(anneal):
    X, y = anneal
    scores = cross_val_score(DL85Classifier(max_depth=2, min_sup=5), X, y, cv=3)
    assert scores.mean() > 0.7


@known_bug("the native object is built in __init__, so set_params is ignored")
def test_grid_search_actually_varies_the_depth(anneal):
    X, y = anneal
    search = GridSearchCV(DL85Classifier(min_sup=5), {"max_depth": [1, 3]}, cv=3).fit(
        X, y
    )
    first, second = search.cv_results_["mean_test_score"]
    assert first != second


@known_bug("the search results are a native object that cannot be pickled")
def test_a_fitted_model_survives_pickling(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y)
    revived = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(revived.predict(X), clf.predict(X))


# --- Labels --------------------------------------------------------------


@known_bug("there is no classes_, and predict returns a list")
def test_it_exposes_classes_and_predicts_an_array(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, [0, 1])
    assert isinstance(clf.predict(X), np.ndarray)


@known_bug("labels outside 0..k-1 make the Rust side panic")
def test_labels_do_not_have_to_start_at_zero(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y + 1)
    assert set(np.unique(clf.predict(X))) <= {1, 2}


@known_bug("labels must be numeric")
def test_labels_can_be_strings(anneal):
    X, y = anneal
    labels = np.where(y == 0, "no", "yes")
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, labels)
    assert set(np.unique(clf.predict(X))) <= {"no", "yes"}


# --- Errors --------------------------------------------------------------


@known_bug("non-binary values are silently truncated to integers")
def test_non_binary_features_are_a_value_error(anneal):
    X, y = anneal
    with pytest.raises(ValueError, match="binary"):
        DL85Classifier(max_depth=2).fit(X * 3.5, y)


@known_bug("an exception in the error function becomes a Rust panic")
def test_an_exception_in_a_custom_error_function_reaches_the_caller(anneal):
    X, y = anneal

    def failing(tids):
        raise ZeroDivisionError("from the user's error function")

    with pytest.raises(ZeroDivisionError, match="user's error function"):
        DL85Classifier(max_depth=2, error_function=failing).fit(X, y)


@known_bug("partial_fit checks has_data the wrong way round")
def test_partial_fit_runs_after_load_data(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.load_data(X, y)
    clf.partial_fit()
