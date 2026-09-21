"""DL85Classifier: the optimal search on binary features.

Tests marked xfail are known bugs. They are strict, so fixing a bug makes
its test fail until the marker is removed.
"""

import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score

from pytrees import DL85Classifier, LGDTClassifier
from pytrees.rules import DiscrepancyRule, GainRule


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
    assert errors(clf, X, y) == clf.train_error_


def test_it_is_never_worse_than_the_greedy_tree_of_the_same_depth(anneal):
    X, y = anneal
    optimal = DL85Classifier(max_depth=3, min_sup=1)
    greedy = LGDTClassifier(max_depth=3, min_sup=1)
    optimal.fit(X, y)
    greedy.fit(X, y)
    assert errors(optimal, X, y) <= errors(greedy, X, y)


def test_tree_holds_the_optimal_tree_in_scikit_learns_layout(anneal):
    X, y = anneal
    tree = DL85Classifier(max_depth=2, min_sup=1).fit(X, y).tree_
    leaves = tree.children_left == -1
    assert (tree.node_count, tree.n_leaves, tree.max_depth) == (7, 4, 2)
    assert np.isnan(tree.threshold[leaves]).all()
    assert (tree.threshold[~leaves] == 0.5).all()
    assert (tree.value[~leaves] == -1).all()
    assert tree.error[0] == 137


def test_a_search_that_finishes_reports_optimal(anneal):
    X, y = anneal
    assert DL85Classifier(max_depth=2).fit(X, y).status_ == "optimal"


# --- scikit-learn contracts ----------------------------------------------


def test_fit_returns_the_estimator(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=1)
    assert clf.fit(X, y) is clf


def test_it_can_be_cloned():
    clone(DL85Classifier(max_depth=2))


def test_set_params_changes_the_search(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=1, min_sup=1)
    clf.set_params(max_depth=3)
    clf.fit(X, y)
    assert errors(clf, X, y) == 112


def test_init_keeps_the_parameters_as_given():
    clf = DL85Classifier(gain=GainRule(), similarity_lb=True)
    assert clf.get_params()["similarity_lb"] is True


def test_it_works_under_cross_validation(anneal):
    X, y = anneal
    scores = cross_val_score(DL85Classifier(max_depth=2, min_sup=5), X, y, cv=3)
    assert scores.mean() > 0.7


def test_grid_search_actually_varies_the_depth(anneal):
    X, y = anneal
    search = GridSearchCV(DL85Classifier(min_sup=5), {"max_depth": [1, 3]}, cv=3).fit(
        X, y
    )
    first, second = search.cv_results_["mean_test_score"]
    assert first != second


def test_a_fitted_model_survives_pickling(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y)
    revived = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(revived.predict(X), clf.predict(X))


# --- Labels --------------------------------------------------------------


def test_it_exposes_classes_and_predicts_an_array(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, [0, 1])
    assert isinstance(clf.predict(X), np.ndarray)


def test_labels_do_not_have_to_start_at_zero(anneal):
    X, y = anneal
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, y + 1)
    assert set(np.unique(clf.predict(X))) <= {1, 2}


def test_labels_can_be_strings(anneal):
    X, y = anneal
    labels = np.where(y == 0, "no", "yes")
    clf = DL85Classifier(max_depth=2)
    clf.fit(X, labels)
    assert set(np.unique(clf.predict(X))) <= {"no", "yes"}


# --- Errors --------------------------------------------------------------


def test_non_binary_features_are_a_value_error(anneal):
    X, y = anneal
    with pytest.raises(ValueError, match="binary"):
        DL85Classifier(max_depth=2).fit(X * 3.5, y)


def test_an_exception_in_a_custom_error_function_reaches_the_caller(anneal):
    X, y = anneal

    def failing(tids):
        raise ZeroDivisionError("from the user's error function")

    with pytest.raises(ZeroDivisionError, match="user's error function"):
        DL85Classifier(max_depth=2, error_function=failing).fit(X, y)


@pytest.mark.parametrize("estimator", [DL85Classifier, LGDTClassifier])
def test_scikit_learn_sees_a_classifier(estimator):
    # With BaseEstimator ahead of ClassifierMixin the tag is lost, and
    # cross-validation silently falls back to unstratified folds.
    assert is_classifier(estimator())


@pytest.mark.parametrize("estimator", [DL85Classifier, LGDTClassifier])
def test_predicting_before_fitting_is_a_not_fitted_error(anneal, estimator):
    X, _ = anneal
    with pytest.raises(NotFittedError):
        estimator().predict(X)


@pytest.mark.parametrize("estimator", [DL85Classifier, LGDTClassifier])
def test_a_single_class_gives_a_tree_that_predicts_it(anneal, estimator):
    X, _ = anneal
    clf = estimator(max_depth=2).fit(X, np.full(len(X), "only"))
    assert clf.train_error_ == 0
    assert (clf.predict(X) == "only").all()


def test_indices_without_an_error_function_is_a_value_error(anneal):
    X, y = anneal
    with pytest.raises(ValueError, match="needs an error_function"):
        DL85Classifier(error_function_input="indices").fit(X, y)


# --- Anytime search ------------------------------------------------------


def test_fit_anytime_reports_only_improvements_and_ends_at_the_optimum(anneal):
    X, y = anneal
    seen = []
    clf = DL85Classifier(max_depth=3, min_sup=1, discrepancy=DiscrepancyRule())
    clf.fit_anytime(X, y, lambda error, seconds, status: seen.append(error))
    assert seen == sorted(set(seen), reverse=True)
    assert seen[-1] == errors(clf, X, y) == 112


def test_an_exception_in_the_callback_reaches_the_caller(anneal):
    X, y = anneal

    def failing(error, seconds, status):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        DL85Classifier(max_depth=2).fit_anytime(X, y, failing)
