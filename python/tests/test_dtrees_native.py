"""The native DL8.5 and LGDT searches check their input themselves: bad
data must reach the caller as a ValueError, not as a panic or a truncation,
even when the Python layer is bypassed."""

import numpy as np
import pytest

from pytrees._native.dtrees import RawDL85, RawLGDT

SEARCHES = [RawDL85, RawLGDT]
X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float64)
Y = np.array([0, 1, 1, 0], dtype=np.int64)


@pytest.mark.parametrize("search", SEARCHES)
def test_an_unfitted_search_says_so(search):
    with pytest.raises(RuntimeError, match="not been fitted"):
        search().tree_arrays()


@pytest.mark.parametrize("search", SEARCHES)
def test_non_binary_features_are_a_value_error(search):
    with pytest.raises(ValueError, match="0 or 1"):
        search().fit(X * 2, Y)


@pytest.mark.parametrize("search", SEARCHES)
@pytest.mark.parametrize("labels", [[0, 2, 2, 0], [0, -1, -1, 0]])
def test_labels_must_be_encoded_as_0_to_k_minus_1(search, labels):
    with pytest.raises(ValueError, match="0..k-1"):
        search().fit(X, np.array(labels, dtype=np.int64))


@pytest.mark.parametrize("search", SEARCHES)
def test_x_and_y_must_have_the_same_rows(search):
    with pytest.raises(ValueError, match="rows"):
        search().fit(X, Y[:3])


def test_an_unknown_criterion_is_a_value_error():
    with pytest.raises(ValueError, match="error, information_gain"):
        RawLGDT(criterion="gini")
