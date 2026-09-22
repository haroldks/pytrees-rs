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


def test_other_threads_keep_running_during_a_search(anneal):
    # A search of about half a second. With the GIL held, this loop would
    # stall for all of it; released, it never waits longer than a thread
    # switch. LGDT releases it the same way, but its searches take a few
    # milliseconds, too short to tell the two apart.
    import threading
    import time

    from pytrees import DL85Classifier

    X, y = anneal
    worker = threading.Thread(
        target=DL85Classifier(max_depth=4, min_sup=40).fit, args=(X, y)
    )
    last = time.perf_counter()
    longest_pause = 0.0
    worker.start()
    while worker.is_alive():
        now = time.perf_counter()
        longest_pause, last = max(longest_pause, now - last), now
    assert longest_pause < 0.1
