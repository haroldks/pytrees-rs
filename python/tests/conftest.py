import pathlib

import numpy as np
import pytest

TEST_DATA = (
    pathlib.Path(__file__).resolve().parents[2] / "crates" / "dtrees" / "test_data"
)


@pytest.fixture(scope="session")
def anneal():
    """A binary dataset: 812 rows, 93 features, labels 0 and 1.

    The Rust test suite (crates/dtrees/tests/dl85.rs) pins its optimal
    errors, so the Python results can be checked against the same numbers.
    """
    data = np.genfromtxt(TEST_DATA / "anneal.txt", delimiter=" ")
    return data[:, 1:], data[:, 0]
