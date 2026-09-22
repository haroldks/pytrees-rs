"""Decision trees learned by search, with a scikit-learn interface.

pytrees wraps Rust implementations of four learners:

- ``DL85Classifier``: optimal decision trees over binary features (DL8.5),
  with optional rules that make the search anytime (LDS-DL8.5, Top-k,
  CA-DL8.5; see ``pytrees.rules``).
- ``LGDTClassifier``: trees grown top-down like CART, but each test is chosen
  with a depth-2 lookahead (LGDT).
- ``ConTreeClassifier``: optimal decision trees over continuous features
  (ConTree), with an anytime variant.
- ``DL85Cluster``: clustering with an optimal decision tree.

Every estimator exposes its fitted tree as ``tree_``, a ``pytrees.tree.Tree``.

Example
-------
>>> from sklearn.datasets import load_iris
>>> from pytrees import ConTreeClassifier
>>> X, y = load_iris(return_X_y=True)
>>> clf = ConTreeClassifier(max_depth=2).fit(X, y)
>>> clf.status_
'optimal'
"""

from .base import DecisionTree
from .exceptions import *
from .supervised import ConTreeClassifier, DL85Classifier, LGDTClassifier
from .unsupervised import DL85Cluster
