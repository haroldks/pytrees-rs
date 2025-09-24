"""

pytrees is a Python wrapper around Rust-based decision tree
algorithms from pytrees-rs. It provides scikit-learn compatible interfaces for
both optimal and greedy decision tree construction with advanced optimization
features.

Key Features
------------
- **Optimal Decision Trees**: DL8.5 algorithm for globally optimal trees
- **Greedy Algorithms**: LGDT variants for fast approximate solutions
- **Rule-based Optimization**: Advanced stopping criteria and search control
- **Scikit-learn Compatible**: Drop-in replacement for sklearn decision trees
- **High Performance**: Rust backend with Python convenience

Main Classes
------------
- `DL85Classifier`: Optimal decision tree classifier using DL8.5 algorithm
- `LGDTClassifier`: Greedy decision tree classifier using LGDT algorithm
- `DL85Cluster`: Unsupervised clustering using optimal decision trees
- `DecisionTree`: Base class with common functionality

Quick Start
-----------
```python
from pytrees import DL85Classifier, LGDTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Optimal decision tree
optimal_clf = DL85Classifier(max_depth=3, min_sup=10)
optimal_clf.fit(X_train, y_train)
optimal_pred = optimal_clf.predict(X_test)

# Greedy decision tree (faster)
greedy_clf = LGDTClassifier(max_depth=3, min_sup=10)
greedy_clf.fit(X_train, y_train)
greedy_pred = greedy_clf.predict(X_test)

print(f"Optimal accuracy: {optimal_clf.accuracy_}")
print(f"Greedy accuracy: {greedy_clf.accuracy_}")
```
For more information, see the individual class documentation and examples.
"""

import common
from .base import DecisionTree
from .exceptions import *
from .supervised import LGDTClassifier, DL85Classifier
from .unsupervised import DL85Cluster
