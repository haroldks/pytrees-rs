"""
Unsupervised Learning Module for PyTrees
========================================

This module provides unsupervised learning algorithms based on optimal decision trees.
Unlike traditional clustering methods, these algorithms create interpretable cluster
definitions using feature-based rules derived from decision tree structures.

Classes
-------
DL85Cluster : Optimal decision tree-based clustering
    Uses the DL8.5 algorithm to create optimal clustering solutions where each
    cluster is defined by a set of interpretable feature-based rules.

Key Features
------------
- **Interpretable Clusters**: Each cluster is defined by clear feature-based rules
- **Optimal Solutions**: Guarantees globally optimal clustering within constraints
- **Flexible Error Functions**: Support for custom clustering objectives
- **Tree Visualization**: Integration with decision tree visualization tools

Example Usage
-------------
```python
from pytrees.unsupervised import DL85Cluster
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)

# Perform optimal clustering
clusterer = DL85Cluster(max_depth=3, min_sup=10)
clusterer.fit(X)

# Get cluster assignments
cluster_labels = clusterer.predict(X)
print(f"Found {len(set(cluster_labels))} clusters")

# Visualize cluster rules
dot_string = clusterer.to_dot()
print("Cluster definitions:")
print(dot_string)
```

See Also
--------
pytrees.supervised : Supervised learning algorithms
pytrees.DecisionTree : Base decision tree functionality
"""

from .clustering import DL85Cluster
