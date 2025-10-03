# Quick Start Guide

## 5-Minute Tutorial

### Basic Classification

```python
from pytrees import DL85Classifier, LGDTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = (X > 0).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimal decision tree (guaranteed best solution)
optimal_clf = DL85Classifier(max_depth=3, min_sup=10)
optimal_clf.fit(X_train, y_train)
optimal_pred = optimal_clf.predict(X_test)
optimal_accuracy = accuracy_score(y_test, optimal_pred)

print(f"Optimal Tree Accuracy: {optimal_accuracy:.3f}")
print(f"Training Time: {optimal_clf.results.duration:.3f}s")

# Greedy decision tree (fast approximate solution)
greedy_clf = LGDTClassifier(max_depth=3, min_sup=10)
greedy_clf.fit(X_train, y_train)
greedy_pred = greedy_clf.predict(X_test)
greedy_accuracy = accuracy_score(y_test, greedy_pred)

print(f"Greedy Tree Accuracy: {greedy_accuracy:.3f}")
print(f"Training Time: {greedy_clf.results.duration:.3f}s")
```

### Tree Visualization

```python
from pytrees import DL85Classifier, LGDTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = (X > 0).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

optimal_clf = DL85Classifier(max_depth=3, min_sup=10)
optimal_clf.fit(X_train, y_train)

dot_string = optimal_clf.to_dot()

# Save to file
with open('tree.dot', 'w') as f:
    f.write(dot_string)

# Or use with graphviz (if installed)
try:
    import graphviz
    graph = graphviz.Source(dot_string)
    graph.view()
    print("Tree visualization saved as tree.png")
except ImportError:
    print("Install graphviz for automatic visualization: pip install graphviz")
```


### Advanced Configuration
```python
from pytrees import DL85Classifier, ExposedHeuristic, ExposedGainRule, ExposedPurityRule
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = (X > 0).astype(float)
# Advanced DL85 with rules and heuristics
advanced_clf = DL85Classifier(
    max_depth=4,
    min_sup=5,
    max_time=300.0,
    heuristic=ExposedHeuristic.InformationGain,
    gain=ExposedGainRule(min_gain=0.01, epsilon=1e-4),
    purity=ExposedPurityRule(min_purity=0.9)
)
advanced_clf.fit(X, y)
print(advanced_clf.score(X, y))
```



## Next Steps

Now that you've got the basics down:

1. **[Python Library](./python/README.md)**: Explore the complete API
2. **[Examples](./examples/classification.md)**: See more real-world applications
3. **[Key Concepts](./concepts.md)**: Understand the algorithms and theory
4. **[Advanced Features](./python/advanced.md)**: Rules, custom functions, and optimization
