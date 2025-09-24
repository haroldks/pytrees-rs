# Introduction

**pytrees-rs** is a high-performance Python library for optimal and greedy decision tree learning, powered by Rust.
The library brings state-of-the-art decision tree algorithms to Python developers. It provides both **optimal** and **fast approximate** solutions, giving the flexibility to choose between guaranteed optimality and computational efficiency.

## Key Features

### **Optimal Decision Trees**
- **DL8.5 Algorithm**: Guarantees globally optimal decision trees within constraints
- **Mathematical Proof**: Every solution comes with optimality guarantees
- **Flexible Constraints**: Control depth, minimum support, training time, and custom objectives

### **High-Performance Greedy Algorithms**
- **LGDT Algorithm**: Fast approximate solutions for large datasets using different search strategies
- **Multiple Search Strategies**: Depth-2 optimal error or Depth-2 optimal information gain
- **Scalable**: Handles large datasets samples and depths efficiently

### **Advanced Features**
- **Rule-based Optimization**: Fine-tune search behavior with discrepancy, gain, and purity rules
- **Custom Error Functions**: Define specialized objectives for your domain
- **Interpretable Clustering**: Decision tree-based unsupervised learning
- **Tree Visualization**: Built-in support for Graphviz visualization

### **Python-First Design**
- **Scikit-learn Compatible**: Drop-in replacement for sklearn decision trees
- **Pythonic API**: Intuitive interface following Python conventions
- **Comprehensive Documentation**: Extensive guides, examples, and API reference


## Quick Example

```python
from pytrees import DL85Classifier, LGDTClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Optimal decision tree
optimal_clf = DL85Classifier(max_depth=4, min_sup=10)
optimal_clf.fit(X, y)
print(f"Optimal accuracy: {optimal_clf.score(X, y):.3f}")

# Less greedy decision tree
greedy_clf = LGDTClassifier(max_depth=4, min_sup=10)
greedy_clf.fit(X, y)
print(f"Greedy accuracy: {greedy_clf.score(X, y):.3f}")

# Visualize the optimal tree
dot_string = optimal_clf.to_dot()
try:
    import graphviz
    graphviz.Source(dot_string).view()
except Exception as e:
    print("Failed to visualize tree")
```

## Core Algorithms

### DL8.5: Optimal Decision Tree Learning

DL8.5 algorithm finds **globally optimal decision trees** using dynamic programming and branch-and-bound search. Unlike greedy methods, DL8.5 guarantees the best possible tree within your specified constraints.

**Key Characteristics:**
- **Optimality**: If given enough time it is guaranteed to find the best solution
- **Flexibility**: Supports custom error functions and constraints
- **Efficiency**: Advanced caching and pruning techniques
- **Completeness**: Will find a solution if one exists within limits

**Academic Reference:** Aglin, G., Nijssen, S., & Schaus, P. (2020). Learning optimal decision trees using caching branch-and-bound search. *AAAI Conference on Artificial Intelligence*.

### LDS-DL8.5 and CADL8.5: Anytime Optimal Learning

An enhanced version of DL8.5 that provides **anytime optimization**. It  gives progressively better solutions as training time increases. This algorithm prevents search stagnation and ensures you always get the best possible tree within your time budget.

**Key Features:**
- **Anytime Property**: Improves solution quality over time
- **Limited Discrepancy Search**: Impose an exploration budget to each node of the search space to provide fast early good solutions to improve on (LDS-DL8.5)
- **Top-k**: Similar to LDS but constrains the number of features to explore at each nodes.
- **Restart Strategy**: Prevents getting stuck in local regions
- **Time-bounded**: Respects strict time constraints

**Academic References:**

- [1]
- [2]

### LGDT: Less Greedy Decision Tree Learning

LGDT provides **fast approximate solutions** by looking ahead two levels instead of making purely greedy choices. This approach significantly improves upon traditional greedy algorithms while maintaining computational efficiency.

**Key Innovations:**
- **Two-level Lookahead**: Less myopic than traditional greedy methods
- **Efficient Data Structures**: Reversible sparse bitsets for speed
- **Multiple Search Strategies**: Adaptable to different scenarios
- **Scalability**: Handles large datasets efficiently


**Academic Reference:**

## Getting Started

Ready to experience optimal decision trees? Here's your next steps:

1. **[Installation](./installation.md)**: Set up pytrees-rs
2. **[Quick Start Guide](./quickstart.md)**: Build your first optimal or less greedy tree.
3. **[Python Library](./python/README.md)**: Explore the complete API
4. **[Examples](./examples/classification.md)**: See ways to uses the library



---

