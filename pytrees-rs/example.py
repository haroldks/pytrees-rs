from pytrees import DL85Classifier, LGDTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = (X > 0).astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
