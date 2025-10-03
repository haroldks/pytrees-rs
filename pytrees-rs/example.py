from pytrees import DL85Classifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Use in pipelines
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("classifier", DL85Classifier(max_depth=3))]
)

# Cross-validation
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = (X > 0).astype(float)
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
