from pytrees import DL85Classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

# DL8.5 needs binary features, so the pipeline binarizes them first.
pipeline = Pipeline(
    [("binarize", Binarizer(threshold=0.0)), ("tree", DL85Classifier(max_depth=3))]
)

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
