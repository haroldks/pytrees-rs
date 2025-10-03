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
    purity=ExposedPurityRule(min_purity=0.9),
)
advanced_clf.fit(X, y)
