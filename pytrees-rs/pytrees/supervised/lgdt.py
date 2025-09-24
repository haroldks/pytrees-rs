from .. import DecisionTree, SearchFailedError
from pytreesrs.enums import ExposedSearchStrategy
from pytreesrs.greedy import lgdt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y


class LGDTClassifier(BaseEstimator, ClassifierMixin, DecisionTree):
    def __init__(
        self,
        min_sup=1,
        max_depth=2,
        search_strategy=ExposedSearchStrategy.LGDTErrorMinimizer,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.search_strategy = search_strategy

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype="float64", y_numeric=True)

        try:
            self.results = lgdt(
                X,
                y,
                self.search_strategy,
                self.min_sup,
                self.max_depth,
            )
            self.refresh_stats()
        except Exception as e:
            raise SearchFailedError
