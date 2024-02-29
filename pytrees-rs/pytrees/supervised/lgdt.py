from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.greedy import lgdt
from .. import ExposedSearchStrategy


class LGDTCLassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=2,
        search_strategy=ExposedSearchStrategy.LessGreedyMurtree,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.search_strategy = search_strategy

        self.results = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype="float64")
        self.results = lgdt(
            X,
            y,
            self.search_strategy,
            self.min_sup,
            self.max_depth,
        )
