import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.greedy import lgdt
from .. import ExposedSearchStrategy, DecisionTree


class LGDTCLassifier(BaseEstimator, ClassifierMixin, DecisionTree):
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
        X, y = check_X_y(X, y, dtype="float64")
        self.results = lgdt(
            X,
            y,
            self.search_strategy,
            self.min_sup,
            self.max_depth,
        )

        tree = json.loads(self.results.tree)
        self.statistics = json.loads(self.results.statistics)
        if len(tree["tree"]) == 1 and tree["tree"][0]["value"]["out"] not in [0, 1]:
            self.tree_ = None
        else:
            self.tree_ = tree
            self.is_fitted_ = True
            self.tree_error_ = self.results.error
            self.set_accuracy()
