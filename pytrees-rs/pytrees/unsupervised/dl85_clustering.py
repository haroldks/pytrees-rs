import json

from .. import *
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import DistanceMetric
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.odt import dl85


class DL85Cluster(BaseEstimator, ClusterMixin, DecisionTree):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=1e10,
        max_time=600,
        cache_init_size=0,
        one_time_sort=True,
        lower_bound=ExposedLowerBoundStrategy.None_,
        branching_type=ExposedBranchingStrategy.Dynamic,
        heuristic=ExposedSearchHeuristic.None_,
        cache_init_strategy=ExposedCacheInitStrategy.None_,
        error_function=None,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.cache_init_size = cache_init_size
        self.one_time_sort = one_time_sort
        self.data_format = ExposedDataFormat.Tids
        self.specialization = ExposedSpecialization.None_
        self.lower_bound = lower_bound
        self.branching_type = branching_type
        self.heuristic = heuristic
        self.cache_init_strategy = cache_init_strategy
        self.error_function = error_function

        self.results = None

    @staticmethod
    def default_error(tids, X):
        if len(tids) == 0:
            return sys.maxsize, sys.maxsize
        dist = DistanceMetric.get_metric("euclidean")
        X_subset = np.asarray([X[index, :] for index in list(tids)], dtype="int32")
        centroid = np.mean(X_subset, axis=0)
        distances = dist.pairwise(X_subset, [centroid])
        return round(np.sum(distances), 2), DL85Cluster.default_leaf_value(tids, X)

    @staticmethod
    def default_leaf_value(tids, X):
        return round(np.mean(X.take(list(tids))), 2)

    def fit(self, X, X_error=None):

        if X_error is not None:
            assert_all_finite(X_error)
            X_error = check_array(X_error, dtype="int32")

        if self.error_function is None:
            if X_error is None:
                self.error_function = lambda tids: self.default_error(tids, X)
            else:
                if X_error.shape[0] == X.shape[0]:
                    self.error_function = lambda tids: self.default_error(tids, X_error)
                else:
                    raise ValueError(
                        "X_error does not have the same number of rows as X"
                    )

        self.results = dl85(
            X,
            X_error,
            self.min_sup,
            self.max_depth,
            self.max_time,
            self.cache_init_size,
            self.max_error,
            self.one_time_sort,
            self.data_format,
            self.specialization,
            self.lower_bound,
            self.branching_type,
            self.heuristic,
            self.cache_init_strategy,
            self.error_function,
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
