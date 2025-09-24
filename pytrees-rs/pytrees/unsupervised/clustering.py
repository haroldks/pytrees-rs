import sys
import json
import numpy as np
from sklearn.metrics import DistanceMetric
from .. import DecisionTree, SearchFailedError
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, assert_all_finite
from pytreesrs.odt import PyDL85
from pytreesrs.enums import *


class DL85Cluster(BaseEstimator, ClusterMixin, DecisionTree):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=0.0,
        max_time=600.0,
        always_sort=True,
        lower_bound_policy=ExposedLowerBoundPolicy.Similarity,
        branching_policy=ExposedBranchingPolicy.Dynamic,
        heuristic=ExposedHeuristic.Disabled,
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.always_sort = always_sort
        self.node_data_type = ExposedNodeDataType.Tids
        self.depth2_policy = ExposedDepth2Policy.Disabled
        self.lower_bound_policy = lower_bound_policy
        self.branching_policy = branching_policy
        self.heuristic = heuristic
        self.discrepancy = discrepancy
        self.gain = gain
        self.topk = topk
        self.purity = purity
        self.restart = restart
        self.error_function = error_function
        self.results = None
        self.__obj = PyDL85(
            min_sup=self.min_sup,
            max_depth=self.max_depth,
            max_error=self.max_error,
            time_limit=self.max_time,
            always_sort=self.always_sort,
            heuristic=self.heuristic,
            depth2_policy=self.depth2_policy,
            lower_bound=self.lower_bound_policy,
            branching_policy=self.branching_policy,
            data_type=self.node_data_type,
            discrepancy=self.discrepancy,
            gain=self.gain,
            topk=self.topk,
            restart=self.restart,
            purity=self.purity,
            error_function=self.error_function,
        )
        self.config = json.loads(self.__obj.config)

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
            X_error = check_array(X_error, dtype="float64")

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

        try:
            self.__obj.fit(X, X_error)
            self.results = self.__obj.stats
            self.refresh_stats()
        except Exception as e:
            raise SearchFailedError
