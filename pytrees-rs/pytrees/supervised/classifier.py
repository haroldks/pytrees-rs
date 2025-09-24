import json
from .. import DecisionTree, SearchFailedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.odt import PyDL85
from pytreesrs.enums import *


class DL85Classifier(BaseEstimator, ClassifierMixin, DecisionTree):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=float("inf"),
        max_time=600.0,
        always_sort=True,
        node_data_type=ExposedNodeDataType.ClassSupports,
        depth2_policy=ExposedDepth2Policy.Enabled,
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
        self.node_data_type = node_data_type
        self.depth2_policy = depth2_policy
        self.lower_bound_policy = lower_bound_policy
        self.branching_policy = branching_policy
        self.heuristic = heuristic
        self.discrepancy = discrepancy
        self.gain = gain
        self.topk = topk
        self.restart = restart
        self.purity = purity
        self.error_function = error_function

        self.results = None

        if any(
            rule is not None
            for rule in [
                self.discrepancy,
                self.gain,
                self.topk,
                self.restart,
                self.purity,
            ]
        ):
            self.lower_bound_policy = ExposedLowerBoundPolicy.Disabled
            self.branching_policy = ExposedBranchingPolicy.Default

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

    def fit(self, X, y=None):

        target_is_need = True if y is not None else False

        if target_is_need:  # target-needed tasks (eg: classification, regression, etc.)
            # Check that X and y have correct shape and raise ValueError if not
            X, y = check_X_y(X, y, dtype="float64", y_numeric=True)

        else:  # target-less tasks (clustering, etc.)
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype="float64")

        try:
            self.__obj.fit(X, y)
            self.results = self.__obj.stats
            self.refresh_stats()
        except Exception:
            self.is_fitted_ = False
            raise SearchFailedError

    def load_data(self, X, y):
        self.__obj.load_data(X, y)

    def partial_fit(self):
        self.__obj.partial_fit()
