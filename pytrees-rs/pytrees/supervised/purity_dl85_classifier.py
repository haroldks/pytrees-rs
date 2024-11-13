import json

from .. import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, assert_all_finite
from pytreesrs.odt import dl85, PyGenericDl85


class PurityDL85Classifier(BaseEstimator, ClassifierMixin, DecisionTree):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=1e10,
        purity=0.5,
        epsilon=0.1,
        max_time=600,
        cache_init_size=0,
        one_time_sort=True,
        data_format=ExposedDataFormat.ClassSupports,
        specialization=ExposedSpecialization.Murtree,
        lower_bound=ExposedLowerBoundStrategy.Similarity,
        branching_type=ExposedBranchingStrategy.Dynamic,
        heuristic=ExposedSearchHeuristic.None_,
        cache_init_strategy=ExposedCacheInitStrategy.None_,
        error_function=None,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.purity = purity
        self.epsilon = epsilon
        self.max_time = max_time
        self.cache_init_size = cache_init_size
        self.one_time_sort = one_time_sort
        self.data_format = data_format
        self.specialization = specialization
        self.lower_bound = lower_bound
        self.branching_type = branching_type
        self.heuristic = heuristic
        self.cache_init_strategy = cache_init_strategy
        self.error_function = error_function

        self.first = True
        self.results = None
        self.is_optimal = False

        self.internal = PyGenericDl85(
            self.min_sup,
            self.max_depth,
            None,
            self.max_time,
            self.purity,
            self.epsilon,
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
            ExposedSearchStrategy.PurityLimit,
        )

    def fit(self, X, y=None):

        target_is_need = True if y is not None else False

        if target_is_need:  # target-needed tasks (eg: classification, regression, etc.)
            # Check that X and y have correct shape and raise ValueError if not
            X, y = check_X_y(X, y, dtype="float64")
            # if opt_func is None and opt_pred_func is None:
            #     print("No optimization criterion defined. Misclassification error is used by default.")
        else:  # target-less tasks (clustering, etc.)
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype="float64")

        self.results = self.internal.fit(X, y)
        self.is_optimal = self.internal.is_optimal()

        tree = json.loads(self.results.tree)
        self.statistics = json.loads(self.results.statistics)
        if len(tree["tree"]) == 1 and tree["tree"][0]["value"]["out"] not in [0, 1]:
            self.tree_ = None
        else:
            self.tree_ = tree
            self.is_fitted_ = True
            self.tree_error_ = self.results.error
            self.set_accuracy()

    def partial_fit(self, X, y=None, runtime=1):

        target_is_need = True if y is not None else False

        if target_is_need:  # target-needed tasks (eg: classification, regression, etc.)
            # Check that X and y have correct shape and raise ValueError if not
            X, y = check_X_y(X, y, dtype="float64")
            # if opt_func is None and opt_pred_func is None:
            #     print("No optimization criterion defined. Misclassification error is used by default.")
        else:  # target-less tasks (clustering, etc.)
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype="float64")

        self.results = self.internal.partial_fit(X, y, self.first, runtime)
        self.first = False
        self.is_optimal = self.internal.is_optimal()

        tree = json.loads(self.results.tree)
        self.statistics = json.loads(self.results.statistics)
        if len(tree["tree"]) == 1 and tree["tree"][0]["value"]["out"] not in [0, 1]:
            self.tree_ = None
        else:
            self.tree_ = tree
            self.is_fitted_ = True
            self.tree_error_ = self.results.error
            self.set_accuracy()
