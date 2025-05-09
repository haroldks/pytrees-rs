import uuid
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from .exceptions import TreeNotFoundError, SearchFailedError


class DecisionTree:
    def __init__(self):
        self.results = None
        self.tree_ = None
        self.tree_error_ = None
        self.accuracy_ = None
        self.is_fitted_ = False
        self.statistics = None

    def predict(self):
        pass

    def set_accuracy(self):
        self.accuracy_ = round(
            1 - self.results.error / self.statistics["num_samples"], 5
        )

    @staticmethod
    def is_leaf_node(node):
        return (node["left"] == 0) and (node["right"] == 0)

    def predict(self, X):
        """Implements the standard predict function for a DL8.5 classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        # Check is fit is called
        # check_is_fitted(self, attributes='tree_') # use of attributes is deprecated. alternative solution is below

        # if hasattr(self, 'sol_size') is False:  # fit method has not been called
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError(
                "Call fit method first" % {"name": type(self).__name__}
            )

        if self.tree_ is None:
            raise TreeNotFoundError(
                "predict(): ",
                "Tree not found during training by DL8.5 - "
                "Check fitting message for more info.",
            )

        if hasattr(self, "tree_") is False:  # normally this case is not possible.
            raise SearchFailedError(
                "PredictionError: ",
                "DL8.5 training has failed. Please contact the developers "
                "if the problem is in the scope supported by the tool.",
            )

        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred.append(self.pred_value_on_dict(X[i, :]))

        return pred

    def pred_value_on_dict(self, instance, tree=None):
        node = tree if tree is not None else self.tree_["tree"][0]
        while not DecisionTree.is_leaf_node(node):
            if instance[node["value"]["test"]] == 1:
                node = self.tree_["tree"][node["right"]]
            else:
                node = self.tree_["tree"][node["left"]]
        return node["value"]["out"]

    def get_dot_body_rec(self, node, parent=None, left=0):
        gstring = ""
        id = str(uuid.uuid4())
        id = id.replace("-", "_")

        if node["right"] == node["left"]:
            gstring += (
                    "leaf_"
                    + id
                    + ' [label="{{class|'
                    + str(node["value"]["out"])
                    + "}|{error|"
                    + str(node["value"]["error"])
                    + '}}"];\n'
            )
            gstring += (
                    "node_"
                    + parent
                    + " -> leaf_"
                    + id
                    + " [label="
                    + str(int(left))
                    + "];\n"
            )
        else:
            gstring += (
                    "node_"
                    + id
                    + ' [label="{{feat|'
                    + str(node["value"]["test"])
                    + '}}"];\n'
            )
            gstring += (
                    "node_" + parent + " -> node_" + id + " [label=" + str(left) + "];\n"
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][node["left"]], id, left=0
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][node["right"]], id, left=1
            )
        return gstring

    def export_to_graphviz_dot(self):
        gstring = "digraph Tree { \n" "graph [ranksep=0]; \n" "node [shape=record]; \n"
        id = str(uuid.uuid4())
        id = id.replace("-", "_")

        root = self.tree_["tree"][0]
        feat = root["value"]["test"]
        if feat is not None:
            gstring += (
                    "node_"
                    + id
                    + ' [label="{{feat|'
                    + str(feat)
                    + "}|{error|"
                    + str(self.tree_error_)
                    + '}}"];\n'
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][root["left"]], parent=id, left=0
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][root["right"]], parent=id, left=1
            )
        gstring += "}"
        return gstring
