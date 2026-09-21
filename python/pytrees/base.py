import uuid
import json
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from .exceptions import TreeNotFoundError, SearchFailedError


class DecisionTree:
    """
    Base class for decision tree implementations in pytrees.

    This class provides common functionality for all decision tree algorithms
    including prediction, tree visualization, and result management. It serves
    as the foundation for both optimal (DL8.5) and greedy (LGDT) implementations.

    Attributes
    ----------
    results : object or None
        Raw results from the underlying Rust algorithm containing tree structure,
        error metrics, and search statistics.

    tree_ : dict or None
        Parsed tree structure in dictionary format. None if no valid tree was found.

    tree_error_ : float or None
        Classification error of the constructed tree.

    accuracy_ : float or None
        Classification accuracy (1 - error_rate) of the constructed tree.

    is_fitted_ : bool
        Flag indicating whether the model has been successfully fitted.

    statistics : dict or None
        Detailed search statistics including nodes explored, cache performance,
        and algorithm-specific metrics.

    Notes
    -----
    This class follows scikit-learn conventions and integrates with the
    sklearn ecosystem. It provides tree visualization capabilities and
    comprehensive error handling for robust usage.

    Examples
    --------
    This class is not meant to be used directly. Use DL85Classifier or
    LGDTClassifier instead:

    >>> from pytrees import DL85Classifier
    >>> clf = DL85Classifier(max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self):
        """
        Initialize a new DecisionTree instance.

        Sets all attributes to their default values indicating an unfitted model.
        """
        self.results = None
        self.tree_ = None
        self.tree_error_ = None
        self.accuracy_ = None
        self.is_fitted_ = False
        self.statistics = None

    def predict(self):
        """Placeholder predict method - implemented by subclasses."""
        pass

    def set_accuracy(self):
        """
        Calculate and set the accuracy based on error rate and sample count.

        The accuracy is computed as: accuracy = 1 - (error / num_samples)
        and rounded to 5 decimal places for consistent reporting.
        """
        self.accuracy_ = round(
            1 - self.results.error / self.statistics["num_samples"], 5
        )

    @staticmethod
    def is_leaf_node(node):
        """
        Check if a tree node is a leaf node.

        Parameters
        ----------
        node : dict
            Tree node dictionary with 'left' and 'right' keys.

        Returns
        -------
        bool
            True if the node is a leaf (no children), False otherwise.
        """
        return (node["left"] == 0) and (node["right"] == 0)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        This method implements the standard scikit-learn predict interface,
        traversing the decision tree to make predictions for each input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict. Should be binary features (0 or 1).

        Returns
        -------
        list of int
            Predicted class labels for each sample.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        TreeNotFoundError
            If no valid tree was found during training.
        SearchFailedError
            If the underlying algorithm failed during training.

        Examples
        --------
        >>> clf = DL85Classifier(max_depth=2)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> print(f"Predicted classes: {predictions}")
        """
        # Check if fit has been called
        if self.is_fitted_ is False:
            raise NotFittedError(
                "Call fit method first" % {"name": type(self).__name__}
            )

        if self.tree_ is None:
            raise TreeNotFoundError(
                "predict(): ",
                "Tree not found during training by DL8.5 - "
                "Check fitting message for more info.",
            )

        if hasattr(self, "tree_") is False:
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
        """
        Predict the class label for a single instance by traversing the tree.

        Parameters
        ----------
        instance : array-like of shape (n_features,)
            Single sample to predict. Features should be binary (0 or 1).
        tree : dict, optional
            Tree structure to use. If None, uses self.tree_.

        Returns
        -------
        int
            Predicted class label for the instance.

        Notes
        -----
        This method assumes binary features where 1 indicates the presence
        of a feature and 0 indicates absence. The tree traversal follows
        the right branch when the feature value is 1, left branch otherwise.
        """
        node = tree if tree is not None else self.tree_["tree"][0]
        while not DecisionTree.is_leaf_node(node):
            if instance[node["value"]["test"]] == 1:
                node = self.tree_["tree"][node["right"]]
            else:
                node = self.tree_["tree"][node["left"]]
        return node["value"]["out"]

    def get_dot_body_rec(self, node, parent=None, left=0):
        """
        Recursively generate DOT format string for tree visualization.

        Parameters
        ----------
        node : dict
            Current tree node to process.
        parent : str, optional
            Parent node identifier for edge creation.
        left : int, optional
            Edge label (0 for left branch, 1 for right branch).

        Returns
        -------
        str
            DOT format string fragment for the current subtree.

        Notes
        -----
        This method generates Graphviz DOT format output for tree visualization.
        Leaf nodes show class and error information, internal nodes show the
        feature being tested.
        """
        gstring = ""
        id = str(uuid.uuid4())
        id = id.replace("-", "_")

        if node["right"] == node["left"]:
            # Leaf node
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
            # Internal node
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

    def to_dot(self):
        """
        Export the decision tree to Graphviz DOT format for visualization.

        Returns
        -------
        str
            Complete DOT format string representing the decision tree.

        Notes
        -----
        The generated DOT string can be used with Graphviz tools to create
        visual representations of the decision tree. The format includes:
        - Internal nodes showing the feature being tested
        - Leaf nodes showing class predictions and error values
        - Edges labeled with branch conditions (0/1)

        Examples
        --------
        >>> clf = DL85Classifier(max_depth=2)
        >>> clf.fit(X_train, y_train)
        >>> dot_string = clf.to_dot()
        >>> # Save to file or use with graphviz library
        >>> with open('tree.dot', 'w') as f:
        ...     f.write(dot_string)
        """
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

    def refresh_stats(self):
        """
        Update internal statistics and tree structure from algorithm results.

        This method processes the raw results from the Rust backend and updates
        the Python object's state including tree structure, error metrics, and
        fitting status. It handles cases where no valid tree was found.

        Notes
        -----
        Called automatically after fitting operations to synchronize the
        Python wrapper state with the underlying algorithm results.
        """
        tree = json.loads(self.results.tree)
        self.statistics = json.loads(self.results.statistics)
        if len(tree["tree"]) == 1 and tree["tree"][0]["value"]["out"] not in [0, 1]:
            # No valid tree found
            self.tree_ = None
        else:
            # Valid tree found
            self.tree_ = tree
            self.is_fitted_ = True
            self.tree_error_ = self.results.error
            self.set_accuracy()
