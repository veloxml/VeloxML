from typing import Literal

from ..core import CDecisionTreeClassification, Criterion, SplitAlgorithm
from ..base import ClassificationBase


class DecisionTreeClassification(ClassificationBase):
    """
    Decision Tree Classification model.

    This class implements a decision tree classifier with configurable
    splitting criteria and algorithms.

    Attributes:
        criterion_ (Criterion): The criterion used to measure the quality of a split.
        split_algorithm_ (SplitAlgorithm): The algorithm used for splitting nodes.
        max_depth_ (int): The maximum depth of the tree.
        min_samples_split_ (int): The minimum number of samples required to split an internal node.
        min_samples_leaf_ (int): The minimum number of samples required in a leaf node.
        max_leaf_nodes_ (int): The maximum number of leaf nodes.
        min_impurity_decrease_ (float): The minimum impurity decrease required for a split.
        max_features_ (int): The number of features to consider when looking for the best split.
        max_bins_ (int): The maximum number of bins for histogram-based splitting.

    Args:
        criterion (Literal["Entropy", "Gini", "Logloss"], optional):
            The function to measure the quality of a split. Defaults to "Gini".
            Must be one of ["Entropy", "Gini", "Logloss"].
        split_algorithm (Literal["Standard", "Histogram"], optional):
            The algorithm used to split nodes. Defaults to "Standard".
            Must be one of ["Standard", "Histogram"].
        max_depth (int, optional): The maximum depth of the tree. Defaults to 5.
        min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 20.
        min_samples_leaf (int, optional): The minimum number of samples required to be in a leaf node. Defaults to 2.
        max_leaf_nodes (int, optional): The maximum number of leaf nodes. Defaults to 5.
        min_impurity_decrease (float, optional): The minimum impurity decrease required for a split. Defaults to 1.0.
        max_features (int, optional): The number of features to consider when looking for the best split. Defaults to 5.
        max_bins (int, optional): The maximum number of bins for histogram-based splitting. Defaults to 256.

    Raises:
        ValueError: If an invalid criterion or split algorithm is provided.

    Methods:
        predict(X):
            Predicts class labels for the given input data.

        predict_proba(X, Y):
            Predicts class probabilities for the given input data.

        feature_importances():
            Returns the feature importances of the trained model.

    Example:
        >>> model = DecisionTreeClassification(criterion="Entropy", max_depth=10, min_samples_split=5)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test, y_test)
        >>> feature_importance = model.feature_importances()
    """

    def __init__(
        self,
        criterion: Literal["Entropy", "Gini", "Logloss"] = "Gini",
        split_algorithm: Literal["Standard", "Histogram"] = "Standard",
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=2,
        max_leaf_nodes=5,
        min_impurity_decrease=1.0,
        max_features=5,
        max_bins=256,
    ):
        """
        Initializes the Decision Tree Classification model with the specified parameters.

        Args:
            criterion (Literal["Entropy", "Gini", "Logloss"], optional):
                The function to measure the quality of a split. Defaults to "Gini".
            split_algorithm (Literal["Standard", "Histogram"], optional):
                The algorithm used to split nodes. Defaults to "Standard".
            max_depth (int, optional): The maximum depth of the tree. Defaults to 5.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 20.
            min_samples_leaf (int, optional): The minimum number of samples required in a leaf node. Defaults to 2.
            max_leaf_nodes (int, optional): The maximum number of leaf nodes. Defaults to 5.
            min_impurity_decrease (float, optional): The minimum impurity decrease required for a split. Defaults to 1.0.
            max_features (int, optional): The number of features to consider when looking for the best split. Defaults to 5.
            max_bins (int, optional): The maximum number of bins for histogram-based splitting. Defaults to 256.

        Raises:
            ValueError: If an invalid criterion or split algorithm is provided.
        """
        if criterion == "Entropy":
            self.criterion_ = Criterion.Entropy
        elif criterion == "Gini":
            self.criterion_ = Criterion.Gini
        elif criterion == "Logloss":
            self.criterion_ = Criterion.Logloss
        else:
            raise ValueError(
                f"Invalid criterion '{criterion}'. Expected one of ['Entropy', 'Gini', 'Logloss']."
            )

        if split_algorithm == "Standard":
            self.split_algorithm_ = SplitAlgorithm.Standard
        elif split_algorithm == "Histogram":
            self.split_algorithm_ = SplitAlgorithm.Histogram
        else:
            raise ValueError(
                f"Invalid split_algorithm '{split_algorithm}'. Expected one of ['Standard', 'Histogram']."
            )

        self.max_depth_ = max_depth
        self.min_samples_split_ = min_samples_split
        self.min_samples_leaf_ = min_samples_leaf
        self.max_leaf_nodes_ = max_leaf_nodes
        self.min_impurity_decrease_ = min_impurity_decrease
        self.max_features_ = max_features
        self.max_bins_ = max_bins

        super().__init__(
            CDecisionTreeClassification(
                self.max_depth_,
                self.min_samples_split_,
                self.max_bins_,
                self.criterion_,
                self.split_algorithm_,
                self.min_samples_leaf_,
                self.max_leaf_nodes_,
                self.min_impurity_decrease_,
                self.max_features_,
            )
        )

    def _importances_check(self):
        """
        Checks if the model has been trained by verifying feature importances.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        if self.feature_importances() == []:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

    def predict(self, X):
        """
        Predicts class labels based on the given input data.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted class labels.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        self._importances_check()
        return super().predict(X)

    def predict_proba(self, X, Y):
        """
        Predicts class probabilities for the given input data.

        Args:
            X (array-like): The input features for making probability predictions.
            Y (array-like): The target class labels (may not be necessary for all implementations).

        Returns:
            array-like: The predicted class probabilities.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        self._importances_check()
        return super().predict_proba(X, Y)

    def feature_importances(self):
        """
        Returns the feature importances of the trained model.

        Feature importance provides insight into which features contribute most
        to the decision-making process of the model.

        Returns:
            array-like: The feature importance scores.
        """
        return self.model.feature_importances()
