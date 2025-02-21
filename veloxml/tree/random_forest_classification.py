from typing import Literal

from ..core import CRandomForestClassification, Criterion, SplitAlgorithm
from ..base import ClassificationBase


class RandomForestClassification(ClassificationBase):
    """
    Random Forest Classification model.

    This class implements a random forest classifier, which is an ensemble learning
    method based on multiple decision trees. It supports configurable criteria,
    splitting algorithms, and parallel execution.

    Attributes:
        n_trees_ (int): The number of trees in the forest.
        criterion_ (Criterion): The function used to measure the quality of a split.
        split_algorithm_ (SplitAlgorithm): The algorithm used for splitting nodes.
        max_depth_ (int): The maximum depth of each tree.
        min_samples_split_ (int): The minimum number of samples required to split an internal node.
        min_samples_leaf_ (int): The minimum number of samples required in a leaf node.
        max_leaf_nodes_ (int): The maximum number of leaf nodes per tree.
        min_impurity_decrease_ (float): The minimum impurity decrease required for a split.
        max_features_ (int): The number of features to consider when looking for the best split.
        max_bins_ (int): The maximum number of bins for histogram-based splitting.
        n_jobs_ (int): The number of parallel jobs for training.
        random_seed_ (int): The seed for random number generation.

    Args:
        n_trees (int, optional): The number of trees in the forest. Defaults to 5.
        criterion (Literal["Entropy", "Gini", "Logloss"], optional):
            The function to measure the quality of a split. Defaults to "Gini".
            Must be one of ["Entropy", "Gini", "Logloss"].
        split_algorithm (Literal["Standard", "Histogram"], optional):
            The algorithm used to split nodes. Defaults to "Standard".
            Must be one of ["Standard", "Histogram"].
        max_depth (int, optional): The maximum depth of each tree. Defaults to 5.
        min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 20.
        min_samples_leaf (int, optional): The minimum number of samples required in a leaf node. Defaults to 2.
        max_leaf_nodes (int, optional): The maximum number of leaf nodes per tree. Defaults to 5.
        min_impurity_decrease (float, optional): The minimum impurity decrease required for a split. Defaults to 1.0.
        max_features (int, optional): The number of features to consider when looking for the best split. Defaults to 5.
        max_bins (int, optional): The maximum number of bins for histogram-based splitting. Defaults to 256.
        n_jobs (int, optional): The number of parallel jobs for training. Defaults to 1.
        random_seed (int, optional): The seed for random number generation. Defaults to -1.

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
        >>> model = RandomForestClassification(n_trees=10, criterion="Entropy", max_depth=10, n_jobs=4)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test, y_test)
        >>> feature_importance = model.feature_importances()
    """

    def __init__(
        self,
        n_trees: int = 5,
        criterion: Literal["Entropy", "Gini", "Logloss"] = "Gini",
        split_algorithm: Literal["Standard", "Histogram"] = "Standard",
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 2,
        max_leaf_nodes: int = 5,
        min_impurity_decrease: float = 1.0,
        max_features: int = 5,
        max_bins: int = 256,
        n_jobs: int = 1,
        random_seed: int = -1,
    ):
        """
        Initializes the Random Forest Classification model with the specified parameters.

        Args:
            n_trees (int, optional): The number of trees in the forest. Defaults to 5.
            criterion (Literal["Entropy", "Gini", "Logloss"], optional):
                The function to measure the quality of a split. Defaults to "Gini".
            split_algorithm (Literal["Standard", "Histogram"], optional):
                The algorithm used to split nodes. Defaults to "Standard".
            max_depth (int, optional): The maximum depth of each tree. Defaults to 5.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 20.
            min_samples_leaf (int, optional): The minimum number of samples required in a leaf node. Defaults to 2.
            max_leaf_nodes (int, optional): The maximum number of leaf nodes per tree. Defaults to 5.
            min_impurity_decrease (float, optional): The minimum impurity decrease required for a split. Defaults to 1.0.
            max_features (int, optional): The number of features to consider when looking for the best split. Defaults to 5.
            max_bins (int, optional): The maximum number of bins for histogram-based splitting. Defaults to 256.
            n_jobs (int, optional): The number of parallel jobs for training. Defaults to 1.
            random_seed (int, optional): The seed for random number generation. Defaults to -1.

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

        self.n_trees_ = n_trees
        self.max_depth_ = max_depth
        self.min_samples_split_ = min_samples_split
        self.min_samples_leaf_ = min_samples_leaf
        self.max_leaf_nodes_ = max_leaf_nodes
        self.min_impurity_decrease_ = min_impurity_decrease
        self.max_features_ = max_features
        self.max_bins_ = max_bins
        self.n_jobs_ = n_jobs
        self.random_seed_ = random_seed

        super().__init__(
            CRandomForestClassification(
                self.n_trees_,
                self.max_depth_,
                self.min_samples_leaf_,
                self.min_samples_split_,
                self.min_impurity_decrease_,
                self.max_leaf_nodes_,
                self.max_bins_,
                self.criterion_,
                self.split_algorithm_,
                self.max_features_,
                self.n_jobs_,
                self.random_seed_,
            )
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
        self._check_initialize()
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
        self._check_initialize()
        return super().predict_proba(X, Y)

    def _check_initialize(self):
        """
        Checks if the model has been trained.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        if not self.model.check_initialize():
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

    def feature_importances(self):
        """
        Returns the feature importances of the trained model.

        Feature importance provides insight into which features contribute most
        to the decision-making process of the model.

        Returns:
            array-like: The feature importance scores.
        """
        return self.model.feature_importances()
