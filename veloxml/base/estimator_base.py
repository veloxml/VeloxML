from abc import ABC, abstractmethod


class EstimatorBase(ABC):
    """
    Abstract base class for all estimators.

    This class defines a common interface for all estimators, enforcing 
    the implementation of `fit` and `predict` methods in derived classes.

    Attributes:
        model (object): The underlying model instance used for estimation.

    Methods:
        fit(X, Y):
            Abstract method for training the model with input data X and target values Y.
            Must be implemented in subclasses.

        predict(X):
            Abstract method for making predictions based on input data X.
            Must be implemented in subclasses.
    """

    def __init__(self, model):
        """
        Initializes the estimator with a given model.

        Args:
            model (object): The model instance to be used for estimation.
        """
        self.model = model

    @abstractmethod
    def fit(self, X, Y):
        """
        Trains the model using the given input data and target values.

        Args:
            X (array-like): The input features used for training.
            Y (array-like): The target values corresponding to X.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts target values based on the given input data.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted target values.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        pass
