from .estimator_base import EstimatorBase


class RegressionBase(EstimatorBase):
    """
    Base class for regression models.

    This class provides a foundation for regression models, extending `EstimatorBase`.
    It defines common functionality for fitting a model to training data and making predictions.

    Attributes:
        model (object): The underlying regression model instance.

    Methods:
        fit(X, Y):
            Trains the model using the given input data and target values.

        predict(X):
            Predicts target values based on the given input data.
    """

    def __init__(self, model):
        """
        Initializes the regression base class with a given model.

        Args:
            model (object): The regression model instance to be used.
        """
        super().__init__(model)

    def fit(self, X, Y):
        """
        Trains the regression model using the given input data and target values.

        Args:
            X (array-like): The input features used for training.
            Y (array-like): The target values corresponding to X.

        Returns:
            RegressionBase: The instance itself after fitting the model.
        """
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        """
        Predicts target values based on the given input data.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted target values.
        """
        return self.model.predict(X)
