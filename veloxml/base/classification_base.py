from .estimator_base import EstimatorBase


class ClassificationBase(EstimatorBase):
    """
    Base class for classification models.

    This class provides a foundation for classification models, extending `EstimatorBase`.
    It defines common functionality for fitting a model to training data and making predictions, 
    including probability-based predictions.

    Attributes:
        model (object): The underlying classification model instance.

    Methods:
        fit(X, Y):
            Trains the model using the given input data and target values.

        predict(X):
            Predicts class labels based on the given input data.

        predict_proba(X, Y):
            Predicts class probabilities for the given input data.
    """

    def __init__(self, model):
        """
        Initializes the classification base class with a given model.

        Args:
            model (object): The classification model instance to be used.
        """
        super().__init__(model)

    def fit(self, X, Y):
        """
        Trains the classification model using the given input data and target values.

        Args:
            X (array-like): The input features used for training.
            Y (array-like): The target class labels corresponding to X.

        Returns:
            ClassificationBase: The instance itself after fitting the model.
        """
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        """
        Predicts class labels based on the given input data.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X, Y):
        """
        Predicts class probabilities for the given input data.

        Args:
            X (array-like): The input features for making probability predictions.
            Y (array-like): The target class labels (this argument may not be necessary, 
                            depending on the implementation of the underlying model).

        Returns:
            array-like: The predicted class probabilities.
        """
        return self.model.predict_proba(X, Y)
