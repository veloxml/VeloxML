from .estimator_base import EstimatorBase


class UnsupervisedEstimatorBase(EstimatorBase):
    """
    Base class for unsupervised learning models.

    This class provides a foundation for unsupervised learning models, extending `EstimatorBase`.
    It defines common functionality such as fitting the model, making predictions,
    and transforming data.

    Attributes:
        model (object): The underlying unsupervised learning model instance.

    Methods:
        fit(X, Y=None):
            Trains the model using the given input data.

        predict(X, Y=None):
            Predicts cluster labels or other outputs depending on the model type.

        transform(X):
            Transforms the input data into a different representation (e.g., dimensionality reduction).

        fit_predict(X):
            Fits the model to the data and returns predictions.

        fit_transform(X):
            Fits the model to the data and transforms it.
    """

    def __init__(self, model):
        """
        Initializes the unsupervised learning base class with a given model.

        Args:
            model (object): The unsupervised learning model instance to be used.
        """
        super().__init__(model)

    def fit(self, X, Y=None):
        """
        Trains the unsupervised learning model using the given input data.

        Args:
            X (array-like): The input features used for training.
            Y (array-like, optional): Unused parameter included for compatibility.

        Returns:
            UnsupervisedEstimatorBase: The instance itself after fitting the model.
        """
        self.model.fit(X)
        return self

    def predict(self, X, Y=None):
        """
        Predicts outputs based on the given input data.

        In clustering models, this typically returns cluster labels.

        Args:
            X (array-like): The input features for making predictions.
            Y (array-like, optional): Unused parameter included for compatibility.

        Returns:
            array-like: The predicted labels or other outputs.
        """
        return self.model.predict(X)

    def transform(self, X):
        """
        Transforms the input data into a different representation.

        Used in dimensionality reduction and feature extraction techniques.

        Args:
            X (array-like): The input data to transform.

        Returns:
            array-like: The transformed representation of the input data.
        """
        return self.model.transform(X)

    def fit_predict(self, X):
        """
        Fits the model to the data and then returns predictions.

        This is commonly used in clustering models.

        Args:
            X (array-like): The input data.

        Returns:
            array-like: The predicted labels or other outputs after fitting the model.
        """
        return self.model.fit_predict(X)

    def fit_transform(self, X):
        """
        Fits the model to the data and then transforms it.

        This is commonly used in dimensionality reduction techniques.

        Args:
            X (array-like): The input data.

        Returns:
            array-like: The transformed representation of the input data.
        """
        return self.model.fit_transform(X)
