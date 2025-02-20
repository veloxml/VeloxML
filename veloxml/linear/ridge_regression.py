from typing import Literal

from ..core import CRidgeRegression
from ..base import RegressionBase


class RidgeRegression(RegressionBase):
    """
    Ridge Regression model with L2 regularization.

    This class implements Ridge Regression, which applies L2 regularization
    to prevent overfitting by penalizing large coefficients.

    Attributes:
        lamda_l2_ (float): The regularization strength (lambda).
        penalize_bias_ (bool): Whether to apply L2 regularization to the bias term.

    Args:
        lamda_l2 (float, optional): The regularization strength.
            Defaults to 1.0.
        penalize_bias (bool, optional): If True, applies regularization to the bias term.
            Defaults to False.

    Methods:
        predict(X):
            Predicts target values for the given input data.

        get_weights():
            Returns the weight coefficients of the trained model.

        get_bias():
            Returns the bias term of the trained model.

    Example:
        >>> model = RidgeRegression(lamda_l2=0.5, penalize_bias=True)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, lamda_l2=1.0, penalize_bias=False):
        """
        Initializes the Ridge Regression model with the specified L2 regularization strength.

        Args:
            lamda_l2 (float, optional): The regularization strength. Defaults to 1.0.
            penalize_bias (bool, optional): If True, applies regularization to the bias term.
                Defaults to False.
        """
        self.lamda_l2_ = lamda_l2
        self.penalize_bias_ = penalize_bias

        super().__init__(CRidgeRegression(self.lamda_l2_, self.penalize_bias_))

    def predict(self, X):
        """
        Predicts target values based on the given input data.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted target values.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        if self.get_weights() == []:
            raise RuntimeError(
                "Model weights are not initialized. Please train the model before making predictions."
            )

        return super().predict(X)

    def get_weights(self):
        """
        Returns the weight coefficients of the trained model.

        Returns:
            array-like: The weight coefficients of the regression model.
        """
        return self.model.get_weights()

    def get_bias(self):
        """
        Returns the bias term of the trained model.

        Returns:
            float: The bias term of the regression model.
        """
        return self.model.get_bias()
