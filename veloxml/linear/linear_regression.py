from typing import Literal

from ..core import CLinearRegression, CLinearDecompMode
from ..base import RegressionBase


class LinearRegression(RegressionBase):
    """
    Linear Regression model with multiple decomposition methods.

    This class implements a linear regression model with different decomposition
    methods for solving the normal equation, including LU decomposition, QR decomposition,
    and Singular Value Decomposition (SVD).

    Attributes:
        solve_mode_ (CLinearDecompMode): The decomposition method used for solving
            the linear regression problem.

    Args:
        mode (Literal["LU", "QR", "SVD"], optional): The decomposition method to use.
            Defaults to "LU". Must be one of ["LU", "QR", "SVD"].

    Raises:
        ValueError: If an invalid mode is provided.

    Methods:
        predict(X):
            Predicts target values for the given input data.

        get_weights():
            Returns the weight coefficients of the trained model.

        get_bias():
            Returns the bias term of the trained model.

    Example:
        >>> model = LinearRegression(mode="QR")
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, mode: Literal["LU", "QR", "SVD"] = "LU"):
        """
        Initializes the Linear Regression model with the specified decomposition method.

        Args:
            mode (Literal["LU", "QR", "SVD"], optional): The decomposition method used
                to solve the normal equation. Defaults to "LU".

        Raises:
            ValueError: If the provided mode is invalid.
        """
        if mode == "LU":
            self.solve_mode_ = CLinearDecompMode.LU
        elif mode == "QR":
            self.solve_mode_ = CLinearDecompMode.QR
        elif mode == "SVD":
            self.solve_mode_ = CLinearDecompMode.SVD
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of ['LU', 'QR', 'SVD']."
            )

        super().__init__(CLinearRegression(self.solve_mode_))

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

    @property
    def solve_mode(self):
        """
        Returns the decomposition method used for solving the regression.

        Returns:
            CLinearDecompMode: The decomposition mode.
        """
        return self.solve_mode_

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
