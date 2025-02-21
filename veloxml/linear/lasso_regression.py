from typing import Literal

from ..core import CLassoRegression, CLassoSolverMode
from ..base import RegressionBase


class LassoRegression(RegressionBase):
    """
    Lasso Regression model with L1 regularization.

    This class implements Lasso Regression, which applies L1 regularization to
    promote sparsity in the model by reducing some coefficients to zero.

    Attributes:
        lambda_l1_ (float): The regularization strength (lambda).
        max_iter_ (int): The maximum number of iterations for optimization.
        tol_ (float): The tolerance for stopping criteria.
        admm_rho_ (float): The penalty parameter for ADMM optimization.
        penalize_bias_ (bool): Whether to apply L1 regularization to the bias term.
        solve_mode_ (CLassoSolverMode): The solver mode used for optimization.

    Args:
        mode (Literal["FISTA", "ADMM"], optional): The solver mode used for optimization.
            Defaults to "FISTA". Must be one of ["FISTA", "ADMM"].
        lambda_l1 (float, optional): The L1 regularization strength. Defaults to 1.0.
        max_iter (int, optional): The maximum number of iterations for optimization. Defaults to 100.
        tol (float, optional): The stopping criteria tolerance. Defaults to 1e-4.
        admm_rho (float, optional): The penalty parameter for ADMM optimization. Defaults to 1.0.
        penalize_bias (bool, optional): If True, applies L1 regularization to the bias term.
            Defaults to False.

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
        >>> model = LassoRegression(mode="ADMM", lambda_l1=0.5, max_iter=200)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        mode: Literal["FISTA", "ADMM"] = "FISTA",
        lambda_l1=1.0,
        max_iter=100,
        tol=1e-4,
        admm_rho=1.0,
        penalize_bias=False,
    ):
        """
        Initializes the Lasso Regression model with the specified parameters.

        Args:
            mode (Literal["FISTA", "ADMM"], optional): The solver mode used for optimization.
                Defaults to "FISTA".
            lambda_l1 (float, optional): The L1 regularization strength. Defaults to 1.0.
            max_iter (int, optional): The maximum number of iterations for optimization. Defaults to 100.
            tol (float, optional): The stopping criteria tolerance. Defaults to 1e-4.
            admm_rho (float, optional): The penalty parameter for ADMM optimization. Defaults to 1.0.
            penalize_bias (bool, optional): If True, applies L1 regularization to the bias term.
                Defaults to False.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        self.lambda_l1_ = lambda_l1
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.admm_rho_ = admm_rho
        self.penalize_bias_ = penalize_bias

        if mode == "FISTA":
            self.solve_mode_ = CLassoSolverMode.FISTA
        elif mode == "ADMM":
            self.solve_mode_ = CLassoSolverMode.ADMM
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of ['FISTA', 'ADMM']."
            )

        super().__init__(
            CLassoRegression(
                self.lambda_l1_,
                self.max_iter_,
                self.tol_,
                self.solve_mode_,
                self.admm_rho_,
                self.penalize_bias_,
            )
        )

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
        Returns the solver mode used for optimization.

        Returns:
            CLassoSolverMode: The solver mode.
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
