from typing import Literal

from ..core import CLogisticRegression, CLogisticRegressionSolverType
from ..base import ClassificationBase


class LogisticRegression(ClassificationBase):
    """
    Logistic Regression model with support for different optimization solvers.

    This class implements Logistic Regression for binary classification, supporting
    L2 regularization and optimization via LBFGS or Newton-CG solvers.

    Attributes:
        solver_type_ (CLogisticRegressionSolverType): The optimization solver used.
        lamda_l2_ (float): The L2 regularization strength.
        tol_ (float): The tolerance for stopping criteria.
        max_iter_ (int): The maximum number of iterations for optimization.
        ls_alpha_init_ (float): Initial step size for line search.
        ls_rho_ (float): Line search reduction factor.
        ls_c_ (float): Line search parameter for Armijo condition.
        history_size_ (int): The history size for LBFGS optimization.

    Args:
        solver_type (Literal["LBFGS", "NEWTONCG"], optional): The solver used for optimization.
            Defaults to "LBFGS". Must be one of ["LBFGS", "NEWTONCG"].
        lamda_l2 (float, optional): The L2 regularization strength. Defaults to 1.0.
        tol (float, optional): The stopping criteria tolerance. Defaults to 1e-6.
        max_iter (int, optional): The maximum number of optimization iterations. Defaults to 100.
        ls_alpha_init (float, optional): The initial step size for line search. Defaults to 1.0.
        ls_rho (float, optional): The reduction factor for line search. Defaults to 0.5.
        ls_c (float, optional): The parameter for the Armijo condition in line search. Defaults to 1e-4.
        history_size (int, optional): The history size for LBFGS optimization. Defaults to 10.

    Raises:
        ValueError: If an invalid solver type is provided.

    Methods:
        predict(X):
            Predicts class labels for the given input data.

        predict_proba(X, Y):
            Predicts class probabilities for the given input data.

        get_weights():
            Returns the weight coefficients of the trained model.

        get_bias():
            Returns the bias term of the trained model.

    Example:
        >>> model = LogisticRegression(solver_type="NEWTONCG", lamda_l2=0.5, max_iter=200)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test, y_test)
    """

    def __init__(
        self,
        solver_type: Literal["LBFGS", "NEWTONCG"] = "LBFGS",
        lamda_l2=1.0,
        tol=1e-6,
        max_iter=100,
        ls_alpha_init=1.0,
        ls_rho=0.5,
        ls_c=1e-4,
        history_size=10,
    ):
        """
        Initializes the Logistic Regression model with the specified parameters.

        Args:
            solver_type (Literal["LBFGS", "NEWTONCG"], optional): The solver used for optimization.
                Defaults to "LBFGS".
            lamda_l2 (float, optional): The L2 regularization strength. Defaults to 1.0.
            tol (float, optional): The stopping criteria tolerance. Defaults to 1e-6.
            max_iter (int, optional): The maximum number of optimization iterations. Defaults to 100.
            ls_alpha_init (float, optional): The initial step size for line search. Defaults to 1.0.
            ls_rho (float, optional): The reduction factor for line search. Defaults to 0.5.
            ls_c (float, optional): The parameter for the Armijo condition in line search. Defaults to 1e-4.
            history_size (int, optional): The history size for LBFGS optimization. Defaults to 10.

        Raises:
            ValueError: If an invalid solver type is provided.
        """
        if solver_type == "LBFGS":
            self.solver_type_ = CLogisticRegressionSolverType.LBFGS
        elif solver_type == "NEWTONCG":
            self.solver_type_ = CLogisticRegressionSolverType.NEWTON
        else:
            raise ValueError(
                f"Invalid solver '{solver_type}'. Expected one of ['LBFGS', 'NEWTONCG']."
            )

        self.lamda_l2_ = lamda_l2
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.ls_alpha_init_ = ls_alpha_init
        self.ls_rho_ = ls_rho
        self.ls_c_ = ls_c
        self.history_size_ = history_size

        super().__init__(
            CLogisticRegression(
                self.solver_type_,
                self.lamda_l2_,
                self.tol_,
                self.max_iter_,
                self.ls_alpha_init_,
                self.ls_rho_,
                self.ls_c_,
                self.history_size_,
                1,
            )
        )

    def _weights_check(self):
        """
        Checks if the model weights have been initialized.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        if self.get_weights() == []:
            raise RuntimeError(
                "Model weights are not initialized. Please train the model before making predictions."
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
        self._weights_check()
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
        self._weights_check()
        return super().predict_proba(X, Y)

    def get_weights(self):
        """
        Returns the weight coefficients of the trained model.

        Returns:
            array-like: The weight coefficients of the logistic regression model.
        """
        return self.model.get_weights()

    def get_bias(self):
        """
        Returns the bias term of the trained model.

        Returns:
            float: The bias term of the logistic regression model.
        """
        return self.model.get_bias()
