from typing import Literal

from ..core import CSVMClassification
from ..base import ClassificationBase


class SVMClassification(ClassificationBase):
    """
    Support Vector Machine (SVM) Classification model.

    This class implements a Support Vector Machine (SVM) classifier with different kernel 
    options, allowing flexibility in handling linear and non-linear classification problems.

    Attributes:
        kernel_ (str): The kernel type used in the model.
        C_ (float): The regularization parameter that controls the trade-off between maximizing 
            the margin and minimizing classification error.
        tol_ (float): The stopping criterion tolerance.
        max_passes_ (int): The maximum number of passes over the training data without improvement.
        gamma_scale_ (bool): Whether to scale the gamma parameter automatically.
        gamma_ (float): The kernel coefficient for RBF and polynomial kernels.
        coef_0_ (float): The independent term in the polynomial and sigmoid kernels.
        degree_ (int): The degree of the polynomial kernel.
        approx_kernel_ (bool): Whether to use an approximate kernel computation.

    Args:
        kernel (Literal["linear", "rbf", "poly"], optional): The kernel type used in the model.
            Defaults to "linear". Must be one of ["linear", "rbf", "poly"].
        C (float, optional): The regularization parameter. Defaults to 1.0.
        tol (float, optional): The stopping criterion tolerance. Defaults to 1e-4.
        max_passes (int, optional): The maximum number of passes over the training data without 
            improvement. Defaults to 100.
        gamma_scale (bool, optional): Whether to scale the gamma parameter automatically. 
            Defaults to True.
        gamma (float, optional): The kernel coefficient for RBF and polynomial kernels. 
            Defaults to 0.1.
        coef_0 (float, optional): The independent term in polynomial and sigmoid kernels. 
            Defaults to 0.0.
        degree (int, optional): The degree of the polynomial kernel. Defaults to 3.
        approx_kernel (bool, optional): Whether to use an approximate kernel computation. 
            Defaults to False.

    Raises:
        ValueError: If an invalid kernel type is provided.

    Methods:
        predict(X):
            Predicts class labels for the given input data.

        predict_proba(X, Y):
            Predicts class probabilities for the given input data.

    Example:
        >>> model = SVMClassification(kernel="rbf", C=1.5, gamma=0.01)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        kernel: Literal["linear", "rbf", "poly"] = "linear",
        C: float = 1.0,
        tol: float = 1e-4,
        max_passes: int = 100,
        gamma_scale: bool = True,
        gamma: float = 0.1,
        coef_0: float = 0.0,
        degree: int = 3,
        approx_kernel: bool = False,
    ):
        """
        Initializes the SVM Classification model with the specified parameters.

        Args:
            kernel (Literal["linear", "rbf", "poly"], optional): The kernel type used in the model.
                Defaults to "linear".
            C (float, optional): The regularization parameter. Defaults to 1.0.
            tol (float, optional): The stopping criterion tolerance. Defaults to 1e-4.
            max_passes (int, optional): The maximum number of passes over the training data without 
                improvement. Defaults to 100.
            gamma_scale (bool, optional): Whether to scale the gamma parameter automatically. 
                Defaults to True.
            gamma (float, optional): The kernel coefficient for RBF and polynomial kernels. 
                Defaults to 0.1.
            coef_0 (float, optional): The independent term in polynomial and sigmoid kernels. 
                Defaults to 0.0.
            degree (int, optional): The degree of the polynomial kernel. Defaults to 3.
            approx_kernel (bool, optional): Whether to use an approximate kernel computation. 
                Defaults to False.

        Raises:
            ValueError: If an invalid kernel type is provided.
        """
        if kernel not in ["linear", "rbf", "poly"]:
            raise ValueError(
                f"Invalid kernel '{kernel}'. Expected one of ['linear', 'rbf', 'poly']."
            )
        else:
            self.kernel_ = kernel

        self.C_ = C
        self.tol_ = tol
        self.max_passes_ = max_passes
        self.gamma_scale_ = gamma_scale
        self.gamma_ = gamma
        self.coef_0_ = coef_0
        self.degree_ = degree
        self.approx_kernel_ = approx_kernel

        super().__init__(
            CSVMClassification(
                self.C_,
                self.tol_,
                self.max_passes_,
                self.kernel_,
                self.gamma_scale_,
                self.gamma_,
                self.coef_0_,
                self.degree_,
                self.approx_kernel_,
            )
        )

    def _fitted_check(self):
        """
        Checks if the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return self.model.check_initialize()

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
        if self._fitted_check():
            return super().predict(X)
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

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
        if self._fitted_check():
            return super().predict_proba(X)
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )
