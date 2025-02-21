from typing import Literal

import numpy as np

from ..core import CPCA
from ..base import UnsupervisedEstimatorBase


class PCA(UnsupervisedEstimatorBase):
    """
    Principal Component Analysis (PCA) model.

    This class implements Principal Component Analysis (PCA), a dimensionality reduction
    technique that transforms data into a lower-dimensional space while preserving
    the maximum variance.

    Attributes:
        n_components_ (int): The number of principal components to retain.

    Args:
        n_components (int, optional): The number of principal components to retain.
            Defaults to 5.

    Methods:
        predict(X):
            Projects the input data into the principal component space.

        transform(X):
            Transforms the input data using the trained PCA model.

        get_components():
            Returns the principal components of the trained model.

    Example:
        >>> model = PCA(n_components=3)
        >>> model.fit(X_train)
        >>> transformed_data = model.transform(X_train)
        >>> components = model.get_components()
    """

    def __init__(self, n_components: int = 5):
        """
        Initializes the PCA model with the specified number of components.

        Args:
            n_components (int, optional): The number of principal components to retain.
                Defaults to 5.
        """
        self.n_components_ = n_components

        super().__init__(
            CPCA(
                self.n_components_,
            )
        )

    def predict(self, X):
        """
        Projects the input data into the principal component space.

        Args:
            X (array-like): The input data to be projected.

        Returns:
            array-like: The projected data in the lower-dimensional space.

        Raises:
            RuntimeError: If the model has not been trained before making predictions.
        """
        if self.model.check_initialize():
            return super().predict(X)
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

    def transform(self, X):
        """
        Transforms the input data using the trained PCA model.

        Args:
            X (array-like): The input data to be transformed.

        Returns:
            array-like: The transformed data.

        Raises:
            RuntimeError: If the model has not been trained before making transformations.
        """
        if self.model.check_initialize():
            return super().transform(X)
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

    def get_components(self):
        """
        Returns the principal components of the trained model.

        Returns:
            np.ndarray: The principal component vectors.

        Raises:
            RuntimeError: If the model has not been trained before retrieving components.
        """
        if self.model.check_initialize():
            return np.array(self.model.get_components()).reshape(
                (-1, self.n_components_), order="F"
            )
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )
