from typing import Literal

import numpy as np

from ..core import CKMeans, KMeansAlgorithm
from ..base import UnsupervisedEstimatorBase


class KMeans(UnsupervisedEstimatorBase):
    """
    K-Means Clustering model.

    This class implements K-Means clustering, an unsupervised learning algorithm used to
    partition data into K clusters based on similarity.

    Attributes:
        algorithm_ (KMeansAlgorithm): The clustering algorithm used.
        n_cluster_ (int): The number of clusters.
        max_iter_ (int): The maximum number of iterations for convergence.
        tol_ (float): The tolerance for convergence.
        use_kdtree_ (bool): Whether to use a KD-tree for nearest centroid search.

    Args:
        algorithm (Literal["Standard", "Elkan", "Hamerly"], optional):
            The clustering algorithm to use. Defaults to "Standard".
            Must be one of ["Standard", "Elkan", "Hamerly"].
        n_cluster (int, optional): The number of clusters. Defaults to 2.
        max_iter (int, optional): The maximum number of iterations for convergence. Defaults to 100.
        tol (float, optional): The stopping criteria tolerance. Defaults to 1e-4.
        use_kdtree (bool, optional): Whether to use a KD-tree for nearest centroid search. Defaults to True.

    Raises:
        ValueError: If an invalid algorithm type is provided.

    Methods:
        predict(X):
            Assigns each data point to the nearest cluster.

        transform(X):
            Computes the distance of each data point to the cluster centroids.

        get_centroids():
            Returns the cluster centroids after training.

    Example:
        >>> model = KMeans(n_cluster=3, algorithm="Elkan", max_iter=200)
        >>> model.fit(X_train)
        >>> labels = model.predict(X_train)
        >>> centroids = model.get_centroids()
    """

    def __init__(
        self,
        algorithm: Literal["Standard", "Elkan", "Hamerly"] = "Standard",
        n_cluster: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        use_kdtree: bool = True,
    ):
        """
        Initializes the K-Means clustering model with the specified parameters.

        Args:
            algorithm (Literal["Standard", "Elkan", "Hamerly"], optional):
                The clustering algorithm to use. Defaults to "Standard".
            n_cluster (int, optional): The number of clusters. Defaults to 2.
            max_iter (int, optional): The maximum number of iterations for convergence. Defaults to 100.
            tol (float, optional): The stopping criteria tolerance. Defaults to 1e-4.
            use_kdtree (bool, optional): Whether to use a KD-tree for nearest centroid search. Defaults to True.

        Raises:
            ValueError: If an invalid algorithm type is provided.
        """
        if algorithm == "Standard":
            self.algorithm_ = KMeansAlgorithm.STANDARD
        elif algorithm == "Elkan":
            self.algorithm_ = KMeansAlgorithm.ELKAN
        elif algorithm == "Hamerly":
            self.algorithm_ = KMeansAlgorithm.HAMERLY
        else:
            raise ValueError(
                f"Invalid algorithm '{algorithm}'. Expected one of ['Standard', 'Elkan', 'Hamerly']."
            )

        self.n_cluster_ = n_cluster
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.use_kdtree_ = use_kdtree

        super().__init__(
            CKMeans(
                self.n_cluster_,
                self.max_iter_,
                self.tol_,
                self.algorithm_,
                self.use_kdtree_,
            )
        )

    def predict(self, X):
        """
        Assigns each data point to the nearest cluster.

        Args:
            X (array-like): The input data for clustering.

        Returns:
            array-like: The predicted cluster labels for each data point.

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
        Computes the distance of each data point to the cluster centroids.

        Args:
            X (array-like): The input data for transformation.

        Returns:
            array-like: The distances from each data point to the centroids.

        Raises:
            RuntimeError: If the model has not been trained before making transformations.
        """
        if self.model.check_initialize():
            return super().transform(X)
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )

    def get_centroids(self):
        """
        Returns the cluster centroids after training.

        Returns:
            np.ndarray: The centroid coordinates for each cluster.

        Raises:
            RuntimeError: If the model has not been trained before retrieving centroids.
        """
        if self.model.check_initialize():
            return np.array(self.model.get_centroids()).reshape(
                (self.n_cluster_, -1), order="F"
            )
        else:
            raise RuntimeError(
                "Model is not initialized. Please train the model before making predictions."
            )
