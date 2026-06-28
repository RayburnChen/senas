"""
A vectorised, reusable implementation of a Kohonen Self-Organising Map (SOM).

A SOM is an unsupervised neural network that projects high-dimensional data onto
a 2D grid so that similar inputs end up near each other -- useful for clustering
and visualisation.
"""
from __future__ import annotations

import numpy as np


class KohonenSOM:
    """A Kohonen Self-Organising Map.

    Parameters
    ----------
    width, height : int
        Grid dimensions. The number of nodes is ``width * height``.
    n_iterations : int
        Number of training iterations (passes over the data).
    initial_learning_rate : float
        Learning rate (alpha_0) at iteration 0.
    random_state : int | None
        Seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_iterations: int = 100,
        initial_learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")

        self.width = width
        self.height = height
        self.n_iterations = n_iterations
        self.initial_learning_rate = initial_learning_rate
        self.random_state = random_state

        self.initial_radius = max(width, height) / 2.0
        # Guard the log(sigma0) edge case: for a small grid sigma0 can be <= 1,
        # which makes log(sigma0) zero (divide-by-zero) or negative (decay runs
        # backwards and blows up). Fall back to a neutral time constant.
        log_radius = np.log(self.initial_radius) if self.initial_radius > 1 else 1.0
        self.time_constant = n_iterations / log_radius

        # Grid coordinates are constant, so compute them once up front instead of
        # rebuilding them inside the training loop.
        xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
        self._coords_x = xs  # shape (width, height)
        self._coords_y = ys

        self.weights: np.ndarray | None = None

    def _decay(self, initial_value: float, t: int) -> float:
        """Exponential decay shared by the radius and the learning rate."""
        return initial_value * np.exp(-t / self.time_constant)

    def _best_matching_unit(self, vector: np.ndarray) -> tuple[int, int]:
        """Return the (x, y) grid coordinates of the node closest to ``vector``."""
        # Squared Euclidean distance from every node to the input. We skip the
        # square root because the closest node is the same either way.
        sq_dist = np.sum((self.weights - vector) ** 2, axis=-1)  # (width, height)
        return np.unravel_index(np.argmin(sq_dist), sq_dist.shape)

    def fit(self, data: np.ndarray) -> "KohonenSOM":
        """Train the SOM on ``data`` of shape (n_samples, n_features)."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("data must be a non-empty 2D array (n_samples, n_features)")

        # Infer the input dimension from the data instead of hard-coding 3.
        n_features = data.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.random((self.width, self.height, n_features))

        for t in range(self.n_iterations):
            radius = self._decay(self.initial_radius, t)
            learning_rate = self._decay(self.initial_learning_rate, t)
            two_radius_sq = 2.0 * radius ** 2

            for vector in data:
                bmu_x, bmu_y = self._best_matching_unit(vector)
                # Squared grid distance from EVERY node to the BMU, in one array
                # operation -- this replaces Sam's pure-Python double loop.
                grid_sq_dist = (self._coords_x - bmu_x) ** 2 + (self._coords_y - bmu_y) ** 2
                influence = np.exp(-grid_sq_dist / two_radius_sq)  # (width, height)
                # Nudge all nodes toward the input at once via broadcasting.
                self.weights += learning_rate * influence[..., None] * (vector - self.weights)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Map each input vector to its BMU coordinates -> array of shape (n, 2)."""
        data = np.asarray(data, dtype=float)
        return np.array([self._best_matching_unit(v) for v in data])

    def quantization_error(self, data: np.ndarray) -> float:
        """Mean distance from each input to its BMU's weights (lower = better fit)."""
        data = np.asarray(data, dtype=float)
        errors = [
            np.sqrt(np.sum((self.weights[x, y] - v) ** 2))
            for v, (x, y) in zip(data, self.predict(data))
        ]
        return float(np.mean(errors))

    def save(self, path: str) -> None:
        """Persist the trained weights (the model artifact) to ``path``.npy."""
        np.save(path, self.weights)

    def load(self, path: str) -> "KohonenSOM":
        """Load trained weights from ``path``."""
        self.weights = np.load(path)
        return self


if __name__ == "__main__":
    # Demo only. matplotlib is imported locally so the core module stays
    # dependency-light for anyone who just wants the SOM.
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    colours = rng.random((10, 3))

    som = KohonenSOM(10, 10, n_iterations=100, random_state=42).fit(colours)
    plt.imsave("100.png", som.weights)
    print("quantization error (10x10):", round(som.quantization_error(colours), 4))

    som = KohonenSOM(100, 100, n_iterations=1000, random_state=42).fit(colours)
    plt.imsave("1000.png", som.weights)
    print("quantization error (100x100):", round(som.quantization_error(colours), 4))
