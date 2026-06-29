"""
A vectorised, reusable implementation of a Kohonen Self-Organising Map (SOM).

A SOM is an unsupervised neural network that projects high-dimensional data onto
a 2D grid so that similar inputs end up near each other -- useful for clustering
and visualisation.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
        # Quantisation error after each iteration, populated by fit(record_history=True).
        # This is what lets you plot a learning curve and compare hyperparameters.
        self.quantization_error_history_: list[float] = []

    def _check_fitted(self) -> None:
        if self.weights is None:
            raise RuntimeError("SOM is not trained yet -- call fit() first.")

    def _decay(self, initial_value: float, t: int) -> float:
        """Exponential decay shared by the radius and the learning rate."""
        return initial_value * np.exp(-t / self.time_constant)

    def _best_matching_unit(self, vector: np.ndarray) -> tuple[int, int]:
        """Return the (x, y) grid coordinates of the node closest to ``vector``."""
        # Squared Euclidean distance from every node to the input. We skip the
        # square root because the closest node is the same either way.
        sq_dist = np.sum((self.weights - vector) ** 2, axis=-1)  # (width, height)
        return np.unravel_index(np.argmin(sq_dist), sq_dist.shape)

    def _distances_to_nodes(self, data: np.ndarray) -> np.ndarray:
        """Euclidean distance from every input to every node -> (n_samples, n_nodes)."""
        flat = self.weights.reshape(-1, self.weights.shape[-1])  # (n_nodes, dim)
        return np.linalg.norm(data[:, None, :] - flat[None], axis=2)

    def fit(self, data: np.ndarray, record_history: bool = False) -> "KohonenSOM":
        """Train the SOM on ``data`` of shape (n_samples, n_features).

        If ``record_history`` is set, the quantisation error is measured after
        every iteration and stored on ``quantization_error_history_`` so the
        learning curve can be inspected.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("data must be a non-empty 2D array (n_samples, n_features)")

        # Infer the input dimension from the data instead of hard-coding 3.
        n_features = data.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.random((self.width, self.height, n_features))
        self.quantization_error_history_ = []
        logger.info(
            "Training SOM: grid=%dx%d, iterations=%d, n_samples=%d, n_features=%d",
            self.width, self.height, self.n_iterations, data.shape[0], n_features,
        )

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

            if record_history:
                qe = self.quantization_error(data)
                self.quantization_error_history_.append(qe)
                logger.debug("iter %d/%d  qe=%.6f", t + 1, self.n_iterations, qe)

        if record_history:
            logger.info("Training complete: final qe=%.6f", self.quantization_error_history_[-1])
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Map each input vector to its BMU coordinates -> array of shape (n, 2)."""
        self._check_fitted()
        data = np.asarray(data, dtype=float)
        nearest = self._distances_to_nodes(data).argmin(axis=1)
        xs, ys = np.unravel_index(nearest, (self.width, self.height))
        return np.stack([xs, ys], axis=1)

    def quantization_error(self, data: np.ndarray) -> float:
        """Mean distance from each input to its BMU's weights (lower = better fit).

        The primary SOM metric: it measures how well the map represents the data,
        which is exactly what training optimises.
        """
        self._check_fitted()
        data = np.asarray(data, dtype=float)
        return float(self._distances_to_nodes(data).min(axis=1).mean())

    def topographic_error(self, data: np.ndarray) -> float:
        """Fraction of inputs whose 1st and 2nd BMUs are not grid-neighbours.

        The secondary SOM metric: it measures topology preservation -- similar
        inputs should land on nearby nodes. Lower is better.
        """
        self._check_fitted()
        data = np.asarray(data, dtype=float)
        nearest_two = np.argsort(self._distances_to_nodes(data), axis=1)[:, :2]
        xs, ys = np.unravel_index(nearest_two, (self.width, self.height))
        non_adjacent = (np.abs(xs[:, 0] - xs[:, 1]) > 1) | (np.abs(ys[:, 0] - ys[:, 1]) > 1)
        return float(np.mean(non_adjacent))

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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(42)
    colours = rng.random((10, 3))

    for (width, height, n_iter, out) in [(10, 10, 100, "100.png"), (100, 100, 1000, "1000.png")]:
        som = KohonenSOM(width, height, n_iterations=n_iter, random_state=42)
        som.fit(colours, record_history=True)
        plt.imsave(out, som.weights)
        print(
            f"{width}x{height}, {n_iter} iters -> "
            f"QE={som.quantization_error(colours):.4f}  "
            f"TE={som.topographic_error(colours):.4f}"
        )
