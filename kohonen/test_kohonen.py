"""Unit tests demonstrating the key recommendations. Run with: pytest -q"""
import numpy as np

from kohonen.productionized_kohonen import KohonenSOM


def test_output_shape():
    data = np.random.default_rng(0).random((8, 3))
    som = KohonenSOM(5, 5, n_iterations=10, random_state=0).fit(data)
    assert som.weights.shape == (5, 5, 3)


def test_reproducible_with_seed():
    data = np.random.default_rng(0).random((8, 3))
    w1 = KohonenSOM(5, 5, n_iterations=10, random_state=42).fit(data).weights
    w2 = KohonenSOM(5, 5, n_iterations=10, random_state=42).fit(data).weights
    assert np.allclose(w1, w2)


def test_infers_input_dimension():
    # Not RGB: 5-dimensional inputs should just work (no hard-coded 3).
    data = np.random.default_rng(0).random((8, 5))
    som = KohonenSOM(4, 4, n_iterations=5, random_state=0).fit(data)
    assert som.weights.shape == (4, 4, 5)


def test_bmu_moves_toward_input():
    target = np.array([[0.2, 0.8, 0.5]])
    som = KohonenSOM(6, 6, n_iterations=50, random_state=1).fit(target)
    x, y = som._best_matching_unit(target[0])
    assert np.linalg.norm(som.weights[x, y] - target[0]) < 0.1


def test_quantization_error_decreases_with_training():
    data = np.random.default_rng(0).random((20, 3))
    err_short = KohonenSOM(10, 10, n_iterations=2, random_state=0).fit(data).quantization_error(data)
    err_long = KohonenSOM(10, 10, n_iterations=100, random_state=0).fit(data).quantization_error(data)
    assert err_long < err_short


def test_small_grid_does_not_crash():
    # The log(sigma0) edge case: a 2x2 grid must not divide by zero or blow up.
    data = np.random.default_rng(0).random((5, 3))
    som = KohonenSOM(2, 2, n_iterations=10, random_state=0).fit(data)
    assert np.all(np.isfinite(som.weights))


def test_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        KohonenSOM(5, 5).fit(np.array([1.0, 2.0, 3.0]))  # 1D, not 2D
