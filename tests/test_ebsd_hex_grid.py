"""Tests for rectangular EBSD-grid resampling helpers."""

import numpy as np
import pytest

from kanapy.ebsd_hex_grid import (
    _idw_weights,
    _normalize_quat,
    make_regular_grid,
    quat_markley_mean,
    resample_ebsd_to_rect_grid,
    resample_phase_majority,
    resample_quat_knn_markley,
    resample_scalar_idw,
)


def test_make_regular_grid_from_point_counts():
    xy = np.array([[0.0, 0.0], [2.0, 1.0]])
    x_grid, y_grid, points = make_regular_grid(xy, nx=3, ny=2)

    assert x_grid.shape == (2, 3)
    assert y_grid.shape == (2, 3)
    assert points.shape == (6, 2)
    assert np.allclose(points[0], [0.0, 0.0])
    assert np.allclose(points[-1], [2.0, 1.0])


def test_make_regular_grid_from_spacing_requires_both_modes():
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    x_grid, y_grid, points = make_regular_grid(xy, dx_out=0.5, dy_out=0.5)
    assert x_grid.shape == (3, 3)
    assert y_grid.shape == (3, 3)
    assert points.shape == (9, 2)

    with pytest.raises(ValueError, match="either"):
        make_regular_grid(xy)


def test_idw_weights_are_row_normalized():
    distances = np.array([[1.0, 2.0], [2.0, 2.0]])
    weights = _idw_weights(distances)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 0] > weights[0, 1]
    assert np.allclose(weights[1], [0.5, 0.5])


def test_phase_majority_and_scalar_idw_resampling():
    phase = np.array([1, 1, 2, 2])
    indices = np.array([[0, 1, 2], [2, 3, 0]])
    assert np.array_equal(resample_phase_majority(phase, indices), [1, 2])

    values = np.array([0.0, 2.0])
    nearest = np.array([[0, 1]])
    distances = np.array([[1.0, 3.0]])
    result = resample_scalar_idw(values, nearest, distances)
    assert np.allclose(result, [0.2])

    exact = resample_scalar_idw(
        values, np.array([[1, 0]]), np.array([[0.0, 1.0]])
    )
    assert np.allclose(exact, [2.0])


def test_quaternion_normalization_and_markley_mean():
    quaternions = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    normalized = _normalize_quat(quaternions)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)

    mean = quat_markley_mean(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(mean, [1.0, 0.0, 0.0, 0.0])


def test_quaternion_knn_resampling_handles_exact_and_weighted_points():
    source = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    indices = np.array([[0, 1], [1, 0]])
    distances = np.array([[0.0, 1.0], [1.0, 1.0]])
    result = resample_quat_knn_markley(source, indices, distances)

    assert result.shape == (2, 4)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
    assert np.allclose(result[0], source[0])


def test_resample_ebsd_to_rect_grid_preserves_shapes_and_exact_samples():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    phase = np.array([1, 2, 3, 4])
    iq = np.array([10.0, 20.0, 30.0, 40.0])
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1))

    x_grid, y_grid, phase_grid, iq_grid, quat_grid = resample_ebsd_to_rect_grid(
        xy,
        phase,
        quat,
        iq,
        nx=2,
        ny=2,
        k_phase=1,
        k_iq=1,
        k_quat=1,
    )

    assert x_grid.shape == (2, 2)
    assert y_grid.shape == (2, 2)
    assert phase_grid.shape == (2, 2)
    assert iq_grid.shape == (2, 2)
    assert quat_grid.shape == (2, 2, 4)
    assert np.array_equal(phase_grid.ravel(), phase)
    assert np.allclose(iq_grid.ravel(), iq)
    assert np.allclose(quat_grid[..., 0], 1.0)
