"""Focused tests for RVE statistical and geometry helpers."""

import json

import numpy as np
import pytest

from kanapy.core.rve_stats import (
    arr2mat,
    bbox,
    con_fun,
    find_rot_axis,
    get_grain_geom,
    get_ln_param,
    project_pts,
    update_grid,
)


def test_arr2mat_builds_symmetric_voigt_matrix():
    matrix = arr2mat(np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(matrix, [[1, 6, 5], [6, 2, 4], [5, 4, 3]])


def test_con_fun_is_positive_for_positive_definite_matrix():
    assert con_fun(np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])) > 0.0
    assert con_fun(np.array([-1.0, 2.0, 2.0, 0.0, 0.0, 0.0])) < 0.0


def test_find_rot_axis_detects_symmetric_pair_or_longest_axis():
    assert find_rot_axis(2.0, 2.0, 3.0) == 2
    assert find_rot_axis(1.0, 2.0, 3.0) == 2


def test_get_ln_param_uses_log_median_and_log_standard_deviation():
    data = np.array([1.0, np.e, np.e**2])
    sigma, scale = get_ln_param(data)
    assert np.isclose(sigma, np.sqrt(2.0 / 3.0))
    assert np.isclose(scale, np.e)


def test_project_pts_removes_component_along_plane_normal():
    points = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
    projected = project_pts(points, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(projected[:, 2], 0.0)
    assert np.allclose(projected[:, :2], points[:, :2])


def test_bbox_returns_2d_lengths_and_unit_vectors():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    lengths = bbox(points, return_vector=False, two_dim=True)
    assert np.allclose(lengths, [1.0, 1.0])

    lengths_and_vectors = bbox(points, return_vector=True, two_dim=True)
    assert np.allclose(lengths_and_vectors[:2], [1.0, 1.0])
    assert np.allclose(
        [np.linalg.norm(vector) for vector in lengths_and_vectors[2:]],
        [1.0, 1.0],
    )


def test_get_grain_geom_requires_explicit_2d_mode():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    with pytest.raises(ModuleNotFoundError, match="not implemented"):
        get_grain_geom(points)


def test_get_grain_geom_raw_returns_2d_geometry():
    points = np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]
    ])
    major, minor, major_axis, minor_axis = get_grain_geom(
        points, method="raw", two_dim=True
    )
    assert np.allclose([major, minor], [1.0, 1.0])
    assert np.isclose(np.linalg.norm(major_axis), 1.0)
    assert np.isclose(np.linalg.norm(minor_axis), 1.0)


def test_update_grid_averages_deformation_gradients(tmp_path):
    data = {
        "units": {"Length": "m"},
        "microstructure": [
            {
                "time": 0,
                "grid": {
                    "grid_size": [1.0, 1.0, 1.0],
                    "grid_spacing": [0.5, 0.5, 0.5],
                },
                "voxels": [],
            },
            {
                "time": 1,
                "grid": {},
                "voxels": [
                    {"deformation_gradient": [[2.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0]]},
                    {"deformation_gradient": [[2.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0]]},
                ],
            },
        ],
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    grid_size, spacing, counts, deformation = update_grid(path)

    assert np.allclose(deformation, np.diag([2.0, 1.0, 1.0]))
    assert counts == [4, 2, 2]
    assert np.allclose(spacing, [0.5, 0.5, 0.5])
    assert np.allclose(grid_size, [2.0, 1.0, 1.0])
