"""Focused tests for the standard ORIX texture helpers."""

import numpy as np
import pytest
from orix.crystal_map import Phase
from orix.quaternion import Orientation

from kanapy.texture import (
    createOrisetRandom,
    find_similar_regions,
    find_similar_regions_by_misorientation,
    get_distinct_colormap,
    get_ipf_colors,
    get_proper_symmetry_quaternions,
    mean_orientation_data,
    neighbors,
)


def test_neighbors_supports_four_and_eight_connectivity():
    assert neighbors(2, 3, connectivity=4) == [
        (3, 3), (1, 3), (2, 4), (2, 2)
    ]
    assert len(neighbors(2, 3, connectivity=8)) == 8


def test_distinct_colormap_returns_requested_rgb_colors():
    colors = get_distinct_colormap(4)
    assert len(colors) == 4
    assert all(len(color) == 3 for color in colors)
    assert np.all((np.asarray(colors) >= 0.0) & (np.asarray(colors) <= 1.0))


def test_proper_symmetry_quaternions_remove_inversion():
    symmetry = Phase(point_group="m-3m").point_group
    quaternions = get_proper_symmetry_quaternions(symmetry)
    assert quaternions.shape == (24, 4)
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0)


def test_mean_orientation_data_is_normalized():
    symmetry = Phase(point_group="m-3m").point_group
    orientations = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
    ])
    mean = mean_orientation_data(orientations, symmetry)
    assert mean.shape == (4,)
    assert np.isclose(np.linalg.norm(mean), 1.0)
    assert mean[0] > 0.0


def test_ipf_colors_match_orientation_count():
    orientations = Orientation.from_euler(np.zeros((3, 3)))
    colors = np.asarray(get_ipf_colors(orientations))
    assert colors.shape == (3, 3)
    assert np.all((colors >= 0.0) & (colors <= 1.0))


def test_find_similar_regions_respects_tolerance_and_connectivity():
    values = np.array([[0.0, 0.02, 1.0], [0.01, 0.0, 1.02]])
    labels, count = find_similar_regions(values, tolerance=0.05, connectivity=1)
    assert count == 2
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 2] == labels[1, 2]
    assert labels[0, 0] != labels[0, 2]


def test_find_similar_regions_by_misorientation_masks_background():
    symmetry = Phase(point_group="m-3m").point_group
    ori_map = np.zeros((2, 2, 4), dtype=float)
    ori_map[..., 0] = 1.0
    phase_mask = np.array([[True, True], [False, True]])
    labels, count = find_similar_regions_by_misorientation(
        ori_map, phase_mask, symmetry, tolerance=0.01
    )
    assert count == 1
    assert np.all(labels[~phase_mask] == 0)
    assert np.all(labels[phase_mask] == 1)


def test_create_oriset_random_rejects_mtex_only_options():
    with pytest.raises(ModuleNotFoundError, match="kanapy-mtex"):
        createOrisetRandom(4, hist=np.ones(4))
