import numpy as np
import pytest

from kanapy.core.voxelization import (
    grain_boundary_voxels, nonmanifold_grain_boundary_voxels,
)


@pytest.mark.parametrize('second', [(2, 2, 1), (2, 2, 2)])
def test_edge_and_vertex_contacts(second):
    grains = np.zeros((4, 4, 4), dtype=int)
    grains[1, 1, 1] = grains[second] = 7
    mask = nonmanifold_grain_boundary_voxels(grains)
    assert mask[1, 1, 1] and mask[second]
    assert np.all(~mask | grain_boundary_voxels(grains))
    np.testing.assert_array_equal(
        mask, nonmanifold_grain_boundary_voxels(grains, chunk_size=3))


@pytest.mark.parametrize('periodic', [False, True])
def test_regular_grain_shapes_and_triple_junction(periodic):
    grains = np.zeros((6, 6, 6), dtype=int)
    assert not nonmanifold_grain_boundary_voxels(grains, periodic=periodic).any()
    grains[:3] = 7
    assert not nonmanifold_grain_boundary_voxels(grains, periodic=periodic).any()
    grains[3:, :3] = 23
    assert not nonmanifold_grain_boundary_voxels(grains, periodic=periodic).any()
    grains[:] = 0
    grains[2:4, 2:4, 2:4] = 7
    assert not nonmanifold_grain_boundary_voxels(grains, periodic=periodic).any()


def test_distinct_grains_touching_at_corner_are_valid():
    # All octants have different IDs: each grain is individually manifold.
    grains = np.arange(8).reshape(2, 2, 2)
    assert not nonmanifold_grain_boundary_voxels(grains).any()


@pytest.mark.parametrize('axis', range(3))
def test_periodic_seam_and_translation(axis):
    grains = np.zeros((5, 5, 5), dtype=int)
    grains[1, 1, 1] = grains[2, 2, 2] = 7
    mask = nonmanifold_grain_boundary_voxels(grains, periodic=True)
    shifted = np.roll(grains, -2, axis=axis)
    np.testing.assert_array_equal(
        nonmanifold_grain_boundary_voxels(shifted, periodic=True),
        np.roll(mask, -2, axis=axis))
    assert not nonmanifold_grain_boundary_voxels(shifted).any()


def test_empty_singleton_and_input_preservation():
    assert nonmanifold_grain_boundary_voxels(np.empty((0, 2, 3))).shape == (0, 2, 3)
    grains = np.array([7, 23, 7]).reshape(1, 3, 1)
    original = grains.copy()
    assert not nonmanifold_grain_boundary_voxels(grains, periodic=True).any()
    np.testing.assert_array_equal(grains, original)


@pytest.mark.parametrize('chunk_size', [0, -1, 1.5, True])
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match='positive integer'):
        nonmanifold_grain_boundary_voxels(np.zeros((2, 2, 2)), chunk_size=chunk_size)
