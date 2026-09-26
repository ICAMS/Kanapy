import numpy as np
import pytest

from kanapy.core.voxelization import grain_boundary_voxels


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('shape', [(5, 4, 3), (1, 4, 2), (1, 1, 1)])
def test_against_neighbour_reference(shape, periodic):
    grains = np.random.default_rng(42).choice([0, 7, 23], size=shape)
    expected = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        for axis in range(3):
            for step in (-1, 1):
                neighbour = list(index)
                neighbour[axis] += step
                if periodic:
                    neighbour[axis] %= shape[axis]
                elif not 0 <= neighbour[axis] < shape[axis]:
                    continue
                expected[index] |= grains[index] != grains[tuple(neighbour)]
    np.testing.assert_array_equal(
        grain_boundary_voxels(grains, periodic=periodic), expected)


@pytest.mark.parametrize('axis', range(3))
def test_planar_interface_and_periodic_seam(axis):
    shape = [1, 1, 1]
    shape[axis] = 6
    grains = np.array([7, 7, 7, 23, 23, 23]).reshape(shape)
    mask = grain_boundary_voxels(grains)
    np.testing.assert_array_equal(np.flatnonzero(mask) + 1, [3, 4])
    mask = grain_boundary_voxels(grains, periodic=True)
    np.testing.assert_array_equal(np.flatnonzero(mask) + 1, [1, 3, 4, 6])


@pytest.mark.parametrize('periodic', [False, True])
def test_only_face_neighbours_are_marked(periodic):
    grains = np.full((5, 5, 5), 7)
    grains[2, 2, 2] = 23
    original = grains.copy()
    expected = np.zeros(grains.shape, dtype=bool)
    expected[1:4, 2, 2] = True
    expected[2, 1:4, 2] = True
    expected[2, 2, 1:4] = True
    mask = grain_boundary_voxels(grains, periodic=periodic)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(grains, original)


def test_single_grain_and_empty_array():
    assert not grain_boundary_voxels(np.ones((3, 4, 5)), periodic=True).any()
    assert grain_boundary_voxels(np.empty((0, 3, 2))).shape == (0, 3, 2)


def test_invalid_dimension():
    with pytest.raises(ValueError, match='3D'):
        grain_boundary_voxels(np.ones((3, 4)))


@pytest.mark.parametrize('periodic', [None, 1, 'false', (True, False, True)])
def test_invalid_periodic(periodic):
    with pytest.raises(ValueError, match='boolean'):
        grain_boundary_voxels(np.ones((3, 4, 5)), periodic=periodic)
