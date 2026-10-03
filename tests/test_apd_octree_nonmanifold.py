from itertools import product

import numpy as np
import pytest

from kanapy.core.apd_octree import APDBackgroundOctree
from kanapy.core.voxelization import nonmanifold_grain_boundary_voxels


def uniform(grains, periodic=False):
    indices = np.array(list(np.ndindex(grains.shape)))
    n = grains.size
    # Stored APD flags intentionally blank: diagnostics must use current labels.
    return APDBackgroundOctree(indices, np.zeros(n, dtype=int), grains.ravel().copy(),
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), grains.shape,
        0, np.array([2., 3., 4.]), periodic)


def adaptive(seed, periodic=False):
    leaves = [(0, (1, 0, 0))]  # Coarse right half, adjacent to depth-2 cells.
    for child in product((0, 1), repeat=3):
        if child in ((1, 0, 0), (1, 1, 1)):
            leaves.extend((2, tuple(2*np.array(child) + grandchild))
                          for grandchild in product((0, 1), repeat=3))
        else:
            leaves.append((1, child))
    levels = np.array([level for level, _ in leaves])
    indices = np.array([index for _, index in leaves])
    labels = np.random.default_rng(seed).choice([-5, 7, 23], len(leaves))
    return APDBackgroundOctree(indices, levels, labels, np.zeros(len(labels), bool),
        np.zeros(len(labels), bool), (2, 1, 1), 2, np.array([2., 3., 4.]), periodic)


def dense_reference(tree):
    shape = tuple(np.array(tree.resolution) * 2**tree.max_depth)
    grains = np.empty(shape, dtype=tree.labels.dtype)
    owner = np.empty(shape, dtype=int)
    for cell, (index, level) in enumerate(zip(tree.indices, tree.levels)):
        size = 2**(tree.max_depth - level)
        region = tuple(slice(i*size, (i+1)*size) for i in index)
        grains[region], owner[region] = tree.labels[cell], cell
    mask = nonmanifold_grain_boundary_voxels(grains, periodic=tree.periodic)
    expected = np.zeros(len(tree.labels), dtype=bool)
    expected[np.unique(owner[mask])] = True
    return expected


@pytest.mark.parametrize('periodic', [False, True])
def test_all_binary_vertex_patterns(periodic):
    for pattern in range(256):
        grains = np.array([(pattern >> i) & 1 for i in range(8)]).reshape(2, 2, 2)
        tree = uniform(grains, periodic)
        np.testing.assert_array_equal(tree.nonmanifold_grain_boundary_cells(),
            nonmanifold_grain_boundary_voxels(grains, periodic=periodic).ravel())


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('seed', range(6))
def test_unbalanced_hanging_nodes_against_dense_reference(periodic, seed):
    tree = adaptive(seed, periodic)
    original = tree.labels.copy()
    actual = tree.nonmanifold_grain_boundary_cells(return_report=True, chunk_size=13)
    np.testing.assert_array_equal(actual['cell_mask'], dense_reference(tree))
    np.testing.assert_array_equal(actual['cell_mask'], tree.nonmanifold_grain_boundary_cells())
    np.testing.assert_array_equal(tree.labels, original)
    contacts = actual['contacts']
    assert len({(*c['vertex_index'], c['grain_id']) for c in contacts}) == len(contacts)
    for contact in contacts:
        assert np.all(tree.labels[contact['cell_indices']] == contact['grain_id'])
        assert np.all(actual['cell_mask'][contact['cell_indices']])
        assert np.all(contact['point'] >= 0)
        assert np.all(contact['point'] <= tree.box_size)
        if periodic:
            assert np.all(contact['point'] < tree.box_size)


def test_coarse_fine_planar_and_triple_junctions_are_manifold():
    tree = adaptive(42)
    tree.labels[:] = np.where(tree.centers[:, 0] < 1., 7, 23)
    assert not tree.nonmanifold_grain_boundary_cells().any()
    tree.labels[(tree.centers[:, 0] < 1.) & (tree.centers[:, 1] < 1.5)] = 42
    assert not tree.nonmanifold_grain_boundary_cells().any()


def test_report_known_corner_contact_and_current_labels():
    grains = np.zeros((4, 4, 4), dtype=int)
    grains[1, 1, 1] = grains[2, 2, 2] = 7
    tree = uniform(grains)
    report = tree.nonmanifold_grain_boundary_cells(return_report=True)
    contact = next(c for c in report['contacts'] if c['grain_id'] == 7)
    np.testing.assert_array_equal(contact['vertex_index'], [2, 2, 2])
    np.testing.assert_allclose(contact['point'], [1., 1.5, 2.])
    assert len(contact['cell_indices']) == 2
    assert report['nonmanifold_vertices'] == 1
    tree.labels[:] = 7
    assert not tree.nonmanifold_grain_boundary_cells().any()


@pytest.mark.parametrize('chunk_size', [0, -1, True, 1.5])
def test_bad_chunk_size(chunk_size):
    with pytest.raises(ValueError, match='chunk_size'):
        adaptive(0).nonmanifold_grain_boundary_cells(chunk_size=chunk_size)


def test_empty_and_singleton():
    for shape in [(0, 1, 1), (1, 1, 1)]:
        tree = uniform(np.zeros(shape, dtype=int), periodic=True)
        report = tree.nonmanifold_grain_boundary_cells(return_report=True)
        assert not report['cell_mask'].any()
        assert not report['contacts']
