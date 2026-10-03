from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.apd_octree import APDBackgroundOctree
from kanapy.core.power_diagram import AnisotropicPowerDiagram


def slab(width=2, periodic=False):
    shape = (8, 4, 4)
    indices = np.array(list(np.ndindex(shape)))
    labels = np.where(indices[:, 0] < 3, 23, 42)
    labels[(indices[:, 0] >= 3) & (indices[:, 0] < 3 + width)] = 7
    return APDBackgroundOctree(indices, np.zeros(len(labels), dtype=int), labels,
        np.ones(len(labels), dtype=bool), np.ones(len(labels), dtype=bool),
        shape, 0, np.ones(3), periodic)


def costs(tree):
    # Original grain 7 wins the APD; 42 is the best eligible replacement.
    return SimpleNamespace(box_size=tree.box_size.copy(), periodic=tree.periodic,
        grain_ids=np.array([7, 23, 42]),
        costs=lambda points: np.tile([0., 3., 1.], (len(points), 1)))


@pytest.mark.parametrize('width', [1, 2, 3])
def test_axial_width_and_boundary_zone(width):
    tree = slab(width)
    expected = tree.labels == 7 if width <= 2 else np.zeros(len(tree.labels), dtype=bool)
    np.testing.assert_array_equal(tree.thin_grain_cells(), expected)
    if width == 2:
        assert not tree.thin_grain_cells(max_width=1).any()
    tree.boundary_cells[:] = False
    assert not tree.thin_grain_cells().any()


def test_cleanup_cost_and_fixed_geometry():
    tree = slab(1)
    before = {name: getattr(tree, name).copy() for name in
              ('indices', 'levels', 'lower', 'upper', 'boundary_cells', 'sampled_boundary')}
    report = tree.clean_thin_grains(costs(tree))
    assert report['changed_cells'] == 16
    assert report['remaining_thin_cells'] == 0
    assert report['eliminated_grains'] == [7]
    assert all(c['new_label'] == 42 for c in report['changes'])
    assert sum(report['volumes_after'].values()) == pytest.approx(1.)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(tree, name), value)


def test_two_cell_feature_and_determinism():
    first, second = slab(), slab()
    a = first.clean_thin_grains(costs(first), batch_size=1)
    b = second.clean_thin_grains(costs(second), batch_size=1000)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert a['changes'] == b['changes']
    assert a['changed_cells'] == 32
    assert a['remaining_thin_cells'] == 0
    assert len({c['cell'] for c in a['changes']}) == a['changed_cells']


def test_periodic_run_and_exterior():
    tree = slab(2, periodic=True)
    labels = tree.labels.reshape(tree.resolution)
    tree.labels = np.roll(labels, -4, axis=0).ravel()
    np.testing.assert_array_equal(tree.thin_grain_cells(), tree.labels == 7)
    report = tree.clean_thin_grains(costs(tree))
    assert report['changed_cells'] == 32
    tree = slab(1)
    tree.labels[:] = 42
    tree.labels[tree.indices[:, 0] == 0] = 7
    assert not tree.thin_grain_cells().any()


def test_coarse_neighbours_and_no_coarse_changes():
    fine = np.array(list(product((2, 3), (0, 1), (0, 1))))
    indices = np.vstack([[[0, 0, 0], [2, 0, 0]], fine])
    tree = APDBackgroundOctree(indices, np.array([0, 0] + [1]*8),
        np.array([23, 42] + [7]*8), np.ones(10, dtype=bool), np.ones(10, dtype=bool),
        (3, 1, 1), 1, np.ones(3), False)
    np.testing.assert_array_equal(tree.thin_grain_cells(), [False, False] + [True]*8)
    report = tree.clean_thin_grains(costs(tree))
    assert report['changed_cells'] == 8
    np.testing.assert_array_equal(tree.labels[:2], [23, 42])


def test_uniform_periodic_grain_is_not_thin():
    tree = slab(periodic=True)
    tree.labels[:] = 7
    assert not tree.thin_grain_cells().any()


def test_real_apd_thin_slab_and_unchanged_cost_function():
    diagram = AnisotropicPowerDiagram([[.5, .5, .5]] * 2,
        [np.eye(3), np.diag([2., 1., 1.])], [1., 1., 1.])
    diagram.weights[:] = [0, .04**2]
    tree = diagram.background_octree(2, max_depth=3)
    original = diagram.labels(tree.centers)
    assert np.count_nonzero(original == 2) > 0
    report = tree.clean_thin_grains(diagram)
    assert report['changed_cells'] == np.count_nonzero(original == 2)
    assert not report['remaining_thin_mask'].any()
    np.testing.assert_array_equal(diagram.labels(tree.centers), original)


def test_atomic_failure_and_validation():
    tree = slab()
    original = tree.labels.copy()
    diagram = costs(tree)
    diagram.costs = lambda points: np.full((len(points), 3), np.nan)
    with pytest.raises(ValueError, match='finite'):
        tree.clean_thin_grains(diagram)
    np.testing.assert_array_equal(tree.labels, original)
    with pytest.raises(ValueError, match='max_width'):
        tree.thin_grain_cells(max_width=3)
    with pytest.raises(ValueError, match='max_passes'):
        tree.clean_thin_grains(costs(tree), max_passes=0)
