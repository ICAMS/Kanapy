import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kanapy.core.power_diagram import AnisotropicPowerDiagram


def planar(periodic=False):
    return AnisotropicPowerDiagram([[.25, .5, .5], [.75, .5, .5]],
        [np.eye(3), np.eye(3)], [1, 1, 1], grain_ids=[7, 23], periodic=periodic)


def test_planar_refinement_and_exact_cover():
    diagram = planar()
    original = diagram.weights.copy()
    tree = diagram.background_octree(2, max_depth=3)
    assert set(tree.levels) == {1, 2, 3}
    assert len(tree.levels) < tree.summary()['uniform_finest_cells']
    assert tree.summary()['total_volume'] == pytest.approx(1.)
    assert np.all(tree.levels[tree.boundary_cells] == 3)
    assert np.all(tree.lower[tree.boundary_cells, 0] <= .5)
    assert np.all(tree.upper[tree.boundary_cells, 0] >= .5)
    # Every finest-lattice voxel must be covered by exactly one leaf.
    coverage = np.zeros((16, 16, 16), dtype=int)
    for index, level in zip(tree.indices, tree.levels):
        stride = 2**(3 - level)
        coverage[tuple(slice(i * stride, (i + 1) * stride) for i in index)] += 1
    assert np.all(coverage == 1)
    np.testing.assert_array_equal(diagram.weights, original)


def test_enclosed_grain_missed_by_root_samples():
    # Grain 23 is a tiny sphere around (.25,.25,.25), missed by all 27 root samples.
    center = [.25, .25, .25]
    diagram = AnisotropicPowerDiagram([center, center], [np.eye(3), 2*np.eye(3)],
                                       [1, 1, 1], grain_ids=[7, 23])
    diagram.weights[:] = [0, .04**2]
    root = diagram.background_octree(1, max_depth=0)
    assert root.boundary_cells[0] and not root.sampled_boundary[0]
    tree = diagram.background_octree(1, max_depth=4)
    assert tree.sampled_boundary.any()
    assert tree.levels.max() == 4


@pytest.mark.parametrize('periodic', [False, True])
def test_interior_bounds_and_batch_independence(periodic):
    diagram = AnisotropicPowerDiagram([[.15, .2, .3], [.8, .7, .6]],
        [np.diag([1., 2., 3.]), np.array([[2., .2, 0], [.2, 1., 0], [0, 0, 1.]])],
        [1, 1, 1], periodic=periodic)
    diagram.weights[:] = [.02, -.03]
    tree = diagram.background_octree((2, 3, 1), max_depth=2)
    other = diagram.background_octree((2, 3, 1), max_depth=2, batch_size=7)
    for name in ('indices', 'levels', 'labels', 'boundary_cells', 'sampled_boundary'):
        np.testing.assert_array_equal(getattr(tree, name), getattr(other, name))
    selected = ~tree.boundary_cells
    rng = np.random.default_rng(123)
    points = tree.lower[selected, None] + tree.sizes[selected, None] * rng.random((selected.sum(), 100, 3))
    if selected.any():
        labels = diagram.labels(points.reshape(-1, 3)).reshape(-1, 100)
        assert np.all(labels == tree.labels[selected, None])


def test_single_grain_rectangular_box_and_periodic_seam():
    diagram = AnisotropicPowerDiagram([[.5, .5, .5]], [np.eye(3)], [2, 3, 4], periodic=True)
    tree = diagram.background_octree((2, 3, 4), max_depth=3)
    assert len(tree.levels) == 24
    assert not tree.boundary_cells.any()
    assert tree.summary()['total_volume'] == pytest.approx(24.)
    seam = planar(periodic=True).background_octree(2, max_depth=2)
    assert np.any(seam.boundary_cells & (seam.lower[:, 0] == 0))
    assert np.any(seam.boundary_cells & (seam.upper[:, 0] == 1))


@pytest.mark.parametrize('kwargs', [dict(resolution=0), dict(resolution=True),
    dict(max_depth=-1), dict(max_depth=31), dict(batch_size=0), dict(max_cells=1),
    dict(resolution=1, max_depth=2, max_cells=8)])
def test_validation_and_budget(kwargs):
    with pytest.raises(ValueError):
        planar().background_octree(**kwargs)


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('color_by', ['level', 'grain', 'boundary'])
def test_plot_slice(axis, color_by):
    tree = planar().background_octree(2, max_depth=1)
    ax = tree.plot_slice(axis, position=1., color_by=color_by)
    assert len(ax.collections[0].get_paths()) > 0
    ax.figure.canvas.draw()
    plt.close(ax.figure)
