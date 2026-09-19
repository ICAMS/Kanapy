"""Geometric checks for the experimental continuous APD and junction network."""

from itertools import product

import numpy as np
import pytest

from kanapy.core.entities import Ellipsoid
from kanapy.core.power_diagram import AnisotropicPowerDiagram


def test_particle_metric_matches_kanapy_surface_and_ignores_duplicates():
    angle = 0.63
    p = Ellipsoid(7, 2., 2., 2., 2., 1., 0.7,
                  np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)]))
    other = Ellipsoid(12, 4., 4., 4., 1., 1., 1., np.array([1., 0., 0., 0.]))
    duplicate = Ellipsoid(99, 8., 2., 2., 2., 1., 0.7, p.quat, dup=7)
    apd = AnisotropicPowerDiagram.from_particles([p, other, duplicate], (6, 6, 6))
    np.testing.assert_array_equal(apd.grain_ids, [7, 12])
    np.testing.assert_allclose(np.linalg.det(apd.matrices), 1.)
    surface_cost = apd.costs(p.surfacePointsGen() + p.get_pos())[:, 0]
    np.testing.assert_allclose(surface_cost, (p.a * p.b * p.c)**(2 / 3))
    assert apd.target_volumes[0] / apd.target_volumes[1] == pytest.approx(1.4)
    assert apd.target_volumes.sum() == pytest.approx(216)
    apd.fit_volumes(n_samples=4096, tolerance=0.05)
    assert (p.a, p.b, p.c) == (2., 1., 0.7)
    np.testing.assert_array_equal(p.get_pos(), [2., 2., 2.])


def test_volume_fit_recovers_analytic_two_grain_boundary():
    apd = AnisotropicPowerDiagram([[.25, .5, .5], [.75, .5, .5]],
                                  np.tile(np.eye(3), (2, 1, 1)), (1, 1, 1), [.3, .7])
    fit = apd.fit_volumes(n_samples=8192, tolerance=0.005)
    assert fit['converged']
    np.testing.assert_allclose(apd.estimate_volumes(16384, seed=21), [.3, .7], atol=0.001)
    np.testing.assert_array_equal(apd.labels([[.299, .4, .6], [.301, .4, .6]]), [1, 2])
    assert apd.weights[0] - apd.weights[1] == pytest.approx(-.2, abs=.001)


def test_periodic_anisotropic_costs_match_lattice_search():
    # Strongly rotated anisotropy: coordinatewise wrapping is not the nearest
    # image in this metric. Search bounds must work beyond the nearest 27 images.
    angle = .43
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    matrix = rotation.T @ np.diag([.1, 10., 1.]) @ rotation
    center = np.array([.91, .16, .42])
    apd = AnisotropicPowerDiagram([center], [matrix], [1., 1., 1.], periodic=True)
    points = np.random.default_rng(8).uniform(size=(70, 3))
    shifts = np.array(list(product(range(-6, 7), repeat=3)))
    delta = points[:, None, :] - center - shifts
    brute = np.einsum('pni,ij,pnj->pn', delta, matrix, delta).min(axis=1)
    np.testing.assert_allclose(apd.costs(points)[:, 0], brute, atol=1e-12)
    np.testing.assert_allclose(apd.costs(points + [2, -3, 4])[:, 0], brute, atol=1e-12)
    wrapped = points - center
    wrapped -= np.round(wrapped)
    assert np.max(np.einsum('pi,ij,pj->p', wrapped, matrix, wrapped) - brute) > .1
    assert len(apd.extract_junctions(4).segments) == 0


def test_spherical_junction_is_exact_and_lines_are_on_lower_envelope():
    centers = np.array([[.2, .2, .2], [.8, .2, .2], [.5, .8, .2], [.5, .5, .8]])
    ids = np.array([10, 20, 30, 40])
    apd = AnisotropicPowerDiagram(centers, np.tile(np.eye(3), (4, 1, 1)), [1, 1, 1], grain_ids=ids)
    network = apd.extract_junctions(8)
    expected = np.linalg.solve(2 * (centers[1:] - centers[0]),
                               np.sum(centers[1:]**2, axis=1) - np.sum(centers[0]**2))
    assert network.vertices.shape == (1, 3)
    np.testing.assert_allclose(network.vertices[0], expected, atol=1e-10)
    assert network.vertex_grain_ids == ((10, 20, 30, 40),)
    assert len(network.boundary_points) == 4
    assert len(network.segments) > 4
    for point, triple in zip(network.segments.mean(axis=1), network.grain_ids):
        scores = apd.costs([point])[0]
        members = np.searchsorted(ids, triple)
        np.testing.assert_allclose(scores[members], scores.min(), atol=1e-9)
    # Every segment end is another segment end, a true junction, or a box exit.
    ends = network.segments.reshape(-1, 3)
    for point in ends:
        on_box = np.any(np.isclose(point, 0) | np.isclose(point, 1))
        assert on_box or np.count_nonzero(np.linalg.norm(ends - point, axis=1) < 1e-8) >= 2


def test_curved_diagram_network_converges_under_refinement():
    centers = [[.18, .2, .25], [.82, .22, .3], [.48, .8, .2], [.52, .52, .82]]
    matrices = np.array([np.diag(a) for a in [[.6, 1.5, 1.1], [1.4, .7, 1.],
                                               [.8, 1.1, 1.4], [1.3, .9, .65]]])
    apd = AnisotropicPowerDiagram(centers, matrices, [1, 1, 1])
    errors = []
    for resolution in [6, 12]:
        network = apd.extract_junctions(resolution)
        assert len(network.vertices) > 0
        scores = apd.costs(network.segments.mean(axis=1))
        ties = np.take_along_axis(scores, network.grain_ids - 1, axis=1)
        errors.append(np.max(ties.max(axis=1) - scores.min(axis=1)))
    assert errors[1] < .5 * errors[0]


def test_periodic_network_matches_on_opposite_box_faces():
    centers = np.random.default_rng(15).uniform(.05, .95, (7, 3))
    apd = AnisotropicPowerDiagram(centers, np.tile(np.eye(3), (7, 1, 1)),
                                  [1, 1, 1], periodic=True)
    network = apd.extract_junctions(8)
    assert len(network.boundary_points) > 0
    for axis in range(3):
        points = network.boundary_points
        lo = points[np.isclose(points[:, axis], 0)]
        hi = points[np.isclose(points[:, axis], 1)].copy()
        hi[:, axis] -= 1
        assert len(lo) == len(hi) > 0
        distances = np.linalg.norm(lo[:, None] - hi[None, :], axis=2)
        assert np.max(distances.min(axis=1)) < 1e-8


def test_degenerate_four_grain_line_is_not_reported_as_junction_vertices():
    centers = [[.2, .2, .5], [.8, .2, .5], [.2, .8, .5], [.8, .8, .5]]
    apd = AnisotropicPowerDiagram(centers, np.tile(np.eye(3), (4, 1, 1)), [1, 1, 1])
    with pytest.warns(RuntimeWarning, match='Non-generic'):
        network = apd.extract_junctions(7)
    assert network.vertices.shape == (0, 3)
    assert network.segments.shape == (0, 2, 3)


def test_empty_network_and_plotting(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    apd = AnisotropicPowerDiagram([[.25, .5, .5], [.75, .5, .5]],
                                  np.tile(np.eye(3), (2, 1, 1)), [1, 1, 1])
    network = apd.extract_junctions(4)
    assert network.segments.shape == (0, 2, 3)
    assert network.vertices.shape == (0, 3)
    original_box = apd.box_size.copy()
    ax = apd.plot_junctions(network)
    np.testing.assert_array_equal(apd.box_size, original_box)
    ax.figure.savefig(tmp_path / 'empty.png')
    ax = apd.plot_slice(resolution=12)
    ax.figure.savefig(tmp_path / 'slice.png')
    plt.close('all')


def test_input_validation():
    with pytest.raises(ValueError, match='positive definite'):
        AnisotropicPowerDiagram([[0, 0, 0]], [-np.eye(3)], [1, 1, 1])
    with pytest.raises(ValueError, match='sum'):
        AnisotropicPowerDiagram([[0, 0, 0]], [np.eye(3)], [1, 1, 1], [.4])
    with pytest.raises(ValueError, match='original particles'):
        AnisotropicPowerDiagram.from_particles([], [1, 1, 1])
    apd = AnisotropicPowerDiagram([[0, 0, 0]], [np.eye(3)], [1, 1, 1])
    with pytest.raises(ValueError, match='power of two'):
        apd.fit_volumes(n_samples=100)
    with pytest.raises(ValueError, match='integer'):
        apd.extract_junctions(1)
