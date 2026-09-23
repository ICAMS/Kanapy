import numpy as np
import pytest

from kanapy.core.entities import Ellipsoid, Simulation_Box
from kanapy.core.collisions import collide_detect
from kanapy.core._packing_relaxation import relax_particles


def sphere(gid, center, radius=1.):
    return Ellipsoid(gid, *center, radius, radius, radius, np.array([1., 0., 0., 0.]))


@pytest.mark.parametrize('periodic, centers', [
    (False, [[4., 5., 5.], [5., 5., 5.]]),
    (True, [[.3, 5., 5.], [9.7, 5., 5.]]),
    (True, [[.3, .3, .3], [9.7, 9.7, 9.7]]),
    (False, [[5., 5., 5.], [5., 5., 5.]]),
])
def test_relaxation_separates_without_growing(periodic, centers):
    particles = [sphere(i+1, c) for i, c in enumerate(centers)]
    for p in particles:
        p.speedx = 2.
        p.force_y = 3.
    box = Simulation_Box((10., 10., 10.))
    result, report = relax_particles(particles, box, periodic)
    assert report['converged']
    assert report['initial_contacts'] > 0
    assert report['remaining_contacts'] == 0
    distance = particles[1].get_pos()-particles[0].get_pos()
    if periodic:
        distance -= np.rint(distance/10)*10
        for image in (p for p in result if p.duplicate is not None):
            parent = particles[image.duplicate-1]
            shift = (image.get_pos()-parent.get_pos())/10
            np.testing.assert_allclose(shift, np.rint(shift), atol=1e-12)
            np.testing.assert_array_equal(image.get_coeffs(), parent.get_coeffs())
    assert np.linalg.norm(distance) > 2.
    for p in particles:
        np.testing.assert_array_equal([p.a, p.b, p.c], [1., 1., 1.])
        np.testing.assert_array_equal([p.xold, p.yold, p.zold], p.get_pos())
        assert p.speedx == p.speedy == p.speedz == 0.
        assert p.force_x == p.force_y == p.force_z == 0.


def test_relaxation_of_rotated_ellipsoid():
    angle = np.pi/4
    p = Ellipsoid(1, 5., 5., 5., 3., 1., 1.,
                  np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)]))
    q = sphere(2, p.get_pos() + np.array([2.5, 0., 0.]) @ p.rotation_matrix, .2)
    before = p.rotation_matrix.copy()
    _, report = relax_particles([p, q], Simulation_Box((12., 12., 12.)), True)
    assert report['converged']
    assert not collide_detect(p.get_coeffs(), q.get_coeffs(), p.get_pos(), q.get_pos(),
                              p.rotation_matrix, q.rotation_matrix)
    np.testing.assert_array_equal(p.rotation_matrix, before)
    np.testing.assert_array_equal(p.get_coeffs(), [3., 1., 1.])


def test_relaxation_preserves_already_separated_positions():
    particles = [sphere(1, [2., 2., 2.]), sphere(2, [7., 7., 7.])]
    before = np.array([p.get_pos() for p in particles])
    _, report = relax_particles(particles, Simulation_Box((10., 10., 10.)), True)
    assert report['converged'] and report['steps'] == 0
    np.testing.assert_array_equal([p.get_pos() for p in particles], before)


def test_relaxation_reports_limit_and_can_be_disabled():
    particles = [sphere(1, [4., 5., 5.]), sphere(2, [5., 5., 5.])]
    box = Simulation_Box((10., 10., 10.))
    before = np.array([p.get_pos() for p in particles])
    result, report = relax_particles(particles, box, True, max_steps=0)
    assert result is particles and report['converged'] is None
    np.testing.assert_array_equal([p.get_pos() for p in particles], before)
    with pytest.warns(RuntimeWarning, match='step_limit'):
        _, report = relax_particles(particles, box, True, max_steps=1)
    assert not report['converged'] and report['remaining_contacts'] > 0


def test_periodic_self_overlap_is_reported_not_shrunk():
    particle = sphere(1, [5., 5., 5.], 5.1)
    with pytest.warns(RuntimeWarning, match='stalled'):
        _, report = relax_particles([particle], Simulation_Box((10., 10., 10.)), True)
    assert not report['converged'] and report['remaining_contacts'] == 3
    assert particle.a == 5.1


def test_nonperiodic_wall_constraint():
    particles = [sphere(1, [.5, 5., 5.]), sphere(2, [1., 5., 5.])]
    _, report = relax_particles(particles, Simulation_Box((10., 10., 10.)), False)
    assert report['converged']
    assert all(1. <= p.x <= 9. for p in particles)
    assert np.linalg.norm(particles[1].get_pos()-particles[0].get_pos()) > 2.


@pytest.mark.parametrize('steps', [-1, True, 1.5])
def test_invalid_relaxation_steps(steps):
    with pytest.raises(ValueError, match='nonnegative integer'):
        relax_particles([], Simulation_Box((10., 10., 10.)), True, max_steps=steps)
