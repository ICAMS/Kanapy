"""Center-cost APD assignment and compatibility with Kanapy mesh consumers."""
import numpy as np
import pytest

from kanapy.core.entities import Ellipsoid, Simulation_Box
from kanapy.core.initializations import mesh_creator
from kanapy.core.voxelization import voxelizationRoutine


def particle(gid, center, axes=(.2, .2, .2), phase=0):
    return Ellipsoid(gid, *center, *axes, np.array([1., 0, 0, 0]), phasenum=phase)


def mesh(dim=(4, 2, 2)):
    result = mesh_creator(dim)
    result.create_voxels(Simulation_Box((1, 1, 1)))
    return result


def test_center_assignment_and_phase_bookkeeping():
    particles = [particle(7, (.25, .5, .5)), particle(20, (.75, .5, .5), phase=2)]
    m = mesh()
    # Insertion order must not change the structured voxel order.
    m.vox_center_dict = dict(reversed(list(m.vox_center_dict.items())))
    assert voxelizationRoutine(particles, m, 3, fit_volumes=False, chunk_size=3) is m
    np.testing.assert_array_equal(m.grains[:, 0, 0], [7, 7, 20, 20])
    np.testing.assert_array_equal(m.phases[:, 0, 0], [0, 0, 2, 2])
    np.testing.assert_array_equal(m.ngrains_phase, [1, 0, 1])
    assert m.grain_phase_dict == {7: 0, 20: 2}
    assert sorted(sum(m.grain_dict.values(), [])) == list(range(1, 17))
    assert particles[0].inside_voxels == m.grain_dict[7]


def test_rotated_anisotropic_weighted_costs_at_centers():
    particles = [particle(7, (.2, .3, .5), (.4, .1, .2)),
                 particle(20, (.8, .6, .5), (.15, .3, .2))]
    theta = .6
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    particles[0].rotation_matrix = rotation
    m = mesh((7, 5, 3))
    weights = [.03, -.02]
    voxelizationRoutine(particles, m, 1, weights=weights)
    points = np.array(list(m.vox_center_dict.values()))
    costs = []
    for p, w in zip(particles, weights):
        axes = np.array([p.a, p.b, p.c])
        delta = (points - p.get_pos()) @ p.rotation_matrix.T
        costs.append(np.sum(delta**2 * np.prod(axes)**(2/3) / axes**2, axis=1) - w)
    expected = np.array([7, 20])[np.argmin(costs, axis=0)]
    np.testing.assert_array_equal(m.grains.ravel(), expected)
    np.testing.assert_allclose(particles[0].get_pos(), [.2, .3, .5])


def test_periodic_duplicates_and_override():
    p = particle(7, (.05, .5, .5))
    q = particle(20, (.55, .5, .5))
    duplicate = particle(99, (1.05, .5, .5)); duplicate.duplicate = 7
    m = mesh((10, 1, 1))
    voxelizationRoutine([p, q, duplicate], m, 1, fit_volumes=False)
    assert m.grains[-1, 0, 0] == 7
    assert set(m.grain_dict) == {7, 20}
    assert duplicate.inside_voxels == []
    voxelizationRoutine([p, q, duplicate], m, 1, periodic=False, fit_volumes=False)
    assert m.grains[-1, 0, 0] == 20


def test_volume_fitting():
    particles = [particle(1, (.25, .5, .5), (.1, .1, .1)),
                 particle(2, (.75, .5, .5), (.1 * 3**(1/3),) * 3)]
    m = mesh((40, 2, 2))
    voxelizationRoutine(particles, m, 1, fit_options={'n_samples': 4096})
    assert len(m.grain_dict[1]) / m.nvox == pytest.approx(.25, abs=.025)


def test_deprecated_fraction_and_exact_ties():
    particles = [particle(7, (.5, .5, .5)), particle(20, (.5, .5, .5), phase=1)]
    m = mesh()
    with pytest.warns(DeprecationWarning, match='prec_vf'):
        voxelizationRoutine(particles, m, 2, .4, fit_volumes=False)
    assert np.all(m.grains == 7)
    assert m.prec_vf_voxels == 1
    np.testing.assert_array_equal(m.ngrains_phase, [1, 0])


@pytest.mark.parametrize('kwargs', [{'chunk_size': 0}, {'weights': [0]},
                                    {'weights': [0, np.nan]}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        voxelizationRoutine([particle(1, (.2, .5, .5)), particle(2, (.8, .5, .5))],
                            mesh(), 1, **kwargs)


def test_invalid_particles():
    with pytest.raises(ValueError, match='original particles'):
        voxelizationRoutine([], mesh(), 1)
    with pytest.raises(ValueError, match='phase numbers'):
        voxelizationRoutine([particle(1, (.5, .5, .5), phase=2)], mesh(), 1)


def test_translated_mesh_uses_same_partition():
    shift = np.array([4., -2., 7.])
    particles = [particle(1, shift + [.25, .5, .5]),
                 particle(2, shift + [.75, .5, .5])]
    m = mesh()
    m.nodes = m.nodes + shift
    m.vox_center_dict = {i: np.asarray(x) + shift for i, x in m.vox_center_dict.items()}
    voxelizationRoutine(particles, m, 1, fit_volumes=False)
    np.testing.assert_array_equal(m.grains[:, 0, 0], [1, 1, 2, 2])
    np.testing.assert_allclose(particles[0].get_pos(), shift + [.25, .5, .5])
