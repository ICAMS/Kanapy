"""Verification gate for the APD background tetrahedralization."""
import numpy as np
import pytest

from kanapy.core.power_diagram import AnisotropicPowerDiagram


def diagram(periodic=False):
    result = AnisotropicPowerDiagram([[.3, .4, .5], [1.7, 2., 3.]],
                                     [np.diag([2., 1., .5]), np.eye(3)],
                                     [2., 3., 4.], grain_ids=[7, 20], periodic=periodic)
    result.weights[:] = [.2, -.1]
    return result


@pytest.mark.parametrize('resolution,shape', [(1, (1, 1, 1)), (3, (3, 3, 3)),
                                             ((2, 3, 4), (2, 3, 4))])
def test_conformity_orientation_and_box_boundary(resolution, shape):
    m = diagram().background_mesh(resolution)
    assert len(m.points) == np.prod(np.array(shape) + 1)
    assert len(m.tetrahedra) == 6 * np.prod(shape)
    np.testing.assert_allclose(m.signed_volumes, 24 / len(m.tetrahedra))
    assert m.summary()['total_volume'] == pytest.approx(24)
    assert m.summary()['all_positive']
    # Independently enumerate face incidences and check opposite vertices lie
    # on opposite sides of every shared face (including inter-cube faces).
    incidence = {}
    for ti, tet in enumerate(m.tetrahedra):
        for omit in range(4):
            face = tuple(sorted(np.delete(tet, omit)))
            incidence.setdefault(face, []).append((ti, tet[omit]))
    assert len(incidence) == len(m.faces)
    for face, neighbors, boundary in zip(m.faces, m.face_tetrahedra, m.boundary_ids):
        entries = incidence[tuple(face)]
        assert sorted(neighbors[neighbors >= 0]) == sorted(t for t, _ in entries)
        a, b, c = m.points[face]
        normal = np.cross(b-a, c-a)
        if boundary == 0:
            assert len(entries) == 2
            assert np.prod([normal @ (m.points[v]-a) for _, v in entries]) < 0
        else:
            assert len(entries) == 1
            axis, side = divmod(int(boundary)-1, 2)
            np.testing.assert_array_equal(m.points[face, axis], side * m.box_size[axis])
    for axis in range(3):
        expected = 2 * np.prod([shape[j] for j in range(3) if j != axis])
        for side in range(2):
            assert np.count_nonzero(m.boundary_ids == 2*axis+side+1) == expected


@pytest.mark.parametrize('periodic', [False, True])
def test_costs_labels_batches_and_fixed_diagram(periodic, monkeypatch):
    apd = diagram(periodic)
    before = [v.copy() for v in (apd.centers, apd.matrices, apd.weights, apd.grain_ids)]
    original = apd.costs
    sizes = []
    def costs(points):
        sizes.append(len(points))
        return original(points)
    monkeypatch.setattr(apd, 'costs', costs)
    m = apd.background_mesh((2, 3, 4), batch_size=7)
    assert max(sizes) <= 7 and sum(sizes) == len(m.points)
    np.testing.assert_allclose(m.costs, original(m.points))
    np.testing.assert_array_equal(m.labels, np.array([7, 20])[original(m.points).argmin(axis=1)])
    for current, saved in zip((apd.centers, apd.matrices, apd.weights, apd.grain_ids), before):
        np.testing.assert_array_equal(current, saved)
    apd.weights[:] = 100
    np.testing.assert_allclose(m.costs, original(m.points) + 100 - before[2])


@pytest.mark.parametrize('resolution', [0, -1, True, 2.5, (2, 3), (1, 0, 2), (1., 2., 3.)])
def test_bad_resolution(resolution):
    with pytest.raises(ValueError, match='resolution'):
        diagram().background_mesh(resolution)


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_bad_batch(batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        diagram().background_mesh(batch_size=batch_size)
