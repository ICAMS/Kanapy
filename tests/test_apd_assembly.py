"""Global sharing, incidence and convergence of interpolated APD partitions."""
import numpy as np
import pytest
from kanapy.core.power_diagram import AnisotropicPowerDiagram


def make(centers, matrices=None, box=(1, 1, 1)):
    return AnisotropicPowerDiagram(centers, np.tile(np.eye(3), (len(centers), 1, 1))
                                  if matrices is None else matrices, box)


def verify(background, mesh):
    assert mesh.region_volumes.sum() == pytest.approx(np.prod(background.box_size))
    np.testing.assert_allclose(np.bincount(mesh.region_tetrahedra, weights=mesh.region_volumes),
                               background.signed_volumes)
    assert len(mesh.vertex_keys) == len(set(mesh.vertex_keys))
    assert len(mesh.faces) == len({tuple(sorted(f)) for f in mesh.faces})
    for fi, (a, b) in enumerate(mesh.face_regions):
        assert fi in mesh.region_faces[a]
        if b >= 0:
            assert fi in mesh.region_faces[b]
            assert mesh.boundary_ids[fi] == 0
            assert mesh.region_face_signs[a][list(mesh.region_faces[a]).index(fi)] == 1
            assert mesh.region_face_signs[b][list(mesh.region_faces[b]).index(fi)] == -1
        else:
            assert mesh.boundary_ids[fi] > 0


@pytest.mark.parametrize('centers', [[[.5, .5, .5]],
                                    [[.25, .5, .5], [.75, .5, .5]],
                                    [[.2,.2,.2],[.8,.2,.2],[.5,.8,.2],[.5,.5,.8]]])
def test_shared_partition_and_order(centers):
    b = make(centers).background_mesh(2)
    m = b.assemble()
    verify(b, m)
    other = b.assemble(tetrahedron_order=np.arange(len(b.tetrahedra))[::-1])
    assert m.vertex_keys == other.vertex_keys
    np.testing.assert_array_equal(m.points, other.points)
    np.testing.assert_array_equal(m.face_regions, other.face_regions)
    for f, g in zip(m.faces, other.faces):
        np.testing.assert_array_equal(f, g)
    if len(centers) == 1:
        assert len(m.points) == len(b.points)
        assert len(m.interface_faces) == 0
    elif len(centers) == 2:
        assert m.grain_volumes == pytest.approx({1: .5, 2: .5})
        for fi in m.interface_faces:
            np.testing.assert_allclose(m.points[m.faces[fi], 0], .5)


def test_non_cubic_and_identical_grains():
    b = make([[.5,.5,.5], [.5,.5,.5]], box=(2,3,4)).background_mesh((1,2,3))
    m = b.assemble()
    verify(b, m)
    assert m.grain_volumes == pytest.approx({1: 24})


def test_curved_boundary_volume_convergence():
    # q1-q2 = x^2-.2: exact volume of grain 1 is sqrt(.2).
    apd = make([[0,0,0], [0,0,0]], [np.diag([2.,1,1]), np.eye(3)])
    apd.weights[:] = [.2, 0]
    errors = []
    for n in (2,4,8):
        b = apd.background_mesh((n,1,1))
        m = b.assemble()
        verify(b, m)
        errors.append(abs(m.grain_volumes[1]-np.sqrt(.2)))
    assert errors[2] < errors[1] < errors[0]


def test_invalid_order_and_tolerance():
    b = make([[.2,.5,.5],[.8,.5,.5]]).background_mesh(1)
    with pytest.raises(ValueError, match='permutation'):
        b.assemble(tetrahedron_order=[0]*6)
    with pytest.raises(ValueError, match='tolerance'):
        b.assemble(tolerance=0)


def test_random_point_ownership_after_assembly():
    apd = make([[.17,.24,.31], [.79,.38,.63], [.43,.81,.52]])
    apd.weights[:] = [.03, -.02, 0]
    b = apd.background_mesh(2)
    m = b.assemble()
    verify(b, m)
    rng = np.random.default_rng(15)
    for ti, corners in enumerate(b.tetrahedra):
        bary = rng.dirichlet(np.ones(4), 20)
        samples = bary @ b.points[corners]
        expected = b.grain_ids[(bary @ b.costs[corners]).argmin(axis=1)]
        coverage = np.zeros(20, dtype=int)
        for ri in np.flatnonzero(m.region_tetrahedra == ti):
            inside = np.ones(20, dtype=bool)
            for fi, sign in zip(m.region_faces[ri], m.region_face_signs[ri]):
                xyz = m.points[m.faces[fi]]
                center = xyz.mean(axis=0)
                normal = sign*np.sum(np.cross(xyz-center, np.roll(xyz,-1,axis=0)-center), axis=0)
                inside &= (samples-center) @ normal < 1e-12
            coverage += inside
            assert np.all(expected[inside] == m.region_grain_ids[ri])
        assert np.all(coverage == 1)


def test_rejects_missing_internal_face(monkeypatch):
    b = make([[.5,.5,.5]]).background_mesh(1)
    partition = b.partition_tetrahedron
    def damaged(index, **kwargs):
        parts = partition(index, **kwargs)
        if index == 0:
            part = parts[0]
            # Remove a face not on a box plane, keeping its volume unchanged.
            for j, face in enumerate(part.faces):
                xyz = part.vertices[face]
                if not any(np.all(xyz[:, axis] == value)
                           for axis in range(3) for value in (0, 1)):
                    part.faces = part.faces[:j] + part.faces[j+1:]
                    break
        return parts
    monkeypatch.setattr(b, 'partition_tetrahedron', damaged)
    with pytest.raises(ValueError, match='Unmatched internal face'):
        b.assemble()


def test_optimized_reference_equivalence_and_timings():
    b = make([[.1,.2,.3], [.8,.7,.6], [4,4,4], [6,5,4]]).background_mesh(2)
    fast = b.assemble()
    reference = b.assemble(optimize=False)
    assert fast.vertex_keys == reference.vertex_keys
    np.testing.assert_allclose(fast.points, reference.points, atol=1e-12)
    np.testing.assert_array_equal(fast.face_regions, reference.face_regions)
    np.testing.assert_array_equal(fast.region_grain_ids, reference.region_grain_ids)
    np.testing.assert_allclose(fast.region_volumes, reference.region_volumes)
    for a, bface in zip(fast.faces, reference.faces):
        np.testing.assert_array_equal(a, bface)
    stats = fast.candidate_statistics
    assert stats['uncut_shortcuts'] > 0
    assert stats['max_candidates'] <= 2
    assert sum(stats['histogram'].values()) == len(b.tetrahedra)
    assert reference.candidate_statistics['histogram'] == {4: len(b.tetrahedra)}
    assert fast.timings['total'] > 0
    assert all(t >= 0 for t in fast.timings.values())
    assert fast.timings['local_partition'] >= fast.timings['pruning']
    stages = ['setup', 'local_partition', 'topology_collection', 'vertex_welding',
              'face_assembly', 'validation']
    assert sum(fast.timings[k] for k in stages) <= fast.timings['total']
