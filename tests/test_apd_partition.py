"""Local APD clipping: analytical partitions and independent coverage checks."""
import numpy as np
import pytest

from kanapy.core.apd_mesh import partition_tetrahedron
from kanapy.core.power_diagram import AnisotropicPowerDiagram

POINTS = np.vstack([np.zeros(3), np.eye(3)])


def verify(parts, costs, points=POINTS):
    expected_volume = abs(np.linalg.det(points[1:] - points[0])) / 6
    assert sum(p.volume for p in parts) == pytest.approx(expected_volume, rel=1e-9)
    rng = np.random.default_rng(7)
    samples = rng.dirichlet(np.ones(4), size=2000)
    owners = np.argmin(samples @ costs, axis=1) + 1
    coverage = np.zeros(len(samples), dtype=int)
    for p in parts:
        assert p.volume > 0
        np.testing.assert_allclose(p.barycentric @ points, p.vertices, atol=1e-12)
        assert np.min(p.barycentric) >= -1e-9
        a, b = p.constraints[:, :3], p.constraints[:, 3]
        assert np.max(p.barycentric[:, 1:] @ a.T + b) <= 1e-9
        mask = np.all(samples[:, 1:] @ a.T + b <= 0, axis=1)
        coverage += mask
        assert np.all(owners[mask] == p.grain_id)
        for vertex, active in zip(p.barycentric[:, 1:], p.vertex_constraints):
            assert len(active) >= 3
            np.testing.assert_allclose(a[list(active)] @ vertex + b[list(active)], 0, atol=1e-9)
        # Closed outward shell: every edge twice, oriented oppositely; surface
        # integration supplies a volume check independent of ConvexHull.volume.
        edges = {}
        volume = 0.
        center = p.vertices.mean(axis=0)
        for face, constraints in zip(p.faces, p.face_constraints):
            assert len(face) >= 3 and constraints
            xyz = p.vertices[face]
            normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
            assert normal @ (xyz.mean(axis=0)-center) > 0
            for start, end in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((start, end))), []).append((start, end))
            for k in range(1, len(xyz)-1):
                volume += np.linalg.det(np.array([xyz[0]-center, xyz[k]-center, xyz[k+1]-center])) / 6
            for constraint in constraints:
                np.testing.assert_allclose(p.barycentric[face, 1:] @ a[constraint] + b[constraint], 0, atol=1e-9)
        assert all(len(e) == 2 and e[0] == e[1][::-1] for e in edges.values())
        assert volume == pytest.approx(p.volume, rel=1e-9)
    assert np.all(coverage == 1)


def test_one_grain():
    costs = np.zeros((4, 1))
    parts = partition_tetrahedron(POINTS, costs)
    verify(parts, costs)
    assert len(parts) == 1 and len(parts[0].faces) == 4


def test_planar_split():
    costs = np.column_stack([POINTS[:, 0] - .5, np.zeros(4)])
    parts = partition_tetrahedron(POINTS, costs)
    verify(parts, costs)
    np.testing.assert_allclose([p.volume for p in parts], [7/48, 1/48])
    for p in parts:
        interfaces = [face for face, active in zip(p.faces, p.face_constraints)
                      if any(p.constraint_sources[c][0] == 'grain' for c in active)]
        assert len(interfaces) == 1
        np.testing.assert_allclose(p.vertices[interfaces[0], 0], .5)


@pytest.mark.parametrize('n', [3, 4])
def test_symmetric_junctions(n):
    costs = -np.eye(4)[:, :n]
    parts = partition_tetrahedron(POINTS, costs)
    verify(parts, costs)
    np.testing.assert_allclose([p.volume for p in parts], np.full(n, 1/(6*n)))
    common = [v for v in parts[0].vertices
              if all(np.any(np.linalg.norm(p.vertices-v, axis=1) < 1e-9) for p in parts[1:])]
    assert len(common) == (2 if n == 3 else 1)
    if n == 4:
        np.testing.assert_allclose(common[0], [.25, .25, .25])


def test_interior_winner_without_winning_corners():
    costs = np.column_stack([-np.eye(4), np.full(4, -.3)])
    assert 4 not in costs.argmin(axis=1)
    parts = partition_tetrahedron(POINTS, costs)
    verify(parts, costs)
    assert len(parts) == 5
    assert parts[-1].grain_id == 5
    assert parts[-1].volume > 0
    assert np.all(parts[-1].barycentric > 0)


def test_identical_costs_dominance_and_zero_volume_contacts():
    costs = np.column_stack([np.zeros(4), np.zeros(4), np.ones(4), POINTS[:, 0]])
    parts = partition_tetrahedron(POINTS, costs)
    verify(parts, costs)
    assert [p.grain_id for p in parts] == [1]
    # x=0 is simultaneously a box constraint and a redundant grain constraint.
    assert any(len(c) == 2 for c in parts[0].face_constraints)


def test_affine_coordinate_and_cost_scaling():
    costs = -np.eye(4)
    points = POINTS @ np.array([[2., .4, 0], [0, 3., .2], [.3, 0, -4.]]) + [8, -4, 20]
    parts = partition_tetrahedron(points, costs * 1e-8 + 12)
    verify(parts, costs, points)


def test_random_partitions():
    rng = np.random.default_rng(42)
    for _ in range(8):
        costs = rng.normal(size=(4, 6))
        verify(partition_tetrahedron(POINTS, costs), costs)


def test_background_integration():
    apd = AnisotropicPowerDiagram([[.2, .3, .4], [.7, .8, .9]],
                                 [np.eye(3), np.eye(3)], [1, 2, 3], grain_ids=[7, 20])
    m = apd.background_mesh(1)
    before = m.costs.copy()
    for i in range(6):
        parts = m.partition_tetrahedron(i)
        assert sum(p.volume for p in parts) == pytest.approx(m.signed_volumes[i])
        assert all(p.grain_id in (7, 20) for p in parts)
    np.testing.assert_array_equal(m.costs, before)
    with pytest.raises(ValueError):
        m.partition_tetrahedron(-1)


@pytest.mark.parametrize('points,costs,kwargs', [
    (np.zeros((4, 3)), np.zeros((4, 1)), {}),
    (POINTS, np.zeros((3, 2)), {}),
    (POINTS, np.full((4, 1), np.nan), {}),
    (POINTS, np.zeros((4, 2)), {'grain_ids': [1, 1]}),
    (POINTS, np.zeros((4, 1)), {'tolerance': 0}),
])
def test_invalid_input(points, costs, kwargs):
    with pytest.raises(ValueError):
        partition_tetrahedron(points, costs, **kwargs)


def test_pruning_matches_reference_and_keeps_interior_winner():
    costs = np.column_stack([-np.eye(4), np.full(4, -.3), np.full(4, 5.)])
    stats = {}
    fast = partition_tetrahedron(POINTS, costs, diagnostics=stats)
    reference = partition_tetrahedron(POINTS, costs, optimize=False)
    assert stats['candidates'] == 5
    assert not stats['shortcut']
    assert [p.grain_id for p in fast] == [p.grain_id for p in reference]
    for a, b in zip(fast, reference):
        assert a.volume == pytest.approx(b.volume)
        np.testing.assert_allclose(sorted(map(tuple, a.vertices)), sorted(map(tuple, b.vertices)))


def test_uncut_shortcut_and_boundary_tie_metadata():
    stats = {}
    costs = np.column_stack([np.zeros(4), np.ones(4), np.arange(4)+2])
    parts = partition_tetrahedron(POINTS, costs, diagnostics=stats)
    assert stats['shortcut'] and stats['candidates'] == 1
    verify(parts, costs)
    # A competitor touching a face must survive pruning.
    touching = np.column_stack([np.zeros(4), POINTS[:, 0]])
    parts = partition_tetrahedron(POINTS, touching, diagnostics=stats)
    assert not stats['shortcut'] and stats['candidates'] == 2
    assert any(len(c) == 2 for c in parts[0].face_constraints)
    verify(parts, touching)
