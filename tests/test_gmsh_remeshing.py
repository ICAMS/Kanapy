"""Integration checks against the optional Gmsh backend."""
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.api import Microstructure
from kanapy.core.apd_geometry import build_grain_geometry
from kanapy.core.gmsh_remeshing import remesh_grain_surface
from kanapy.core.power_diagram import AnisotropicPowerDiagram


def geometry(centers=((.25, .5, .5), (.75, .5, .5)), resolution=2):
    diagram = AnisotropicPowerDiagram(centers, [np.eye(3)] * len(centers),
                                     [1, 1, 1], grain_ids=[7, 20, 31][:len(centers)])
    return build_grain_geometry(diagram, dict.fromkeys(diagram.grain_ids, 0), resolution)


@pytest.fixture
def gmsh():
    return pytest.importorskip('gmsh')


def test_shared_planar_interface_and_size_control(gmsh):
    g = geometry()
    original = g['Surface'].points.copy()
    coarse = remesh_grain_surface(g, .3)
    fine = remesh_grain_surface(g, .15)
    assert len(fine.surface.triangles) > len(coarse.surface.triangles)
    assert fine.report['grain_volumes_after'] == pytest.approx({7: .5, 20: .5})
    ids = [i for i, pair in enumerate(fine.surface.face_grains) if pair == (7, 20)]
    assert fine.surface.areas[ids].sum() == pytest.approx(1.)
    np.testing.assert_allclose(fine.surface.normals[ids], np.tile([1, 0, 0], (len(ids), 1)), atol=1e-10)
    np.testing.assert_array_equal(g['Surface'].points, original)
    assert set(fine.surface.boundary_ids) == set(range(7))
    assert not gmsh.isInitialized()
    # Full remeshing can cross old polygon edges, unlike subdivision/refinement.
    crosses = []
    for tri, fi in zip(coarse.surface.triangles, coarse.surface.source_faces):
        polygon = g['Boundary'].points[g['Boundary'].faces[fi]]
        normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            signed = np.cross(b-a, coarse.surface.points[tri]-a) @ normal
            crosses.append(np.min(signed) < -1e-10)
    assert any(crosses)


def test_triple_junction_is_conforming(gmsh):
    result = remesh_grain_surface(geometry(((.2, .2, .5), (.8, .2, .5), (.5, .8, .5))), .2)
    surf = result.surface
    vertices = {}
    for pair in ((7, 20), (7, 31), (20, 31)):
        vertices[pair] = set(surf.triangles[[i for i, p in enumerate(surf.face_grains) if p == pair]].ravel())
    shared = set.intersection(*vertices.values())
    assert len(shared) >= 2
    np.testing.assert_allclose(surf.points[sorted(shared), :2], np.tile([.5, .425], (len(shared), 1)), atol=1e-8)


def test_periodic_box_nodes(gmsh):
    g = geometry(((.5, .5, .5),))
    result = remesh_grain_surface(g, .2, periodic=True)
    assert result.periodic_nodes
    for slaves, masters, shift in result.periodic_nodes:
        np.testing.assert_allclose(result.surface.points[slaves] - result.surface.points[masters],
                                   np.tile(shift, (len(slaves), 1)), atol=1e-8)
    assert result.report['grain_volumes_after'] == pytest.approx({7: 1.})


def test_periodic_apd_auto_detection(gmsh):
    diagram = AnisotropicPowerDiagram([[.2, .5, .5], [.7, .5, .5]],
                                     [np.eye(3)] * 2, [1, 1, 1], periodic=True)
    g = build_grain_geometry(diagram, {1: 0, 2: 0}, 3, periodic_images=False)
    result = remesh_grain_surface(g, .2)
    assert result.report['periodic']
    assert result.periodic_nodes
    assert result.report['grain_volumes_after'] == pytest.approx(result.report['grain_volumes_before'])
    # Every triangle on a slave box plane has a translated master triangle.
    surface = result.surface
    for axis in range(3):
        mapping = {}
        for slaves, masters, shift in result.periodic_nodes:
            if shift[axis] > 0:
                mapping.update(zip(slaves, masters))
        masters = {tuple(sorted(t)) for t in surface.triangles[surface.boundary_ids == 2*axis+1]}
        slaves = surface.triangles[surface.boundary_ids == 2*axis+2]
        assert {tuple(sorted(mapping[n] for n in t)) for t in slaves} == masters


def test_existing_session_and_options_survive(gmsh):
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add('user_model')
        gmsh.model.geo.addPoint(2, 3, 4)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMax', 123)
        models_before = gmsh.model.list()
        remesh_grain_surface(geometry(), .25)
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == 'user_model'
        assert gmsh.model.list() == models_before
        assert len(gmsh.model.getEntities(0)) == 1
        assert gmsh.option.getNumber('Mesh.MeshSizeMax') == 123
    finally:
        gmsh.finalize()


def test_api_atomic_storage_and_export(gmsh, tmp_path):
    ms = SimpleNamespace(geometry=geometry(), name='remeshed')
    reference = ms.geometry['Surface']
    result = Microstructure.remesh_grains(ms, .25)
    assert ms.geometry['Remeshed'] is result
    assert ms.geometry['Surface'] is reference
    Microstructure.write_stl(ms, boundary=result.surface, include_exterior=True, path=tmp_path)
    assert (tmp_path / 'remeshed.stl').exists()
    with pytest.raises(ValueError, match='positive'):
        Microstructure.remesh_grains(ms, -1)
    assert ms.geometry['Remeshed'] is result


@pytest.mark.parametrize('size', [0, -1, float('nan'), float('inf')])
def test_invalid_size(size):
    with pytest.raises(ValueError, match='mesh_size'):
        remesh_grain_surface(None, size)


def test_missing_boundary():
    with pytest.raises(ValueError, match='APDBoundaryComplex'):
        remesh_grain_surface({}, .1)
    with pytest.raises(ValueError, match='generate_grains'):
        Microstructure.remesh_grains(SimpleNamespace(geometry=None), .1)


def test_closed_curved_interface_and_polygon_preserving_mode(gmsh):
    diagram = AnisotropicPowerDiagram([[.5, .5, .5]] * 2,
                                     [2*np.eye(3), np.eye(3)], [1, 1, 1])
    diagram.weights[:] = [.12, 0]
    g = build_grain_geometry(diagram, {1: 0, 2: 1}, 4)
    result = remesh_grain_surface(g, .2)
    assert set(result.report['grain_volumes_after']) == {1, 2}
    assert max(abs(v) for v in result.report['relative_volume_changes'].values()) < .05
    faceted = remesh_grain_surface(g, .2, compound=False)
    assert faceted.report['grain_volumes_after'] == pytest.approx(faceted.report['grain_volumes_before'])


def test_disconnected_grain(gmsh):
    diagram = AnisotropicPowerDiagram([[.5, .5, .5]] * 2,
                                     [np.eye(3), np.diag([2., 1, 1])], [1, 1, 1])
    diagram.weights[:] = [0, .0625]
    g = build_grain_geometry(diagram, {1: 0, 2: 0}, (4, 1, 1))
    result = remesh_grain_surface(g, .2)
    assert result.report['grain_volumes_after'] == pytest.approx({1: .5, 2: .5})


def test_cleanup_after_mesher_failure(gmsh, monkeypatch):
    gmsh.initialize([], readConfigFiles=False)
    try:
        previous = gmsh.model.getCurrent()
        models = gmsh.model.list()
        old_size = gmsh.option.getNumber('Mesh.MeshSizeMax')
        def fail(*args):
            raise RuntimeError('mesher failed')
        monkeypatch.setattr(gmsh.model.mesh, 'generate', fail)
        ms = SimpleNamespace(geometry=geometry())
        ms.geometry['Remeshed'] = sentinel = object()
        with pytest.raises(RuntimeError, match='mesher failed'):
            Microstructure.remesh_grains(ms, .2)
        assert ms.geometry['Remeshed'] is sentinel
        assert gmsh.model.getCurrent() == previous
        assert gmsh.model.list() == models
        assert gmsh.option.getNumber('Mesh.MeshSizeMax') == old_size
    finally:
        gmsh.finalize()


def test_missing_optional_dependency(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'gmsh', None)
    with pytest.raises(ImportError, match=r'kanapy\[gmsh\]'):
        remesh_grain_surface(geometry(), .2)
