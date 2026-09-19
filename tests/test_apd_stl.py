"""One triangulation per shared face and ASCII STL export."""
from types import SimpleNamespace
import numpy as np
import pytest
from kanapy.core.api import Microstructure
from kanapy.core.power_diagram import AnisotropicPowerDiagram


@pytest.fixture
def boundary():
    apd = AnisotropicPowerDiagram([[.25,.5,.5],[.75,.5,.5]],
                                  [np.eye(3),np.eye(3)], [1,1,1])
    return apd.background_mesh(2).assemble().boundary_complex()


def test_unique_faces_area_normals_and_closed_shells(boundary):
    surface = boundary.triangulate(include_exterior=True)
    assert len({tuple(sorted(t)) for t in surface.triangles}) == len(surface.triangles)
    assert np.all(surface.areas > 0)
    for fi, face in enumerate(boundary.faces):
        xyz = boundary.points[face]
        vector = np.sum(np.cross(xyz-xyz.mean(axis=0), np.roll(xyz,-1,axis=0)-xyz.mean(axis=0)), axis=0)/2
        mask = surface.source_faces == fi
        np.testing.assert_allclose(surface.area_vectors[mask].sum(axis=0), vector, atol=1e-14)
        assert surface.areas[mask].sum() == pytest.approx(np.linalg.norm(vector))
    for grain in (1,2):
        edges = {}
        for tri, pair in zip(surface.triangles, surface.face_grains):
            if grain not in pair:
                continue
            loop = tri if pair[0] == grain else tri[::-1]
            for a,b in zip(loop,np.roll(loop,-1)):
                edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
        assert all(len(e)==2 and e[0]==e[1][::-1] for e in edges.values())
    internal = boundary.triangulate()
    assert internal.areas.sum() == pytest.approx(1.)  # not two copies
    np.testing.assert_allclose(internal.normals, np.tile([1,0,0],(len(internal.triangles),1)))


@pytest.mark.parametrize('exterior', [False,True])
@pytest.mark.parametrize('pretriangulated', [False,True])
def test_api_stl_roundtrip(boundary,tmp_path,exterior,pretriangulated):
    ms = SimpleNamespace(name='test_apd')
    source = boundary.triangulate(include_exterior=True) if pretriangulated else boundary
    Microstructure.write_stl(ms, path=tmp_path, boundary=source, include_exterior=exterior)
    lines = (tmp_path/'test_apd.stl').read_text().splitlines()
    vertices = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.strip().startswith('vertex')]).reshape(-1,3,3)
    normals = np.array([[float(x) for x in l.split()[2:]] for l in lines if l.strip().startswith('facet normal')])
    expected = boundary.triangulate(include_exterior=exterior)
    np.testing.assert_allclose(vertices, expected.points[expected.triangles])
    np.testing.assert_allclose(normals, expected.normals)
    keys = [tuple(sorted(map(tuple,t))) for t in vertices]
    assert len(set(keys)) == len(keys)
    assert expected.areas.sum() == pytest.approx(7 if exterior else 1)


def test_invalid_input_does_not_create_file(tmp_path):
    with pytest.raises(ValueError, match='generate_grains'):
        Microstructure.write_stl(SimpleNamespace(name='bad'), path=tmp_path, boundary=None)
    assert not (tmp_path/'bad.stl').exists()


def test_duplicate_rejected(boundary,tmp_path):
    surface = boundary.triangulate()
    surface.triangles = np.vstack([surface.triangles,surface.triangles[0]])
    with pytest.raises(ValueError,match='Duplicate'):
        surface.write_stl(tmp_path/'bad.stl')
    assert not (tmp_path/'bad.stl').exists()
