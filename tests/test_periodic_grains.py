"""Whole-grain lifts retain volumes, shared faces and lattice correspondence."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.apd_geometry import build_grain_geometry as _build_grain_geometry
from kanapy.core.power_diagram import AnisotropicPowerDiagram
from kanapy.core.periodic_grains import unwrap_periodic_grains
from kanapy.core.surface_validation import validate_surface
from kanapy.core.api import Microstructure


def build_grain_geometry(*args, **kwargs):
    """These tests retain the earlier fragment-unwrapping path explicitly."""
    return _build_grain_geometry(*args, periodic_images=False, **kwargs)


def periodic_geometry(offset=.15, resolution=3):
    centers=np.array(list(product([offset,offset+.5],repeat=3)))
    apd=AnisotropicPowerDiagram(centers,[np.eye(3)]*8,[1,1,1],
                               grain_ids=[7,10,15,22,41,53,68,90],periodic=True)
    return build_grain_geometry(apd,{int(g):i%2 for i,g in enumerate(apd.grain_ids)},resolution)


@pytest.mark.parametrize('offset', [.15,.25])
def test_whole_grains_seams_and_exact_volumes(offset):
    g=periodic_geometry(offset,4)
    original=g['Surface'].points.copy()
    w=unwrap_periodic_grains(g)
    assert not np.any(w.surface.boundary_ids)
    assert len(w.paired_faces)>0
    validate_surface(w.surface.points,w.surface.triangles,w.surface.face_grains)
    view=w.as_geometry()
    for grain in view['Grains'].values():
        assert grain['Volume'] == pytest.approx(.125)
        np.testing.assert_allclose(np.ptp(grain['Points'],axis=0),.5,atol=1e-9)
        np.testing.assert_allclose(grain['Covariance'],np.eye(3)/48,atol=1e-10)
    assert sum(view['PhaseVolumes'].values()) == pytest.approx(1)
    if offset==.15: assert w.surface.points.min() < 0
    np.testing.assert_array_equal(original,g['Surface'].points)
    assert_pairs(w)


def assert_pairs(w):
    for (a,b),shift in zip(w.paired_faces,w.pair_translations):
        ta,tb=w.surface.triangles[a],w.surface.triangles[b]
        opposite={int(w.vertex_classes[v]):w.surface.points[v] for v in tb}
        np.testing.assert_allclose(np.array([opposite[int(w.vertex_classes[v])] for v in ta]),
                                   w.surface.points[ta]+shift*w.box_size,atol=1e-9)
        assert np.dot(w.surface.area_vectors[a],w.surface.area_vectors[b])<0


def test_api_export(tmp_path, monkeypatch):
    ms=SimpleNamespace(geometry=periodic_geometry(.15,2),name='whole',
                       rve=SimpleNamespace(size=[1,1,1]))
    w=Microstructure.unwrap_grains(ms)
    assert ms.geometry['WholeGrains'] is w
    Microstructure.write_stl(ms,file='whole.stl',path=tmp_path)
    assert (tmp_path/'whole.stl').read_text().count('endfacet')==len(w.surface.triangles)
    plotted=[]
    monkeypatch.setattr('kanapy.core.api.plot_polygons_3D',lambda geometry, **kwargs: plotted.append(geometry))
    Microstructure.plot_grains(ms)
    assert plotted[-1]['Representation']=='PeriodicWholeGrains'
    Microstructure.plot_grains(ms,geometry=ms.geometry)
    assert plotted[-1] is ms.geometry


def test_default_periodic_reconstruction_and_orientation_export(tmp_path, monkeypatch):
    import json
    reference = periodic_geometry(.15, 2)
    apd = reference['APD']
    phases = {g: data['Phase'] for g, data in reference['Grains'].items()}
    orientations = {g: np.array([.1, .2, .3]) for g in phases}
    mesh = SimpleNamespace(apd=apd, grain_phase_dict=phases, grain_ori_dict=orientations)
    ms = SimpleNamespace(mesh=mesh, particles=[], geometry=None, nphases=2,
                         name='images', rve=SimpleNamespace(size=[1, 1, 1],
                         periodic=True, phase_names=['A', 'B']))
    Microstructure.generate_grains(ms, resolution=2)
    geometry = ms.geometry
    assert geometry['PeriodicImageGeometry']
    whole = geometry['WholeGrains']
    assert_pairs(whole)
    stats = validate_surface(whole.surface.points, whole.surface.triangles,
                             whole.surface.face_grains)
    assert stats['grain_volumes'] == pytest.approx({g: .125 for g in phases})
    assert geometry['PhaseVolumes'] == pytest.approx({0: .5, 1: .5})
    json.dumps(whole.report)
    for gid in phases:
        np.testing.assert_array_equal(whole.grain_orientations[gid], orientations[gid])
        assert whole.grain_orientations[gid] is not orientations[gid]
    Microstructure.write_stl(ms, file='images.stl', path=tmp_path)
    assert (tmp_path / 'images.stl').read_text().count('endfacet') == len(whole.surface.triangles)
    plotted = []
    monkeypatch.setattr('kanapy.core.api.plot_polygons_3D',
                        lambda geometry, **kwargs: plotted.append(geometry))
    Microstructure.plot_grains(ms)
    assert plotted[-1]['Representation'] == 'PeriodicWholeGrains'


@pytest.mark.parametrize('damage, message', [
    ('empty', 'nonempty'), ('nonfinite', 'finite'),
    ('degenerate', 'Degenerate'), ('duplicate', 'Duplicate'),
    ('open', 'closed'), ('flipped', 'oriented'), ('inverted', 'Nonpositive'),
])
def test_surface_validation_rejects_invalid_shells(damage, message):
    # A unit tetrahedron with outward-oriented faces provides an independent
    # fixture for validation, without relying on reconstruction to create it.
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if damage == 'empty':
        triangles = triangles[:0]
    elif damage == 'nonfinite':
        points[0, 0] = np.nan
    elif damage == 'degenerate':
        triangles[0] = [0, 0, 1]
    elif damage == 'duplicate':
        triangles = np.vstack([triangles, triangles[0]])
    elif damage == 'open':
        triangles = triangles[:-1]
    elif damage == 'flipped':
        triangles[0] = triangles[0, ::-1]
    else:
        triangles = triangles[:, ::-1]
    with pytest.raises(ValueError, match=message):
        validate_surface(points, triangles, ((7, None),) * len(triangles))


def test_winding_grains_and_nonperiodic_rejected():
    apd=AnisotropicPowerDiagram([[.5,.5,.5]],[np.eye(3)],[1,1,1],periodic=True)
    g=build_grain_geometry(apd,{1:0},2)
    with pytest.raises(ValueError,match='compact hull'): unwrap_periodic_grains(g,split_winding=False)
    apd=AnisotropicPowerDiagram([[.25,.5,.5],[.75,.5,.5]],[np.eye(3)]*2,[1,1,1],periodic=True)
    g=build_grain_geometry(apd,{1:0,2:0},4)
    with pytest.raises(ValueError,match='wind'): unwrap_periodic_grains(g,split_winding=False)
    g['APD'].periodic=False
    with pytest.raises(ValueError,match='periodic APD'):unwrap_periodic_grains(g)


def test_winding_split_keeps_parent_volume_phase_and_orientation():
    apd=AnisotropicPowerDiagram([[.25,.5,.5],[.75,.5,.5]],[np.eye(3)]*2,[1,1,1],periodic=True)
    g=build_grain_geometry(apd,{1:0,2:1},4)
    labels=g['Partition'].region_grain_ids.copy()
    orientations={1:np.array([.1,.2,.3]),2:np.array([.4,.5,.6])}
    ms=SimpleNamespace(geometry=g,mesh=SimpleNamespace(grain_ori_dict=orientations))
    w=Microstructure.unwrap_grains(ms)
    assert w.report['split_parent_grains']==[1,2]
    assert len(w.grain_parent_ids)>2
    assert w.report['parent_grain_volumes']==pytest.approx({1:.5,2:.5})
    assert w.report['split_face_ids']
    assert_pairs(w)
    for entity,parent in w.grain_parent_ids.items():
        assert w.phase_by_grain[entity]==g['Grains'][parent]['Phase']
        np.testing.assert_array_equal(w.grain_orientations[entity],orientations[parent])
        assert w.grain_orientations[entity] is not orientations[parent]
    np.testing.assert_array_equal(labels,g['Partition'].region_grain_ids)


def test_single_periodic_grain_split_and_resolution_gate():
    apd=AnisotropicPowerDiagram([[.5,.5,.5]],[np.eye(3)],[1,1,1],periodic=True)
    g=build_grain_geometry(apd,{1:0},4)
    w=unwrap_periodic_grains(g)
    assert len(w.grain_parent_ids)==8
    assert set(w.grain_parent_ids.values())=={1}
    assert w.report['parent_grain_volumes'][1]==pytest.approx(1)
    assert_pairs(w)
    coarse=build_grain_geometry(apd,{1:0},2)
    with pytest.raises(ValueError,match='resolution >= 3'):unwrap_periodic_grains(coarse)
