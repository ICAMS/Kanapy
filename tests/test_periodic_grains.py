"""Whole-grain lifts retain volumes, shared faces and lattice correspondence."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.apd_geometry import build_grain_geometry as _build_grain_geometry
from kanapy.core.power_diagram import AnisotropicPowerDiagram
from kanapy.core.periodic_grains import unwrap_periodic_grains
from kanapy.core.surface_regularization import regularize_grain_surface, _validate
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
    _validate(w.surface.points,w.surface.triangles,w.surface.face_grains)
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


def test_auto_periodic_regularization_and_legacy_opt_out():
    g=periodic_geometry(.15,2)
    points=g['Surface'].points.copy()
    result=regularize_grain_surface(g,iterations=2,patch_retriangulation=True,
                                    simplify_junctions=True,target_angle=20)
    assert result.report['periodic'] and result.report['whole_grains']
    assert result.periodic_geometry is not None
    assert not np.any(result.surface.boundary_ids)
    assert_pairs(result.periodic_geometry)
    assert sum(result.report['after']['grain_volumes'].values()) == pytest.approx(1)
    assert result.report['max_vertex_displacement'] <= result.report['max_displacement']+1e-12
    np.testing.assert_array_equal(points,g['Surface'].points)
    legacy=regularize_grain_surface(g,iterations=1,periodic=False)
    assert not legacy.report['periodic']
    assert legacy.periodic_geometry is None
    assert np.any(legacy.surface.boundary_ids)


def test_api_export_and_atomic_failure(tmp_path, monkeypatch):
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
    result=Microstructure.regularize_grains(ms,iterations=1,max_displacement=0)
    Microstructure.write_stl(ms,file='regularized.stl',path=tmp_path)
    assert (tmp_path/'regularized.stl').read_text().count('endfacet')==len(result.surface.triangles)
    Microstructure.plot_grains(ms,geometry=ms.geometry)
    assert plotted[-1] is ms.geometry
    with pytest.raises(ValueError,match='does not meet'):
        Microstructure.regularize_grains(ms,iterations=1,min_quality=1)
    assert ms.geometry['Regularized'] is result


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
    r=Microstructure.regularize_grains(ms,iterations=1,max_displacement=0)
    assert_pairs(r.periodic_geometry)
    assert r.report['parent_grain_volumes']==pytest.approx({1:.5,2:.5})
    assert r.periodic_geometry.grain_orientations


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


def test_periodic_collision_guard_sees_translated_obstacle():
    from kanapy.core.periodic_grains import _PeriodicCollisionGuard
    from kanapy.core.apd_boundary import APDBoundaryTriangles
    # Second triangle lies at x=1.95 but its translated copy intersects x=.95.
    points=np.array([[.8,.1,.5],[1.1,.1,.5],[.8,.4,.5],
                     [1.95,.15,.3],[1.95,.15,.7],[1.95,.35,.5]])
    tri=np.array([[0,1,2],[3,4,5]])
    surface=APDBoundaryTriangles(points,tri,np.array([-1,-1]),((1,2),(3,4)),np.zeros(2,int))
    guard=_PeriodicCollisionGuard(surface,np.arange(6),np.zeros((6,3),int),np.ones(3),.1)
    assert not guard.allows(np.array([0]),points[tri[:1]])
    assert guard.rejections==1


def test_nonperiodic_default_unchanged_and_periodic_report_serializable():
    import json
    g=periodic_geometry(.15,2)
    g['APD'].periodic=False
    auto=regularize_grain_surface(g,iterations=2)
    explicit=regularize_grain_surface(g,iterations=2,periodic=False)
    np.testing.assert_array_equal(auto.surface.points,explicit.surface.points)
    np.testing.assert_array_equal(auto.surface.triangles,explicit.surface.triangles)
    g['APD'].periodic=True
    periodic=regularize_grain_surface(g,iterations=1)
    json.dumps(periodic.report)
