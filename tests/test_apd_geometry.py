"""APD generate_grains consumers: moments, statistics, plots and exports."""
from types import SimpleNamespace
import numpy as np
import pytest
from kanapy.core.api import Microstructure
from kanapy.core.apd_geometry import build_grain_geometry, label_geometry_points
from kanapy.core.power_diagram import AnisotropicPowerDiagram
from kanapy.core.rve_stats import get_stats_poly
from kanapy.core.plotting import plot_polygons_3D


def diagram():
    return AnisotropicPowerDiagram([[.25,.5,.5],[.75,.5,.5]],
                                  [np.eye(3),np.eye(3)],[1,1,1],grain_ids=[7,20])


def test_moments_shared_areas_and_statistics():
    g = build_grain_geometry(diagram(), {7:0,20:1}, 2)
    assert g['Ngrains'] == 2
    np.testing.assert_allclose(g['GBarea'], [[7,20,1.]])
    for gid, x in [(7,.25),(20,.75)]:
        grain = g['Grains'][gid]
        assert grain['Volume'] == pytest.approx(.5)
        assert grain['Area'] == pytest.approx(4.)
        np.testing.assert_allclose(grain['Center'], [x,.5,.5], atol=1e-14)
        np.testing.assert_allclose(grain['Covariance'], np.diag([.25,1,1])/12, atol=1e-14)
        stats = get_stats_poly(g['Grains'], iphase=grain['Phase'], show_plot=False)
        np.testing.assert_allclose(stats['eqd'], [(3/np.pi)**(1/3)])
    assert g['PhaseVolumes'] == pytest.approx({0:.5,1:.5})


def test_nonconvex_disconnected_slices_and_moments():
    apd = AnisotropicPowerDiagram([[.5,.5,.5]]*2,
                                 [np.eye(3),np.diag([2.,1,1])],[1,1,1])
    apd.weights[:] = [0,.0625]
    g = build_grain_geometry(apd,{1:0,2:0}, (4,1,1))
    assert len(g['Grains'][1]['Shells']) == 2
    np.testing.assert_array_equal(label_geometry_points(g, [[.1,.5,.5],[.5,.5,.5],[.9,.5,.5]]),[1,2,1])
    np.testing.assert_allclose(g['Grains'][1]['Center'], [.5,.5,.5])
    assert g['Grains'][1]['Volume'] == pytest.approx(.5)
    assert g['Grains'][1]['Covariance'][0,0] == pytest.approx(7/48)


def test_api_geometry_exports_and_atomic_failure(tmp_path,monkeypatch):
    from kanapy.core import api
    ms = SimpleNamespace(mesh=SimpleNamespace(apd=diagram(),grain_phase_dict={7:0,20:1}),
                         particles=[],nphases=2,rve=SimpleNamespace(phase_names=['A','B']),
                         name='apd',geometry=None)
    Microstructure.generate_grains(ms,resolution=2)
    old = ms.geometry
    Microstructure.write_stl(ms,path=tmp_path)
    Microstructure.write_centers(ms,path=tmp_path)
    assert (tmp_path/'apd.stl').exists()
    np.testing.assert_allclose(np.loadtxt(tmp_path/'apd_centroid.csv',delimiter=','),[[.25,.5,.5],[.75,.5,.5]])
    def fail(*args,**kwargs):
        raise ValueError('geometry failed')
    monkeypatch.setattr(api,'build_grain_geometry',fail)
    with pytest.raises(ValueError,match='geometry failed'):
        Microstructure.generate_grains(ms,2)
    assert ms.geometry is old


def test_plot_shared_triangles_once(monkeypatch):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    g=build_grain_geometry(diagram(),{7:0,20:1},2)
    original=Axes3D.plot_trisurf
    seen=[]
    def record(self,*args,**kwargs):
        seen.extend(map(tuple,kwargs['triangles']))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Axes3D,'plot_trisurf',record)
    fig=plot_polygons_3D(g,silent=True)
    assert len(seen)==len(g['Surface'].triangles)
    assert len(set(tuple(sorted(t)) for t in seen))==len(seen)
    plt.close(fig)


def test_missing_apd():
    ms=SimpleNamespace(mesh=SimpleNamespace(),geometry=None)
    with pytest.raises(ValueError,match='No ellipsoids'):
        Microstructure.generate_grains(ms)


@pytest.mark.parametrize('cut', ['xy','xz','yz'])
def test_api_slice_sparse_ids_and_noncubic_sampling(tmp_path,monkeypatch,cut):
    monkeypatch.chdir(tmp_path)
    g=build_grain_geometry(diagram(),{7:0,20:1},2)
    ms=SimpleNamespace(geometry=g,rve=SimpleNamespace(size=(1,1,1),dim=(4,3,2)))
    filename=Microstructure.output_ang(ms,ori={7:[.1,.2,.3],20:[.4,.5,.6]},
                                      cut=cut,data='poly',plot=False,pos='bottom')
    rows=np.loadtxt(filename,comments='#')
    expected_rows={'xy':12,'xz':8,'yz':6}[cut]
    assert len(rows)==expected_rows
    assert set(rows[:,0]) <= {.1,.4}


@pytest.mark.parametrize('periodic', [False, True])
def test_generate_without_voxelization(periodic, monkeypatch):
    from kanapy.core.entities import Ellipsoid, Simulation_Box
    particles = [Ellipsoid(7,.25,.5,.5,.2,.2,.2,np.array([1.,0,0,0]),phasenum=0),
                 Ellipsoid(20,.75,.5,.5,.2,.2,.2,np.array([1.,0,0,0]),phasenum=1)]
    duplicate = Ellipsoid(99,1.25,.5,.5,.2,.2,.2,np.array([1.,0,0,0]),dup=7)
    particles.append(duplicate)
    ms = SimpleNamespace(mesh=None, particles=particles, geometry=None, nphases=2,
                         simbox=Simulation_Box((1,1,1)),
                         rve=SimpleNamespace(size=(1,1,1),periodic=periodic,phase_names=['A','B']))
    before = [p.get_pos().copy() for p in particles]
    fitted=[]
    original=AnisotropicPowerDiagram.fit_volumes
    def fit(self, **kwargs):
        fitted.append(self)
        return original(self, n_samples=1024)
    monkeypatch.setattr(AnisotropicPowerDiagram,'fit_volumes',fit)
    Microstructure.generate_grains(ms,resolution=4)
    assert ms.mesh is None
    assert fitted == [ms.geometry['APD']]
    assert ms.geometry['APD'].periodic is periodic
    assert set(ms.geometry['Grains']) == {7,20}
    assert ms.geometry['PhaseVolumes'] == pytest.approx({0:.5,1:.5})
    for p, position in zip(particles,before):
        np.testing.assert_array_equal(p.get_pos(),position)
