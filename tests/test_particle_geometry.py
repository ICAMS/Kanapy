"""Direct matrix/particle surfaces, including clipping and public consumers."""
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.api import Microstructure
from kanapy.core.entities import Ellipsoid, Simulation_Box
from kanapy.core.particle_geometry import build_particle_geometry
from kanapy.core.plotting import plot_polygons_3D


def particle(x=.5):
    return Ellipsoid(7, x, .5, .5, .2, .15, .1,
                     np.array([1., 0., 0., 0.]), phasenum=1)


def test_surface_moments_and_resolution():
    p = particle()
    coarse = build_particle_geometry([p], [1, 1, 1], 3)
    g = build_particle_geometry([p], [1, 1, 1], 16)
    grain = g['Grains'][7]
    exact = 4*np.pi*.2*.15*.1/3
    assert abs(grain['Volume']-exact) < abs(coarse['Grains'][7]['Volume']-exact)
    assert grain['Volume'] == pytest.approx(exact, rel=.02)
    np.testing.assert_allclose(grain['Center'], [.5]*3, atol=1e-12)
    np.testing.assert_allclose(grain['SemiAxes'], [.2, .15, .1], rtol=.02)
    assert sum(g['PhaseVolumes'].values()) == pytest.approx(1)
    assert g['GBarea'][0] == pytest.approx([0, 7, grain['Area']])
    surface = g['Surface']
    assert set(surface.face_grains) == {(7, 0)}
    radial = surface.points[surface.triangles].mean(axis=1)-p.get_pos()
    assert np.all(np.einsum('ij,ij->i', radial, surface.normals) > 0)


@pytest.mark.parametrize('periodic', [False, True])
def test_box_clipping_and_periodic_fragments(periodic):
    g = build_particle_geometry([particle(0)], [1, 1, 1], 8, periodic=periodic)
    full = build_particle_geometry([particle()], [1, 1, 1], 8)
    assert g['Grains'][7]['Volume'] == pytest.approx(
        full['Grains'][7]['Volume'] * (1 if periodic else .5))
    assert len(g['Grains'][7]['Shells']) == (2 if periodic else 1)
    assert np.all(g['Points'] >= 0) and np.all(g['Points'] <= 1)
    surface = g['Surface']
    assert np.any(surface.boundary_ids)
    for pair, bid in zip(surface.face_grains, surface.boundary_ids):
        assert pair == (7, None if bid else 0)
    # Each clipped fragment is a closed, consistently oriented shell.
    for shell in g['Grains'][7]['Shells']:
        faces = surface.triangles[[i for i, _ in shell]]
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        assert np.all(counts == 2)


def test_api_matrix_geometry_plot_stl_and_atomic_errors(tmp_path, monkeypatch):
    from kanapy.core import api
    import matplotlib.pyplot as plt

    def fail_apd(*args, **kwargs):
        pytest.fail('Matrix surface generation must not build APD geometry')
    monkeypatch.setattr(api, 'build_grain_geometry', fail_apd)
    ms = SimpleNamespace(mesh=SimpleNamespace(apd=object()), particles=[particle(0)],
                         precipit=.03, nphases=2, geometry=None, name='particles',
                         simbox=Simulation_Box((1, 1, 1)),
                         rve=SimpleNamespace(size=(1, 1, 1), periodic=True,
                                             matrix_phase=0, phase_names=['Matrix', 'Pores']))
    Microstructure.generate_grains(ms, 6)
    g = ms.geometry
    assert g['Representation'] == 'Particles'
    fig = plot_polygons_3D(g, silent=True)
    plt.close(fig)
    Microstructure.write_stl(ms, path=tmp_path)
    Microstructure.write_stl(ms, 'closed.stl', path=tmp_path, include_exterior=True)
    assert (tmp_path/'particles.stl').read_text().count('facet normal') == np.count_nonzero(
        g['Surface'].boundary_ids == 0)
    assert (tmp_path/'closed.stl').read_text().count('facet normal') == len(g['Facets'])
    ms.particles[0].inner = object()
    with pytest.raises(ValueError, match='inner structure'):
        Microstructure.generate_grains(ms)
    assert ms.geometry is g


def test_duplicates_are_not_meshed_twice():
    p = particle()
    duplicate = particle()
    duplicate.id, duplicate.duplicate = 99, 7
    g = build_particle_geometry([p, duplicate], [1, 1, 1], periodic=True)
    assert set(g['Grains']) == {7}
    assert len(g['Grains'][7]['Shells']) == 1
