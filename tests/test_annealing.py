"""Physical evolution, boundary conditions and API consistency checks."""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('fipy')
from kanapy.core.annealing import grain_growth
from kanapy.core.api import Microstructure


def inclusion():
    a = np.full((16, 16, 3), 7, dtype=int)
    x, y = np.ogrid[:16, :16]
    a[(x-7.5)**2 + (y-7.5)**2 < 16, :] = 42
    return a


def test_curved_grain_shrinks_and_stops_at_first_crossing():
    a = inclusion()
    before = a.copy()
    b, r = grain_growth(a, max_steps=300)
    assert np.array_equal(a, before)
    assert r['reason'] == 'volume_change'
    assert r['max_volume_change'] >= .1
    assert np.all(r['max_volume_change_history'][:-1] < .1)
    assert np.sum(b == 42) < np.sum(a == 42)
    assert r['final_volumes'].sum() == a.size


def test_periodic_translation_equivariance():
    a = inclusion()
    b, r = grain_growth(a, periodic=True, max_steps=12)
    shifted, _ = grain_growth(np.roll(a, 6, axis=0), periodic=True, max_steps=12)
    assert np.array_equal(shifted, np.roll(b, 6, axis=0))
    assert r['reason'] == 'max_steps'


def test_single_grain_and_validation():
    a = np.ones((3, 4, 2), dtype=int)
    b, r = grain_growth(a)
    assert np.array_equal(a, b) and r['steps'] == 0
    for kw in [dict(dt=-1), dict(mobility=np.nan), dict(max_steps=0),
               dict(spacing=(1, 0, 1)), dict(volume_change=2)]:
        with pytest.raises(ValueError):
            grain_growth(a, **kw)


def test_api_updates_voxels_and_invalidates_geometry():
    ms = Microstructure.__new__(Microstructure)
    ms.precipit = None
    ms.particles = None
    a = inclusion()
    ms.nphases = 1
    ms.rve = SimpleNamespace(periodic=False)
    ms.geometry = {'old': True}
    ms.mesh = SimpleNamespace(grains=a, dim=a.shape,
        nodes=np.array([[0, 0, 0], list(a.shape)]),
        grain_phase_dict={7: 0, 42: 0}, grain_ori_dict={7: [1, 2, 3], 42: [3, 2, 1]})
    report = ms.anneal(max_steps=300)
    assert report['reason'] == 'volume_change'
    for gid, voxels in ms.mesh.grain_dict.items():
        assert np.all(ms.mesh.grains.ravel()[np.array(voxels)-1] == gid)
    assert sum(map(len, ms.mesh.grain_dict.values())) == a.size
    assert ms.geometry is None and ms.mesh.apd is None
    assert ms.Ngr == 2 and np.array_equal(ms.vf_vox, [1.])
    with pytest.raises(ValueError, match='annealed'):
        ms.generate_grains()
    ms.mesh.grain_phase_dict[42] = 1
    with pytest.raises(ValueError, match='single-phase'):
        ms.anneal()
