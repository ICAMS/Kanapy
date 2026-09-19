"""Multiphase workflows with GRAIN0 reserved for the unoriented PHASE0 matrix."""
import json
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from kanapy.core.api import Microstructure
from kanapy.core.entities import Ellipsoid, Simulation_Box
from kanapy.core.initializations import RVE_creator, mesh_creator
from kanapy.core.input_output import import_voxels
from kanapy.core.voxelization_legacy import voxelizationRoutine_legacy


@pytest.fixture
def three_phases():
    first = {
        'Grain type': 'Equiaxed',
        'Equivalent diameter': {'sig': 0.15, 'scale': 3.0, 'loc': 0.,
                                'cutoff_min': 2.4, 'cutoff_max': 3.6},
        'RVE': {'sideX': 8, 'sideY': 8, 'sideZ': 8, 'Nx': 8, 'Ny': 8, 'Nz': 8, 'ialloy': 4},
        'Simulation': {'periodicity': False, 'output_units': 'um'},
        'Phase': {'Name': 'Grains A', 'Volume fraction': 0.2}}
    second = deepcopy(first)
    second['Phase'] = {'Name': 'Grains B', 'Volume fraction': 0.3}
    second['RVE']['ialloy'] = 5
    matrix = {'Grain type': 'Matrix', 'Phase': {'Name': 'Matrix', 'Volume fraction': 0.5}}
    return [first, second, matrix]  # Matrix is canonicalized to PHASE0.


def assert_phase_consistency(ms):
    mesh = ms.mesh
    counts = np.zeros(ms.nphases, dtype=int)
    volumes = np.zeros(ms.nphases, dtype=int)
    voxel_ids = []
    for gid, voxels in mesh.grain_dict.items():
        pid = mesh.grain_phase_dict[gid]
        indices = np.asarray(voxels, dtype=int) - 1
        assert np.all(mesh.grains.ravel()[indices] == gid)
        assert np.all(mesh.phases.ravel()[indices] == pid)
        counts[pid] += 1
        volumes[pid] += len(voxels)
        voxel_ids.extend(voxels)
    assert sorted(voxel_ids) == list(range(1, mesh.nvox + 1))
    np.testing.assert_array_equal(mesh.ngrains_phase, counts)
    np.testing.assert_allclose(ms.vf_vox, volumes / mesh.nvox)
    assert np.sum(ms.vf_vox) == pytest.approx(1)
    assert mesh.grain_phase_dict[0] == 0
    assert [g for g, p in mesh.grain_phase_dict.items() if p == 0] == [0]


@pytest.fixture
def voxelized_three_phases_legacy(three_phases):
    random.seed(12)
    np.random.seed(12)
    ms = Microstructure(three_phases, name='three_phases')
    ms.init_RVE(nsteps=20)
    ms.pack(save_files=False, verbose=False)
    ms.voxelize_legacy()
    return ms


def test_three_phase_workflow(voxelized_three_phases_legacy, tmp_path):
    ms = voxelized_three_phases_legacy
    assert ms.nphases == 3
    assert ms.rve.phase_names == ['Matrix', 'Grains A', 'Grains B']
    assert ms.rve.phase_vf == [0.5, 0.2, 0.3]
    assert ms.rve.ialloy == [None, 4, 5]
    assert ms.nparticles[0] == 0
    assert 0 < ms.nparticles[1] < ms.nparticles[2]
    assert {p.phasenum for p in ms.particles} == {1, 2}
    assert all(p.id > 0 for p in ms.particles)
    assert_phase_consistency(ms)
    assert np.all(ms.vf_vox > 0)
    assert ms.mesh.prec_vf_voxels == pytest.approx(ms.vf_vox[1] + ms.vf_vox[2])

    ms.generate_orientations('random', Nbase=50, iphase=1)
    saved = {g: o.copy() for g, o in ms.mesh.grain_ori_dict.items()}
    ms.generate_orientations('random', Nbase=50, iphase=2)
    for gid, orientation in saved.items():
        np.testing.assert_array_equal(ms.mesh.grain_ori_dict[gid], orientation)
    assert set(ms.mesh.grain_ori_dict) == set(ms.mesh.grain_dict) - {0}
    ms.generate_grains(resolution=2)
    assert set(ms.geometry['Grains']) <= {p.id for p in ms.particles}
    assert_phase_consistency(ms)
    from kanapy.core.rve_stats import get_stats_vox
    assert len(get_stats_vox(ms.mesh)['eqd']) == len(ms.mesh.grain_dict) - 1

    ms.write_voxels(file='three.json', path=tmp_path)
    loaded = import_voxels('three.json', path=tmp_path)
    assert_phase_consistency(loaded)
    assert loaded.precipit == pytest.approx(1 - ms.vf_vox[0])
    assert loaded.mesh.grain_ori_dict.keys() == ms.mesh.grain_ori_dict.keys()
    for gid in ms.mesh.grain_ori_dict:
        np.testing.assert_allclose(loaded.mesh.grain_ori_dict[gid], ms.mesh.grain_ori_dict[gid])

    (tmp_path / 'a.inc').write_text('1, 2, 3\n')
    (tmp_path / 'b.inc').write_text('4, 5, 6\n')
    kwargs = dict(ialloy=[0, 4, 5], props_file=[None, 'a.inc', 'b.inc'],
                  crystal_plasticity=[False, True, True], path=tmp_path)
    ms.write_abq(file='three_geom.inp', **kwargs)
    geometry = (tmp_path / 'three_geom.inp').read_text()
    material = (tmp_path / 'three_mat.inp').read_text()
    assert '*Solid Section, elset=GRAIN0_SET, material=PHASE0_MAT' in geometry
    assert '*Material, name=PHASE0_MAT' in geometry
    assert '*Material, name=GRAIN0_MAT' not in material
    for gid, pid in ms.mesh.grain_phase_dict.items():
        if gid == 0:
            continue
        block = material.split(f'*Material, name=GRAIN{gid}_MAT\n')[1].split('*Material')[0]
        assert f'\n{float([0, 4, 5][pid])}, ' in block
        assert f'*Include, input="{[None, "a.inc", "b.inc"][pid]}"' in block
    ms.write_abq_ori(file='ori_mat.inp', **kwargs)
    assert (tmp_path / 'ori_mat.inp').read_text() == material

    ms.write_abq(file='standard_geom.inp', path=tmp_path, dual_phase=True,
                 props_file=[None, None, None], crystal_plasticity=[False, False, False])
    standard = (tmp_path / 'standard_geom.inp').read_text()
    for pid in range(3):
        assert f'*Solid Section, elset=PHASE{pid}_SET, material=PHASE{pid}_MAT' in standard


@pytest.mark.parametrize('use_file', [False, True])
def test_descriptor_normalization(three_phases, tmp_path, use_file):
    original = deepcopy(three_phases)
    if use_file:
        file = tmp_path / 'phases.json'
        file.write_text(json.dumps(three_phases))
        ms = Microstructure(file=file)
    else:
        ms = Microstructure(three_phases)
    ms.init_RVE(nsteps=1)
    assert ms.nphases == 3
    assert [d['Phase']['Number'] for d in ms.descriptor] == [0, 1, 2]
    assert ms.rve.nparticles[0] == 0
    assert [p['Phase'] for p in ms.rve.particle_data] == [1, 2]
    assert three_phases == original


@pytest.mark.parametrize('use_file', [False, True])
def test_legacy_implicit_matrix(three_phases, tmp_path, use_file):
    descriptor = three_phases[0]
    file = tmp_path / 'one.json'
    file.write_text(json.dumps(descriptor))
    ms = Microstructure(file=file) if use_file else Microstructure(descriptor)
    ms.init_RVE(nsteps=1)
    assert ms.nphases == 2
    assert ms.rve.phase_vf == [0.8, 0.2]
    assert ms.rve.nparticles[0] == 0
    assert ms.precipit == pytest.approx(0.2)


@pytest.mark.parametrize('fractions', [[0.2, 0.2], [-0.2, 1.2], [float('nan'), 0.5], [float('inf'), 0.5]])
def test_invalid_fractions(three_phases, fractions):
    descriptors = three_phases[:2]
    for d, vf in zip(descriptors, fractions):
        d['Phase']['Volume fraction'] = vf
    with pytest.raises(ValueError, match='fraction'):
        RVE_creator(descriptors, from_voxels=True)


@pytest.mark.parametrize('phase_ids', [[0, 0, 1, 1, 1, 2], [0, 1, 0], [2, 0, 2, 0, 2]])
def test_orientations_unequal_interleaved_and_empty_phases(phase_ids, monkeypatch):
    import kanapy.texture
    gids = [10 + i * 3 for i in range(len(phase_ids))]
    mesh = SimpleNamespace(grains=np.ones((1, 1, 1)), grain_dict=dict.fromkeys(gids, [1]),
                           grain_phase_dict=dict(zip(gids, phase_ids)), grain_ori_dict=None)
    ms = SimpleNamespace(mesh=mesh, ngrains=np.bincount(phase_ids, minlength=3))
    monkeypatch.setattr(kanapy.texture, 'createOrisetRandom',
                        lambda n, **kw: np.arange(n * 3).reshape(n, 3))
    Microstructure.generate_orientations(ms, 'random')
    for pid in range(3):
        phase_grains = [g for g in gids if mesh.grain_phase_dict[g] == pid]
        for i, gid in enumerate(phase_grains):
            np.testing.assert_array_equal(mesh.grain_ori_dict[gid], np.arange(3) + i * 3)


def test_partial_orientation_roundtrip(voxelized_three_phases_legacy, tmp_path):
    ms = voxelized_three_phases_legacy
    # Highest grain ID (phase 2) has no orientation; phase 1 must survive import.
    ms.generate_orientations('random', iphase=1, Nbase=50)
    ms.write_voxels(file='partial.json', path=tmp_path)
    loaded = import_voxels('partial.json', path=tmp_path)
    assert loaded.mesh.grain_ori_dict.keys() == ms.mesh.grain_ori_dict.keys()
    loaded.write_voxels(file='again.json', path=tmp_path)
    second = import_voxels('again.json', path=tmp_path)
    assert second.mesh.grain_ori_dict == loaded.mesh.grain_ori_dict


def test_phase_lost_during_voxelization_legacy(monkeypatch):
    import kanapy.core.voxelization_legacy as voxelization
    particles = [Ellipsoid(7, 1, 1, 1, .5, .5, .5, np.array([1., 0, 0, 0]), phasenum=1),
                 Ellipsoid(20, 1, 1, 1, .5, .5, .5, np.array([1., 0, 0, 0]), phasenum=2)]
    def assign(*args, **kwargs):
        particles[0].inside_voxels = [1, 2, 3]
        particles[1].inside_voxels = []
    monkeypatch.setattr(voxelization, 'assign_voxels_to_ellipsoid_legacy', assign)
    mesh = mesh_creator((2, 2, 2))
    mesh.create_voxels(Simulation_Box((2, 2, 2)))
    voxelizationRoutine_legacy(particles, mesh, 3, prec_vf=0.4)
    np.testing.assert_array_equal(mesh.ngrains_phase, [1, 1, 0])
    assert mesh.grain_phase_dict == {7: 1, 0: 0}
    assert set(mesh.grain_dict[0]) == {4, 5, 6, 7, 8}


def test_two_grain_phases_without_matrix(three_phases):
    descriptors = three_phases[:2]
    descriptors[0]['Phase']['Volume fraction'] = .4
    descriptors[1]['Phase']['Volume fraction'] = .6
    random.seed(5)
    np.random.seed(5)
    ms = Microstructure(descriptors)
    ms.init_RVE(nsteps=20)
    ms.pack(save_files=False, verbose=False)
    ms.voxelize()
    ms.generate_orientations('random', Nbase=50)
    assert ms.nphases == 2
    assert ms.precipit is None
    assert 0 not in ms.mesh.grain_dict
    assert set(ms.mesh.grain_phase_dict.values()) == {0, 1}
    assert len(ms.mesh.grain_ori_dict) == ms.Ngr
    assert sum(map(len, ms.mesh.grain_dict.values())) == ms.mesh.nvox
    assert sum(ms.vf_vox) == pytest.approx(1)


def test_stats_skip_matrix_without_renumbering(three_phases, monkeypatch):
    from kanapy.core import api
    ms = Microstructure(three_phases)
    ms.init_RVE(nsteps=1)
    monkeypatch.setattr(api, 'plot_init_stats', lambda *args, **kw: None)
    _, descriptors = ms.plot_stats_init(silent=True, return_descriptors=True)
    assert [d['phase'] for d in descriptors] == [1, 2]
    ms.mesh = SimpleNamespace(grain_phase_dict={1: 1, 0: 0, 5: 2})
    ms.particles = []
    visited = []
    def particle_stats(*args, iphase=None, **kwargs):
        visited.append(iphase)
        return dict.fromkeys(('a_sig', 'b_sig', 'c_sig', 'a_scale', 'b_scale', 'c_scale',
                              'ind_rot', 'ar_scale', 'ar_sig', 'eqd_scale', 'eqd_sig'), 1.)
    monkeypatch.setattr(api, 'get_stats_part', particle_stats)
    comparisons = []
    monkeypatch.setattr(api, 'plot_output_stats',
                        lambda *args, **kwargs: comparisons.append(kwargs['gs_param']))
    assert len(ms.plot_stats(data='p', phases=True, silent=True,
                             gs_param=['matrix', 'phase1', 'phase2'])) == 2
    assert visited == [1, 2]
    assert comparisons == ['phase1', 'phase2']


def test_legacy_matrix_unchanged_when_apd_geometry_fails(voxelized_three_phases_legacy, monkeypatch):
    ms = voxelized_three_phases_legacy
    original = deepcopy(ms.mesh.grain_dict)
    from kanapy.core import api
    def fail(*args, **kwargs):
        raise ValueError('geometry failure')
    monkeypatch.setattr(api, 'build_grain_geometry', fail)
    with pytest.raises(ValueError, match='geometry failure'):
        ms.generate_grains()
    for gid in original:
        np.testing.assert_array_equal(original[gid], ms.mesh.grain_dict[gid])
    assert_phase_consistency(ms)


def test_missing_cp_orientation_rejected_before_export(voxelized_three_phases_legacy, tmp_path):
    ms = voxelized_three_phases_legacy
    ms.generate_orientations('random', iphase=1)
    with pytest.raises(ValueError, match='Missing orientations'):
        ms.write_abq(file='missing_geom.inp', path=tmp_path, ialloy=[0, 4, 5],
                     crystal_plasticity=[False, True, True])
    assert not (tmp_path / 'missing_geom.inp').exists()


def test_matrix_rejects_cp(voxelized_three_phases_legacy, tmp_path):
    ms = voxelized_three_phases_legacy
    with pytest.raises(ValueError, match='Grain 0 requires standard plasticity'):
        ms.write_abq(file='invalid_geom.inp', path=tmp_path,
                     crystal_plasticity=[True, True, True])
    assert not (tmp_path / 'invalid_geom.inp').exists()


def test_three_phase_export_with_empty_phase(tmp_path):
    mesh = mesh_creator((2, 2, 2))
    mesh.create_voxels(Simulation_Box((2, 2, 2)))
    mesh.grain_dict = {7: [1, 2], 0: [3, 4, 5, 6, 7, 8]}
    mesh.grain_phase_dict = {7: 1, 0: 0}
    mesh.grain_ori_dict = {7: [10, 20, 30]}
    ms = SimpleNamespace(mesh=mesh, nphases=3, rve=SimpleNamespace(ialloy=[0, 4, 5], units='um', periodic=False))
    Microstructure.write_abq(ms, file='empty_geom.inp', path=tmp_path,
                            crystal_plasticity=[False, True, True])
    text = (tmp_path / 'empty_mat.inp').read_text()
    assert text.count('*Material, name=GRAIN') == 1
    Microstructure.write_abq(ms, file='phase_geom.inp', path=tmp_path, dual_phase=True,
                            crystal_plasticity=[False, False, False])
    text = (tmp_path / 'phase_geom.inp').read_text()
    assert 'PHASE2_SET' not in text


def test_legacy_full_rve_fills_residual_voxels_without_creating_matrix(monkeypatch):
    import kanapy.core.voxelization_legacy as voxelization
    particles = [Ellipsoid(1, .5, .5, .5, .5, .5, .5, np.array([1., 0, 0, 0]), phasenum=0),
                 Ellipsoid(2, 1.5, 1.5, 1.5, .5, .5, .5, np.array([1., 0, 0, 0]), phasenum=1)]
    def assign(*args, **kwargs):
        particles[0].inside_voxels = [1, 2, 3]
        particles[1].inside_voxels = [5, 6, 7, 8]
    monkeypatch.setattr(voxelization, 'assign_voxels_to_ellipsoid_legacy', assign)
    mesh = mesh_creator((2, 2, 2))
    mesh.create_voxels(Simulation_Box((2, 2, 2)))
    voxelizationRoutine_legacy(particles, mesh, 2)
    assert 0 not in mesh.grain_dict
    assert sorted(v for vox in mesh.grain_dict.values() for v in vox) == list(range(1, 9))
    for gid, vox in mesh.grain_dict.items():
        indices = np.array(vox) - 1
        assert np.all(mesh.grains.ravel()[indices] == gid)
        assert np.all(mesh.phases.ravel()[indices] == mesh.grain_phase_dict[gid])


@pytest.mark.parametrize('matrix_fraction', [0., 1.])
def test_degenerate_matrix_rejected(three_phases, matrix_fraction):
    three_phases[0]['Phase']['Volume fraction'] = (1 - matrix_fraction) * .4
    three_phases[1]['Phase']['Volume fraction'] = (1 - matrix_fraction) * .6
    three_phases[2]['Phase']['Volume fraction'] = matrix_fraction
    with pytest.raises(ValueError, match='Matrix volume fraction'):
        RVE_creator(three_phases, from_voxels=True)


def test_multiple_matrices_rejected(three_phases):
    with pytest.raises(ValueError, match='at most one matrix'):
        Microstructure(three_phases + [deepcopy(three_phases[-1])])


@pytest.mark.parametrize('grain,phase', [('0', 1), ('1', 0), ('1', 3)])
def test_import_rejects_invalid_phase_mapping(tmp_path, grain, phase):
    data = {
        'Model': {'Size': [2, 2, 2], 'Phase_names': ['Matrix', 'A', 'B'],
                  'Periodicity': False, 'Units': {'Length': 'um'}, 'Material': 'invalid'},
        'Data': {'Shape': [2, 2, 2], 'Order': 'C', 'Values': [0, 0, 0, 0, 1, 1, 2, 2]},
        'Grains': {'0': {'Phase': 0}, '1': {'Phase': 1}, '2': {'Phase': 2}},
    }
    data['Grains'][grain]['Phase'] = phase
    (tmp_path / 'invalid.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='PHASE0|phase ID'):
        import_voxels('invalid.json', path=tmp_path)


def test_reinitialize_updates_phase_count_and_matrix(three_phases):
    ms = Microstructure(three_phases[0])
    assert ms.nphases == 2
    ms.init_RVE(descriptor=three_phases, nsteps=1)
    assert ms.nphases == 3
    assert ms.precipit == pytest.approx(.5)
    descriptor = deepcopy(three_phases[0])
    descriptor['Phase']['Volume fraction'] = 1.
    ms.init_RVE(descriptor=descriptor, nsteps=1)
    assert ms.nphases == 1
    assert ms.precipit is None


@pytest.mark.parametrize('phase_sets', [False, True])
def test_legacy_matrix_export_rejected(tmp_path, phase_sets):
    mesh = mesh_creator((2, 2, 2))
    mesh.create_voxels(Simulation_Box((2, 2, 2)))
    mesh.grain_dict = {1: [1, 2], 0: [3, 4, 5, 6, 7, 8]}
    mesh.grain_phase_dict = {1: 0, 0: 1}  # obsolete convention
    mesh.grain_ori_dict = None
    ms = SimpleNamespace(mesh=mesh, nphases=2,
                         rve=SimpleNamespace(ialloy=None, units='um', periodic=False))
    with pytest.raises(ValueError, match='migrate legacy'):
        Microstructure.write_abq(ms, file='old_geom.inp', path=tmp_path,
                                 dual_phase=phase_sets, crystal_plasticity=[False, False])
    assert not (tmp_path / 'old_geom.inp').exists()


def test_matrix_ebsd_source_phase_mapping(monkeypatch):
    import kanapy.texture
    visited = []
    class EBSD:
        def calcORI(self, n, iphase, **kwargs):
            visited.append(iphase)
            return np.zeros((n, 3))
    monkeypatch.setattr(kanapy.texture, 'EBSDmap', EBSD)
    ms = SimpleNamespace(precipit=.5, ngrains=[1, 2, 1],
                         mesh=SimpleNamespace(grains=np.ones((1, 1, 1)),
                             grain_dict={0: [1], 3: [2], 9: [3], 7: [4]},
                             grain_phase_dict={0: 0, 3: 1, 9: 1, 7: 2}, grain_ori_dict=None))
    Microstructure.generate_orientations(ms, EBSD())
    assert visited == [0, 1]
    assert set(ms.mesh.grain_ori_dict) == {3, 9, 7}
    visited.clear()
    Microstructure.generate_orientations(ms, EBSD(), iphase=2, ebsd_phase_map={2: 4})
    assert visited == [4]


def test_saved_matrix_examples_use_phase_zero():
    from pathlib import Path
    example_root = Path(__file__).resolve().parents[1] / 'examples'
    checked = 0
    for file in example_root.rglob('*.json'):
        if '.ipynb_checkpoints' in file.parts:
            continue
        data = json.loads(file.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('Grains'), dict):
            continue
        grains = data['Grains']
        if '0' not in grains:
            continue
        assert grains['0']['Phase'] == 0, str(file)
        assert all(row['Phase'] != 0 for gid, row in grains.items()
                   if gid != '0' and isinstance(row, dict)), str(file)
        checked += 1
    assert checked > 0


def test_low_level_matrix_export_requires_phase_mapping(tmp_path):
    from kanapy.core.input_output import export2abaqus
    with pytest.raises(ValueError, match='explicit phase ID'):
        export2abaqus(None, tmp_path / 'missing.inp', {0: [1], 1: [2]}, {})
    assert not (tmp_path / 'missing.inp').exists()
