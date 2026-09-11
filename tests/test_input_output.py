#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from kanapy.core.initializations import RVE_creator, mesh_creator
from kanapy.core.input_output import *
from kanapy.core.entities import Ellipsoid, Simulation_Box, Cuboid


def test_particleStatGenerator():
    st_dict = {'Grain type': 'Elongated',
               'Equivalent diameter': {'sig': 0.531055, 'loc': 0.0, 'scale': 2.76736, 'cutoff_min': 2.0, 'cutoff_max': 4.0},
               'Aspect ratio': {'sig': 0.3, 'loc': 0.0, 'scale': 2.5, 'cutoff_min': 2.0, 'cutoff_max': 4.0},
               'Tilt angle': {'kappa': 2.1, 'loc': 1.6, 'cutoff_min': 1.0, 'cutoff_max': 2.0},
               'RVE': {'sideX': 8, 'sideY': 8, 'sideZ': 8, 'Nx': 15, 'Ny': 15, 'Nz': 15},
               'Simulation': {'periodicity': 'True', 'output_units': 'um'},
               'Phase': {'Name': 'XXXX', 'Number': 0, 'Volume fraction': 1.0}}

    rve = RVE_creator([st_dict])
    simbox = Simulation_Box(rve.size)
    mesh = mesh_creator(rve.size)
    mesh.nphases = 1
    mesh.create_voxels(simbox)
    assert rve.dim[0] == 15
    assert len(rve.particle_data) == 1
    assert mesh.nodes.shape[1] == 3
    assert mesh.voxel_dict[1][7] == 8
    

@pytest.fixture
def temp_dump():

    # Initialize the Ellipsoids
    ell1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, np.array(
        [0.52532199, 0., -0., 0.85090352]))
    ell2 = Ellipsoid(2, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0,
                     np.array([0.52532199, 0., -0., 0.85090352]))
    ells = [ell1, ell2]

    # Inititalize the simulation box
    sbox = Simulation_Box((10, 10, 10))
    write_dump(ells, sbox)
    return sbox


def test_write_dump(temp_dump):
    cwd = os.getcwd()
    assert os.path.isfile(
        cwd + '/dump_files/particle.{0}.dump'.format(temp_dump.sim_ts))
    return


def test_read_dump(temp_dump):
    cwd = os.getcwd()
    # Test if FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        read_dump(cwd + '/dump_files/nonExistingFile.dump')
    
    # test the remainder of the code
    gen_sbox, genEll = read_dump(cwd + '/dump_files/particle.{0}.dump'.format(temp_dump.sim_ts))

    assert isinstance(gen_sbox, Cuboid)
    for gel in genEll:
        assert isinstance(gel, Ellipsoid)
    return

def test_export_abaqus():

    nodes = np.array(
             [[1., 0., 1.], [1., 0., 0.], [0., 0., 0.], [0., 0., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.],
             [0., 1., 1.], [2., 0., 1.], [2., 0., 0.], [2., 1., 1.], [2., 1., 0.], [3., 0., 1.], [3., 0., 0.], 
             [3., 1., 1.], [3., 1., 0.], [1., 2., 1.], [1., 2., 0.], [0., 2., 0.], [0., 2., 1.], [2., 2., 1.], 
             [2., 2., 0.], [3., 2., 1.], [3., 2., 0.], [1., 3., 1.], [1., 3., 0.], [0., 3., 0.], [0., 3., 1.],
             [2., 3., 1.], [2., 3., 0.], [3., 3., 1.], [3., 3., 0.], [1., 0., 2.], [0., 0., 2.], [1., 1., 2.], 
             [0., 1., 2.], [2., 0., 2.], [2., 1., 2.], [3., 0., 2.], [3., 1., 2.], [1., 2., 2.], [0., 2., 2.], 
             [2., 2., 2.], [3., 2., 2.], [1., 3., 2.], [0., 3., 2.], [2., 3., 2.], [3., 3., 2.], [1., 0., 3.], 
             [0., 0., 3.], [1., 1., 3.], [0., 1., 3.], [2., 0., 3.], [2., 1., 3.], [3., 0., 3.], [3., 1., 3.], 
             [1., 2., 3.], [0., 2., 3.], [2., 2., 3.], [3., 2., 3.], [1., 3., 3.], [0., 3., 3.], [2., 3., 3.], 
             [3., 3., 3.]])

    ed = {1: [1, 2, 3, 4, 5, 6, 7, 8], 2: [9, 10, 2, 1, 11, 12, 6, 5], 3: [13, 14, 10, 9, 15, 16, 12, 11],
          4: [5, 6, 7, 8, 17, 18, 19, 20], 5: [11, 12, 6, 5, 21, 22, 18, 17], 6: [15, 16, 12, 11, 23, 24, 22, 21],
          7: [17, 18, 19, 20, 25, 26, 27, 28], 8: [21, 22, 18, 17, 29, 30, 26, 25], 9: [23, 24, 22, 21, 31, 32, 30, 29],
          10: [33, 1, 4, 34, 35, 5, 8, 36], 11: [37, 9, 1, 33, 38, 11, 5, 35], 12: [39, 13, 9, 37, 40, 15, 11, 38],
          13: [35, 5, 8, 36, 41, 17, 20, 42], 14: [38, 11, 5, 35, 43, 21, 17, 41], 15: [40, 15, 11, 38, 44, 23, 21, 43],
          16: [41, 17, 20, 42, 45, 25, 28, 46], 17: [43, 21, 17, 41, 47, 29, 25, 45], 18: [44, 23, 21, 43, 48, 31, 29, 47],
          19: [49, 33, 34, 50, 51, 35, 36, 52], 20: [53, 37, 33, 49, 54, 38, 35, 51], 21: [55, 39, 37, 53, 56, 40, 38, 54],
          22: [51, 35, 36, 52, 57, 41, 42, 58], 23: [54, 38, 35, 51, 59, 43, 41, 57], 24: [56, 40, 38, 54, 60, 44, 43, 59],
          25: [57, 41, 42, 58, 61, 45, 46, 62], 26: [59, 43, 41, 57, 63, 47, 45, 61], 27: [60, 44, 43, 59, 64, 48, 47, 63]}

    esd = {1: [1, 2, 4, 5, 10, 11, 13, 3, 6, 7, 8, 9, 16],
           2: [20, 22, 23, 17, 25, 26, 12, 15, 18, 21, 24, 27, 19, 14]}

    
    name ='/kanapy_{0}grains.inp'.format(len(esd))
    cwd = os.getcwd()
    export2abaqus(nodes, cwd+name, esd, ed, units='um')
    assert os.path.isfile(cwd + name)
    os.remove(cwd + name)


@pytest.mark.parametrize('selector', [None, 4])
def test_cp_material_include(tmp_path, selector):
    props = tmp_path / 'shared.inc'
    content = '** shared properties\n1, 2, 3, 4, 5, 6, 7, 8\n9\n'
    props.write_text(content)
    writeAbaqusMat(selector, {1: [10, 20, 30], 2: [40, 50, 60]},
                   file='test_mat.inp', path=tmp_path, props_file='shared.inc')
    result = (tmp_path / 'test_mat.inp').read_text()
    assert result.count('*User Material, constants=17') == 2
    assert result.count('*Include, input="shared.inc"') == 2
    assert f'{float(selector or 0)}, 10, 20, 30, 0., 0., 0., 0.' in result
    assert props.read_text() == content


def test_cp_material_legacy(tmp_path):
    writeAbaqusMat(4, {1: [10, 20, 30]}, file='legacy.inp', path=tmp_path)
    assert (tmp_path / 'legacy.inp').read_text().endswith(
        '*User Material, constants=4\n4.0, 10, 20, 30\n')


def test_cp_material_bad_include(tmp_path):
    (tmp_path / 'bad.inc').write_text('1, 2\n3, 4\n')
    with pytest.raises(ValueError, match='eight constants'):
        writeAbaqusMat(4, {1: [0, 0, 0]}, file='bad_mat.inp',
                       path=tmp_path, props_file='bad.inc')
    assert not (tmp_path / 'bad_mat.inp').exists()


@pytest.mark.parametrize('selector', [None, 4])
def test_write_abq_include_api(tmp_path, monkeypatch, selector):
    from types import SimpleNamespace
    from kanapy.core import api

    (tmp_path / 'shared.inc').write_text('1, 2, 3\n')
    mesh = SimpleNamespace(nodes=np.zeros((8, 3)), voxel_dict={}, grain_dict={1: [1]},
                           grain_ori_dict={1: [10, 20, 30]})
    ms = SimpleNamespace(mesh=mesh, nphases=1,
                         rve=SimpleNamespace(ialloy=None, units='um', periodic=False))
    captured = {}
    monkeypatch.setattr(api, 'export2abaqus', lambda *args, **kwargs: captured.update(kwargs))
    api.Microstructure.write_abq(ms, nodes='v', file='test_geom.inp', path=tmp_path,
                                 ialloy=selector, props_file='shared.inc')
    assert captured['ialloy'] == [selector or 0]
    assert '*User Material, constants=11' in (tmp_path / 'test_mat.inp').read_text()


def test_cp_material_phase_includes(tmp_path):
    (tmp_path / 'a.inc').write_text('1\n')
    (tmp_path / 'b.inc').write_text('2, 3\n')
    writeAbaqusMat([4, 5], {1: [0, 0, 0], 2: [1, 2, 3]},
                   file='phases.inp', path=tmp_path, grain_phase_dict={1: 0, 2: 1},
                   props_file=['a.inc', 'b.inc'])
    result = (tmp_path / 'phases.inp').read_text()
    assert '*User Material, constants=9\n4.0' in result
    assert '*User Material, constants=10\n5.0' in result
