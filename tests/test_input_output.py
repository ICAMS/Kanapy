#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import pytest
import json

from kanapy.cli import write_position_weights
from kanapy.initializations import RVEcreator
from kanapy.input_output import *
from kanapy.entities import Ellipsoid, Simulation_Box, Cuboid

def test_particleStatGenerator():

    # Test if FileNotFoundError is raised
    #with pytest.raises(FileNotFoundError):        
    #    particleStatGenerator('inp.json')

    # create an temporary input file for user defined statistics
    cwd = os.getcwd()
    stat_inp = cwd + '/input_test.json'   
            
    # Test if ValueError is raised w.r.t output_units
    to_write = {'Grain type': 'Elongated', 'Equivalent diameter': {'std': 0.531055, 'mean': 2.76736, 'cutoff_min': 2.0, 'cutoff_max': 4.0},
                'Aspect ratio': {'std':0.3, 'mean': 2.5, 'cutoff_min': 2.0, 'cutoff_max': 4.0}, 'Tilt angle': {'std': 28.8, 'mean': 87.4, 
                "cutoff_min": 75.0, "cutoff_max": 105.0}, 'RVE': {'sideX': 8, 'sideY': 8, 'sideZ': 8, 'Nx': 15, 'Ny': 15, 'Nz': 15}, 
                'Simulation': {'periodicity': 'True', 'output_units': 'm'},
                'Phase': {'Name': 'XXXX', 'Number': 0, 'Volume fraction': 1.0}}

    with open(stat_inp, 'w') as outfile:
        json.dump(to_write, outfile, indent=2)       

    with pytest.raises(ValueError):            
        RVEcreator(to_write, save_files=True)
    os.remove(stat_inp)
                            
    # Test the remaining code
    to_write = {'Grain type': 'Elongated', 'Equivalent diameter': {'std': 0.531055, 'mean': 2.76736, 'cutoff_min': 2.0, 'cutoff_max': 4.0},
                'Aspect ratio': {'std':0.3, 'mean': 2.5, 'cutoff_min': 2.0, 'cutoff_max': 4.0}, 'Tilt angle': {'std': 28.8, 'mean': 87.4, 
                "cutoff_min": 75.0, "cutoff_max": 105.0}, 'RVE': {'sideX': 6, 'sideY': 6, 'sideZ': 6, 'Nx': 15, 'Ny': 15, 'Nz': 15}, 
                'Simulation': {'periodicity': 'True', 'output_units': 'mm'},
                'Phase': {'Name': 'XXXX', 'Number': 0, 'Volume fraction': 1.0}}    

    with open(stat_inp, 'w') as outfile:
        json.dump(to_write, outfile, indent=2) 
    
    RVEcreator(to_write, save_files=True)

    # Read the json files written by the function
    json_dir = cwd + '/json_files'
    with open(json_dir + '/particle_data.json') as json_file:
        pd = json.load(json_file)
    with open(json_dir + '/RVE_data.json') as json_file:
        rd = json.load(json_file)
    with open(json_dir + '/simulation_data.json') as json_file:
        sd = json.load(json_file)

    # Dictionaries to verify against
    compare_rd = {'RVE_sizeX': 6, 'RVE_sizeY': 6, 'RVE_sizeZ': 6,
                  'Voxel_numberX': 15, 'Voxel_numberY': 15, 'Voxel_numberZ': 15,
                  'Voxel_resolutionX': 0.4, 'Voxel_resolutionY': 0.4, 'Voxel_resolutionZ': 0.4,
                  'Periodic': True, 'Units': 'mm'}
    compare_sd = {'Time steps': 1000, 'Periodicity': 'True', 'Output units': 'mm'}
    
    # Verify
    assert rd == compare_rd
    assert sd == compare_sd

    os.remove(stat_inp)
    shutil.rmtree(json_dir)
    

@pytest.fixture
def temp_dump():

    # Initialize the Ellipsoids
    ell1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, np.array(
        [0.52532199, 0., -0., 0.85090352]))
    ell2 = Ellipsoid(2, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0,
                     np.array([0.52532199, 0., -0., 0.85090352]))
    ells = [ell1, ell2]

    # Inititalize the simulation box
    sbox = Simulation_Box(10, 10, 10)
    write_dump(ells, sbox)
    return sbox


def test_write_dump(temp_dump):

    cwd = os.getcwd()
    assert os.path.isfile(
        cwd + '/dump_files/particle.{0}.dump'.format(temp_dump.sim_ts))


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


def test_write_position_weights(temp_dump):

    # Test if FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        write_position_weights(26957845264856)
        
    # Test the remainder of the code    
    write_position_weights(0)
    cwd = os.getcwd()
    assert os.path.isfile(cwd + '/sphere_positions.txt')
    assert os.path.isfile(cwd + '/sphere_weights.txt')

    os.remove(cwd + '/sphere_positions.txt')
    os.remove(cwd + '/sphere_weights.txt')


def test_export_abaqus():

    nodes = [[1., 0., 1.], [1., 0., 0.], [0., 0., 0.], [0., 0., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.], 
             [0., 1., 1.], [2., 0., 1.], [2., 0., 0.], [2., 1., 1.], [2., 1., 0.], [3., 0., 1.], [3., 0., 0.], 
             [3., 1., 1.], [3., 1., 0.], [1., 2., 1.], [1., 2., 0.], [0., 2., 0.], [0., 2., 1.], [2., 2., 1.], 
             [2., 2., 0.], [3., 2., 1.], [3., 2., 0.], [1., 3., 1.], [1., 3., 0.], [0., 3., 0.], [0., 3., 1.],
             [2., 3., 1.], [2., 3., 0.], [3., 3., 1.], [3., 3., 0.], [1., 0., 2.], [0., 0., 2.], [1., 1., 2.], 
             [0., 1., 2.], [2., 0., 2.], [2., 1., 2.], [3., 0., 2.], [3., 1., 2.], [1., 2., 2.], [0., 2., 2.], 
             [2., 2., 2.], [3., 2., 2.], [1., 3., 2.], [0., 3., 2.], [2., 3., 2.], [3., 3., 2.], [1., 0., 3.], 
             [0., 0., 3.], [1., 1., 3.], [0., 1., 3.], [2., 0., 3.], [2., 1., 3.], [3., 0., 3.], [3., 1., 3.], 
             [1., 2., 3.], [0., 2., 3.], [2., 2., 3.], [3., 2., 3.], [1., 3., 3.], [0., 3., 3.], [2., 3., 3.], 
             [3., 3., 3.]]

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
    json_dir = cwd + '/json_files'

    # Test if FileNotFoundError is raised
    #with pytest.raises(FileNotFoundError):
    #    abaqusoutput()
        
    export2abaqus(nodes, cwd+name, esd, ed, units='um')
    assert os.path.isfile(cwd + name)
    os.remove(cwd + name)
    
    '''
    CLI function cannot be tested properly in current setting
    
    # Test the remainder of the code
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/simulation_data.json', 'w') as outfile:
        json.dump(simData, outfile)
        
    with open(json_dir + '/nodes_v.csv', 'w') as f:
        for v in nodes:
            f.write('{0}, {1}, {2}\n'.format(v[0], v[1], v[2]))

    with open(json_dir + '/elmtDict.json', 'w') as outfile:
        json.dump(ed, outfile)

    with open(json_dir + '/elmtSetDict.json', 'w') as outfile:
        json.dump(esd, outfile)

    
    abaqusoutput()
    assert os.path.isfile(cwd + name)
    os.remove(cwd + name)'''

    
    

