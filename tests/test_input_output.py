#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

import pytest
import numpy as np

import kanapy
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
                'Simulation': {'periodicity': 'True', 'output_units': 'm'}}

    with open(stat_inp, 'w') as outfile:
        json.dump(to_write, outfile, indent=2)       

    with pytest.raises(ValueError):            
        RVEcreator(to_write, save_files=True)
    os.remove(stat_inp)
                            
    # Test the remaining code
    to_write = {'Grain type': 'Elongated', 'Equivalent diameter': {'std': 0.531055, 'mean': 2.76736, 'cutoff_min': 2.0, 'cutoff_max': 4.0},
                'Aspect ratio': {'std':0.3, 'mean': 2.5, 'cutoff_min': 2.0, 'cutoff_max': 4.0}, 'Tilt angle': {'std': 28.8, 'mean': 87.4, 
                "cutoff_min": 75.0, "cutoff_max": 105.0}, 'RVE': {'sideX': 6, 'sideY': 6, 'sideZ': 6, 'Nx': 15, 'Ny': 15, 'Nz': 15}, 
                'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}    

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
    compare_rd = {'RVE_sizeX': 6, 'RVE_sizeY': 6, 'RVE_sizeZ': 6, 'Voxel_numberX': 15, 'Voxel_numberY': 15, 
                  'Voxel_numberZ': 15, 'Voxel_resolutionX': 0.4, 'Voxel_resolutionY': 0.4, 'Voxel_resolutionZ': 0.4}
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
    write_dump(ells, sbox, len(ells))
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


def test_write_abaqus_inp():

    nd = {1: (1.0, 0.0, 1.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 1.0),
          5: (1.0, 1.0, 1.0), 6: (1.0, 1.0, 0.0), 7: (0.0, 1.0, 0.0), 8: (0.0, 1.0, 1.0),
          9: (2.0, 0.0, 1.0), 10: (2.0, 0.0, 0.0), 11: (2.0, 1.0, 1.0), 12: (2.0, 1.0, 0.0),
          13: (3.0, 0.0, 1.0), 14: (3.0, 0.0, 0.0), 15: (3.0, 1.0, 1.0), 16: (3.0, 1.0, 0.0),
          17: (1.0, 2.0, 1.0), 18: (1.0, 2.0, 0.0), 19: (0.0, 2.0, 0.0), 20: (0.0, 2.0, 1.0),
          21: (2.0, 2.0, 1.0), 22: (2.0, 2.0, 0.0), 23: (3.0, 2.0, 1.0), 24: (3.0, 2.0, 0.0),
          25: (1.0, 3.0, 1.0), 26: (1.0, 3.0, 0.0), 27: (0.0, 3.0, 0.0), 28: (0.0, 3.0, 1.0),
          29: (2.0, 3.0, 1.0), 30: (2.0, 3.0, 0.0), 31: (3.0, 3.0, 1.0), 32: (3.0, 3.0, 0.0),
          33: (1.0, 0.0, 2.0), 34: (0.0, 0.0, 2.0), 35: (1.0, 1.0, 2.0), 36: (0.0, 1.0, 2.0),
          37: (2.0, 0.0, 2.0), 38: (2.0, 1.0, 2.0), 39: (3.0, 0.0, 2.0), 40: (3.0, 1.0, 2.0),
          41: (1.0, 2.0, 2.0), 42: (0.0, 2.0, 2.0), 43: (2.0, 2.0, 2.0), 44: (3.0, 2.0, 2.0),
          45: (1.0, 3.0, 2.0), 46: (0.0, 3.0, 2.0), 47: (2.0, 3.0, 2.0), 48: (3.0, 3.0, 2.0),
          49: (1.0, 0.0, 3.0), 50: (0.0, 0.0, 3.0), 51: (1.0, 1.0, 3.0), 52: (0.0, 1.0, 3.0),
          53: (2.0, 0.0, 3.0), 54: (2.0, 1.0, 3.0), 55: (3.0, 0.0, 3.0), 56: (3.0, 1.0, 3.0),
          57: (1.0, 2.0, 3.0), 58: (0.0, 2.0, 3.0), 59: (2.0, 2.0, 3.0), 60: (3.0, 2.0, 3.0),
          61: (1.0, 3.0, 3.0), 62: (0.0, 3.0, 3.0), 63: (2.0, 3.0, 3.0), 64: (3.0, 3.0, 3.0)}

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

    simData = {'Periodicity': 'True', 'Output units': 'mm'}
    
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'

    # Test if FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        write_abaqus_inp()
    
    # Test the remainder of the code
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/simulation_data.json', 'w') as outfile:
        json.dump(simData, outfile)
        
    with open(json_dir + '/nodeDict.json', 'w') as outfile:
        json.dump(nd, outfile)

    with open(json_dir + '/elmtDict.json', 'w') as outfile:
        json.dump(ed, outfile)

    with open(json_dir + '/elmtSetDict.json', 'w') as outfile:
        json.dump(esd, outfile)

    write_abaqus_inp()
    assert os.path.isfile(cwd + '/kanapy_{0}grains.inp'.format(len(esd)))
    os.remove(cwd + '/kanapy_{0}grains.inp'.format(len(esd)))

