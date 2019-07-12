import os
import shutil

import pytest
import numpy as np
from scipy.spatial import cKDTree

import kanapy
from src.kanapy.input_output import *
from src.kanapy.entities import Ellipsoid, Simulation_Box, Cuboid


def test_particleStatGenerator():

    # create an temporary input file for user defined statistics
    cwd = os.getcwd()
    stat_inp = cwd + '/stat_input.txt'

    to_write = ['@ Equivalent diameter', 'std = 0.531055', 'mean = 2.76736', 'cutoff_min = 2.0', 'cutoff_max = 4.0',
                ' ', '@ Aspect ratio', 'mean = 2.5', ' ', '@ Orientation', 'sigma = 28.8', 'mean = 87.4', ' ',
                '@ RVE', 'side_length = 8', 'voxel_per_side = 15', ' ', '@ Simulation', 'nsteps = 1000', 'periodicity = True']

    with open(stat_inp, 'w') as fd:
        for text in to_write:
            fd.write('{0}\n'.format(text))

    particleStatGenerator(stat_inp)

    # Read the json files written by the function
    json_dir = cwd + '/json_files'
    with open(json_dir + '/particle_data.txt') as json_file:
        pd = json.load(json_file)
    with open(json_dir + '/RVE_data.txt') as json_file:
        rd = json.load(json_file)
    with open(json_dir + '/simulation_data.txt') as json_file:
        sd = json.load(json_file)

    # Dictionaries to verify against
    compare_pd = {'Number': 23, 'Equivalent_diameter': [2.34257201982466, 2.5889424698079426, 
                                    2.861223926202678, 2.861223926202678, 3.1621414733414213, 
                                    3.1621414733414213, 3.1621414733414213, 3.4947067951778186, 
                                    3.4947067951778186, 3.4947067951778186, 3.4947067951778186, 
                                    3.4947067951778186, 3.4947067951778186, 3.862248317231872, 
                                    3.862248317231872, 3.862248317231872, 3.862248317231872, 
                                    3.862248317231872, 3.862248317231872, 3.862248317231872, 
                                    3.862248317231872, 3.862248317231872, 3.862248317231872], 
                                 'Major_diameter': [4.31505455443384, 4.76887280347015, 
                                    5.270419534397091, 5.270419534397091, 5.824714395473459, 
                                    5.824714395473459, 5.824714395473459, 6.437304955973842, 
                                    6.437304955973842, 6.437304955973842, 6.437304955973842, 
                                    6.437304955973842, 6.437304955973842, 7.114322228126523, 
                                    7.114322228126523, 7.114322228126523, 7.114322228126523, 
                                    7.114322228126523, 7.114322228126523, 7.114322228126523, 
                                    7.114322228126523, 7.114322228126523, 7.114322228126523], 
                                'Minor_diameter1': [1.726021821773536, 1.90754912138806, 
                                    2.1081678137588367, 2.1081678137588367, 2.3298857581893833, 
                                    2.3298857581893833, 2.3298857581893833, 2.5749219823895366, 
                                    2.5749219823895366, 2.5749219823895366, 2.5749219823895366, 
                                    2.5749219823895366, 2.5749219823895366, 2.845728891250609, 
                                    2.845728891250609, 2.845728891250609, 2.845728891250609, 
                                    2.845728891250609, 2.845728891250609, 2.845728891250609, 
                                    2.845728891250609, 2.845728891250609, 2.845728891250609], 
                                'Minor_diameter2': [1.726021821773536, 1.90754912138806, 
                                2.1081678137588367, 2.1081678137588367, 2.3298857581893833, 
                                2.3298857581893833, 2.3298857581893833, 2.5749219823895366, 
                                2.5749219823895366, 2.5749219823895366, 2.5749219823895366, 
                                2.5749219823895366, 2.5749219823895366, 2.845728891250609, 
                                2.845728891250609, 2.845728891250609, 2.845728891250609, 
                                2.845728891250609, 2.845728891250609, 2.845728891250609, 
                                2.845728891250609, 2.845728891250609, 2.845728891250609], 
                                
                                'Orientation': [106.60626000152192, 124.66957053792352, 
                                80.56970459162754, 107.95838248673792, 116.5477111808218, 
                                67.6560160145182, 126.18149470917848, 91.5633739708368, 
                                100.15057313553775, 42.5488697344453, 117.77433283603096, 
                                75.0556605235559, 53.398438937765995, 89.29551572293823, 
                                96.36003794723608, 81.06039225691705, 34.55477918647256, 
                                80.89054323025383, 57.39767896614846, 125.34531514146595, 
                                85.1573824453464, 79.66291278991923, 127.29461815001449]}

    compare_rd = {'RVE_size': 8.0, 'Voxel_number_per_side': 15, 
                  'Voxel_resolution': 0.5333333333333333}
    compare_sd = {'Time steps': 1000.0, 'Periodicity': 'True'}

    # Verify
    for k, v in pd.items():
        if k != 'Orientation':              # Don't check for orientation as it is random
            assert pd[k] == compare_pd[k]

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
    gen_sbox, genEll, gen_centerDict, gen_centerTree = read_dump(
        cwd + '/dump_files/particle.{0}.dump'.format(temp_dump.sim_ts))

    assert isinstance(gen_sbox, Cuboid)
    for gel in genEll:
        assert isinstance(gel, Ellipsoid)
    for k, v in gen_centerDict.items():
        assert isinstance(v, Ellipsoid)
    assert isinstance(gen_centerTree, cKDTree)


def test_write_position_weights(temp_dump):

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

    cwd = os.getcwd()
    json_dir = cwd + '/json_files'

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/nodeDict.txt', 'w') as outfile:
        json.dump(nd, outfile)

    with open(json_dir + '/elmtDict.txt', 'w') as outfile:
        json.dump(ed, outfile)

    with open(json_dir + '/elmtSetDict.txt', 'w') as outfile:
        json.dump(esd, outfile)

    write_abaqus_inp()
    assert os.path.isfile(cwd + '/kanapy.inp')
    os.remove(cwd + '/kanapy.inp')


if __name__ == "__main__":
    test_particleStatGenerator()
    test_write_dump()
    test_read_dump()
    test_write_position_weights()
    test_write_abaqus_inp()
