00#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse grain geometries of polycrystals, extract statistical information
form 2D slices, comaparable to experiment, fit ellipsoids to 3D grains
(polyhedra) and exctract statistical information

@author: Alexander Hartmaier, ICAMS / Ruhr-University Bochum, Germany
August 2022

"""

import numpy as np
import kanapy as knpy

texture = 'random'  # Implemented textures are goss, copper, random
matname = 'Simulanium fcc'
matnumber = 4  # UMAT number for fcc Iron
Nv = 30
size = 40
periodic = False
ms_elong = {'Grain type': 'Elongated',
          'Equivalent diameter': {'std': 1.0, 'mean': 12.0, 'offs': 4.0, 'cutoff_min': 8.0, 'cutoff_max': 18.0},
          'Aspect ratio': {'std':1.5, 'mean': 2.0, 'offs': 0.8, 'cutoff_min': 1.0, 'cutoff_max': 4.0},
          'Tilt angle': {'std': 15., 'mean': 90., "cutoff_min": 0.0, "cutoff_max": 180.0},
          'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv},
          'Simulation': {'periodicity': str(periodic), 'output_units': 'mm'}}

ms = knpy.Microstructure(ms_elong)
ms.init_stats()
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels()
ms.generate_grains()
ms.plot_grains()

# compare microstructure on three surfaces
# for voxelated and polygonalized grains
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='poly')

# Test cases for misorientation distribution (MDF)
'''adapted from
Miodownik, M., et al. "On boundary misorientation distribution functions
and how to incorporate them into three-dimensional models of microstructural
evolution." Acta materialia 47.9 (1999): 2661-2668.
https://doi.org/10.1016/S1359-6454(99)00137-8
'''
if knpy.MTEX_AVAIL:
    mdf_freq = {
      "High": [0.0013303016, 0.208758929, 0.3783092708, 0.7575794912, 1.6903069613,
               2.5798481069, 5.0380640643, 10.4289690569, 21.892113657, 21.0,
               22.1246762077, 13.9000439533],
      "Low":  [4.5317, 18.6383, 25, 20.755, 12.5984, 7.2646, 4.2648, 3.0372, 2.5,
               1, 0.31, 0.1],
      "Random": [0.1, 0.67, 1.9, 3.65, 5.8, 8.8, 11.5, 15.5, 20, 16.7, 11.18, 4.2]
    }
    mdf_bins = np.linspace(62.8/12,62.8,12)

    # generate grain orientations and write ang file
    '''
    Different textures can be choosen and assinged to the RVE geometry that has
    been defined above.
    Texture is defined by the orientation of the ideal component in Euler space
    ang and a kernel halfwidth omega. Kernel used here is deLaValleePoussin.
    The function createOriset will first create an artificial EBSD by sampling
    a large number of discrete orientations from the ODF defined by ang and
    omega. Then a reduction method is applied to reconstruct this initial ODF
    with a smaller number of discrete orientations Ngr. The reduced set of
    orientations is returned by the function.
    For more information on the method see:
    https://doi.org/10.1107/S1600576719017138
    '''
    if texture == 'goss':
        ang = [0, 45, 0]    # Euler angle for Goss texture
        omega = 7.5         # kernel half-width
    elif texture == 'copper':
        ang = [90, 35, 45]
        omega = 7.5
    elif texture == 'random':
        #For Random texture, no subsampling and ODF recreation is neccessary
        ori_rve = knpy.createOrisetRandom(ms.Ngr)
    else:
        raise ValueError('texture not defined. Take goss, copper or random')
    if texture != 'random':
        ori_rve = knpy.createOriset(ms.Ngr, ang, omega, hist=mdf_freq['Low'],
                                    shared_area=ms.shared_area)
    fname = ms.output_ang(ori=ori_rve, matname=matname, cut='xy', data='poly',
                          plot=False)
    # analyze result
    ebsd = knpy.EBSDmap(fname, matname)
    #ebsd.showIPF()

    # write Euler angles of grains into Abaqus input file
    knpy.writeAbaqusMat(matnumber, ori_rve)

    # write Abaqus input file for voxelated structure
    ms.output_abq('v')
