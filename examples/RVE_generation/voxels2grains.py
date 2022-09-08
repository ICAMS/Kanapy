#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse grain geometries of polycrystals

@author: alexander Hartmaier, ICAMS / Ruhr-University Bochum, Germany
August 2022

"""

import kanapy as knpy

Nv = 40
size = 50
periodic = True
matname='Simulanium fcc'
matnumber = 4  # UMAT number for fcc Iron
ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 1.0, 'mean': 15.0, 'offs': 7.0, 'cutoff_min': 10.0, 'cutoff_max': 20.0},
          'Aspect ratio': {'std':1.5, 'mean': 1.5, 'offs': 1.0, 'cutoff_min': 1.0, 'cutoff_max': 3.0},
          'Tilt angle': {'std': 15., 'mean': 60., "cutoff_min": 0.0, "cutoff_max": 180.0}, 
          'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv}, 
          'Simulation': {'periodicity': str(periodic), 'output_units': 'um'},
          'Phase': {'Name': 'Simulanium fcc', 'Number': 0, 'Volume fraction': 1.0}}

ms = knpy.Microstructure(ms_elong)
ms.init_stats()
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels()
ms.generate_grains()
ms.plot_grains()
ms.plot_stats()

# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='xz', pos='top', data='voxels')
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='yz', pos='top', data='voxels')
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='voxels')
ms.plot_slice(cut='xy', pos='top', data='poly')

# generate grain orientations and write ang file
# requires Kanapy's texture module
if knpy.MTEX_AVAIL:
    ang = [0, 45, 0]    # Euler angles for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = knpy.createOriset(ms.Ngr, ang, omega)
    fname = ms.output_ang(ori=ori_rve, cut='xy', pos='top', data='voxels',
                          matname=matname, plot=False)
    
    # analyze result
    ebsd = knpy.EBSDmap(fname, matname)
    #ebsd.showIPF()
    
    # write Euler angles of grains into Abaqus input file
    knpy.writeAbaqusMat(matnumber, ori_rve)
    
    # write Abaqus input file for voxelated structure
    ms.output_abq('v')