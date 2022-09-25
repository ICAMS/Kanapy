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
ms.write_voxels(__file__, file='test.json', source=ms_elong)

ms2 = knpy.import_voxels('test.json')
ms2.plot_voxels()
ms2.generate_grains()
ms2.plot_grains()
# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='xz', pos='top', data='voxels')
ms2.plot_slice(cut='xz', pos='top', data='voxels')
ms.plot_slice(cut='yz', pos='top', data='poly')
ms2.plot_slice(cut='yz', pos='top', data='poly')

ms.output_abq()
ms2.output_abq()