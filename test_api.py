#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:56:39 2021

@author: alexander
"""

import kanapy as knpy

ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 0.3, 'mean': 3.0, 'cutoff_min': 10.0, 'cutoff_max': 25.0},
          'Aspect ratio': {'std':0.4, 'mean': 0.1, 'cutoff_min': 0.7, 'cutoff_max': 1.4}, 
          'Tilt angle': {'std': 15.88, 'mean': 87.4, "cutoff_min": 75.0, "cutoff_max": 105.0}, 
          'RVE': {'sideX': 50, 'sideY': 50, 'sideZ': 50, 'Nx': 50, 'Ny': 50, 'Nz': 50}, 
          'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms_equi = {'Grain type': 'Equiaxed', 
          'Equivalent diameter': {'std': 0.3, 'mean': 3.0, 'cutoff_min': 10.0, 'cutoff_max': 25.0},
          'RVE': {'sideX': 50, 'sideY': 50, 'sideZ': 50, 'Nx': 50, 'Ny': 50, 'Nz': 50}, 
          'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms = knpy.Microstructure(ms_elong)
#ms = knpy.Microstructure(file="examples/ellipsoid_packing/stat_input.json")
ms.create_stats()
ms.create_RVE(save_files=False)
ms.pack()
ms.voxelize()
#ms.smoothen()
#ms.abq_output