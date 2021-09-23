#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:56:39 2021

@author: alexander
"""

import kanapy as knpy

ms_def = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 0.4, 'mean': 10.0, 'cutoff_min': 5.0, 'cutoff_max': 15.0},
          'Aspect ratio': {'std':0.2, 'mean': 1.0, 'cutoff_min': 0.8, 'cutoff_max': 2.0}, 
          'Tilt angle': {'std': 15.88, 'mean': 87.4, "cutoff_min": 75.0, "cutoff_max": 105.0}, 
          'RVE': {'sideX': 50, 'sideY': 50, 'sideZ': 50, 'Nx': 25, 'Ny': 25, 'Nz': 25}, 
          'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms = knpy.Microstructure(ms_def)
#ms = knpy.Microstructure(file="examples/ellipsoid_packing/stat_input.json")
ms.create_RVE()
ms.pack()
ms.voxelize()
#ms.smoothen()
#ms.abq_output