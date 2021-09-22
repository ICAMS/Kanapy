#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:56:39 2021

@author: alexander
"""

import kanapy as knpy

ms_def = {'Grain type': 'Elongated', 'Equivalent diameter': {'std': 0.39, 'mean': 2.418, 'cutoff_min': 8.0, 'cutoff_max': 25.0},
            'Aspect ratio': {'std':0.3, 'mean': 0.918, 'cutoff_min': 1.0, 'cutoff_max': 4.0}, 'Tilt angle': {'std': 15.88, 'mean': 87.4, 
            "cutoff_min": 75.0, "cutoff_max": 105.0}, 'RVE': {'sideX': 86, 'sideY': 86, 'sideZ': 86, 'Nx': 65, 'Ny': 65, 'Nz': 65}, 
            'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms = knpy.Microstructure()
ms.create_RVE(ms_def)
ms.pack()
ms.voxelize()
#ms.smoothen()
#ms.abq_output