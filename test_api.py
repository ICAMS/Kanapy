#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:56:39 2021

@author: alexander
"""

import kanapy as knpy

ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 0.3, 'mean': 3.0, 'cutoff_min': 12.0, 'cutoff_max': 25.0},
          'Aspect ratio': {'std':0.4, 'mean': 0.8, 'cutoff_min': 0.7, 'cutoff_max': 3.5}, 
          'Tilt angle': {'std': 15.88, 'mean': 80., "cutoff_min": 60.0, "cutoff_max": 100.0}, 
          'RVE': {'sideX': 50, 'sideY': 50, 'sideZ': 50, 'Nx': 50, 'Ny': 50, 'Nz': 50}, 
          'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms_equi = {'Grain type': 'Equiaxed', 
          'Equivalent diameter': {'std': 0.3, 'mean': 3.0, 'cutoff_min': 12.0, 'cutoff_max': 25.0},
          'RVE': {'sideX': 50, 'sideY': 50, 'sideZ': 50, 'Nx': 50, 'Ny': 50, 'Nz': 50}, 
          'Simulation': {'periodicity': 'True', 'output_units': 'mm'}}
ms_ex_sphere = { "Grain type": "Equiaxed",
  "Equivalent diameter":  { "std": 0.39, "mean": 2.418, "cutoff_min": 8.0, "cutoff_max": 25.0},
  "RVE": {"sideX": 85.9, "sideY": 85.9, "sideZ": 85.9, "Nx": 40, "Ny": 40, "Nz": 40},
  "Simulation": { "periodicity": "True", "output_units": "mm"}}
ms_ex_ellips = {"Grain type": "Elongated",
  "Equivalent diameter": {"std": 0.39, "mean": 2.418, "cutoff_min": 8.0, "cutoff_max": 25.0},
  "Aspect ratio": {"std": 0.3, "mean": 0.918, "cutoff_min": 1.0, "cutoff_max": 4.0},           
  "Tilt angle":{"std": 15.8774, "mean": 87.4178,  "cutoff_min": 75.0, "cutoff_max": 105.0},
  "RVE": {"sideX": 86, "sideY": 86, "sideZ": 86, "Nx": 65, "Ny": 65, "Nz": 65},
  "Simulation":{"periodicity": "True", "output_units": "mm"}}

ms = knpy.Microstructure(ms_elong)
#ms = knpy.Microstructure(file="examples/ellipsoid_packing/stat_input.json")
ms.create_stats()
ms.create_RVE(save_files=False)
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_3D()
ms.output_stats()
ms.plot_stats()
ms.smoothen()
ms.abq_output()
ms.output_neper()