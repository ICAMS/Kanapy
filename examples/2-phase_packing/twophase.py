#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:20:56 2022

@author: golsatolooeieshlaghi
"""

import kanapy as knpy
from math import pi
import json
import csv
import numpy as np

with open('./ms_stats_0.json') as json_file:
    ms_stats_0 = json.load(json_file)

with open('./ms_stats_1.json') as json_file:
    ms_stats_1 = json.load(json_file)

# ms stats should be included in a list.
ms_stats = [ms_stats_0, ms_stats_1]

ms = knpy.Microstructure(descriptor=ms_stats, name='RVE')
ms.init_stats()

# create and visualize synthetic RVE
ms.init_RVE()
ms.pack(k_rep=0.0, k_att=0.0)
#ms.pack(k_rep=0.1, k_att=0.1)
ms.plot_ellipsoids(dual_phase=True)
ms.voxelize(dual_phase=True)
ms.plot_voxels(sliced=True,dual_phase=True)
ms.plot_slice(cut='xz', pos='top', data='voxels', dual_phase=True)
ms.plot_slice(cut='yz', pos='top', data='voxels', dual_phase=True)
ms.plot_slice(cut='xy', pos='top', data='voxels', dual_phase=True)
ms.generate_grains(dual_phase=True)
ms.plot_grains(dual_phase=True)
ms.plot_stats(dual_phase=True)

# # compare microstructure on three surfaces 
# # for voxelated and polygonalized grains
# ms.plot_slice(cut='xz', pos='top', data='poly', dual_phase=False)
# ms.plot_slice(cut='yz', pos='top', data='poly', dual_phase=False)
# ms.plot_slice(cut='xy', pos='top', data='poly', dual_phase=False)

# write Abaqus input file for voxelated structure
ms.output_abq('v')
# write phaseID of grains into Abaqus input file
knpy.writeAbaqusPhase(ms.RVE_data['Grains'])

# smoothen voxelated structure and write Abaqus .inp file
ms.smoothen()
ms.output_abq('s')


