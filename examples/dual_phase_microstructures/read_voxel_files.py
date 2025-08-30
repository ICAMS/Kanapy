#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for loading voxel structures

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

January 2024
"""
import kanapy as knpy
import os

name_d = 'dual_phase_voxels.json'
name_p = 'porous_voxels.json'

if not (os.path.isfile(name_p) or os.path.isfile(name_d)):
    raise FileNotFoundError('Please run pathon scripts "porosity.py" and "dual_phase.py" first to ' +
                            'generate JSON files for voxel structures.')
ms_porous = knpy.import_voxels(name_p)
ms_porous.plot_voxels()
ms_porous.plot_voxels(phases=True)
ms_porous.plot_stats(show_all=True)

ms_dual = knpy.import_voxels(name_d)
ms_dual.plot_voxels()
ms_dual.plot_voxels(phases=True)
ms_dual.generate_grains()
ms_dual.plot_grains()
ms_dual.plot_stats(show_all=True)
