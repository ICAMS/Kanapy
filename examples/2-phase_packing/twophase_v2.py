#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 2-phase microstructure

Authors: Golsa Tolooei Eshlaghi, Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

December 2023
"""
import kanapy as knpy
from math import pi
import matplotlib
import logging
logging.basicConfig(level=20)
#matplotlib.use('MacOSX')

periodic = False
vf0 = 0.7
vf1 = 1. - vf0
name0 = 'Austenite'
name1 = 'Martensite'
nvox = 40
lside = 50

ms_stats_0 = {
    "Grain type": "Elongated",
    "Equivalent diameter": {
        "std": 1.1, "mean": 12.7, "offs": 8.0, "cutoff_min": 10.0, "cutoff_max": 20.0},
    "Aspect ratio": {
        "std": 0.75, "mean": 1.6, "offs": 0.87, "cutoff_min": 1.0, "cutoff_max": 3.0},
    "Tilt angle": {
        "std": 0.5, "mean": 1.4, "cutoff_min": 0.0, "cutoff_max": 2*pi},
    "RVE": {
        "sideX": lside, "sideY": lside, "sideZ": lside,
        "Nx": nvox, "Ny": nvox, "Nz": nvox},
    "Simulation": {
        "periodicity": str(periodic), "output_units": "um"},
    "Phase": {
        "Name": name0, "Number": 0, "Volume fraction": vf0}
}

ms_stats_1 = {
    "Grain type": "Elongated",
    "Equivalent diameter": {
        "std": 1.1, "mean": 6.0, "offs": 4.0, "cutoff_min": 5.0, "cutoff_max": 11.0},
    "Aspect ratio": {
        "std": 0.72, "mean": 2.4, "offs": 1.0, "cutoff_min": 1.2, "cutoff_max": 5.0},
    "Tilt angle": {
        "std": 0.54, "mean": 1.4, "cutoff_min": 0.0, "cutoff_max": 2*pi},
    "RVE": {
        "sideX": lside, "sideY": lside, "sideZ": lside,
        "Nx": nvox, "Ny": nvox, "Nz": nvox},
    "Simulation": {
        "periodicity": str(periodic), "output_units": "um"},
    "Phase": {
        "Name": name1, "Number": 1, "Volume fraction": vf1}
}

# Generate microstructure object
ms = knpy.Microstructure(descriptor=[ms_stats_0, ms_stats_1], name='dual_phase')
ms.plot_stats_init()

# create and visualize synthetic RVE
ms.init_RVE()
ms.pack(k_rep=0.01, k_att=0.01)
ms.plot_ellipsoids(dual_phase=True)
ms.voxelize()
ms.plot_voxels(sliced=True, dual_phase=True)
#ms.plot_slice(cut='xz', pos='top', data='voxels', dual_phase=True)
ms.generate_grains()
ms.plot_grains(dual_phase=True)
ms.plot_stats()

ms.write_voxels(script_name=__file__)