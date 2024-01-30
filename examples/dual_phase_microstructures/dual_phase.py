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

periodic = True  # create periodic RVE
vf0 = 0.75  # volume fraction of phase 1 (dense phase)
vf1 = 1. - vf0  # volume fraction of phase 1 (porosity)
name0 = 'Austenite'
name1 = 'Martensite'
nvox = 30  # number of voxels in each Cartesian direction (cuboid RVE)
lside = 40  # side length in micron in each Cartesian direction

ms_stats_0 = {  # statistical data for dense phase
    "Grain type": "Elongated",
    "Equivalent diameter": {
        "sig": 1.0, "scale": 12.0, "loc": 5.0, "cutoff_min": 8.0, "cutoff_max": 14.0},
    "Aspect ratio": {
        "sig": 0.5, "scale": 1.0, "loc": 0.6, "cutoff_min": 1.0, "cutoff_max": 2.4},
    "Tilt angle": {
        "kappa": 1.0, "loc": 0.5*pi, "cutoff_min": 0.0, "cutoff_max": 2*pi},
    "RVE": {
        "sideX": lside, "sideY": lside, "sideZ": lside,
        "Nx": nvox, "Ny": nvox, "Nz": nvox},
    "Simulation": {
        "periodicity": str(periodic), "output_units": "um"},
    "Phase": {
        "Name": name0, "Number": 0, "Volume fraction": vf0}
}

ms_stats_1 = {  # statistical data for porosity (will not be considered explicitly)
    "Grain type": "Elongated",
    "Equivalent diameter": {
        "sig": 1.5, "scale": 9.0, "loc": 6.0, "cutoff_min": 8.0, "cutoff_max": 12.0},
    "Aspect ratio": {
        "sig": 1.0, "scale": 3.5, "loc": 1.0, "cutoff_min": 1.6, "cutoff_max": 5.0},
    "Tilt angle": {
        "kappa": 1.5, "loc": 0.5*pi, "cutoff_min": 0.0, "cutoff_max": 2*pi},
    "RVE": {
        "sideX": lside, "sideY": lside, "sideZ": lside,
        "Nx": nvox, "Ny": nvox, "Nz": nvox},
    "Simulation": {
        "periodicity": str(periodic), "output_units": "um"},
    "Phase": {
        "Name": name1, "Number": 1, "Volume fraction": vf1}
}

# Generate microstructure object
ms = knpy.Microstructure(descriptor=[ms_stats_0, ms_stats_1], name='dual_phase')  # generate microstructure object
ms.plot_stats_init()  # plot initial microstructure statistics and cut-offs for both phases

# Create and visualize synthetic RVE
ms.init_RVE()  # initialize RVE and generate particle distribution according to statistical data
ms.pack(k_rep=0.01, k_att=0.01)  # packing will be stopped when desired volume fraction is reached
ms.plot_ellipsoids(dual_phase=True)  # plot particles at the end of growth phase (dual_phase=False plots colored ellips)
ms.voxelize()  # assigning particles to voxels, empty voxels will be considered as phase 1 (porosity)
ms.plot_voxels(sliced=True, dual_phase=True)  # plot voxels, dual_phase=False will plot colored voxels for phase 0
ms.generate_grains()  # construct polyhedral hull for each grain
ms.plot_grains(dual_phase=True)  # plot grain structure (dual_phase=True will plot colored grains for phase 0)
ms.plot_stats()  # plot statistical distribution of grain sizes and aspect ratios and compare to input statistics

# Write voxel structure to JSON file
ms.write_voxels(script_name=__file__, mesh=False, system=False)
