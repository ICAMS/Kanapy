#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create porous microstructure

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

January 2024
"""
import kanapy as knpy
import numpy as np
from math import pi

periodic = True  # create periodic RVE
vf_pores = 0.10  # volume fraction of pores (or precipitates)
name = 'Pores'
nvox = 40  # number of voxels in each Cartesian direction (cuboid RVE)
lside = 30  # side length in micron in each Cartesian direction

ms_stats = {  # statistical data for dense phase
    "Grain type": "Elongated",
    "Equivalent diameter": {
        "sig": 1.0, "scale": 10.0, "loc": 3.5, "cutoff_min": 6.0, "cutoff_max": 8.0},
    "Aspect ratio": {
        "sig": 1.0, "scale": 1.0, "loc": -0.1, "cutoff_min": 0.15, "cutoff_max": 0.85},
    "Tilt angle": {
        "kappa": 1.0, "loc": 0.5*pi, "cutoff_min": 0.0, "cutoff_max": pi},
    "RVE": {
        "sideX": lside, "sideY": lside, "sideZ": lside,
        "Nx": nvox, "Ny": nvox, "Nz": nvox},
    "Simulation": {
        "periodicity": str(periodic), "output_units": "um"},
    "Phase": {
        "Name": name, "Number": 0, "Volume fraction": vf_pores}
}

# Generate microstructure object
ms = knpy.Microstructure(descriptor=ms_stats, name='porous')  # generate microstructure object
ms.plot_stats_init()  # plot initial microstructure statistics and cut-offs for both phases

# Create and visualize synthetic RVE
ms.init_RVE()  # setup RVE geometry
ms.pack(k_rep=0.01, k_att=0.01)  # packing will be stopped when desired volume fraction is reached
ms.plot_ellipsoids()  # plot particles at the end of growth phase
ms.voxelize()  # assigning particles to voxels, empty voxels will be considered as phase 1 (matrix)
ms.plot_voxels(sliced=True, dual_phase=False)  # plot voxel structure, dual_phase=True will plot green/red contrast

# plot voxels of porous phase
mask = np.full(ms.mesh.dim, False, dtype=bool)
for igr, ip in ms.mesh.grain_phase_dict.items():
    if ip == 0:
        for nv in ms.mesh.grain_dict[igr]:
            i, j, k = np.unravel_index(nv-1, ms.mesh.dim, order='F')
            mask[i, j, k] = True
knpy.plot_voxels_3D(ms.mesh.grains, Ngr=ms.Ngr, mask=mask)
#ms.generate_grains()  # construct polyhedral hull for each grain
#ms.plot_grains()  # plot grain structure
#ms.plot_stats()  # plot statistical distribution of grain sizes and aspect ratios and compare to input statistics

# Write voxel structure to JSON file
ms.write_voxels(script_name=__file__, mesh=False, system=False)
