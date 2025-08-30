#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create RVE with regular paatern of rectangular cuboid grains and
predefined crystallographic texture.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

February 2024
"""
import kanapy as knpy
import numpy as np
import itertools
from kanapy.core.initializations import RVE_creator, mesh_creator
from kanapy.core.entities import Simulation_Box

# define basic parameters of RVE
texture = 'unimodal'  # chose texture type 'random' or 'unimodal'
matname = 'Simulanium'  # Material name
ngr = (5, 5, 5)  # number of grains per axis
nv_gr = (3, 3, 3)  # number of voxels per grain dimension
dim = (ngr[0]*nv_gr[0], ngr[1]*nv_gr[1], ngr[2]*nv_gr[2])  # number of voxels per axis
size = (45, 45, 45)  # size of RVE in micron
nphases = 1
periodic = False

# define basic stats dict
stats_dict = {
    'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
            'Nx': dim[0], 'Ny': dim[1], 'Nz': dim[2]},
    'Simulation': {'periodicity': periodic,
                   'output_units': 'um'},
    'Phase': {'Name': matname, 'Volume fraction': 1.0}
}

# Create microstructure object
ms = knpy.Microstructure('from_voxels')
ms.name = matname
ms.Ngr = np.prod(ngr)
ms.nphases = 1
ms.descriptor = [stats_dict]
ms.ngrains = [ms.Ngr]
ms.rve = RVE_creator(ms.descriptor, from_voxels=True)
ms.simbox = Simulation_Box(size)

# initialize voxel structure (= mesh)
ms.mesh = mesh_creator(dim)
ms.mesh.create_voxels(ms.simbox)

# assign voxels to grains (-> grain_dict) and create grains-array
grains = np.zeros(dim, dtype=int)
grain_dict = dict()
grain_phase_dict = dict()
for ih in range(ngr[0]):
    for ik in range(ngr[1]):
        for il in range(ngr[2]):
            igr = il + ik*ngr[1] + ih*ngr[0]*ngr[1] + 1
            grain_dict[igr] = []
            grain_phase_dict[igr] = 0
            ind0 = np.arange(nv_gr[0], dtype=int) + ih*nv_gr[0]
            ind1 = np.arange(nv_gr[1], dtype=int) + ik*nv_gr[1]
            ind2 = np.arange(nv_gr[2], dtype=int) + il*nv_gr[2]
            listOLists = [ind0, ind1, ind2]
            ind_list = itertools.product(*listOLists)
            for ind in ind_list:
                nv = np.ravel_multi_index(ind, dim, order='F')
                grain_dict[igr].append(nv + 1)
            grains[ind0[0]:ind0[-1]+1,
                   ind1[0]:ind1[-1]+1,
                   ind2[0]:ind2[-1]+1] = igr

ms.mesh.grains = grains
ms.mesh.grain_dict = grain_dict
ms.mesh.phases = np.zeros(dim, dtype=int)
ms.mesh.grain_phase_dict = grain_phase_dict
ms.mesh.ngrains_phase = ms.ngrains

# Done. Plot voxelated grains
ms.plot_voxels()

# create grain orientations for unimodal Goss texture
# Note: angles are given in degrees
ms.generate_orientations(texture, ang=[0, 45, 0], omega=7.5)

# output rve in voxel format
ms.write_voxels(script_name=__file__, mesh=False, system=False)
