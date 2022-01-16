#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse grain geometries of polycrystals

@author: alexander Hartmaier, ICAMS / Ruhr-University Bochum, Germany
December 2021

Definitions:
    Vertices: Connection points of 4 up to 8 grains, intersection points of 3 
        or more edges
    Edges: Triple or quadruple lines shared by 3 or 4 grains, respecitvely. 
        Border lines of facets.
    Facets: grain boundary area between 2 grains, described by convex hull of 
        all shared nodes in voxelated structure
"""

import numpy as np
import matplotlib.pyplot as plt
import kanapy as knpy

Nv = 60
size = 70
periodic = False
matname='Simulanium fcc'
matnumber = 4  # UMAT number for fcc Iron
ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 1.0, 'mean': 15.0, 'offs': 6.0, 'cutoff_min': 12.0, 'cutoff_max': 27.0},
          'Aspect ratio': {'std':1.5, 'mean': 1.5, 'offs': 1.0, 'cutoff_min': 1.0, 'cutoff_max': 3.0}, 
          'Tilt angle': {'std': 15., 'mean': 60., "cutoff_min": 0.0, "cutoff_max": 180.0}, 
          'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv}, 
          'Simulation': {'periodicity': str(periodic), 'output_units': 'um'}}

ms = knpy.Microstructure(ms_elong)
ms.init_stats()
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels()
ms.analyze_RVE()
ms.plot_polygons()
ms.plot_stats()

# plot selected grains as voxels and polygons
for igr in range(1,6):
    # plot voxels of grain
    ind = np.array(ms.elmtSetDict[igr]) - 1
    ix = ind % Nv
    ih = np.array((ind - ix) / Nv, dtype=int)
    iy = ih % Nv
    iz = np.array((ih - iy) / Nv, dtype=int)
    mask = np.full(ms.voxels.shape, False, dtype=bool)
    mask[ix,iy,iz] = True
    ax = knpy.plot_voxels_3D(ms.voxels, mask=mask, Ngr=len(ms.elmtSetDict.keys()))
    # plot simplices of grain boundary
    grain = ms.RVE_data['Grains'][igr]
    coord = grain['Points']/ms.RVE_data['Voxel_resolutionX']
    ax.scatter(coord[:, 0], coord[:, 1], 
               coord[:, 2], marker='o', s=30, c='b')
    ax.plot_trisurf(coord[:, 0], coord[:, 1], coord[:, 2],
                    triangles=grain['Simplices'], color=[1, 0, 0, 0.3], 
                    edgecolor=[0.5,0.5,0.5,0.3], linewidth=1)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Grain #{}'.format(igr))
    ax.view_init(30, 60)
    plt.show()

# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='xz', pos='top', data='voxels')
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='yz', pos='top', data='voxels')
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='voxels')
ms.plot_slice(cut='xy', pos='top', data='poly')

# generate grain orientations and write ang file
# requires Kanapy's texture module
if knpy.MTEX_AVAIL:
    ang = [0, 45, 0]    # Euler angles for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = knpy.createOriset(ms.Ngr, ang, omega)
    fname = ms.output_ang(ori=ori_rve, cut='xy', pos='top', data='voxels',
                          matname=matname, plot=False)
    
    # analyze result
    ebsd = knpy.EBSDmap(fname, matname)
    #ebsd.showIPF()
    
    # write Euler angles of grains into Abaqus input file
    knpy.writeAbaqusMat(matnumber, ori_rve)
    
    # write Abaqus input file for voxelated structure
    ms.output_abq('v')