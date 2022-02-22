#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse grain geometries of polycrystals, extract statistical information 
form 2D slices, comaparable to experiment, fit ellipsoids to 3D grains 
(polyhedra) and exctract statistical information

@author: alexander Hartmaier, ICAMS / Ruhr-University Bochum, Germany
January 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import kanapy as knpy

matname = 'Simulanium fcc'
matnumber = 4  # UMAT number for fcc Iron
Nv = 60
size = 70
periodic = True
ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'std': 1.0, 'mean': 20.0, 'offs': 12.0, 'cutoff_min': 18.0, 'cutoff_max': 25.0},
          'Aspect ratio': {'std':1.5, 'mean': 1.8, 'offs': 1.0, 'cutoff_min': 1.0, 'cutoff_max': 3.0}, 
          'Tilt angle': {'std': 15., 'mean': 60., "cutoff_min": 0.0, "cutoff_max": 180.0}, 
          'RVE': {'sideX': size, 'sideY':size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv}, 
          'Simulation': {'periodicity': str(periodic), 'output_units': 'um'},
          'Phase': {'Name': 'Simulanium fcc', 'Number': 0, 'Volume fraction': 1.0}}

ms_elong = [ms_elong]

ms = knpy.Microstructure(ms_elong)
ms.init_stats()
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels()
ms.analyze_RVE()
ms.plot_polygons()

# plot selected grains as voxels and polygons
ndisp = 0
igr = 1
while ndisp < 10 and igr <= ms.Ngr:
    # select split grains
    if np.any(ms.RVE_data['Grains'][igr]['Split']):
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
        ndisp += 1
    igr += 1

# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='poly')

# Test cases for misorientation distribution (MDF)
'''adapted from 
Miodownik, M., et al. "On boundary misorientation distribution functions 
and how to incorporate them into three-dimensional models of microstructural 
evolution." Acta materialia 47.9 (1999): 2661-2668.
https://doi.org/10.1016/S1359-6454(99)00137-8
'''
if knpy.MTEX_AVAIL:
    mdf_freq = {
      "High": [0.0013303016, 0.208758929, 0.3783092708, 0.7575794912, 1.6903069613,
               2.5798481069, 5.0380640643, 10.4289690569, 21.892113657, 21.0,
               22.1246762077, 13.9000439533],
      "Low":  [4.5317, 18.6383, 25, 20.755, 12.5984, 7.2646, 4.2648, 3.0372, 2.5,
               1, 0.31, 0.1],
      "Random": [0.1, 0.67, 1.9, 3.65, 5.8, 8.8, 11.5, 15.5, 20, 16.7, 11.18, 4.2] 
    }
    mdf_bins = np.linspace(62.8/12,62.8,12)
    
    # generate grain orientations and write ang file
    ang = [0, 45, 0]    # Euler angle for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = knpy.createOriset(ms.Ngr, ang, omega, hist=mdf_freq['Low'],
                                shared_area=ms.shared_area)
    fname = ms.output_ang(ori=ori_rve, matname=matname, cut='xy', data='poly',
                          plot=False)
    # analyze result
    ebsd = knpy.EBSDmap(fname, matname)
    #ebsd.showIPF()
    
    # write Euler angles of grains into Abaqus input file
    knpy.writeAbaqusMat(matnumber, ori_rve)
    
    # write Abaqus input file for voxelated structure
    ms.output_abq('v')
