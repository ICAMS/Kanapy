#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subroutines for plotting of microstructures

Created on Mon Oct  4 07:52:55 2021

@author: Alexander Hartmaier, Golsa Tolooei Eshlaghi
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_microstructure_3D(ms=None, voxels=None, sliced=True, dual_phase=False, \
                           cmap='prism', test=False):
    if voxels is None:
        if ms is None:
            raise ValueError('Either microstructure instance (ms) or voxel array must be given for plotting.')
        lx = int(ms.RVE_data['RVE_sizeX'])
        ly = int(ms.RVE_data['RVE_sizeY'])
        lz = int(ms.RVE_data['RVE_sizeZ'])
        Nx = int(ms.RVE_data['Voxel_numberX'])
        Ny = int(ms.RVE_data['Voxel_numberY'])
        Nz = int(ms.RVE_data['Voxel_numberZ'])
        Ngr = int(ms.particle_data['Number'])
        hh = np.zeros(Nx*Ny*Nz)
        for ih, il in ms.elmtSetDict.items():
            il = np.array(il) - 1
            hh[il] = ih
        voxels = np.reshape(hh, (Nx,Ny,Nz), order='F')
        
        if test:
            # test consistency of voxel array with element dict
            for igr, iel in ms.elmtSetDict.items():
                ic = 0
                for i in iel:
                    iv = ms.vox_centerDict[i]
                    ih = voxels[int(Nx*iv[0]/lx), int(Ny*iv[1]/ly), int(Nz*iv[2]/lz)]
                    if ih != igr:
                        print('Wrong voxel number',igr, ih, i, iv)
                    else:
                        ic += 1
                if ic==len(iel):
                    print('Correct voxel assignment to grain:',igr)
    else:
        Nx = voxels.shape[0]
        Ny = voxels.shape[1]
        Nz = voxels.shape[2]
    
    if dual_phase:
        # phase assignment should be stored in elmtSetDict
        phase_0 = voxels%2==0
        phase_1 = voxels%2==1
        vox_b = phase_0 | phase_1
        colors = np.empty(voxels.shape, dtype=object)
        colors[phase_0] = 'blue'
        colors[phase_1] = 'red'
    else:
        vox_b = np.full(voxels.shape, True, dtype=bool)
        cm = plt.cm.get_cmap(cmap, Ngr)
        colors = cm(voxels.astype(int))
    
    if sliced:
        ix = int(Nx/2)
        iy = int(Ny/2)
        iz = int(Nz/2)
        vox_b[ix:Nx,iy:Ny,iz:Nz] = False

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(vox_b, facecolors=colors, edgecolors=colors, shade = True)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    
    ax.view_init(30, 30)
    ax.set_xlim(right=Nx)
    ax.set_ylim(top=Ny)
    ax.set_zlim(top=Nz)
    plt.show()

    return 