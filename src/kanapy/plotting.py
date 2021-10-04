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
        Nx = ms.RVE_data['RVE_sizeX']
        Ny = ms.RVE_data['RVE_sizeY']
        Nz = ms.RVE_data['RVE_sizeZ']
        Ngr = ms.particle_data['Number']
        hh = np.zeros(Nx*Ny*Nz)
        for ir in range(ms.particle_data['Number']):
            ih = ir + 1
            il = np.array(ms.elmtSetDict[ih]) - 1
            hh[il] = ih
        voxels = np.reshape(hh, (Nx,Ny,Nz), order='F')
        
        if test:
            # test consistency of voxel array with Element dict
            for ir in range(Ngr):
                igr = ir + 1
                iel = ms.elmtSetDict[igr]
                ic = 0
                for i in iel:
                    iv = ms.vox_centerDict[i]
                    if voxels[int(iv[0]), int(iv[1]), int(iv[2])] != igr:
                        print('Wrong voxel number',ih, i, iv)
                    else:
                        ic += 1
                if ic==len(iel):
                    print('Correct voxel assignment to grain',igr)
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
    return cm