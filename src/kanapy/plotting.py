#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subroutines for plotting of microstructures

Created on Mon Oct  4 07:52:55 2021

@author: Alexander Hartmaier, Golsa Tolooei Eshlaghi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_microstructure_3D(voxels, Ngr=1, sliced=True, dual_phase=False, \
                           cmap='prism', test=False):

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

def plot_ellipsoids(particles, cmap='prism', test=False):
              
    fig = plt.figure(figsize=plt.figaspect(1),dpi=1200) 
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(30, 30)
    
    #Npa = len(particles)
        
    for pa in particles:
        qw, qx, qy, qz = pa.quat
        x_c, y_c, z_c, a, b, c = pa.x, pa.y, pa.z, pa.a, pa.b, pa.c
        #Rotation
        r = R.from_quat([qx, qy, qz, qw])
        #Local coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_local = (a * np.outer(np.cos(u), np.sin(v))).reshape((10000,))
        y_local = (b * np.outer(np.sin(u), np.sin(v))).reshape((10000,))
        z_local = (c * np.outer(np.ones_like(u), np.cos(v))).reshape((10000,))
        points_local = list(np.array([x_local, y_local, z_local]).transpose())
        #Global coordinates
        points_global = r.apply(points_local, inverse = True) 
        x = (points_global[:,0] + np.ones_like(points_global[:,0])*x_c).reshape((100,100))
        y = (points_global[:,1] + np.ones_like(points_global[:,1])*y_c).reshape((100,100))
        z = (points_global[:,2] + np.ones_like(points_global[:,2])*z_c).reshape((100,100))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=.0001)
    plt.show()

        

