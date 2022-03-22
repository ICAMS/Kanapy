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

def plot_voxels_3D(voxels, voxels_phase = None, Ngr=1, sliced=False, dual_phase=False,
                   mask=None, cmap='prism', alpha=1.0):
    '''
    Plot voxeles in microstructure, each grain with a different color. Sliced 
    indicates whether one eights of the box should be removed to see internal 
    structure. With alpha, the transparency of the voxels can be adjusted.

    Parameters
    ----------
    voxels : int array
        Grain number associated to each voxel
    Ngr : int, optional
        Number of grains. The default is 1.
    sliced : Boolean, optional
        Indicate if one eighth of box is invisible. The default is False.
    dual_phase : Boolean, optional
        Indicate if microstructure is dual phase. The default is False.
    mask : bool array, optional
        Mask for voxels to be displayed. The default is None, in which case
        all voxels will be plotted (except sliced).
    cmap : color map, optional
        Color map for voxels. The default is 'prism'.
    alpha : float, optional
        Adjust transparaency of voxels in alpha channel of color map. 
        The default is 1.0.

    Returns
    -------
    ax : matplotlib.axes
        Axes handle for 3D subplot

    '''
    Nx = voxels.shape[0]
    Ny = voxels.shape[1]
    Nz = voxels.shape[2]

    if dual_phase:
        # phase assignment should be stored in elmtSetDict
        phase_0 = voxels_phase%2==0
        phase_1 = voxels_phase%2==1
        vox_b = phase_0 | phase_1
        colors = np.empty(voxels_phase.shape, dtype=object)
        colors[phase_0] = 'red'
        colors[phase_1] = 'green'
        #cm = plt.cm.get_cmap(cmap, Ngr)
        #colors = cm(voxels_phase.astype(int))
        if sliced:
            ix = int(Nx/2)
            iy = int(Ny/2)
            iz = int(Nz/2)
            vox_b[ix:Nx,iy:Ny,iz:Nz] = False            
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(vox_b, facecolors=colors, edgecolors=colors, shade = True)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ax.set_title('Dual-phase microstructure')
        ax.view_init(30, 30)
        ax.set_xlim(right=Nx)
        ax.set_ylim(top=Ny)
        ax.set_zlim(top=Nz)

    if mask is None:
        vox_b = np.full(voxels.shape, True, dtype=bool)
    else:
        vox_b = mask
    cm = plt.cm.get_cmap(cmap, Ngr)
    colors = cm(voxels.astype(int))
    
    if sliced:
        ix = int(Nx/2)
        iy = int(Ny/2)
        iz = int(Nz/2)
        vox_b[ix:Nx,iy:Ny,iz:Nz] = False

    ax = plt.figure().add_subplot(projection='3d')
    colors[:,:,:,-1] = alpha   # add semitransparency
    ax.voxels(vox_b, facecolors=colors, edgecolors=colors, shade = True)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Voxelated microstructure')
    ax.view_init(30, 30)
    ax.set_xlim(right=Nx)
    ax.set_ylim(top=Ny)
    ax.set_zlim(top=Nz)
    # plt.show()

    return ax

def plot_polygons_3D(grains, cmap='prism', alpha=0.4, ec=[0.5,0.5,0.5,0.1], dual_phase=False):
    '''
    Plot triangularized convex hulls of grains, based on vertices, i.e. 
    connection points of 4 up to 8 grains or the end pointss of triple or quadriiple 
    lines between grains.

    Parameters
    ----------
    grains : dict
        Dictonary with 'vertices' (node numbers) and 'simplices' (triangles)
    nodes : (N,nx,ny,nz)-array
        Nodal positions
    cmap : color map, optional
        Color map for triangles. The default is 'prism'.
    alpha : float, optional
        Transparency of plotted objects. The default is 0.4.
    ec : color, optional
        Color of edges. The default is [0.5,0.5,0.5,0.1].

    Returns
    -------
    None.

    '''
    cm = plt.cm.get_cmap(cmap, len(grains.keys()))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for igr in grains.keys():
        pts = grains[igr]['Points']
        if len(pts) > 6:
            if dual_phase==True:
                if grains[igr]['PhaseID'] == 0:
                    col = 'red'
                else: 
                    col = 'green'
            else:
                col = list(cm(igr))
                col[-1] = alpha   # change alpha channel to create semi-transparency
            ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                            triangles=grains[igr]['Simplices'], color=col, 
                            edgecolor=ec, linewidth=1)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Polygonized microstructure')
    ax.view_init(30, 30)
    plt.show()
    
def plot_ellipsoids_3D(particles, cmap='prism', dual_phase=False):
    '''
    Display ellipsoids during or after packing procedure

    Parameters
    ----------
    particles : Class particles
        Ellipsoids in microstructure before voxelization.
    cmap : color map, optional
        Color map for ellipsoids. The default is 'prism'.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=plt.figaspect(1),dpi=1200) 
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(30, 30)
    
    Npa = len(particles)
    cm = plt.cm.get_cmap(cmap, Npa)
        
    for i, pa in enumerate(particles):
        if dual_phase==True:
            if pa.phasenum == 0:
                col = 'red'
            else: 
                col = 'green'
            #col = cm(pa.phasenum)
        else:
            col = cm(i+1)  # set to 'b' for only blue ellipsoids
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
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=col, linewidth=0)
    plt.show()

        

