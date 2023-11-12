#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subroutines for plotting of microstructures

Created on Mon Oct  4 07:52:55 2021

@author: Alexander Hartmaier, Golsa Tolooei Eshlaghi
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import lognorm


def plot_voxels_3D(voxels, voxels_phase = None, Ngr=1, sliced=False, dual_phase=False,
                   mask=None, cmap='prism', alpha=1.0, show=True):
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
    show : bool
        Indicate whether to show the plot or to return the axis for further use

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
    if show:
        plt.show()
        return
    else:
        return ax

def plot_polygons_3D(rve, cmap='prism', alpha=0.4, ec=[0.5,0.5,0.5,0.1],
                     dual_phase=False):
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
    grains = rve['Grains']
    pts = rve['Points']
    Ng = np.amax(list(grains.keys()))
    cm = plt.cm.get_cmap(cmap, Ng)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for igr in grains.keys():
        if not grains[igr]['Simplices']:
            continue
        if dual_phase:
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
        if pa.duplicate is not None:
            continue
        if dual_phase:
            if pa.phasenum == 0:
                col = 'red'
            else: 
                col = 'green'
            #col = cm(pa.phasenum)
        else:
            col = cm(i+1)  # set to 'b' for only blue ellipsoids
        qw, qx, qy, qz = pa.quat
        x_c, y_c, z_c = pa.x, pa.y, pa.z
        a, b, c = pa.a, pa.b, pa.c
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


def plot_output_stats(dataDict, gs_data=None, gs_param=None,
                      ar_data=None, ar_param=None, save_files=False):
    r"""
    Evaluates particle- and output RVE grain statistics with respect to Major, Minor & Equivalent diameters and plots the distributions
    """
    print('')
    print('Plotting input & output statistics')
    cwd = os.getcwd()

    par_eqDia = np.sort(np.asarray(dataDict['Particle_Equivalent_diameter']))
    grain_eqDia = np.sort(np.asarray(dataDict['Grain_Equivalent_diameter']))

    # Convert to micro meter for plotting
    if dataDict['Unit_scale'] == 'mm':
        par_eqDia *= 1.e-3
        grain_eqDia *= 1.e-3

    # Concatenate both arrays to compute shared bins
    # NOTE: 'doane' produces better estimates for non-normal datasets
    total_eqDia = np.concatenate([par_eqDia, grain_eqDia])
    shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')

    # Get the mean & std of the underlying normal distribution
    par_data = np.log(par_eqDia)
    mu_par = np.mean(par_data)
    std_par = np.std(par_data)
    ind = np.nonzero(grain_eqDia > 1.e-5)[0]
    grain_data = np.log(grain_eqDia[ind])
    mu_gr = np.mean(grain_data)
    std_gr = np.std(grain_data)

    # NOTE: lognorm takes mean & std of normal distribution
    par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par))
    grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr))

    # read the data from the file
    if dataDict['Grain type'] == 'Equiaxed':
        # Plot the histogram & PDF
        sns.set(color_codes=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))

        # Plot histogram
        ax[0].hist([par_eqDia, grain_eqDia], density=False, bins=len(shared_bins), label=['Input', 'Output'])
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)

        # Plot PDF
        ypdf1 = par_lognorm.pdf(par_eqDia)
        ypdf2 = grain_lognorm.pdf(grain_eqDia)
        ax[1].plot(par_eqDia, ypdf1, linestyle='-', linewidth=3.0, label='Input')
        ax[1].fill_between(par_eqDia, 0, ypdf1, alpha=0.3)
        ax[1].plot(grain_eqDia, ypdf2, linestyle='-', linewidth=3.0, label='Output')
        ax[1].fill_between(grain_eqDia, 0, ypdf2, alpha=0.3)

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if save_files:
            plt.savefig(cwd + "/Equivalent_diameter.png", bbox_inches="tight")
        plt.show()


    elif dataDict['Grain type'] == 'Elongated':
        total_eqDia = np.concatenate([par_eqDia, grain_eqDia])
        # Find the corresponding shared bin edges
        # NOTE: 'doane' produces better estimates for non-normal datasets
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')
        binNum = len(shared_bins)
        name = 'Equivalent'

        # Plot the histogram & PDF
        sns.set(color_codes=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        data = [par_eqDia, grain_eqDia]
        label = ['Input', 'Output']
        if gs_data is not None:
            data.append(gs_data)
            label.append('Experiment')
        # Plot histogram
        ax[0].hist(data, density=False, bins=binNum, label=label)
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('{} diameter (μm)'.format(name), fontsize=18)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)

        # Plot PDF
        ypdf1 = par_lognorm.pdf(par_eqDia)
        area = np.trapz(ypdf1, par_eqDia)
        ypdf1 /= area
        ypdf2 = grain_lognorm.pdf(grain_eqDia)
        area = np.trapz(ypdf2, grain_eqDia)
        ypdf2 /= area
        if gs_param is not None:
            x0 = np.minimum(np.amin(grain_eqDia), np.amin(par_eqDia))
            x1 = np.maximum(np.amax(grain_eqDia), np.amax(par_eqDia))
            x = np.linspace(x0, x1, num=50)
            y = lognorm.pdf(x, gs_param[0], loc=gs_param[1], scale=gs_param[2])
            area = np.trapz(y, x)
            y /= area
            ax[1].plot(x, y, '--k', label='Experiment')
        ax[1].plot(par_eqDia, ypdf1, linestyle='-', linewidth=3.0, label='Input')
        ax[1].fill_between(par_eqDia, 0, ypdf1, alpha=0.3)
        ax[1].plot(grain_eqDia, ypdf2, linestyle='-', linewidth=3.0, label='Output')
        ax[1].fill_between(grain_eqDia, 0, ypdf2, alpha=0.3)

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('{} diameter (μm)'.format(name), fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if save_files:
            plt.savefig(cwd + "/{0}_diameter.png".format(name), bbox_inches="tight")
            print("    '{0}_diameter.png' is placed in the current working directory\n".format(name))
        plt.show()


        # Plot the aspect ratio comparison
        par_AR = np.sort(np.asarray(dataDict['Particle_Major_diameter']) /
                         np.asarray(dataDict['Particle_Minor_diameter']))
        ind = np.nonzero(dataDict['Grain_Minor_diameter'] > 1.e-5)[0]
        grain_AR = np.sort(np.asarray(dataDict['Grain_Major_diameter'][ind]) /
                           np.asarray(dataDict['Grain_Minor_diameter'][ind]))

        # Concatenate corresponding arrays to compute shared bins
        total_AR = np.concatenate([par_AR, grain_AR])

        # Find the corresponding shared bin edges
        shared_AR = np.histogram_bin_edges(total_AR, bins='doane')

        # Get the mean & std of the underlying normal distribution
        '''par_data, grain_data = np.log(par_AR), np.log(grain_AR)
        mu_par, std_par = np.mean(par_data), np.std(par_data)
        mu_gr, std_gr = np.mean(grain_data), np.std(grain_data)'''
        std_par, offs_par, sc_par = lognorm.fit(par_AR)
        std_gr, offs_gr, sc_gr = lognorm.fit(grain_AR)

        #par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par))
        #grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr))
        par_lognorm = lognorm(std_par, loc=offs_par, scale=sc_par)
        grain_lognorm = lognorm(std_gr, loc=offs_gr, scale=sc_gr)
        data = [par_AR, grain_AR]
        label = ['Input', 'Output']
        if ar_data is not None:
            data.append(ar_data)
            label.append('Experiment')

        # Plot the histogram & PDF
        sns.set(color_codes=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))

        # Plot histogram
        ax[0].hist(data, density=False, bins=len(shared_AR), label=label)
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('Aspect ratio', fontsize=18)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)

        # Plot PDF
        ypdf1 = par_lognorm.pdf(par_AR)
        area = np.trapz(ypdf1, par_AR)
        ypdf1 /= area
        ypdf2 = grain_lognorm.pdf(grain_AR)
        area = np.trapz(ypdf2, grain_AR)
        ypdf2 /= area
        if ar_param is not None:
            x0 = np.minimum(np.amin(grain_AR), np.amin(par_AR))
            x1 = np.maximum(np.amax(grain_AR), np.amax(par_AR))
            x = np.linspace(x0, x1, num=100)
            y = lognorm.pdf(x, ar_param[0], loc=ar_param[1], scale=ar_param[2])
            area = np.trapz(y, x)
            y /= area
            ax[1].plot(x, y, '--k', label='Experiment')
        ax[1].plot(par_AR, ypdf1, linestyle='-', linewidth=3.0, label='Input')
        ax[1].fill_between(par_AR, 0, ypdf1, alpha=0.3)
        ax[1].plot(grain_AR, ypdf2, linestyle='-', linewidth=3.0, label='Output')
        ax[1].fill_between(grain_AR, 0, ypdf2, alpha=0.3)

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if save_files:
            plt.savefig(cwd + "/Aspect_ratio.png", bbox_inches="tight")
            print("    'Aspect_ratio.png' is placed in the current working directory\n")
        plt.show()

    print('---->DONE!\n')
    return
