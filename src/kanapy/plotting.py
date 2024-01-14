#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subroutines for plotting of microstructures

Created on Mon Oct  4 07:52:55 2021

@author: Alexander Hartmaier, Golsa Tolooei Eshlaghi
"""
import logging
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import lognorm


def plot_voxels_3D(data, Ngr=1, sliced=False, dual_phase=False,
                   mask=None, cmap='prism', alpha=1.0, show=True):
    """
    Plot voxels in microstructure, each grain with a different color. Sliced
    indicates whether one eighth of the box should be removed to see internal
    structure. With alpha, the transparency of the grains can be adjusted.

    Parameters
    ----------
    data : int array
        Grain number or phase number associated to each voxel
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
        Adjust transparency of voxels in alpha channel of color map.
        The default is 1.0.
    show : bool
        Indicate whether to show the plot or to return the axis for further use

    Returns
    -------
    ax : matplotlib.axes
        Axes handle for 3D subplot

    """
    Nx = data.shape[0]
    Ny = data.shape[1]
    Nz = data.shape[2]

    if mask is None:
        vox_b = np.full(data.shape, True, dtype=bool)
    else:
        vox_b = mask
    if dual_phase:
        Ngr = 2
    cm = plt.cm.get_cmap(cmap, Ngr)
    colors = cm(data.astype(int))

    if sliced:
        ix = int(Nx / 2)
        iy = int(Ny / 2)
        iz = int(Nz / 2)
        vox_b[ix:Nx, iy:Ny, iz:Nz] = False

    ax = plt.figure().add_subplot(projection='3d')
    colors[:, :, :, -1] = alpha  # add semi-transparency
    ax.voxels(vox_b, facecolors=colors, edgecolors=colors, shade=True)
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


def plot_polygons_3D(geometry, cmap='prism', alpha=0.4, ec=[0.5, 0.5, 0.5, 0.1],
                     dual_phase=False):
    """
    Plot triangularized convex hulls of grains, based on vertices, i.e.
    connection points of 4 up to 8 grains or the end points of triple or quadruple
    lines between grains.

    Parameters
    ----------
    geometry : dict
        Dictionary with 'vertices' (node numbers) and 'simplices' (triangles)
    cmap : color map, optional
        Color map for triangles. The default is 'prism'.
    alpha : float, optional
        Transparency of plotted objects. The default is 0.4.
    ec : color, optional
        Color of edges. The default is [0.5,0.5,0.5,0.1].
    dual_phase : bool, optional
        Whether to plot red/green contrast for dual phase microstructure or colored grains

    Returns
    -------
    None.

    """
    grains = geometry['Grains']
    pts = geometry['Points']
    Ng = np.amax(list(grains.keys()))
    cm = plt.cm.get_cmap(cmap, Ng)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for igr in grains.keys():
        if not grains[igr]['Simplices']:
            continue
        if dual_phase:
            if grains[igr]['Phase'] == 0:
                col = 'red'
            else:
                col = 'green'
        else:
            col = list(cm(igr))
            col[-1] = alpha  # change alpha channel to create semi-transparency
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=grains[igr]['Simplices'], color=col,
                        edgecolor=ec, linewidth=1)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Polygonized microstructure')
    ax.view_init(30, 30)
    plt.show()


def plot_ellipsoids_3D(particles, cmap='prism', dual_phase=False):
    """
    Display ellipsoids during or after packing procedure

    Parameters
    ----------
    particles : Class particles
        Ellipsoids in microstructure before voxelization.
    cmap : color map, optional
        Color map for ellipsoids. The default is 'prism'.
    dual_phase : bool, optional
        Whether to display the ellipsoids in red/green contrast or in colors

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=plt.figaspect(1), dpi=1200)
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
            # col = cm(pa.phasenum)
        else:
            col = cm(i + 1)  # set to 'b' for only blue ellipsoids
        qw, qx, qy, qz = pa.quat
        x_c, y_c, z_c = pa.x, pa.y, pa.z
        a, b, c = pa.a, pa.b, pa.c
        # Rotation
        r = Rotation.from_quat([qx, qy, qz, qw])
        # Local coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_local = (a * np.outer(np.cos(u), np.sin(v))).reshape((10000,))
        y_local = (b * np.outer(np.sin(u), np.sin(v))).reshape((10000,))
        z_local = (c * np.outer(np.ones_like(u), np.cos(v))).reshape((10000,))
        points_local = list(np.array([x_local, y_local, z_local]).transpose())
        # Global coordinates
        points_global = r.apply(points_local, inverse=True)
        x = (points_global[:, 0] + np.ones_like(points_global[:, 0]) * x_c).reshape((100, 100))
        y = (points_global[:, 1] + np.ones_like(points_global[:, 1]) * y_c).reshape((100, 100))
        z = (points_global[:, 2] + np.ones_like(points_global[:, 2]) * z_c).reshape((100, 100))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=col, linewidth=0)
    plt.show()


def plot_output_stats(dataDict,
                      gs_data=None, gs_param=None,
                      ar_data=None, ar_param=None,
                      plot_particles=True,
                      save_files=False):
    r"""
    Evaluates particle- and output RVE grain statistics with respect to Major, Minor & Equivalent diameters and plots
    the distributions.
    """

    grain_eqDia = np.sort(np.asarray(dataDict['Grain_Equivalent_diameter']))
    data = [grain_eqDia]
    label = ['Grains']
    # Convert to micro meter for plotting
    if dataDict['Unit_scale'] == 'mm':
        grain_eqDia *= 1.e-3
    if plot_particles and 'Particle_Equivalent_diameter' in dataDict.keys():
        par_eqDia = np.sort(np.asarray(dataDict['Particle_Equivalent_diameter']))
        data.append(par_eqDia)
        label.append('Particles')
        if dataDict['Unit_scale'] == 'mm':
            par_eqDia *= 1.e-3
        total_eqDia = np.concatenate([grain_eqDia, par_eqDia])
        par_data = np.log(par_eqDia)
        mu_par = np.mean(par_data)
        std_par = np.std(par_data)
        par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par))
        particles = True
    else:
        par_eqDia = None
        total_eqDia = grain_eqDia
        particles = False
    if gs_data is not None:
        data.append(gs_data)
        label.append('Experiment')
        total_eqDia = np.concatenate([total_eqDia, gs_data])
    # NOTE: 'doane' produces better estimates for non-normal datasets
    shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')
    # Get the mean & std of the underlying normal distribution
    ind = np.nonzero(grain_eqDia > 1.e-5)[0]
    grain_data = np.log(grain_eqDia[ind])
    mu_gr = np.mean(grain_data)
    std_gr = np.std(grain_data)
    # NOTE: lognorm takes mean & std of normal distribution
    grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr))
    binNum = len(shared_bins)

    # Plot the histogram & PDF for equivalent diameter
    sns.set(color_codes=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 9))

    # Plot histogram
    ax[0].hist(data, density=False, bins=binNum, label=label)
    ax[0].legend(loc="upper right", fontsize=16)
    ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=18)
    ax[0].set_ylabel('Frequency', fontsize=18)
    ax[0].tick_params(labelsize=14)

    # Plot PDF
    ypdf2 = grain_lognorm.pdf(grain_eqDia)
    area = np.trapz(ypdf2, grain_eqDia)
    if np.isclose(area, 0.):
        logging.debug(f'Grain AREA interval: {area}')
        logging.debug(np.amin(grain_eqDia))
        logging.debug(np.amax(grain_eqDia))
        area = 1.
    ypdf2 /= area
    ax[1].plot(grain_eqDia, ypdf2, linestyle='-', linewidth=3.0, label='Grains')
    ax[1].fill_between(grain_eqDia, 0, ypdf2, alpha=0.3)
    if particles:
        ypdf1 = par_lognorm.pdf(par_eqDia)
        area = np.trapz(ypdf1, par_eqDia)
        if np.isclose(area, 0.):
            logging.debug(f'Particle AREA interval: {area}')
            logging.debug(np.amin(par_eqDia))
            logging.debug(np.amax(par_eqDia))
            area = 1.
        ypdf1 /= area
        ax[1].plot(par_eqDia, ypdf1, linestyle='-', linewidth=3.0, label='Particles')
        ax[1].fill_between(par_eqDia, 0, ypdf1, alpha=0.3)
    if gs_param is not None:
        x0 = np.amin(grain_eqDia)
        x1 = np.amax(grain_eqDia)
        x = np.linspace(x0, x1, num=50)
        y = lognorm.pdf(x, gs_param[0], loc=gs_param[1], scale=gs_param[2])
        area = np.trapz(y, x)
        if np.isclose(area, 0.):
            logging.debug(f'Expt. AREA interval: {x0}, {x1}')
            logging.debug(np.amin(grain_eqDia))
            logging.debug(np.amax(grain_eqDia))
            area = 1.
        y /= area
        ax[1].plot(x, y, '--k', label='Experiment')

    ax[1].legend(loc="upper right", fontsize=16)
    ax[1].set_xlabel('Equivalent diameter (μm)', fontsize=18)
    ax[1].set_ylabel('Density', fontsize=18)
    ax[1].tick_params(labelsize=14)
    if save_files:
        plt.savefig("Equivalent_diameter.png", bbox_inches="tight")
        print("    'Equivalent_diameter.png' is placed in the current working directory\n")
    plt.show()

    if 'Grain_Minor_diameter' in dataDict.keys():
        # Plot the aspect ratio comparison
        ind = np.nonzero(dataDict['Grain_Minor_diameter'] > 1.e-5)[0]
        grain_AR = np.sort(np.asarray(dataDict['Grain_Major_diameter'][ind]) /
                           np.asarray(dataDict['Grain_Minor_diameter'][ind]))
        # Get the mean & std of the underlying normal distribution
        std_gr, offs_gr, sc_gr = lognorm.fit(grain_AR)
        grain_lognorm = lognorm(std_gr, loc=offs_gr, scale=sc_gr)
        if particles and 'Particle_Minor_diameter' in dataDict.keys():
            par_AR = np.sort(np.asarray(dataDict['Particle_Major_diameter']) /
                             np.asarray(dataDict['Particle_Minor_diameter']))
            # Concatenate corresponding arrays to compute shared bins
            total_AR = np.concatenate([par_AR, grain_AR])
            std_par, offs_par, sc_par = lognorm.fit(par_AR)
            par_lognorm = lognorm(std_par, loc=offs_par, scale=sc_par)
            data = [par_AR, grain_AR]
            label = ['Particles', 'Grains']
        else:
            total_AR = grain_AR
            data = [grain_AR]
            label = ['Grains']
        if ar_data is not None:
            data.append(ar_data)
            label.append('Experiment')
        # Find the corresponding shared bin edges
        shared_AR = np.histogram_bin_edges(total_AR, bins='doane')

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
        if particles and 'Particle_Minor_diameter' in dataDict.keys():
            ypdf1 = par_lognorm.pdf(par_AR)
            area = np.trapz(ypdf1, par_AR)
            ypdf1 /= area
            ax[1].plot(par_AR, ypdf1, linestyle='-', linewidth=3.0, label='Particles')
            ax[1].fill_between(par_AR, 0, ypdf1, alpha=0.3)
        ypdf2 = grain_lognorm.pdf(grain_AR)
        area = np.trapz(ypdf2, grain_AR)
        ypdf2 /= area
        ax[1].plot(grain_AR, ypdf2, linestyle='-', linewidth=3.0, label='Grains')
        ax[1].fill_between(grain_AR, 0, ypdf2, alpha=0.3)
        if ar_param is not None:
            x0 = np.amin(1.0)
            x1 = np.amax(grain_AR)
            x = np.linspace(x0, x1, num=100)
            y = lognorm.pdf(x, ar_param[0], loc=ar_param[1], scale=ar_param[2])
            area = np.trapz(y, x)
            y /= area
            ax[1].plot(x, y, '--k', label='Experiment')

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if save_files:
            plt.savefig("Aspect_ratio.png", bbox_inches="tight")
            print("    'Aspect_ratio.png' is placed in the current working directory\n")
        plt.show()
    return


def plot_init_stats(stats_dict, gs_data=None, ar_data=None, save_files=False):
    r"""
    Plot initial microstructure descriptors, including cut-offs, based on user defined statistics
    """

    # Extract grain diameter statistics info from input file
    sd = stats_dict["Equivalent diameter"]["std"]
    mean = stats_dict["Equivalent diameter"]["mean"]
    if "offs" in stats_dict["Equivalent diameter"]:
        offs = stats_dict["Equivalent diameter"]["offs"]
    else:
        offs = None
    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]

    # Equivalent diameter statistics
    # NOTE: SCIPY's lognorm takes in sigma & mu of Normal distribution
    # https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma/13837335#13837335

    # Compute the Log-normal PDF & CDF.
    if offs is None:
        frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
    else:
        frozen_lognorm = lognorm(s=sd, loc=offs, scale=mean)

    xaxis = np.linspace(0.1, 200, 1000)
    ypdf = frozen_lognorm.pdf(xaxis)

    # Find the location at which CDF > 0.99
    # cdf_idx = np.where(ycdf > 0.99)[0][0]
    # x_lim = xaxis[cdf_idx]
    x_lim = dia_cutoff_max * 1.5

    if stats_dict["Grain type"] == "Equiaxed":
        sns.set(color_codes=True)
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        plt.ion()

        # Plot grain size distribution
        plt.plot(xaxis, ypdf, linestyle='-', linewidth=3.0)
        ax.fill_between(xaxis, 0, ypdf, alpha=0.3)
        ax.set_xlim(left=0.0, right=x_lim)
        ax.set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(labelsize=14)
        ax.axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                   label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax.axvline(dia_cutoff_max, linestyle='-', linewidth=3.0,
                   label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        if gs_data is not None:
            ind = np.nonzero(gs_data < x_lim)[0]
            ax.hist(gs_data[ind], bins=80, density=True, label='experimental data')
        plt.title("Grain equivalent diameter distribution", fontsize=20)
        plt.legend(fontsize=16)
        plt.show()
    elif stats_dict["Grain type"] == "Elongated":
        # Extract mean grain aspect ratio value info from input file
        sd_AR = stats_dict["Aspect ratio"]["std"]
        mean_AR = stats_dict["Aspect ratio"]["mean"]
        if "offs" in stats_dict["Aspect ratio"]:
            offs_AR = stats_dict["Aspect ratio"]["offs"]
        else:
            offs_AR = None
        ar_cutoff_min = stats_dict["Aspect ratio"]["cutoff_min"]
        ar_cutoff_max = stats_dict["Aspect ratio"]["cutoff_max"]

        sns.set(color_codes=True)
        fig, ax = plt.subplots(2, 1, figsize=(9, 9))
        plt.ion()

        # Plot grain size distribution
        ax[0].plot(xaxis, ypdf, linestyle='-', linewidth=3.0)
        ax[0].fill_between(xaxis, 0, ypdf, alpha=0.3)
        ax[0].set_xlim(left=0.0, right=x_lim)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[0].set_ylabel('Density', fontsize=18)
        ax[0].tick_params(labelsize=14)
        ax[0].axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax[0].axvline(dia_cutoff_max, linestyle='-', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        if gs_data is not None:
            ax[0].hist(gs_data, bins=80, density=True, label='experimental data')
        ax[0].legend(fontsize=14)

        # Plot aspect ratio statistics
        # Compute the Log-normal PDF & CDF.
        if offs_AR is None:
            frozen_lognorm = lognorm(s=sd_AR, scale=np.exp(mean_AR))
        else:
            frozen_lognorm = lognorm(sd_AR, loc=offs_AR, scale=mean_AR)
        xaxis = np.linspace(0.5, 2 * ar_cutoff_max, 500)
        ypdf = frozen_lognorm.pdf(xaxis)
        ax[1].plot(xaxis, ypdf, linestyle='-', linewidth=3.0)
        ax[1].fill_between(xaxis, 0, ypdf, alpha=0.3)
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        ax[1].axvline(ar_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(ar_cutoff_min))
        ax[1].axvline(ar_cutoff_max, linestyle='-', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(ar_cutoff_max))
        if ar_data is not None:
            ax[1].hist(ar_data, bins=15, density=True, label='experimental data')
        ax[1].legend(fontsize=14)
        plt.show()
    else:
        raise ValueError('Value for "Grain_type" must be either "Equiaxed" or "Elongated".')

    if save_files:
        plt.savefig("Input_distribution.png", bbox_inches="tight")
        plt.draw()
        plt.pause(0.001)
        print(' ')
        input("    Press [enter] to continue")
        print("    'Input_distribution.png' is placed in the current working directory\n")
    return
