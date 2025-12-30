# -*- coding: utf-8 -*-
"""
Subroutines for plotting of microstructures

Created on Mon Oct  4 07:52:55 2021

@author: Alexander Hartmaier, Golsa Tolooei Eshlaghi, Ronak Shoghi
"""
import sys
import logging
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import lognorm

#from PyQt5.QtWidgets import QApplication

#def dpi_system():
#    """
#    A function to get the working system's DPI
#    """
#    app = QApplication(sys.argv)
#    screen = app.screens()[0]
#    dpi = screen.physicalDotsPerInch()
#    app.quit()
#    return dpi

#def plot_dpi():
#    """
#    Calculates the scaled DPI value for matplotlib
#    """
#    system_dpi = 200 #dpi_system()
#    dpi_scale = 100/system_dpi
#    matplotlib_dpi = 100 * dpi_scale
#    return matplotlib_dpi


def plot_voxels_3D(data, Ngr=None, sliced=False, dual_phase=None,
                   mask=None, cmap='prism', alpha=1.0, silent=False,
                   clist=None, asp_arr=None,
                   phases=False, cols=None):
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
    clist : (Ngr, 3)-ndarray
        List of RGB colors for each grain

    Returns
    -------
    ax : matplotlib.axes
        Axes handle for 3D subplot

    """
    if dual_phase is not None:
        print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
        phases = dual_phase
    if Ngr is not None:
        print('Use of "Ngr" is depracted. Value will be determined automatically.')
    if cols is None:
        cols = ['red', 'green', 'lightblue', 'orange', 'gray']
    Nx = data.shape[0]
    Ny = data.shape[1]
    Nz = data.shape[2]
    if asp_arr is None:
        asp_arr = [1, 1, 1]
    if mask is None:
        vox_b = np.full(data.shape, True, dtype=bool)
    else:
        vox_b = mask
    if phases:
        Ngr = len(np.unique(data))
    else:
        Ngr = np.max(data)
        if np.min(data) == 0:
            Ngr += 1
    if clist is None:
        cm = plt.cm.get_cmap(cmap, Ngr)
        colors = cm(data.astype(int))
    else:
        colors = np.ones((Nx, Ny, Nz, 4))
        grl = np.unique(data)
        if grl[0] == 0:
            grl = grl[1:]
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    igr = data[i, j, k]
                    if igr > 0:
                        ind = np.where(grl == igr)[0]
                        colors[i, j, k, 0:3] = clist[ind[0]]
                    else:
                        colors[i, j, k, 0:3] = [0.3, 0.3, 0.3]

    if sliced:
        ix = int(Nx / 2)
        iy = int(Ny / 2)
        iz = int(Nz / 2)
        vox_b[ix:Nx, iy:Ny, iz:Nz] = False

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors[:, :, :, -1] = alpha  # add semi-transparency
    ax.voxels(vox_b, facecolors=colors, edgecolors=colors, shade=True)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Voxelated microstructure')
    ax.view_init(30, 30)
    ax.set_xlim(right=Nx)
    ax.set_ylim(top=Ny)
    ax.set_zlim(top=Nz)
    ax.set_box_aspect(asp_arr)
    if silent:
        return fig
    else:
        plt.show(block=True)


def plot_polygons_3D(geometry, cmap='prism', alpha=0.4, ec=None,
                     dual_phase=None, silent=False, asp_arr=None,
                     phases=False, cols=None):
    """
    Plot triangularized convex hulls of grains based on vertices and simplices

    Parameters
    ----------
    geometry : dict
        Dictionary containing grain geometry with keys:
        - 'Grains': dict with simplices (triangles) for each grain
        - 'Points': ndarray of vertex coordinates (N, 3)
    cmap : str, optional
        Colormap for coloring the triangles. Default is 'prism'.
    alpha : float, optional
        Transparency of the polygons. Default is 0.4.
    ec : color or list, optional
        Edge color for triangles. Default is [0.5, 0.5, 0.5, 0.1].
    dual_phase : bool, optional
        Deprecated. Use `phases` instead. If True, grains are colored
        by phase in red/green contrast.
    silent : bool, optional
        If True, returns the figure object instead of showing it. Default is False.
    asp_arr : list of 3 floats, optional
        Aspect ratio for the 3D axes (x, y, z). Default is [1, 1, 1].
    phases : bool, optional
        If True, colors grains according to their phase number.
    cols : list of str, optional
        Colors to use for each phase when `phases=True`.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object if `silent=True`.

    Notes
    -----
    - The `dual_phase` parameter is deprecated. Use `phases` to control
      coloring by phase.
    - Each grain is plotted using its convex hull triangles defined in
      `geometry['Grains'][igr]['Simplices']`.
    """
    if dual_phase is not None:
        print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
        phases = dual_phase
    if cols is None:
        cols = ['red', 'green', 'lightblue', 'orange', 'gray']
    if ec is None:
        ec = [0.5, 0.5, 0.5, 0.1]
    if asp_arr is None:
        asp_arr = [1, 1, 1]
    grains = geometry['Grains']
    pts = geometry['Points']
    Ng = np.amax(list(grains.keys()))
    cm = plt.cm.get_cmap(cmap, Ng)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for igr in grains.keys():
        if not grains[igr]['Simplices']:
            continue
        if phases:
            icol = grains[igr]['Phase']
            if icol < len(cols):
                col = cols[icol]
            else:
                col = 'gray'
        else:
            col = list(cm(igr))
            col[-1] = alpha  # change alpha channel to create semi-transparency
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=grains[igr]['Simplices'], color=col,
                        edgecolor=ec, linewidth=1)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_title('Polygonized microstructure')
    ax.view_init(30, 30)
    ax.set_box_aspect(asp_arr)
    if silent:
        return fig
    else:
        plt.show(block=True)



def plot_ellipsoids_3D(particles, cmap='prism', dual_phase=None, silent=False, asp_arr=None,
                       phases=False, cols=None):
    """
    Display ellipsoids after the packing procedure

    Parameters
    ----------
    particles : list
        List of ellipsoid objects representing particles in the microstructure.
    cmap : str, optional
        Color map for ellipsoids. Default is 'prism'.
    dual_phase : bool, optional
        Deprecated. Use `phases` instead. If True, ellipsoids are colored
        according to phase (red/green contrast).
    silent : bool, optional
        If True, returns the figure object without displaying it. Default is False.
    asp_arr : list of 3 floats, optional
        Aspect ratio for the 3D axes (x, y, z). Default is [1, 1, 1].
    phases : bool, optional
        If True, ellipsoids are colored according to their phase number.
    cols : list of str, optional
        List of colors used for each phase when `phases=True`.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object if `silent=True`.

    Notes
    -----
    - The `dual_phase` parameter is deprecated. Use `phases` to control
      coloring by phase.
    - Each ellipsoid is represented by its surface mesh generated via
      `surfacePointsGen` method of the ellipsoid object.
    """
    if asp_arr is None:
        asp_arr = [1, 1, 1]
    if dual_phase is not None:
        print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
        phases = dual_phase
    if cols is None:
        cols = ['red', 'green', 'lightblue', 'orange', 'gray']
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(30, 30)
    Npa = len(particles)
    cm = plt.get_cmap(cmap, Npa+1) if not phases else None

    for i, pa in enumerate(particles):
        if pa.duplicate:
            continue
        if phases:
            icol = pa.phasenum
            if icol < len(cols):
                color = cols[icol]
            else:
                color = 'gray'
        else:
            color = cm(i + 1)
        pts = pa.surfacePointsGen(nang=100)
        ctr = pa.get_pos()
        x = (pts[:, 0] + ctr[0]).reshape((100, 100))
        y = (pts[:, 1] + ctr[1]).reshape((100, 100))
        z = (pts[:, 2] + ctr[2]).reshape((100, 100))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=0, antialiased=False)
    ax.set_box_aspect(asp_arr)
    if silent:
        return fig
    else:
        plt.show(block=True)



def plot_particles_3D(particles, cmap='prism', dual_phase=False, plot_hull=True,
                      silent=False, asp_arr=None):
    """
    Display inner polyhedra and optional hulls of ellipsoids after packing procedure

    Parameters
    ----------
    particles : list
        List of ellipsoid objects containing inner polygons and convex hulls.
    cmap : str, optional
        Colormap for ellipsoids. Default is 'prism'.
    dual_phase : bool, optional
        If True, colors ellipsoids according to phase (red/green contrast).
    plot_hull : bool, optional
        If True, plots the outer hull of the ellipsoid. Default is True.
    silent : bool, optional
        If True, returns the figure object instead of displaying the plot.
    asp_arr : list of 3 floats, optional
        Aspect ratio for the 3D axes (x, y, z). Default is [1, 1, 1].

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object if `silent=True`.

    Notes
    -----
    - Each ellipsoid must have an inner polygon (`pa.inner`) to be plotted.
    - The convex hull of the ellipsoid is displayed using `plot_trisurf`.
    - If `plot_hull` is True, the outer surface of the ellipsoid is also plotted
      using `plot_surface`.
    """
    if asp_arr is None:
        asp_arr = [1, 1, 1]
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.view_init(30, 30)

    Npa = len(particles)
    cm = plt.cm.get_cmap(cmap, Npa)

    for i, pa in enumerate(particles):
        if pa.duplicate is not None:
            continue
        if pa.inner is None:
            logging.error(f'Ellipsoid {pa.id} without inner polygon cannot be plotted.')
            continue
        if dual_phase:
            if pa.phasenum == 0:
                col = 'red'
            elif pa.phasenum == 1:
                col = 'green'
            else:
                col = 'lightblue'
        else:
            col = cm(i + 1)  # set to 'b' for only blue ellipsoids
        pa.sync_poly()
        pts = pa.inner.points
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=pa.inner.convex_hull, color=col,
                        edgecolor='k', linewidth=1)
        if plot_hull:
            col = [0.7, 0.7, 0.7, 0.3]
            pts = pa.surfacePointsGen(nang=100)
            ctr = pa.get_pos()
            x = (pts[:, 0] + ctr[None, 0]).reshape((100, 100))
            y = (pts[:, 1] + ctr[None, 1]).reshape((100, 100))
            z = (pts[:, 2] + ctr[None, 2]).reshape((100, 100))
            ax.plot_surface(x, y, z, rstride=4, cstride=4, color=col, linewidth=0)
    ax.set_box_aspect(asp_arr)
    if silent:
        return fig
    else:
        plt.show(block=True)


def plot_output_stats(data_list, labels, iphase=None,
                      gs_data=None, gs_param=None,
                      ar_data=None, ar_param=None,
                      save_files=False, silent=False, enhanced_plot=False):
    """
    Evaluate and plot particle- and RVE grain statistics, including equivalent diameters and aspect ratios

    Parameters
    ----------
    data_list : list of dict
        List of dictionaries containing particle/grain statistics. Each dictionary
        should include keys like 'eqd', 'eqd_scale', 'eqd_sig', 'ar', 'ar_scale', 'ar_sig'.
    labels : list of str
        Labels indicating the type of data in `data_list`, e.g., ['Grains', 'Partcls'].
    iphase : int, optional
        Phase index to label the plots. Default is None.
    gs_data : array-like, optional
        Experimental equivalent diameter data for comparison. Default is None.
    gs_param : tuple, optional
        Parameters for lognormal distribution of experimental data (s, loc, scale). Default is None.
    ar_data : array-like, optional
        Experimental aspect ratio data for comparison. Default is None.
    ar_param : tuple, optional
        Parameters for lognormal distribution of experimental aspect ratio data (s, loc, scale). Default is None.
    save_files : bool, optional
        If True, saves the generated figures as PNG files. Default is False.
    silent : bool, optional
        If True, returns the figure objects instead of showing them. Default is False.
    enhanced_plot : bool, optional
        If True, produces higher-resolution comparison plots using lognormal PDFs. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        Figure object if `silent=True`.

    Notes
    -----
    - The function can compare statistical distributions between grains, particles, and experimental data.
    - Both histograms and lognormal probability density functions are used to display distributions.
    - If `enhanced_plot=True`, PDFs for equivalent diameter and aspect ratio are plotted using a dense x-axis.
    - `labels` must contain at least 'Grains' or 'Voxels'. Otherwise, a ValueError is raised.
    """
    if 'Grains' not in labels and 'Voxels' not in labels:
        print(f'Argument "labels": {labels}')
        raise ValueError('Either grains or voxels must be provided in labaels for statistical analysis.')
    if 'Grains' in labels:
        # if grain information is given plot this for comparison
        ind = labels.index('Grains')
    else:
        # otherwise use voxel statistics
        ind = labels.index('Voxels')
    # process equiv. diameter data
    grain_eqDia = data_list[ind]['eqd']
    data = [grain_eqDia]
    label = [labels[ind]]
    mu_gr = data_list[ind]['eqd_scale']
    std_gr = data_list[ind]['eqd_sig']
    grain_lognorm = lognorm(std_gr, scale=mu_gr)
    # process aspect ratio data
    grain_AR = data_list[ind]['ar']
    sig_ar = data_list[ind]['ar_sig']
    sc_ar = data_list[ind]['ar_scale']
    ar_lognorm = lognorm(sig_ar, scale=sc_ar)

    if 'Partcls' in labels:
        ind = labels.index('Partcls')
        par_eqDia = data_list[ind]['eqd']
        data.append(par_eqDia)
        label.append('Particles')

        total_eqDia = np.concatenate(data)
        mu_par = data_list[ind]['eqd_scale']
        std_par = data_list[ind]['eqd_sig']
        par_lognorm = lognorm(s=std_par, scale=mu_par)
        par_AR = data_list[ind]['ar']
        particles = True
    else:
        par_eqDia = None
        total_eqDia = data[0]
        particles = False
    if gs_data is not None:
        data.append(gs_data)
        label.append('Experiment')
        total_eqDia = np.concatenate([total_eqDia, gs_data])
    if enhanced_plot:
        # Determine x-axis limits for equivalent diameter
        dia_cutoff_min = np.min(total_eqDia)
        dia_cutoff_max = np.max(total_eqDia)
        x_lim_dia = dia_cutoff_max # * 1.5

        # Plot the equivalent diameter PDF
        sns.set(color_codes=True)
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        xaxis_dia = np.linspace(dia_cutoff_min, x_lim_dia, 1000)

        # plot lognormal distribution of stats parameters
        if gs_param is not None:
            y = lognorm.pdf(xaxis_dia, gs_param[0], loc=gs_param[1], scale=gs_param[2])
            """area = np.trapz(y, xaxis_dia)
            if np.isclose(area, 0.):
                logging.debug(f'Expt. AREA interval: {np.min(xaxis_dia)}, {np.max(xaxis_dia)}')
                logging.debug(np.amin(grain_eqDia))
                logging.debug(np.amax(grain_eqDia))
                area = 1.
            y /= area"""
            ax[0].plot(xaxis_dia, y, linestyle='-', linewidth=3.0, label='Input')
            ax[0].fill_between(xaxis_dia, 0, y, alpha=0.3)

        # Plot PDF for equivalent diameter
        ypdf2 = grain_lognorm.pdf(xaxis_dia)
        """area = np.trapz(ypdf2, xaxis_dia)
        if np.isclose(area, 0.):
            logging.debug(f'Grain AREA interval: {area}')
            logging.debug(np.amin(xaxis_dia))
            logging.debug(np.amax(xaxis_dia))
            area = 1.
        ypdf2 /= area"""
        ax[0].plot(xaxis_dia, ypdf2, linestyle='-', linewidth=3.0, label='Output')
        ax[0].fill_between(xaxis_dia, 0, ypdf2, alpha=0.3)

        ax[0].legend(loc="upper right", fontsize=12)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=16)
        ax[0].set_ylabel('Density', fontsize=16)
        ax[0].tick_params(labelsize=12)
        ax[0].set_xlim(left=dia_cutoff_min*0.9, right=x_lim_dia*1.1)
        if iphase is not None:
            ax[0].set_title(f'Equivalent Diameter Distribution - Phase {iphase}', fontsize=20)

        # Determine x-axis limits for aspect ratio
        ar_cutoff_min = np.min(grain_AR)
        ar_cutoff_max = np.max(grain_AR)
        if particles:
            ar_cutoff_min = min(ar_cutoff_min, np.min(par_AR))
            ar_cutoff_max = max(ar_cutoff_max, np.max(par_AR))
        x_lim_ar = ar_cutoff_max  # * 1.5
        xaxis_ar = np.linspace(ar_cutoff_min, x_lim_ar, 1000)

        # plot lognormal distribution for aspect ratio parameters
        if ar_param is not None:
            y = lognorm.pdf(xaxis_ar, ar_param[0], loc=ar_param[1], scale=ar_param[2])
            """area = np.trapz(y, xaxis_ar)
            y /= area"""
            ax[1].plot(xaxis_ar, y, linestyle='-', linewidth=3.0, label='Input')
            ax[1].fill_between(xaxis_ar, 0, y, alpha=0.3)
        # Plot the PDF for aspect ratio
        ypdf2 = ar_lognorm.pdf(xaxis_ar)
        """area = np.trapz(ypdf2, xaxis_ar)
        if np.isclose(area, 0.0):
            logging.debug('Small area for aspect ratio of grains.')
            logging.debug(ypdf2, xaxis_ar)
            area = 1.0
        ypdf2 /= area"""
        ax[1].plot(xaxis_ar, ypdf2, linestyle='-', linewidth=3.0, label='Output')
        ax[1].fill_between(xaxis_ar, 0, ypdf2, alpha=0.3)

        ax[1].legend(loc="upper right", fontsize=12)
        ax[1].set_xlabel('Aspect ratio', fontsize=16)
        ax[1].set_ylabel('Density', fontsize=16)
        ax[1].tick_params(labelsize=12)
        ax[1].set_xlim(left=ar_cutoff_min*0.9, right=x_lim_ar*1.1)
        if iphase is not None:
            ax[1].set_title(f'Aspect Ratio Distribution - Phase {iphase}', fontsize=20)

        if save_files:
            fname = 'equiv_diameter_and_aspect_ratio_comp.png'
            plt.savefig(fname, bbox_inches="tight")
            print(f"'{fname}' is placed in the current working directory\n")

        if silent:
            return fig
        else:
            plt.show(block=True)
    else:

        # NOTE: 'fd' produces better estimates for non-normal datasets
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='fd')
        binNum = len(shared_bins)

        # Plot the histogram & PDF for equivalent diameter
        sns.set(color_codes=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Plot histogram
        ax[0].hist(data, density=True, bins=binNum, label=label)
        ax[0].legend(loc="upper right", fontsize=12)
        ax[0].set_xlabel('equivalent diameter (μm)', fontsize=14)
        ax[0].set_ylabel('frequency', fontsize=12)
        ax[0].tick_params(labelsize=12)
        if iphase is not None:
            ax[0].set_title(f'Phase {iphase}')

        # Plot PDF
        ypdf2 = grain_lognorm.pdf(grain_eqDia)
        """area = np.trapz(ypdf2, grain_eqDia)
        if np.isclose(area, 0.):
            logging.debug(f'Grain AREA interval: {area}')
            logging.debug(np.amin(grain_eqDia))
            logging.debug(np.amax(grain_eqDia))
            area = 1.
        ypdf2 /= area"""
        ax[1].plot(grain_eqDia, ypdf2, linestyle='-', linewidth=3.0, label=label[0])
        ax[1].fill_between(grain_eqDia, 0, ypdf2, alpha=0.3)
        if particles:
            #par_lognorm = lognorm(std_par, scale=mu_par)
            ypdf1 = par_lognorm.pdf(par_eqDia)
            """area = np.trapz(ypdf1, par_eqDia)
            if np.isclose(area, 0.):
                logging.debug(f'Particle AREA interval: {area}')
                logging.debug(np.amin(par_eqDia))
                logging.debug(np.amax(par_eqDia))
                area = 1.
            ypdf1 /= area"""
            ax[1].plot(par_eqDia, ypdf1, linestyle='-', linewidth=3.0, label='Particles')
            ax[1].fill_between(par_eqDia, 0, ypdf1, alpha=0.3)
        if gs_param is not None:
            x0 = np.min(grain_eqDia)
            x1 = np.max(grain_eqDia)
            if particles:
                x0 = min(np.min(par_eqDia), x0)
                x1 = max(np.max(par_eqDia), x1)
            x = np.linspace(x0, x1, num=50)
            y = lognorm.pdf(x, gs_param[0], loc=gs_param[1], scale=gs_param[2])
            """area = np.trapz(y, x)
            if np.isclose(area, 0.):
                logging.debug(f'Expt. AREA interval: {x0}, {x1}')
                logging.debug(np.amin(grain_eqDia))
                logging.debug(np.amax(grain_eqDia))
                area = 1.
            y /= area"""
            ax[1].plot(x, y, '--k', label='Experiment')

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('equivalent diameter (μm)', fontsize=18)
        ax[1].set_ylabel('density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if iphase is not None:
            ax[1].set_title(f'Phase {iphase}')
        if save_files:
            fname = 'equiv_diameter_comp.png'
            plt.savefig(fname, bbox_inches="tight")
            print(f"    '{fname}' is placed in the current working directory\n")
        plt.show()

        # Plot the aspect ratio comparison
        if particles:
            ind = labels.index('Partcls')
            par_AR = data_list[ind]['ar']
            # Concatenate corresponding arrays to compute shared bins
            total_AR = np.concatenate([grain_AR, par_AR])
            sig_par = data_list[ind]['ar_sig']
            sc_par = data_list[ind]['ar_scale']
            par_lognorm = lognorm(sig_par, scale=sc_par)
            data = [grain_AR, par_AR]
        else:
            total_AR = grain_AR
            data = [grain_AR]
        if ar_data is not None:
            data.append(ar_data)
            label.append('Experiment')
        # Find the corresponding shared bin edges
        shared_AR = np.histogram_bin_edges(total_AR, bins='fd')

        # Plot the histogram & PDF
        sns.set(color_codes=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        # Plot histogram
        ax[0].hist(data, density=True, bins=len(shared_AR), label=label)
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('aspect ratio', fontsize=18)
        ax[0].set_ylabel('frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)
        if iphase is not None:
            ax[0].set_title(f'Phase {iphase}')

        # Plot PDF
        ypdf2 = ar_lognorm.pdf(grain_AR)
        """area = np.trapz(ypdf2, grain_AR)
        if np.isclose(area, 0.0):
            logging.debug('Small area for aspect ratio of grains.')
            logging.debug(ypdf2, grain_AR)
            area = 1.0
        ypdf2 /= area"""
        ax[1].plot(grain_AR, ypdf2, linestyle='-', linewidth=3.0, label=label[0])
        ax[1].fill_between(grain_AR, 0, ypdf2, alpha=0.3)
        if particles:
            ypdf1 = par_lognorm.pdf(par_AR)
            """area = np.trapz(ypdf1, par_AR)
            ypdf1 /= area"""
            ax[1].plot(par_AR, ypdf1, linestyle='-', linewidth=3.0, label='Particles')
            ax[1].fill_between(par_AR, 0, ypdf1, alpha=0.3)
        if ar_param is not None:
            x0 = np.min(grain_AR)
            x1 = np.max(grain_AR)
            if particles:
                x0 = min(x0, np.min(par_AR))
                x1 = max(x1, np.max(par_AR))
            x = np.linspace(x0, x1, num=100)
            y = lognorm.pdf(x, ar_param[0], loc=ar_param[1], scale=ar_param[2])
            """area = np.trapz(y, x)
            y /= area"""
            ax[1].plot(x, y, '--k', label='Experiment')

        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('aspect ratio', fontsize=18)
        ax[1].set_ylabel('density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        if iphase is not None:
            ax[1].set_title(f'Phase {iphase}')
        if save_files:
            fname = "aspect_ratio_comp.png"
            plt.savefig(fname, bbox_inches="tight")
            print(f"    '{fname}' is placed in the current working directory\n")
        plt.show()
        return


def plot_init_stats(stats_dict,
                    gs_data=None, ar_data=None,
                    gs_param=None, ar_param=None,
                    save_files=False, silent=False):

    """
    Plot initial microstructure descriptors (grain size and aspect ratio) based on user-defined statistics.

    Parameters
    ----------
    stats_dict : dict
        Dictionary containing initial microstructure statistics. Expected keys:
        - "Equivalent diameter": dict with 'sig', 'scale', optionally 'loc', 'cutoff_min', 'cutoff_max'
        - "Aspect ratio": dict with 'sig', 'scale', optionally 'loc', 'cutoff_min', 'cutoff_max' (for elongated grains)
        - "Grain type": str, one of "Equiaxed", "Elongated", or "Free"
        - "Semi axes": dict with 'scale', 'sig', 'cutoff_min', 'cutoff_max' (for free grains)
        - "Phase": dict with 'Number' and 'Name', optional
    gs_data : array-like, optional
        Experimental equivalent diameter data. Default is None.
    ar_data : array-like, optional
        Experimental aspect ratio data. Default is None.
    gs_param : tuple, optional
        Parameters (s, loc, scale, min, max) for lognormal distribution of experimental equivalent diameters. Default is None.
    ar_param : tuple, optional
        Parameters (s, loc, scale, min, max) for lognormal distribution of experimental aspect ratios. Default is None.
    save_files : bool, optional
        If True, save the figure to file "Input_distribution.png". Default is False.
    silent : bool, optional
        If True, return the figure object instead of displaying it. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The figure object if `silent=True`.

    Notes
    -----
    - The function automatically selects plotting style depending on "Grain type":
        - "Equiaxed": only equivalent diameter distribution
        - "Elongated": both equivalent diameter and aspect ratio distributions
        - "Free": compute aspect ratio from semi-axes and rotation axis
    - Vertical dashed lines mark the minimum and maximum cut-offs from `stats_dict`.
    - Overlays experimental data (histogram) and lognormal fits if `gs_data` or `ar_data` are provided.
    - Uses lognormal PDFs for both grain size and aspect ratio.
    """
    def plot_lognorm(x, sig, loc, scale, axis):
        """
        Plot a log-normal probability density function (PDF) on a given matplotlib axis

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.
        sig : float
            Shape parameter (standard deviation of log of the variable).
        loc : float
            Location parameter (shifts the distribution along x-axis).
        scale : float
            Scale parameter (exp(mean of log of the variable)).
        axis : matplotlib.axes.Axes
            Matplotlib axis to plot the PDF on.
        """
        y = lognorm.pdf(x, sig, loc=loc, scale=scale)
        """area = np.trapz(y, xv)
        if np.isclose(area, 0.0):
            area = 1.
        y /= area"""
        axis.plot(x, y, linestyle='-', linewidth=3.0, label='Output stats')
        axis.fill_between(x, 0, y, alpha=0.3)

    # Extract grain diameter statistics info from input file
    sd = stats_dict["Equivalent diameter"]["sig"]
    scale = stats_dict["Equivalent diameter"]["scale"]
    if 'loc' in stats_dict["Equivalent diameter"].keys():
        loc = stats_dict["Equivalent diameter"]["loc"]
    else:
        loc = 0.

    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]

    if "Phase" in stats_dict.keys():
        title = (f'Microstructure statistics of phase {stats_dict["Phase"]["Number"]} '
                 f'({stats_dict["Phase"]["Name"]})')
    else:
        title = ('Microstructure statistics')

    x_lim_eqd = dia_cutoff_max * 1.5
    if gs_param is not None:
        x_lim_eqd = max(x_lim_eqd, gs_param[4]*1.1)

    # Compute the Log-normal PDF
    xaxis = np.linspace(0.1, x_lim_eqd, 1000)
    ypdf = lognorm.pdf(xaxis, sd, loc=loc, scale=scale)
    # normalize density to region between min and max cutoff
    ind = np.nonzero(np.logical_and(xaxis >= dia_cutoff_min, xaxis <= dia_cutoff_max))[0]
    """if gs_data is None:
        area = np.trapz(ypdf[ind], xaxis[ind])
        if np.isclose(area, 0.0):
            area = 1.
        ypdf /= area"""
    # set colorcodes
    sns.set(color_codes=True)
    if stats_dict["Grain type"] == "Equiaxed":
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)
        # plt.ion()

        # Plot grain size distribution
        ax.axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                   label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax.axvline(dia_cutoff_max, linestyle='--', linewidth=3.0,
                   label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        plt.plot(xaxis, ypdf, linestyle='-', linewidth=3.0, label='Input stats')
        ax.fill_between(xaxis[ind], 0, ypdf[ind], alpha=0.3)
        ax.set_xlim(left=0.0, right=x_lim_eqd)
        ax.set_xlabel('Equivalent diameter (μm)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.tick_params(labelsize=12)
        if gs_data is not None:
            idata = np.nonzero(gs_data < x_lim_eqd)[0]
            ax.hist(gs_data[idata], bins=80, density=True, label='Experimental data')
        if gs_param is not None:
            if len(gs_param) < 5:
                xv = xaxis
            else:
                xv = np.linspace(gs_param[3], gs_param[4], 200)
            plot_lognorm(xv, gs_param[0], gs_param[1], gs_param[2], ax[0])
        plt.title(title, fontsize=18)
        plt.legend(fontsize=12)
    elif stats_dict["Grain type"] in ["Elongated"]:
        # Extract grain aspect ratio descriptors from input file
        sd_AR = stats_dict["Aspect ratio"]["sig"]
        scale_AR = stats_dict["Aspect ratio"]["scale"]
        if 'loc' in stats_dict["Aspect ratio"].keys():
            loc_AR = stats_dict["Aspect ratio"]["loc"]
        else:
            loc_AR = 0.0
        ar_cutoff_min = stats_dict["Aspect ratio"]["cutoff_min"]
        ar_cutoff_max = stats_dict["Aspect ratio"]["cutoff_max"]

        x_lim_ar = ar_cutoff_max * 3
        if ar_param is not None:
            x_lim_ar = max(x_lim_ar, ar_param[4] * 1.1)

        # Plot grain size distribution
        fig, ax = plt.subplots(2, 1, figsize=(8, 9))
        ax[0].axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax[0].axvline(dia_cutoff_max, linestyle='--', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        ax[0].plot(xaxis, ypdf, linestyle='-', linewidth=3.0, label='Input stats')
        ax[0].fill_between(xaxis[ind], 0, ypdf[ind], alpha=0.3)
        ax[0].set_xlim(left=0.0, right=x_lim_eqd)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=14)
        ax[0].set_ylabel('Density', fontsize=14)
        ax[0].tick_params(labelsize=12)
        if gs_data is not None:
            ax[0].hist(gs_data, bins=80, density=True, label='experimental data')
        if gs_param is not None:
            if len(gs_param) < 5:
                xv = xaxis
            else:
                xv = np.linspace(gs_param[3], gs_param[4], 200)
            plot_lognorm(xv, gs_param[0], gs_param[1], gs_param[2], ax[0])
        ax[0].legend(fontsize=12)
        ax[0].set_title(title, fontsize=18)

        # Plot aspect ratio statistics
        # Compute the Log-normal PDF
        xaxis = np.linspace(0.5 * ar_cutoff_min, 2 * ar_cutoff_max, 500)
        ypdf = lognorm.pdf(xaxis, sd_AR, loc=loc_AR, scale=scale_AR)
        # normalize density to region between min and max cutoff
        ind = np.nonzero(np.logical_and(xaxis >= ar_cutoff_min, xaxis <= ar_cutoff_max))[0]
        """if ar_data is None:
            area = np.trapz(ypdf[ind], xaxis[ind])
            if np.isclose(area, 0.0):
                area = 1.
            ypdf /= area"""
        ax[1].axvline(ar_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(ar_cutoff_min))
        ax[1].axvline(ar_cutoff_max, linestyle='--', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(ar_cutoff_max))
        ax[1].plot(xaxis, ypdf, linestyle='-', linewidth=3.0, label='Input stats')
        ax[1].fill_between(xaxis[ind], 0, ypdf[ind], alpha=0.3)
        ax[1].set_xlim(left=0.0, right=x_lim_ar)
        ax[1].set_xlabel('Aspect ratio', fontsize=14)
        ax[1].set_ylabel('Density', fontsize=14)
        ax[1].tick_params(labelsize=12)
        if ar_data is not None:
            ax[1].hist(ar_data, bins=15, density=True, label='Experimental data')
        if ar_param is not None:
            if len(gs_param) < 5:
                xv = xaxis
            else:
                xv = np.linspace(ar_param[3], ar_param[4], 200)
            plot_lognorm(xv, ar_param[0], ar_param[1], ar_param[2], ax[1])
        ax[1].legend(fontsize=12)
    elif stats_dict["Grain type"] in ["Free"]:
        # Compute reference grain aspect ratio descriptors from stats dict
        # Identify most likely rotation axis of ellepsoid with free semi-axes
        from kanapy.rve_stats import find_rot_axis
        semi_axs = stats_dict["Semi axes"]["scale"]
        cut_min = stats_dict["Semi axes"]["cutoff_min"]
        cut_max = stats_dict["Semi axes"]["cutoff_max"]
        irot = find_rot_axis(semi_axs[0], semi_axs[1], semi_axs[2])
        if irot == 0:
            ar_scale = 2.0 * semi_axs[0] / (semi_axs[1] + semi_axs[2])
            ar_cutoff_min = 2.0 * cut_min[0] / (cut_min[1] + cut_min[2])
            ar_cutoff_max = 2.0 * cut_max[0] / (cut_max[1] + cut_max[2])
        elif irot == 1:
            ar_scale = 2.0 * semi_axs[1] / (semi_axs[0] + semi_axs[2])
            ar_cutoff_min = 2.0 * cut_min[1] / (cut_min[0] + cut_min[2])
            ar_cutoff_max = 2.0 * cut_max[1] / (cut_max[0] + cut_max[2])
        elif irot == 2:
            ar_scale = 2.0 * semi_axs[2] / (semi_axs[1] + semi_axs[0])
            ar_cutoff_min = 2.0 * cut_min[2] / (cut_min[1] + cut_min[0])
            ar_cutoff_max = 2.0 * cut_max[2] / (cut_max[1] + cut_max[0])
        else:
            raise ValueError('Rotation axis not identified properly')
        ar_sig = stats_dict["Semi axes"]["sig"][irot]

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        # Plot grain size distribution
        ax[0].axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax[0].axvline(dia_cutoff_max, linestyle='--', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        ax[0].plot(xaxis, ypdf, linestyle='-', linewidth=3.0, label='Input stats')
        ax[0].fill_between(xaxis[ind], 0, ypdf[ind], alpha=0.3)
        ax[0].set_xlim(left=0.0, right=x_lim_eqd)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=14)
        ax[0].set_ylabel('Density', fontsize=14)
        ax[0].tick_params(labelsize=12)
        if gs_data is not None:
            ax[0].hist(gs_data, bins=80, density=True, label='Experimental data')
        if gs_param is not None:
            if len(gs_param) < 5:
                xv = xaxis
            else:
                xv = np.linspace(gs_param[3], gs_param[4], 200)
            plot_lognorm(xv, gs_param[0], gs_param[1], gs_param[2], ax[0])
        ax[0].legend(fontsize=12)
        ax[0].set_title(title, fontsize=18)

        # Plot aspect ratio statistics
        # Compute the Log-normal PDF
        xaxis = np.linspace(0.5 * ar_cutoff_min, 2 * ar_cutoff_max, 500)
        ypdf = lognorm.pdf(xaxis, ar_sig, scale=ar_scale)
        # normalize density to region between min and max cutoff
        ind = np.nonzero(np.logical_and(xaxis >= ar_cutoff_min, xaxis <= ar_cutoff_max))[0]
        """area = np.trapz(ypdf[ind], xaxis[ind])
        if np.isclose(area, 0.0):
            area = 1.
        ypdf /= area"""
        ax[1].axvline(ar_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(ar_cutoff_min))
        ax[1].axvline(ar_cutoff_max, linestyle='--', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(ar_cutoff_max))
        ax[1].plot(xaxis, ypdf, linestyle='-', linewidth=3.0, label='Input stats')
        ax[1].fill_between(xaxis[ind], 0, ypdf[ind], alpha=0.3)
        ax[1].set_xlabel('Aspect ratio', fontsize=14)
        ax[1].set_ylabel('Density', fontsize=14)
        ax[1].tick_params(labelsize=12)
        if ar_data is not None:
            ax[1].hist(ar_data, bins=15, density=True, label='Experimental data')
        if ar_param is not None:
            if len(gs_param) < 5:
                xv = xaxis
            else:
                xv = np.linspace(ar_param[3], ar_param[4], 200)
            plot_lognorm(xv, ar_param[0], ar_param[1], ar_param[2], ax[1])
        ax[1].legend(fontsize=12)
    else:
        raise ValueError('Value for "Grain_type" must be either "Equiaxed" or "Elongated".')

    if silent:
        return fig
    else:
        plt.show(block=True)
        if save_files:
            fig.savefig("Input_distribution.png", bbox_inches="tight")
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue")
            print("Input_distribution.png saved in the current working directory\n")


def plot_stats_dict(sdict, title=None, save_files=False):
    """
    Plot statistical data on semi-axes of effective ellipsoids in RVE as histogram

    Parameters
    ----------
    sdict : dict
        Dictionary containing semi-axis data and log-normal parameters.
        Keys should include 'a', 'b', 'c', and their corresponding '_sig' and '_scale'.
    title : str, optional
        Title for the plot
    save_files : bool, optional
        Whether to save the plot as PNG in the current working directory
    """
    shared_bins = np.histogram_bin_edges(sdict['eqd'], bins='fd')
    binNum = len(shared_bins)
    # Plot the histogram & PDF for equivalent diameter
    sns.set(color_codes=True)
    # Plot PDF

    loc = 0.0
    """
    sig = sdict['eqd_sig']
    sc = sdict['eqd_scale']
    xval = np.linspace(np.min(sdict['eqd']), np.max(sdict['eqd']), 50, endpoint=True)
    ypdf = lognorm.pdf(xval, sig, loc=loc, scale=sc)
    plt.figure(figsize=(12, 9))
    plt.plot(xval, ypdf, linestyle='-', linewidth=3.0, label='PDF')
    plt.fill_between(xval, 0, ypdf, alpha=0.3)
    # Plot histogram
    plt.hist(sdict['eqd'], density=True, bins=binNum, label='Data')
    plt.legend(loc="upper right", fontsize=16)
    plt.xlabel('equivalent diameter (μm)', fontsize=16)
    plt.ylabel('density', fontsize=16)
    plt.tick_params(labelsize=14)
    if title is not None:
        plt.title(title, fontsize=20)
    if save_files:
        if title is None:
            fname = 'equiv_diameter.png'
        else:
            fname = f'equiv_diameter_{title}.png'
        plt.savefig(fname, bbox_inches="tight")
        print(f"    '{fname}' is placed in the current working directory\n")
    plt.show()
    """

    # plot statistics of semi-axes
    cts = []
    val = []
    xmin_gl = np.inf
    xmax_gl = 0.
    label = []
    nb = 0
    for key in ['a', 'b', 'c']:
        counts, bins = np.histogram(sdict[key])
        cts.append(counts)
        val.append(bins[:-1])
        label.append(f'Semi-axis {key}')
        nb = np.maximum(nb, len(bins))
        xmin_gl = min(xmin_gl, min(sdict[key]))
        xmax_gl = max(xmax_gl, max(sdict[key]))

    # Plot the histogram & PDF
    sns.set(color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Plot histogram
    """ax[0].hist(val, weights=cts, density=False, bins=nb, label=label)
    ax[0].legend(loc="upper right", fontsize=12)
    ax[0].set_xlabel('length of semi-axis (μm)', fontsize=14)
    ax[0].set_ylabel('frequency', fontsize=14)
    ax[0].tick_params(labelsize=12)"""
    # Plot PDF
    xval = np.linspace(xmin_gl, xmax_gl, 50, endpoint=True)
    for i, key in enumerate(['a', 'b', 'c']):
        ypdf = lognorm.pdf(xval, sdict[f'{key}_sig'], loc=loc, scale=sdict[f'{key}_scale'])
        ax.plot(xval, ypdf, linestyle='-', linewidth=3.0, label=label[i])
        ax.fill_between(xval, ypdf, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xlabel('Length of semi-axis (μm)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(labelsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)
        #ax[1].set_title(title, fontsize=16)
    if save_files:
        if title is None:
            fname = 'semi_axes.png'
        else:
            fname = f'semi_axes_{title}.png'
        plt.savefig(fname, bbox_inches="tight")
        print(f"    '{fname}' is placed in the current working directory\n")
    plt.show()
    return

