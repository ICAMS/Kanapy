# -*- coding: utf-8 -*-
"""
Subroutines for analysis of statistical descriptors of RVEs:
 * original particles (or their inner polyhedra)
 * voxel structure
 * polyhedral grain structure

@author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
March 2024
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from kanapy.plotting import plot_stats_dict


def get_stats(particle_data, geometry, units, nphases, ngrains):
    """
    Compare the geometries of particles used for packing and the resulting
    grains.

    Parameters
    ----------


    Returns
    -------
    output_data : dict
        Statistical information about particle and grain geometries.

    """
    if particle_data is not None and nphases == len(particle_data):
        # convert particle geometries to dicts
        par_eqDia = dict()
        par_majDia = dict()
        par_minDia = dict()
        for ip, part in enumerate(particle_data):
            # Analyse geometry of particles used for packing algorithm
            par_eqDia[ip] = np.array(part['Equivalent_diameter'])
            if part['Type'] == 'Elongated':
                par_majDia[ip] = np.array(part['Major_diameter'])
                par_minDia[ip] = np.array(part['Minor_diameter1'])

    # convert grain geometries to dicts
    grain_eqDia = dict()
    grain_majDia = dict()
    grain_minDia = dict()
    for i in range(nphases):
        grain_eqDia[i] = []
        grain_majDia[i] = []
        grain_minDia[i] = []
    for i, igr in enumerate(geometry['Grains'].keys()):
        ip = geometry['Grains'][igr]['Phase']
        grain_eqDia[ip].append(geometry['Grains'][igr]['eqDia'])
        grain_minDia[ip].append(geometry['Grains'][igr]['minDia'])
        grain_majDia[ip].append(geometry['Grains'][igr]['majDia'])

    output_data_list = []
    for ip in range(nphases):
        # Create dictionaries to store the data generated
        output_data = {'Number': ngrains[ip],
                       'Unit_scale': units,
                       'Grain_Equivalent_diameter': np.array(grain_eqDia[ip]),
                       'Grain_Major_diameter': np.array(grain_majDia[ip]),
                       'Grain_Minor_diameter': np.array(grain_minDia[ip])}
        if particle_data is not None:
            # Compute the L1-error between particle and grain geometries for phases
            error = l1_error_est(par_eqDia[ip], grain_eqDia[ip])
            print('\n    L1 error phase {} between particle and grain geometries: {}'
                  .format(ip, round(error, 5)))
            # Store Particle data in output dict
            output_data['Grain_type'] = particle_data[ip]['Type']
            output_data['Particle_Equivalent_diameter'] = par_eqDia[ip]
            output_data['L1-error'] = error
            if particle_data[ip]['Type'] == 'Elongated':
                output_data['Particle_Major_diameter'] = par_majDia[ip]
                output_data['Particle_Minor_diameter'] = par_minDia[ip]
        output_data_list.append(output_data)
    return output_data_list


def arr2mat(mc):
    """
    Converts a numpy array into a symmetric matrix.

    Parameters
    ----------
    mc

    Returns
    -------

    """
    return np.array([[mc[0], mc[5], mc[4]],
                     [mc[5], mc[1], mc[3]],
                     [mc[4], mc[3], mc[2]]])


def con_fun(mc):
    """Constraint: matrix must be positive definite, for symmetric matrix all eigenvalues
    must be positive. Constraint will penalize negative eigenvalues.
    """
    eigval, eigvec = np.linalg.eig(arr2mat(mc))
    return np.min(eigval) * 1000


def find_rot_axis(len_a, len_b, len_c):
    """
    Find the rotation axis of an ellipsoid defined be three semi-axes and return the
    equivalent values for the semi-axes along and transversal to the rotation axis.

    Parameters
    ----------
    len_a
    len_b
    len_c

    Returns
    -------
    rot_ax
    trans_ax

    """
    ar_list = [np.abs(len_a / len_b - 1.0), np.abs(len_b / len_a - 1.0),
               np.abs(len_b / len_c - 1.0), np.abs(len_c / len_b - 1.0),
               np.abs(len_c / len_a - 1.0), np.abs(len_a / len_c - 1.0)]
    minval = np.min(ar_list)
    if minval > 0.15:
        # no clear rotational symmetry, choose longest axis as rot_ax diameter
        rot_ax = len_a
        trans_ax = 0.5 * (len_b + len_c)
    else:
        ind = np.argmin(ar_list)  # identify two axes with aspect ratio closest to 1
        if ind in [0, 1]:
            rot_ax = len_c  # rot_ax diameter along the rotational axis of grain
            trans_ax = 0.5 * (len_a + len_b)  # trans_ax diameter is average of transversal axes
        elif ind in [2, 3]:
            rot_ax = len_a
            trans_ax = 0.5 * (len_c + len_b)
        else:
            rot_ax = len_b
            trans_ax = 0.5 * (len_a + len_c)
    return rot_ax, trans_ax


def get_ln_param(data):
    # sig, loc, sc = lognorm.fit(sdict['eqd'])
    ind = np.nonzero(data > 1.e-5)[0]
    log_data = np.log(data[ind])
    scale = np.exp(np.median(log_data))
    sig = np.std(log_data)
    return sig, scale


def pts_in_ellips(Mcomp, pts):
    """ Check how well points fulfill equation of ellipsoid
    (pts-ctr)^T M (pts-ctr) = 1

    Parameters
    ----------
    Mcomp
    pts

    Returns
    -------

    """
    if Mcomp.shape != (6,):
        raise ValueError(f'Matrix components must be given as array with shape (6,), not {Mcomp.shape}')
    ctr = np.average(pts, axis=0)
    pcent = pts - ctr[np.newaxis, :]
    score = 0.
    for x in pcent:
        mp = np.matmul(arr2mat(Mcomp), x)
        score += np.abs(np.dot(x, mp) - 1.0)
    return score / len(pts)


def calc_stats_dict(a, b, c, eqd):
    arr_a = np.sort(a)
    arr_b = np.sort(b)
    arr_c = np.sort(c)
    arr_eqd = np.sort(eqd)
    a_sig, a_sc = get_ln_param(arr_a)
    b_sig, b_sc = get_ln_param(arr_b)
    c_sig, c_sc = get_ln_param(arr_c)
    e_sig, e_sc = get_ln_param(arr_eqd)
    vrot, vtrans = find_rot_axis(a_sc, b_sc, c_sc)
    sd = {
        'a': arr_a,
        'b': arr_b,
        'c': arr_c,
        'eqd': arr_eqd,
        'a_sig': a_sig,
        'b_sig': b_sig,
        'c_sig': c_sig,
        'eqd_sig': e_sig,
        'a_scale': a_sc,
        'b_scale': b_sc,
        'c_scale': c_sc,
        'eqd_scale': e_sc,
        'ax_rot': vrot,
        'ax_trans': vtrans,
        'aspect_ratio': vrot / vtrans
    }
    return sd


def get_stats_vox(mesh, minval=1.e-5, show_plot=True, verbose=False, ax_max=None):
    """
    Get statistics about the microstructure from voxels, by fitting a 3D ellipsoid to
    each grain.

    Parameters
    ----------
    mesh
    minval
    show_plot
    verbose
    ax_max

    Returns
    -------

    """
    if ax_max is not None:
        minval = max(minval, 1. / ax_max ** 2)
    cons = ({'type': 'ineq', 'fun': con_fun})  # constraints for minimization
    opt = {'maxiter': 200}  # options for minimization
    mc = np.array([1., 1., 1., 0., 0., 0.])  # start value of matrix for minimization
    gfac = 3.0 / (4.0 * np.pi)
    arr_a = []
    arr_b = []
    arr_c = []
    arr_eqd = []
    for igr, vlist in mesh.grain_dict.items():
        nodes = set()
        for iv in vlist:
            nodes.update(mesh.voxel_dict[iv])
        ind = np.array(list(nodes), dtype=int) - 1
        pts_all = mesh.nodes[ind, :]
        hull = ConvexHull(pts_all)
        eqd = 2.0 * (gfac * hull.volume) ** (1.0 / 3.0)
        pts = hull.points[hull.vertices]  # outer nodes of grain no. igr
        # find best fitting ellipsoid to points
        rdict = minimize(pts_in_ellips, x0=mc, args=(pts,), method='SLSQP',
                         constraints=cons, options=opt)
        if not rdict['success']:
            if verbose:
                print(f'Optimization failed for grain {igr}')
                print(rdict['message'])
            continue
        eval, evec = np.linalg.eig(arr2mat(rdict['x']))
        if any(eval <= minval):
            if verbose:
                print(f'Matrix for grain {igr} not positive definite or semi-axes too large. '
                      f'Eigenvalues: {eval}')
            continue
        if verbose:
            print(f'Optimization succeeded for grain {igr} after {rdict["nit"]} iterations.')
            print(f'Eigenvalues: {eval}')
            print(f'Eigenvectors: {evec}')
        # Semi-axes of ellipsoid
        ea = 1. / np.sqrt(eval[0])
        eb = 1. / np.sqrt(eval[1])
        ec = 1. / np.sqrt(eval[2])
        arr_a.append(ea)
        arr_b.append(eb)
        arr_c.append(ec)
        arr_eqd.append(eqd)

        """ Plot points on hull with fitted ellipsoid
        # Points on the outer surface of optimal ellipsoid
        nang = 100
        col = [0.7, 0.7, 0.7, 0.5]
        ctr = np.average(pts, axis=0)
        u = np.linspace(0, 2 * np.pi, nang)
        v = np.linspace(0, np.pi, nang)

        # Cartesian coordinates that correspond to the spherical angles:
        xval = ea * np.outer(np.cos(u), np.sin(v))
        yval = eb * np.outer(np.sin(u), np.sin(v))
        zval = ec * np.outer(np.ones_like(u), np.cos(v))

        # combine the three 2D arrays element wise
        surf_pts = np.stack(
            (xval.ravel(), yval.ravel(), zval.ravel()), axis=1)
        surf_pts = surf_pts.dot(evec.transpose())  # rotate to eigenvector frame
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = (surf_pts[:, 0] + ctr[None, 0]).reshape((100, 100))
        y = (surf_pts[:, 1] + ctr[None, 1]).reshape((100, 100))
        z = (surf_pts[:, 2] + ctr[None, 2]).reshape((100, 100))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=col, linewidth=0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')
        plt.show()"""
    # calculate and print statistical parameters
    vox_stats_dict = calc_stats_dict(arr_a, arr_b, arr_c, arr_eqd)
    print('\n--------------------------------------------------------')
    print('Statistical microstructure parameters in voxel structure')
    print('--------------------------------------------------------')
    print('Median lengths of semi-axes of fitted ellipsoids in micron')
    print(f'a: {vox_stats_dict["a_scale"]:.3f}, b: {vox_stats_dict["b_scale"]:.3f}, '
          f'c: {vox_stats_dict["c_scale"]:.3f}')
    av_std = np.mean([vox_stats_dict['a_sig'], vox_stats_dict['b_sig'], vox_stats_dict['c_sig']])
    print(f'Average standard deviation of semi-axes: {av_std:.4f}')
    print('\nAssuming rotational symmetry in grains')
    print(f'Median grain size along rotational axis: {vox_stats_dict["ax_rot"]:.3f} micron')
    print(f'Median grain size transversal to rotational axis: {vox_stats_dict["ax_trans"]:.3f} micron')
    print(f'Median aspect ratio: {vox_stats_dict["aspect_ratio"]:.3f}')
    print('\nGrain size')
    print(f'Median equivalent grain diameter: {vox_stats_dict["eqd_scale"]:.3f} micron')
    print(f'Standard deviation of equivalent grain diameter: {vox_stats_dict["eqd_sig"]:.4f}')
    print('--------------------------------------------------------')
    if show_plot:
        plot_stats_dict(vox_stats_dict, title='Statistics of voxel structure')

    return vox_stats_dict


def get_stats_poly(grains, minval=1.e-5, show_plot=True, verbose=False, ax_max=None):
    """ Extract statistics about the microstructure from polyhedral grains
        by fitting a 3D ellipsoid to each polyhedron.

        Parameters
        ----------
        geom
        minval
        show_plot
        verbose
        ax_max

        Returns
        -------

        """
    if ax_max is not None:
        minval = max(minval, 1. / ax_max ** 2)
    cons = ({'type': 'ineq', 'fun': con_fun})  # constraints for minimization
    opt = {'maxiter': 200}  # options for minimization
    mc = np.array([1., 1., 1., 0., 0., 0.])  # start value of matrix for minimization
    arr_a = []
    arr_b = []
    arr_c = []
    arr_eqd = []
    for gid, pc in grains.items():
        pts = pc['Points']
        rdict = minimize(pts_in_ellips, x0=mc, args=(pts,), method='SLSQP',
                         constraints=cons, options=opt)
        if not rdict['success']:
            if verbose:
                print(f'Optimization failed for particle {gid}')
                print(rdict['message'])
            continue
        eval, evec = np.linalg.eig(arr2mat(rdict['x']))
        if any(eval <= minval):
            if verbose:
                print(f'Matrix for particle {gid} not positive definite or semi-axes too large. '
                      f'Eigenvalues: {eval}')
            continue
        if verbose:
            print(f'Optimization succeeded for particle {gid} after {rdict["nit"]} iterations.')
            print(f'Eigenvalues: {eval}')
            print(f'Eigenvectors: {evec}')
        # Semi-axes of ellipsoid
        ea = 1. / np.sqrt(eval[0])
        eb = 1. / np.sqrt(eval[1])
        ec = 1. / np.sqrt(eval[2])
        eqd = 2.0 * (ea * eb * ec) ** (1.0 / 3.0)
        arr_a.append(ea)
        arr_b.append(eb)
        arr_c.append(ec)
        arr_eqd.append(eqd)

    # calculate statistical parameters
    poly_stats_dict = calc_stats_dict(arr_a, arr_b, arr_c, arr_eqd)
    print('\n----------------------------------------------------------')
    print('Statistical microstructure parameters of polyhedral grains')
    print('----------------------------------------------------------')
    print('Median lengths of semi-axes of fitted ellipsoids in micron')
    print(f'a: {poly_stats_dict["a_scale"]:.3f}, b: {poly_stats_dict["b_scale"]:.3f}, '
          f'c: {poly_stats_dict["c_scale"]:.3f}')
    av_std = np.mean([poly_stats_dict['a_sig'], poly_stats_dict['b_sig'], poly_stats_dict['c_sig']])
    print(f'Average standard deviation of semi-axes: {av_std:.4f}')
    print('\nAssuming rotational symmetry in grains')
    print(f'Median grain size along rotational axis: {poly_stats_dict["ax_rot"]:.3f} micron')
    print(f'Median grain size transversal to rotational axis: {poly_stats_dict["ax_trans"]:.3f} micron')
    print(f'Median aspect ratio: {poly_stats_dict["aspect_ratio"]:.3f}')
    print('\nGrain size')
    print(f'Median equivalent grain diameter: {poly_stats_dict["eqd_scale"]:.3f} micron')
    print(f'Standard deviation of equivalent grain diameter: {poly_stats_dict["eqd_sig"]:.4f}')
    print('--------------------------------------------------------')
    if show_plot:
        title = 'Statistics of polyhedral grains'
        plot_stats_dict(poly_stats_dict, title=title)

    return poly_stats_dict


def get_stats_part(part, minval=1.e-5, show_plot=True, verbose=False, ax_max=None):
    """ Extract statistics about the microstructure from particles. If inner structure is contained
    by fitting a 3D ellipsoid to each structure.

    Parameters
    ----------
    part
    minval
    show_plot
    verbose
    ax_max

    Returns
    -------

    """
    if ax_max is not None:
        minval = max(minval, 1. / ax_max ** 2)
    cons = ({'type': 'ineq', 'fun': con_fun})  # constraints for minimization
    opt = {'maxiter': 200}  # options for minimization
    mc = np.array([1., 1., 1., 0., 0., 0.])  # start value of matrix for minimization
    arr_a = []
    arr_b = []
    arr_c = []
    arr_eqd = []
    for pc in part:
        if pc.inner is not None:
            pts = pc.inner.points
            rdict = minimize(pts_in_ellips, x0=mc, args=(pts,), method='SLSQP',
                             constraints=cons, options=opt)
            if not rdict['success']:
                if verbose:
                    print(f'Optimization failed for particle {pc.id}')
                    print(rdict['message'])
                continue
            eval, evec = np.linalg.eig(arr2mat(rdict['x']))
            if any(eval <= minval):
                if verbose:
                    print(f'Matrix for particle {pc.id} not positive definite or semi-axes too large. '
                          f'Eigenvalues: {eval}')
                continue
            if verbose:
                print(f'Optimization succeeded for particle {pc.id} after {rdict["nit"]} iterations.')
                print(f'Eigenvalues: {eval}')
                print(f'Eigenvectors: {evec}')
            # Semi-axes of ellipsoid
            ea = 1. / np.sqrt(eval[0])
            eb = 1. / np.sqrt(eval[1])
            ec = 1. / np.sqrt(eval[2])
            eqd = 2.0 * (ea * eb * ec) ** (1.0 / 3.0)
            arr_a.append(ea)
            arr_b.append(eb)
            arr_c.append(ec)
            arr_eqd.append(eqd)
        else:
            eqd = 2.0 * (pc.a * pc.b * pc.c) ** (1.0 / 3.0)
            arr_a.append(pc.a)
            arr_b.append(pc.b)
            arr_c.append(pc.c)
            arr_eqd.append(eqd)

    # calculate statistical parameters
    part_stats_dict = calc_stats_dict(arr_a, arr_b, arr_c, arr_eqd)
    print('\n--------------------------------------------------')
    print('Statistical microstructure parameters of particles')
    print('--------------------------------------------------')
    print('Median lengths of semi-axes of fitted ellipsoids in micron')
    print(f'a: {part_stats_dict["a_scale"]:.3f}, b: {part_stats_dict["b_scale"]:.3f}, '
          f'c: {part_stats_dict["c_scale"]:.3f}')
    av_std = np.mean([part_stats_dict['a_sig'], part_stats_dict['b_sig'], part_stats_dict['c_sig']])
    print(f'Average standard deviation of semi-axes: {av_std:.4f}')
    print('\nAssuming rotational symmetry in grains')
    print(f'Median grain size along rotational axis: {part_stats_dict["ax_rot"]:.3f} micron')
    print(f'Median grain size transversal to rotational axis: {part_stats_dict["ax_trans"]:.3f} micron')
    print(f'Median aspect ratio: {part_stats_dict["aspect_ratio"]:.3f}')
    print('\nGrain size')
    print(f'Median equivalent grain diameter: {part_stats_dict["eqd_scale"]:.3f} micron')
    print(f'Standard deviation of equivalent grain diameter: {part_stats_dict["eqd_sig"]:.4f}')
    print('--------------------------------------------------------')
    if show_plot:
        if part[0].inner is None:
            title = 'Particle statistics'
        else:
            title = 'Statistics of inner particle structures'
        plot_stats_dict(part_stats_dict, title=title)

    return part_stats_dict


def l1_error_est(par_eqDia, grain_eqDia):
    r"""
    Evaluates the L1-error between the particle- and output RVE grain
    statistics with respect to Major, Minor & Equivalent diameters.

    .. note:: 1. Particle information is read from (.json) file generated by
                 :meth:`kanapy.input_output.particleStatGenerator`.
                 And RVE grain information is read from the (.json) files
                 generated by :meth:`kanapy.voxelization.voxelizationRoutine`.

              2. The L1-error value is written to the 'output_statistics.json'
                 file.
    """

    print('')
    print('Computing the L1-error between input and output diameter distributions.',
          end="")

    # Concatenate both arrays to compute shared bins
    # NOTE: 'doane' produces better estimates for non-normal datasets
    total_eqDia = np.concatenate([par_eqDia, grain_eqDia])
    shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')

    # Compute the histogram for particles and grains
    hist_par, _ = np.histogram(par_eqDia, bins=shared_bins)
    hist_gr, _ = np.histogram(grain_eqDia, bins=shared_bins)

    # Normalize the values
    hist_par = hist_par / np.sum(hist_par)
    hist_gr = hist_gr / np.sum(hist_gr)

    # Compute the L1-error between particles and grains
    l1_value = np.sum(np.abs(hist_par - hist_gr))
    return l1_value
