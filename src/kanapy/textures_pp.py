"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from kanapy.util import MTEX_DIR, ROOT_DIR, MAIN_DIR
# from kanapy.ebsd_utils.ebsd import EBSD
from orix import io, plot, quaternion
from orix.vector import Miller
from scipy.stats import lognorm, vonmises
import logging


def find_similar_regions(array, tolerance=0.087, connectivity=1):
    """
    Identifies connected regions of similar values in a 2D array.

    Parameters:
        array (ndarray): 2D NumPy array with values.
        tolerance (float): Max allowed difference between connected values.
        connectivity (int): 1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        labeled_array (ndarray): 2D array of region labels.
        num_features (int): Total number of connected regions found.
    """

    def neighbors(r, c):
        if connectivity == 1:
            return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        else:
            return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                    if not (i == 0 and j == 0)]

    array = np.asarray(array)

    visited = np.full(array.shape, False, dtype=bool)
    labeled_array = np.zeros(array.shape, dtype=int)
    current_label = 1

    rows, cols = array.shape

    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                ref_val = array[r, c]
                stack = [(r, c)]

                while stack:
                    i, j = stack.pop()
                    if (0 <= i < rows) and (0 <= j < cols) and not visited[i, j]:
                        if abs(array[i, j] - ref_val) <= tolerance:
                            visited[i, j] = True
                            labeled_array[i, j] = current_label
                            stack.extend(neighbors(i, j))

                current_label += 1

    return labeled_array, current_label - 1


class EBSDmap:
    """Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate
    synthetic RVEs
    """

    def __init__(self, fname, matname=None, gs_min=3, vf_min=0.03, show_plot=True, hist=None):
        """
        Generate microstructural data from EBSD maps

        Parameters
        ----------
        fname : str
            filname incl. path to EBSD file.
        matname : str, optional
            Name of material, depracted. The default is None.
        gs_min : int, optional
            Minimum grain size in pixels, smaller grains will be disregarded
            for the statistical analysis. The default is 3.
        vf_min : int, optional
            Minimum volume fracture, phases with smaller values will be
            disregarded. The default is 0.03.
        plot : bool, optional
            Plot microstructures. The default is True.

        Returns
        -------
        None.
        
        Attributes
        ----------
        ms_data : list of dict
            List of dictionaries with phase specific microstructure
            information.  
            name : name of phase  
            vf : volume fraction
            ngrain : number of grains in phase
            ori : matlab object with grain orientations
            cs : matlab object with crystal structure
            grains : matlab object with grains in each phase
            omega : orintations of major grain axis
            gs_param : statistical grain size parameters
            gs_data : grain sizesvonmises
            ar_param
            ar_data
            om_param
            om_data
            
            
            
        eng : handle for Matlab engine with MTEX data

        """
        if matname is not None:
            logging.warning('Use of parameter "matname" is depracted.')
        if hist is None:
            hist = plot

        # read EBSD map and return the matlab.object of MTEX class EBSD
        # emap = EBSD(fname)
        emap = io.load(fname)
        h, w = emap.shape

        # determine number of phases and generate histogram
        Nphase = len(emap.phases.ids)  # number of phases
        offs = 0 if 0 in emap.phases.ids else 1  # in CTX maps, there is no phase "0"
        npx = emap.size  # total number of pixels in EBSD map
        if h * w != npx:
            raise ValueError(f"Size of map ({npx} px) does not match its shape: {h, w}")
        phist = np.histogram(emap.phase_id, Nphase + offs)

        if show_plot and 'ci' in emap.prop.keys():
            val = emap.prop['ci']
            if len(val) == npx:
                bmap = val
            else:
                bmap = np.zeros(npx)
                bmap[emap.phase_id == 0] = val
            plt.imshow(bmap.reshape((h, w)))
            plt.title('CI values in EBSD map')
            plt.colorbar(label="CI")
            plt.show()

        # read phase names and calculate volume fractions
        self.ms_data = []
        for i, ind in enumerate(emap.phases.ids):
            if ind == -1:
                continue
            vf = phist[0][i + offs] / npx
            if vf < vf_min:
                continue
            data = dict()  # initialize data dictionary
            data['vf'] = vf
            data['name'] = emap.phases.names[i]
            data['index'] = ind

            # generate phase-specific orientations
            ori_e = emap[emap.phase_id == ind].orientations.in_euler_fundamental_region()
            data['ori'] = quaternion.Orientation.from_euler(ori_e)
            data['cs'] = emap.phases[ind].point_group.laue
            # analyze grain boundaries
            array = np.zeros(npx)
            val = data['ori'].angle
            array[emap.phase_id == 0] = val
            labels, n_regions = find_similar_regions(array.reshape((h, w)),
                                                     tolerance=5*np.pi/180,
                                                     connectivity=2)

            print(f"Phase #{data['index']} ({data['name']}): Identified {n_regions} grains")

            """grains_full = eng.calcGrains(ebsd_h, 'boundary', 'tight', 'angle',
                                         5 * (np.pi / 180.0))
            # filter out small grains
            eng.workspace["grains_w"] = grains_full
            data['grains'] = eng.eval("grains_w(grains_w.grainSize > {})"
                                      .format(gs_min))

            # use MTEX function to analyze grains and plot ellipses around grain
            # centres; calculate orientation, long and short axes of ellipses
            omega_r, ha, hb = eng.principalComponents(data['grains'],
                                                      nargout=3)
            omega = np.array(omega_r)[:, 0]
            data['omega'] = omega
            data['ngrain'] = len(omega)"""
            if show_plot:
                # plot identified regions
                plt.imshow(labels, cmap='flag')
                plt.title(f"Phase #{data['index']} ({data['name']}): Identified Grains: {n_regions}")
                plt.colorbar(label='Grain Number')
                plt.show()

                # plot misorientation field
                val = data['ori'].angle
                if len(val) == npx:
                    bmap = val
                else:
                    bmap = np.zeros(npx)
                    bmap[emap.phase_id == ind] = val
                plt.imshow(bmap.reshape((h, w)))
                plt.title('Misorientation angle wrt reference')
                plt.colorbar(label="Misorientation (rad)")
                plt.show()

                # plot EBSD map for current phase
                # Get IPF colors
                ipf_key = plot.IPFColorKeyTSL(data['cs'])
                rgb_val = ipf_key.orientation2color(data['ori'])
                # set pixels of other phases to black
                rgb_all = np.zeros((npx, 3))
                rgb_all[emap.phase_id == ind] = rgb_val
                fig = emap.plot(
                    rgb_all,
                    return_figure=True,
                    figure_kwargs={"figsize": (12, 8)},
                )
                fig.show()

                # plot inverse pole figure
                # <111> poles in the sample reference frame
                t_fe = Miller(uvw=[1, 1, 1], phase=emap.phases[ind]).symmetrise(unique=True)
                t_fe_all = data['ori'].inv().outer(t_fe)
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="stereographic")
                ax.scatter(t_fe_all)
                ax.set_labels("X", "Y", None)
                ax.set_title(data['name'] + r" $\left<111\right>$ PF")
                plt.show()
                """
                eng.plot(ebsd_h, data['ori'])
                eng.hold('on', nargout=0)
                # plot grain boundaries into EBSD map
                eng.plot(eng.getfield(data['grains'], 'boundary'), 'linewidth', 2.0,
                         'micronbar', 'on')
                # evalute centres of grains
                centres = eng.getfield(data['grains'], 'centroid')
                eng.plotEllipse(centres, ha, hb, omega_r, 'lineColor', 'r',
                                'linewidth', 2.0, nargout=0)
                eng.hold('off', nargout=0)"""
                # eng.exportgraphics(gcf,'ebsd_map.png','Resolution',300)

            # Evaluate grain shape statistics
            # generate dict for statistical input for geometry module

            # grain equivalent diameter
            deq = 2.0 * np.array(eng.equivalentRadius(data['grains']))[:, 0]
            # dsig, doffs, dscale = lognorm.fit(deq)  # fit log normal distribution
            doffs = 0.
            deq_log = np.log(deq)
            dscale = med_eq = np.exp(np.median(deq_log))  # lognorm.median(dsig, loc=doffs, scale=dscale)
            dsig = std_eq = np.std(deq_log)  # lognorm.std(dsig, loc=doffs, scale=dscale)
            data['gs_param'] = np.array([dsig, doffs, dscale])
            data['gs_data'] = deq
            data['gs_moments'] = [med_eq, std_eq]
            if hist:
                # plot distribution of grain sizes
                fig, ax = plt.subplots()
                x = np.linspace(np.amin(deq), np.amax(deq), 150)
                y = lognorm.pdf(x, dsig, loc=doffs, scale=dscale)
                ax.plot(x, y, '-r', label='fit')
                ax.hist(deq, bins=20, density=True, label='data')
                plt.legend()
                plt.title('Histogram of grain equivalent diameters')
                plt.xlabel('Equivalent diameter (micron)')
                plt.ylabel('Normalized frequency')
                plt.show()

            # grain aspect ratio
            asp = np.array(eng.aspectRatio(data['grains']))[:, 0]
            # asig, aoffs, ascale = lognorm.fit(asp)  # fit log normal distribution
            aoffs = 0.
            asp_log = np.log(asp)
            ascale = med_ar = np.exp(np.median(asp_log))  # lognorm.median(asig, loc=aoffs, scale=ascale)
            asig = std_ar = np.std(asp_log)  # lognorm.std(asig, loc=aoffs, scale=ascale)
            data['ar_param'] = np.array([asig, aoffs, ascale])
            data['ar_data'] = asp
            data['ar_moments'] = [med_ar, std_ar]
            if hist:
                # plot distribution of aspect ratios
                fig, ax = plt.subplots()
                x = np.linspace(np.amin(asp), np.amax(asp), 150)
                y = lognorm.pdf(x, asig, loc=aoffs, scale=ascale)
                ax.plot(x, y, '-r', label='fit lognorm')
                ax.hist(asp, bins=20, density=True, label='density')
                plt.legend()
                plt.title('Histogram of grain aspect ratio')
                plt.xlabel('aspect ratio')
                plt.ylabel('normalized frequency')
                plt.show()

            # angles of main axis
            # fit von Mises distribution (circular normal distribution) to data
            omega_p = 2.0 * omega - np.pi  # rescale angles from [0, pi] to [-pi,pi] for von Mises distr. fit
            kappa, oloc, oscale = vonmises.fit(omega_p)
            med_om = vonmises.median(kappa, loc=oloc)  # scale parameter has no effect on vonmises distribution
            std_om = vonmises.std(kappa, loc=oloc)
            data['om_param'] = np.array([kappa, oloc])
            data['om_data'] = omega_p
            data['om_moments'] = [med_om, std_om]
            if hist:
                fig, ax = plt.subplots()
                x = np.linspace(-np.pi, np.pi, 200)  # np.amin(omega), np.amax(omega), 150)
                y = vonmises.pdf(x, kappa, loc=oloc)
                ax.plot(0.5 * (x + np.pi), 2 * y, '-r', label='fit')
                ax.hist(omega, bins=40, density=True, label='data')
                plt.legend()
                plt.title('Histogram of tilt angles of major axes')
                plt.xlabel('angle (rad)')
                plt.ylabel('normalized frequency')
                plt.show()

            print('Analyzed microstructure with {} grains.'
                  .format(data['ngrain']))
            print(f'Median values: equiv. diameter: {med_eq.round(3)} micron, ' +
                  f'aspect ratio: {med_ar.round(3)}, ' +
                  f'tilt angle: {(med_om * 180 / np.pi).round(3)}°')
            print(f'Std. dev: equivalent diameter: {std_eq.round(3)} micron, ' +
                  f'aspect ratio: {std_ar.round(3)}, ' +
                  f'tilt angle: {(std_om * 180 / np.pi).round(3)}°')
            self.ms_data.append(data)
        return

    def calcORI(self, Ng, iphase=0, shared_area=None, nbins=12):
        """
        Estimate optimum kernel half-width and produce reduced set of
        orientations for given number of grains.
        
        Parameters
        ----------
        Ng : int
            Numbr of grains for which orientation is requested.
        iphase : int, optional
            Phase for which data is requested. The default is 0.
        shared_area : array, optional
            Grain boundary data. The default is None.
        nbins : int, optional
            number of bins for GB texture. The default is 12.

        Returns
        -------
        ori : (Ng, 3)-array
            Array with Ng Euler angles.

        """
        ms = self.ms_data[iphase]
        orired, odfred, ero = \
            self.eng.textureReconstruction(Ng, 'orientation',
                                           ms['ori'], 'grains', ms['grains'], nargout=3)

        if shared_area is None:
            return np.array(self.eng.Euler(orired))
        else:
            orilist, ein, eout, mbin = \
                self.eng.gb_textureReconstruction(ms['grains'], orired,
                                                  matlab.double(shared_area), nbins, nargout=4)
            return np.array(self.eng.Euler(orilist))

    def showIPF(self):
        """
        Plot IPF key.

        Returns
        -------
        None.

        """
        for ms in self.ms_data:
            ipfKey = self.eng.ipfColorKey(ms['ori'], nargout=1)
            self.eng.plot(ipfKey, nargout=0)


def get_ipf_colors(ori_list, color_key=0):
    """
    Get colors of list of orientations (in radians).
    Assumes cubic crystal symmetry and cubic specimen symmetry.

    Parameters
    ----------
    ori_list: (N, 3) ndarray
        List of N Euler angles in radians
    
    Returns
    -------
    colors: (N, 3) ndarray
        List of RGB values

    """
    # start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(MTEX_DIR, nargout=0)
    eng.addpath(ROOT_DIR, nargout=0)
    mat_path = os.path.join(MAIN_DIR, 'src', 'kanapy')
    eng.addpath(mat_path, nargout=0)
    eng.startup(nargout=0)

    # get colors
    """Create possibility to pass CS to MTEX"""
    colors = eng.get_ipf_col(ori_list, color_key, nargout=1)
    return np.array(colors)


def createOriset(num, ang, omega, hist=None, shared_area=None,
                 cs='m-3m', Nbase=10000):
    """
    Create a set of num Euler angles according to the ODF defined by the 
    set of Euler angles ang and the kernel half-width omega.
    Example: Goss texture: ang = [0, 45, 0], omega = 5

    Parameters
    ----------
    num : int
        Number of Euler angles in set to be created.
    ang : (3, ) or (M, 3) array
        Set of Euler angles (in degrees) defining the ODF.
    omega : float
        Half-width of kernel in degrees.
    hist : array, optional
        Histogram of MDF. The default is None.
    shared_area: array, optional
        The shared area between pairs of grains. The default in None.
    cs : str, optional
        Crystal symmetry group. The default is 'm3m'.
    Nbase : int, optional
        Base number of orientations for random texture. The default is 10000

    Returns
    -------
    ori : (num, 3) array
        Set of Euler angles

    """
    # start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(MTEX_DIR, nargout=0)
    eng.addpath(ROOT_DIR, nargout=0)
    mat_path = os.path.join(MAIN_DIR, 'src', 'kanapy')
    eng.addpath(mat_path, nargout=0)
    eng.startup(nargout=0)

    # prepare parameters
    deg = np.pi / 180.
    omega *= deg
    ang = np.array(ang) * deg
    cs_ = eng.crystalSymmetry(cs)
    ori = eng.orientation('Euler', float(ang[0]), float(ang[1]),
                          float(ang[2]), cs_)
    psi = eng.deLaValleePoussinKernel('halfwidth', omega)
    odf = eng.calcKernelODF(ori, 'kernel', psi)

    # create artificial EBSD
    o = eng.calcOrientations(odf, Nbase)
    ori, odfred, ero = \
        eng.textureReconstruction(num, 'orientation', o, 'kernel', psi, nargout=3)
    if hist is None:
        return np.array(eng.Euler(ori))
    else:
        if shared_area is None:
            raise ValueError('Microstructure.shared_area must be provided if hist is given.')
        orilist, ein, eout, mbin = \
            eng.gb_textureReconstruction(matlab.double(hist), ori,
                                         matlab.double(shared_area), len(hist), nargout=4)
        return np.array(eng.Euler(orilist))


def createOrisetRandom(num, omega=7.5, hist=None, shared_area=None,
                       cs='m-3m', Nbase=5000):
    """
    Create a set of num Euler angles for Random texture. 
    Other than knpy.createOriset() this method does not create an artificial
    EBSD which is reduced in a second step to num discrete orientations but 
    directly samples num randomly distributed orientations.s

    Parameters
    ----------
    num : int
        Number of Euler angles in set to be created.
    omega : float
        Halfwidth of kernel in degrees (optional, default: 7.5)
    hist : array, optional
        Histogram of MDF. The default is None.
    shared_area: array, optional
        The shared area between pairs of grains. The default in None.
    cs : str, optional
        Crystal symmetry group. The default is 'm3m'.
    Nbase : int, optional
        Base number of orientations for random texture. The default is 5000

    Returns
    -------
    ori : (num, 3) array
        Set of Euler angles

    """
    # start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(MTEX_DIR, nargout=0)
    eng.addpath(ROOT_DIR, nargout=0)
    mat_path = os.path.join(MAIN_DIR + 'src' + 'kanapy')
    eng.addpath(mat_path, nargout=0)
    eng.startup(nargout=0)

    omega = omega * np.pi / 180.
    cs_ = eng.crystalSymmetry(cs)
    ot = eng.orientation.rand(Nbase, cs_)
    psi = eng.deLaValleePoussinKernel('halfwidth', omega)
    ori, odfred, ero = \
        eng.textureReconstruction(num, 'orientation', ot, 'kernel', psi,
                                  nargout=3)
    # ori = eng.project2FundamentalRegion(ori)
    if hist is None:
        return np.array(eng.Euler(ori))
    else:
        if shared_area is None:
            raise ValueError('Shared grain boundary area (geometry["GBarea"]) must be provided if hist is given.')
        orilist, ein, eout, mbin = \
            eng.gb_textureReconstruction(matlab.double(hist), ori,
                                         matlab.double(shared_area), len(hist), nargout=4)
        return np.array(eng.Euler(orilist))
