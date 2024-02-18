"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import os
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from kanapy.util import MTEX_DIR, ROOT_DIR, MAIN_DIR
from scipy.stats import lognorm, vonmises
import logging


class EBSDmap:
    """Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate
    synthetic RVEs
    """

    def __init__(self, fname, matname=None, gs_min=3, vf_min=0.03, plot=True):
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
        # start MATLAB Engine to use MTEX commands
        eng = matlab.engine.start_matlab()
        eng.addpath(MTEX_DIR, nargout=0)
        eng.addpath(ROOT_DIR, nargout=0)
        mat_path = os.path.normpath(MAIN_DIR + '/src/kanapy')
        eng.addpath(mat_path, nargout=0)
        eng.startup(nargout=0)
        self.eng = eng

        # read EBSD map and return the matlab.object of MTEX class EBSD
        ebsd_full = eng.EBSD.load(fname, 'convertSpatial2EulerReferenceFrame', 'setting 2')
        # remove not indexed pixels
        eng.workspace["ebsd_w"] = ebsd_full
        if plot:
            eng.plot(ebsd_full)
        ebsd = eng.eval("ebsd_w('indexed')")  # select only indexed pixels in EBSD
        eng.workspace["ebsd"] = ebsd
        # get list of phases at each pixel in EBSD map
        plist = np.array(eng.getfield(ebsd, 'phase'))[:, 0]
        # determine number of phases and generate histogram
        Nphase = len(np.unique(plist))
        npx = len(plist)  # total number of pixels in EBSD map
        phist = np.histogram(plist, Nphase)
        # read phase names and calculate volume fractions
        self.ms_data = []
        for i in range(Nphase):
            data = dict()  # initialize data dictionary
            # generate phase-specific ebsd object in MTEX
            ebsd_h = eng.eval(f"ebsd('{i+1}')")
            data['name'] = eng.getfield(ebsd_h, 'mineral')
            vf = phist[0][i] / npx
            if vf < vf_min:
                break
            data['vf'] = vf

            # Texture analysis: orientation set from the EBSD
            ori0 = eng.getfield(ebsd_h, 'orientations')
            data['ori'] = eng.project2FundamentalRegion(ori0)
            data['cs'] = eng.getfield(ebsd_h, 'CS')
            # analyze grain boundaries with MTEX function
            grains_full = eng.calcGrains(ebsd_h, 'boundary', 'tight', 'angle',
                                         5 * (np.pi / 180.0))
            # filter out small grains
            eng.workspace["grains_w"] = grains_full
            data['grains'] = eng.eval("grains_w(grains_w.grainSize > {})"
                                      .format(gs_min))

            # use MTEX function to analye grains and plot ellipses around grain
            # centres; calculate orientation, long and short axes of ellipses
            omega_r, ha, hb = eng.principalComponents(data['grains'],
                                                      nargout=3)
            omega = np.array(omega_r)[:, 0]
            data['omega'] = omega
            data['ngrain'] = len(omega)
            if plot:
                # plot EBSD map
                eng.plot(ebsd_h, data['ori'])
                eng.hold('on', nargout=0)
                # plot grain boundaries into EBSD map
                eng.plot(eng.getfield(data['grains'], 'boundary'), 'linewidth', 2.0,
                         'micronbar', 'on')
                # evalute centres of grains
                centres = eng.getfield(data['grains'], 'centroid')
                eng.plotEllipse(centres, ha, hb, omega_r, 'lineColor', 'r',
                                'linewidth', 2.0, nargout=0)
                eng.hold('off', nargout=0)
                # eng.exportgraphics(gcf,'ebsd_map.png','Resolution',300)

                ''' ODF plotting produces system failure, use Matlab functions
                
                # plot ODF 
                # estimate the ODF using KDE
                psi = eng.deLaValleePoussinKernel('halfwidth', 5*np.pi/180.)
                odf = eng.calcKernelODF(self.ori, 'kernel', psi)
                
                h = [eng.Miller(1, 0, 0, self.cs),
                     eng.Miller(1, 1, 0, self.cs),
                     eng.Miller(1, 1, 1, self.cs)]
                
                # plot pole figure
                try:
                    eng.plotPDF(odf,h,'contourf', nargout=0)
                    eng.hold('on', nargout=0)
                    eng.mtexColorbar
                except:
                    logging.warning('ODF too large for plotting')'''
                # workaround:
                eng.workspace["ori"] = data['ori']
                eng.workspace["cs"] = data['cs']
                try:
                    eng.eval(f"plotPDF(ori,Miller(0,0,1,cs),'points','all')", nargout=0)
                except:
                    logging.warning('Plotting of orientations failed.')
            # Evaluate grain shape statistics
            # generate dict for statistical input for geometry module

            # grain equivalent diameter
            deq = 2.0 * np.array(eng.equivalentRadius(data['grains']))[:, 0]
            dsig, doffs, dscale = lognorm.fit(deq)  # fit log normal distribution
            med_eq = lognorm.median(dsig, loc=doffs, scale=dscale)
            std_eq = lognorm.std(dsig, loc=doffs, scale=dscale)
            data['gs_param'] = np.array([dsig, doffs, dscale])
            data['gs_data'] = deq
            data['gs_moments'] = [med_eq, std_eq]
            if plot:
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
            asig, aoffs, ascale = lognorm.fit(asp)  # fit log normal distribution
            med_ar = lognorm.median(asig, loc=aoffs, scale=ascale)
            std_ar = lognorm.std(asig, loc=aoffs, scale=ascale)
            data['ar_param'] = np.array([asig, aoffs, ascale])
            data['ar_data'] = asp
            data['ar_moments'] = [med_ar, std_ar]
            if plot:
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
            kappa, oloc, oscale = vonmises.fit(omega)
            med_om = vonmises.median(kappa, loc=oloc, scale=oscale)
            std_om = vonmises.std(kappa, loc=oloc, scale=oscale)
            data['om_param'] = np.array([kappa, oloc, oscale])
            data['om_data'] = omega
            data['om_moments'] = [med_om, std_om]
            if plot:
                fig, ax = plt.subplots()
                x = np.linspace(-np.pi, np.pi, 200) # np.amin(omega), np.amax(omega), 150)
                y = vonmises.pdf(x, kappa, loc=oloc, scale=oscale)
                ax.plot(x, y, '-r', label='fit')
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
            
            
def get_ipf_colors(ori_list):
    """
    Get colors of list of orientations (in radians).
    Assumes cubic crystal symmtry and cubic specimen symmetry.

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
    colors = eng.get_ipf_col(ori_list, nargout=1)
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
    mat_path = os.path.normpath(MAIN_DIR + '/src/kanapy')
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
