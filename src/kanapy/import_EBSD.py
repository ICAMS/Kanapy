"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import os
import json
import warnings
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from kanapy.util import MTEX_DIR, ROOT_DIR
from scipy.stats import lognorm, norm

class EBSDmap:
    '''Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate 
    synthetic RVEs
    '''
    def __init__(self, fname, matname, gs_min=3, plot=True):
        # start MATLAB Engine to use MTEX commands
        eng = matlab.engine.start_matlab()
        eng.addpath(MTEX_DIR, nargout=0)
        eng.addpath(ROOT_DIR, nargout=0)
        eng.startup(nargout=0)
        self.eng = eng
        
        # read EBSD map and return the matlab.object of MTEX class EBSD
        fmt = fname[-3:]
        if fmt!='ang' and fmt!='ctf':
            raise ValueError(f'Unknown EBSD format: {fmt}')
        ebsd_full = eng.EBSD.load(fname, matname, 'interface', fmt,
                                  'convertSpatial2EulerReferenceFrame', 'setting 2')  # 'silent')
        # remove not indexed pixels
        eng.workspace["ebsd_w"] = ebsd_full
        ebsd = eng.eval("ebsd_w('indexed')")  # select only indexed pixels in EBSD
        # Texture analysis: if multiple phases exist, select one single phase
        ori0 = eng.getfield(ebsd, 'orientations')  # orientation set from the EBSD
        self.ori = eng.project2FundamentalRegion(ori0)
        self.cs = eng.getfield(ebsd, 'CS')
        # analyze grain boundaries with MTEX function
        grains_full = eng.calcGrains(ebsd, 'boundary', 'tight', 'angle',
                                     5*(np.pi/180.0))
        # filter out small grains
        eng.workspace["grains_w"] = grains_full
        self.grains = eng.eval("grains_w(grains_w.grainSize > {})".format(gs_min))
        
        # use MTEX function to analye grains and plot ellipses around grain centres
        # calculate orientation, long and short axes of ellipses
        omega_r, ha, hb = eng.principalComponents(self.grains, nargout=3)
        omega = np.array(omega_r)[:, 0]
        hist, bin_edges = np.histogram(omega, bins=30)
        im = np.argmax(hist)
        hw = bin_edges[-1] - bin_edges[0]
        hc = bin_edges[im]
        hh = (hc-bin_edges[0])/hw
        if hh < 0.35:
            # maximum of distribution in lower quartile
            # shift large omegas to negative values
            ind = np.nonzero(omega > hc+0.35*hw)[0]
            omega[ind] -= np.pi
        elif hh > 0.65:
            ind = np.nonzero(omega < hc-0.35*hw)[0]
            omega[ind] += np.pi
        self.omega=omega
        self.ngrain = len(omega)
        if plot:
            # plot EBSD map
            eng.plot(ebsd, self.ori)
            eng.hold('on', nargout=0)
            # plot grain boundaries into EBSD map
            eng.plot(eng.getfield(self.grains, 'boundary'), 'linewidth', 2.0, 
                     'micronbar', 'on')
            # evalute centres of grains
            centres = eng.getfield(self.grains, 'centroid')
            eng.plotEllipse(centres, ha, hb, omega_r, 'lineColor', 'r',
                            'linewidth', 2.0, nargout=0)
            eng.hold('off', nargout=0)
            #eng.exportgraphics(gcf,'ebsd_map.png','Resolution',300)
            
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
                warnings.warn('ODF too large for plotting')'''
        
        # Evaluate grain shape statistics
        # generate dict for statistical input for geometry module
        
        # grain equivalent diameter
        deq = 2.0*np.array(eng.equivalentRadius(self.grains))[:,0]
        dsig, doffs, dscale = lognorm.fit(deq)  # fit log normal distribution
        self.gs_param = [dsig, doffs, dscale]
        self.gs_data = deq
        if plot:
            # plot distribution of grain sizes
            fig, ax = plt.subplots()
            x = np.linspace(np.amin(deq), np.amax(deq), 150)
            y = lognorm.pdf(x, dsig, loc=doffs, scale=dscale)
            ax.plot(x, y, '-r', label='fit')
            dfreq, dbins, art = ax.hist(deq, bins=20, density=True, label='data')
            plt.legend()
            plt.title('Histogram of grain equivalent diameters')
            plt.xlabel('Equivalent diameter (micron)')
            plt.ylabel('Normalized frequency')
            plt.show()
        
        # grain aspect ratio
        asp = np.array(eng.aspectRatio(self.grains))[:,0]
        asig, aoffs, ascale = lognorm.fit(asp)  # fit log normal distribution
        self.ar_param = [asig, aoffs, ascale]
        self.ar_data = asp
        if plot:
            # plot distribution of aspect ratios
            fig, ax = plt.subplots()
            x = np.linspace(np.amin(asp), np.amax(asp), 150)
            y = lognorm.pdf(x, asig, loc=aoffs, scale=ascale)
            ax.plot(x, y, '-r', label='fit lognorm')
            afreq, abins, art = ax.hist(asp, bins=20, density=True, label='density')
            plt.legend()
            plt.title('Histogram of grain aspect ratio')
            plt.xlabel('aspect ratio')
            plt.ylabel('normalized frequency')
            plt.show()
        
        # angles of main axis
        omean, osig = norm.fit(omega)  # fit normal distribution
        self.om_param = [osig, omean]
        self.om_data = omega
        if plot:
            fig, ax = plt.subplots()
            x = np.linspace(np.amin(omega), np.amax(omega), 150)
            y = norm.pdf(x, scale=osig, loc=omean)
            ax.plot(x, y, '-r', label='fit')
            ofreq, bins, art = \
                ax.hist(omega, bins=20, density=True, label='data')
            plt.legend()
            plt.title('Histogram of tilt angles of major axes')
            plt.xlabel('angle (rad)')
            plt.ylabel('normalized frequency')
            plt.show()
            
        print('Analyzed microstructure with {} grains.'.format(self.ngrain))
        print('Average grain size = {} micron, average aspect ratio = {}, \
        average tilt angle = {}°'.format(self.gs_param[2].round(3), 
        self.ar_param[2].round(3), (self.om_param[1]*180/np.pi).round(3)))
        return
    
    def calcORI(self, Ng, shared_area=None, nbins = 12):
        # estimate optimum kernel half-width and produce reduced set of orientations
        orired, odfred, ero = \
            self.eng.textureReconstruction(Ng, 'orientation', self.ori,
                                          'grains', self.grains, nargout=3)
        orired = self.eng.project2FundamentalRegion(orired)
        # this should work, too, but produces an error
        # psi = eng.calcKernel(eng.getfield(grains,'meanOrientation'))
        # orired, odfred, ero = eng.textureReconstruction(Ng,
        # 'orientation', ori, 'kernelShape', psi, nargout=3)
        if shared_area is None:
            return np.array(self.eng.Euler(orired))
        else:
            orilist, ein, eout, mbin = \
                self.eng.gb_textureReconstruction(self.grains, orired, 
                    matlab.double(shared_area), nbins, nargout=4)
            return np.array(self.eng.Euler(orilist))
        
    def showIPF(self):
        ipfKey = self.eng.ipfColorKey(self.ori, nargout=1)
        self.eng.plot(ipfKey, nargout=0)
        
def set_stats(grains, ar, omega, deq_min=None, deq_max=None,
              asp_min=None, asp_max=None, omega_min=None, omega_max=None,
              size=100, voxels=60, gtype='Elongated', rveunit = 'mm',
              periodicity=True, VF = None, phasename = None, phasenum = None, save_file=False):
    '''
    grains = [std deviation, offset , mean grain sizeof lognorm distrib.]
    ar = [std deviation, offset , mean aspect ratio of gamma distrib.]
    omega = [std deviation, mean tilt angle]
    '''
    
    # define cutoff values
    # cutoff deq
    if deq_min is None:
        deq_min = 1.3*grains[1]   # 316L: 8
    if deq_max is None:
        deq_max = grains[1] + grains[2] + 6.*grains[0]  # 316L: 30
    # cutoff asp
    if asp_min is None:
        asp_min = np.maximum(1., ar[1])   # 316L: 1
    if asp_max is None:
        asp_max = ar[2] + ar[0]  # 316L: 3
    # cutoff omega
    if omega_min is None:
        omega_min = omega[1] - 2*omega[0]
    if omega_max is None:
        omega_max = omega[1] + 2*omega[0]

    # RVE box size
    lx = ly = lz = size  # size of box in each direction

    # number of voxels
    nx = ny = nz = voxels  # number of voxels in each direction
    # specify RVE info
    # type of grains either 'Elongated' or 'Equiaxed'
    if not (gtype=='Elongated' or gtype=='Equiaxed'):
        raise ValueError('Wrong grain type given in set_stats: {}'
                         .format(gtype))
    pbc = 'True' if periodicity else 'False'

    # check grain type
    # create the corresponding dict with statistical grain geometry information
    ms_stats = {'Grain type': gtype,
                'Equivalent diameter':
                    {'std': grains[0], 'mean': grains[2], 'offs': grains[1],
                     'cutoff_min': deq_min, 'cutoff_max': deq_max},
                'Aspect ratio':
                    {'std': ar[0], 'mean': ar[2], 'offs': ar[1],
                     'cutoff_min': asp_min, 'cutoff_max': asp_max},
                'Tilt angle':
                    {'std': omega[0], 'mean': omega[1],
                     "cutoff_min": omega_min, "cutoff_max": omega_max},
                'RVE':
                    {'sideX': lx, 'sideY': ly, 'sideZ': lz,
                     'Nx': nx, 'Ny': ny, 'Nz': nz},
                'Simulation': {'periodicity': pbc,
                               'output_units': rveunit},
                'Phase':{'Name':phasename,
                         'Number':phasenum,
                         'Volume fraction':VF}}
    if save_file:
        cwd = os.getcwd()     
        json_dir = cwd + '/json_files'   # Folder to store the json files
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(json_dir + '/stat_info.json', 'w') as outfile:
            json.dump(ms_stats, outfile, indent=2)
    
    return ms_stats

def createOriset(num, ang, omega, hist=None, shared_area=None,
                 cs='m-3m', Nbase=10000):
    """
    Create a set of num Euler angles according to the ODF defined by the 
    set of Euler angles ang and the kernel half-width omega.
    Example: Goss texture: ang = [0, 45, 0], omega = 5

    Parameters
    ----------
    num : int
        Numberof Euler angles in set to be created.
    ang : (3,) or (M, 3) array
        Set of Euler angles (in degrees) defining the ODF.
    omega : float
        Half-wodth of kernel in degrees.
    hist : array, optional
        Histogram of MDF. The default is None.
    shared_area: array, optional
        The shared area between pairs of grains. The default in None.
    cs : str, optional
        Crystal symmetry group. The default is 'm3m'.

    Returns
    -------
    ori : (num, 3) array
        Set of Euler angles

    """
    # start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(MTEX_DIR, nargout=0)
    eng.addpath(ROOT_DIR, nargout=0)
    eng.startup(nargout=0)
    
    # prepare parameters
    deg = np.pi/180.
    omega *= deg
    ang = np.array(ang)*deg
    
    '''
    # support for higher dim arrays for orientation angles
    N = len(ang)
    sh = ang.shape
    if N==3 and sh==(3,):
        ang=np.array([ang])
    elif sh==(N,3):
        raise ValueError('"ang" must be a single Euler angle, i.e. (3,) array,'+\
                         'or an (N,3) array of Euler angles')'''
        
    cs_ = eng.crystalSymmetry(cs)
    
    ori = eng.orientation('Euler', float(ang[0]), float(ang[1]),
                          float(ang[2]), cs_)
    psi = eng.deLaValleePoussinKernel('halfwidth', omega)
    odf = eng.calcKernelODF(ori, 'kernel', psi)

    # create artificial EBSD
    o = eng.calcOrientations(odf, Nbase);

    ori, odfred, ero = \
        eng.textureReconstruction(num, 'orientation', o, 'kernel', psi,
                                  nargout=3)
    ori = eng.project2FundamentalRegion(ori)
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
                 cs='m-3m', Nbase=10000, file=None):
    
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

    Returns
    -------
    ori : (num, 3) array
        Set of Euler angles

    """
    # start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(MTEX_DIR, nargout=0)
    eng.addpath(ROOT_DIR, nargout=0)
    eng.startup(nargout=0)
    
    omega = omega*np.pi/180.
    cs_ = eng.crystalSymmetry(cs)
    ot = eng.orientation.rand(Nbase, cs_)
    psi = eng.deLaValleePoussinKernel('halfwidth', omega)
    ori, odfred, ero = \
        eng.textureReconstruction(num, 'orientation', ot, 'kernel', psi,
                                  nargout=3)
    ori = eng.project2FundamentalRegion(ori)
    if hist is None:
        return np.array(eng.Euler(ori))
    else:
        if shared_area is None:
            raise ValueError('Microstructure.shared_area must be provided if hist is given.')
        orilist, ein, eout, mbin = \
            eng.gb_textureReconstruction(matlab.double(hist), ori, 
                matlab.double(shared_area), len(hist), nargout=4)
        return np.array(eng.Euler(orilist))
    
