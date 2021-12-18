"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import os, json
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from kanapy.util import MTEX_DIR, ROOT_DIR
from scipy.stats import lognorm, norm, gamma

class EBSDmap:
    '''Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate 
    synthetic RVEs
    '''
    def __init__(self, fname, matname, plot=True):
        # start MATLAB Engine to use MTEX commands
        eng = matlab.engine.start_matlab()
        eng.addpath(MTEX_DIR, nargout=0)
        eng.addpath(ROOT_DIR, nargout=0)
        eng.startup(nargout=0)
        self.eng = eng
        
        # read EBSD map and return the matlab.object of MTEX class EBSD
        ebsd_full = eng.EBSD.load(fname, matname, 'interface', 'ang',
                                  'convertSpatial2EulerReferenceFrame', 'silent')
        # remove not indexed pixels
        eng.workspace["ebsd_w"] = ebsd_full
        ebsd = eng.eval("ebsd_w('indexed')")  # select only indexed pixels in EBSD
        # Texture analysis: if multiple phases exist, select one single phase

        self.ori = eng.getfield(ebsd, 'orientations')  # orientation set from the EBSD
        
        # analyze grain boundaries with MTEX function
        grains_full = eng.calcGrains(ebsd, 'boundary', 'tight', 'angle',
                                     5*(np.pi/180.0))
        # filter out small grains
        eng.workspace["grains_w"] = grains_full
        self.grains = eng.eval("grains_w(grains_w.grainSize > 3)")
        
        # use MTEX function to analye grains and plot ellipses around grain centres
        # calculate orientation, long and short axes of ellipses
        omega_r, ha, hb = eng.principalComponents(self.grains, nargout=3)
        omega = np.array(omega_r)[:, 0]
        self.ngrain = len(omega)
        if plot:
            # plot EBSD map
            eng.plot(ebsd, self.ori)
            eng.hold('on', nargout=0)
            # plot grain boundaries into EBSD map
            eng.plot(eng.getfield(self.grains, 'boundary'), 'linewidth', 2.0, 
                     'micronbar', 'off')
            # evalute centres of grains
            centres = eng.getfield(self.grains, 'centroid')
            eng.plotEllipse(centres, ha, hb, omega_r, 'lineColor', 'r',
                            'linewidth', 2.0, nargout=0)
            eng.hold('off', nargout=0)
        
            '''ODF too large for plotting
            # estimate the ODF using KDE
            odf = eng.calcKernelODF(ori,'kernel',psi)
            
            # describe poles for the polefigure plot ()
            cs = eng.getfield(odf,'CS')
            h = [eng.Miller(1,0,0,cs),eng.Miller(1,1,0,cs),eng.Miller(1,1,1,cs)]
            
            # plot pole figure
            eng.plotPDF(odf,h,'contourf', nargout=0)
            eng.hold('on', nargout=0)
            eng.mtexColorbar'''
        
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
        gsig, goffs, gscale = gamma.fit(asp)
        self.ar_param = [gsig, goffs, gscale]
        self.ar_data = asp
        if plot:
            # plot distribution of aspect ratios
            fig, ax = plt.subplots()
            x = np.linspace(np.amin(asp), np.amax(asp), 150)
            y = lognorm.pdf(x, asig, loc=aoffs, scale=ascale)
            yg = gamma.pdf(x, gsig, loc=1., scale=gscale)
            ax.plot(x, y, '-r', label='fit lognorm')
            ax.plot(x, yg, '.g', label='fit gamma')
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
        
def set_stats(grains, ar, omega, deq_min=None, deq_amx=None,
              asp_min=None, asp_max=None, omega_min=0., omega_max=np.pi,
              size=None, voxels=None, gtype='Elongated', rveunit = 'um',
              periodicity=True, save_file=False):
    '''
    grains = [std deviation, mean grain size, offset of lognorm distrib.]
    ar = [std deviation, mean aspect ration, offset of gamma distrib.]
    omega = [std deviation, mean tilt angle]
    '''
    
    # define cutoff values
    # cutoff deq
    cut1_deq = 8.0
    cut2_deq = 30.0
    # cutoff asp
    cut1_asp = 1.0
    cut2_asp = 3.0

    # RVE box size
    if size is None:
        lx = ly = lz = 100  # size of box in each direction
    else:
        lx = ly = lz = size

    # number of voxels
    if voxels is None:
        nx = ny = nz = 60  # number of voxels in each direction
    # specify RVE info
    # type of grains either 'Elongated' or 'Equiaxed'
    if not (gtype=='Elongated' or gtype=='Equiaxed'):
        raise ValueError('Wrong grain type given in set_stats: {}'
                         .format(gtype))
    if periodicity:
        pbc = 'True'
    else:
        pbc = 'False'

    # check grain type
    # create the corresponding dict with statistical grain geometry information
    ms_stats = {'Grain type': gtype,
                'Equivalent diameter':
                    {'std': grains[0], 'mean': grains[2], 'offs': grains[1],
                     'cutoff_min': cut1_deq, 'cutoff_max': cut2_deq},
                'Aspect ratio':
                    {'std': ar[0], 'mean': ar[2], 'offs': ar[1],
                     'cutoff_min': cut1_asp, 'cutoff_max': cut2_asp},
                'Tilt angle':
                    {'std': omega[0], 'mean': omega[1],
                     "cutoff_min": omega_min, "cutoff_max": omega_max},
                'RVE':
                    {'sideX': lx, 'sideY': ly, 'sideZ': lz,
                     'Nx': nx, 'Ny': ny, 'Nz': nz},
                'Simulation': {'periodicity': pbc,
                               'output_units': rveunit}}
    if save_file:
        cwd = os.getcwd()     
        json_dir = cwd + '/json_files'   # Folder to store the json files
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(json_dir + '/stat_info.json', 'w') as outfile:
            json.dump(ms_stats, outfile, indent=2)
    
    return ms_stats
       
            
       

        