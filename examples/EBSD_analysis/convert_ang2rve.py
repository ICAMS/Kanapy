import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from kanapy import Microstructure, MTEX_DIR, ROOT_DIR
from scipy import stats


eng = matlab.engine.start_matlab()
eng.addpath(MTEX_DIR,nargout=0)
eng.addpath(ROOT_DIR,nargout=0)
eng.startup(nargout=0)

fname = 'ebsd_316L.ang'
matname = 'Iron fcc'

#read EBSD map and return the matlab.object of MTEX class EBSD
ebsd_full = eng.EBSD.load(fname,matname,'interface','ang','convertSpatial2EulerReferenceFrame','silent')
#remove not indexed pixels
eng.workspace["ebsd_w"] = ebsd_full
ebsd = eng.eval("ebsd_w('indexed')") # select only indexed pixels in EBSD
## Texture analysis: if multiple phases exist, select one single phase

#plot EBSD map
ori = eng.getfield(ebsd,'orientations') # orientation set from the EBSD
eng.plot(ebsd,ori)
eng.hold('on', nargout=0)

#analyze grain boundaries with MTEX function
grains_full = eng.calcGrains(ebsd,'boundary','tight','angle',5*(np.pi/180.0))
#filter out small grains
eng.workspace["grains_w"] = grains_full
grains = eng.eval("grains_w(grains_w.grainSize > 3)")

#plot grain boundaries into EBSD map
eng.plot(eng.getfield(grains,'boundary'),'linewidth',2.0,'micronbar','off')

#use MTEX function to analye grains and plot ellipses around grain centres
omega_r,ha,hb = eng.principalComponents(grains, nargout=3) # calculate orientation, long and short axes of ellipses
omega = np.array(omega_r)[:,0]
centres = eng.getfield(grains,'centroid')  # evalute centres of grains
ngrain = len(centres)
eng.plotEllipse(centres,ha,hb,omega_r,'lineColor','r','linewidth',2.0, nargout=0)
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

# Evaluate grain shape statistics and generate dict for statistical input for geometry modeule

# grain equivalent diameter
deq = 2.0*np.array(eng.equivalentRadius(grains))
dsig, dmu, dscale = stats.lognorm.fit(deq) # fit log normal distribution
#plot distribution of grain sizes
fig, ax = plt.subplots()
x = np.linspace(np.amin(deq),np.amax(deq),150)
y = stats.lognorm.pdf(x, dsig, loc=dmu, scale=dscale)
ax.plot(x,y,'-r', label='fit')
dfreq, dbins, art = ax.hist(deq, bins=20, density=True, label='data')
plt.legend()
plt.show()

# grain aspect ratio 
asp = np.array(eng.aspectRatio(grains))
asig, amu, ascale = stats.lognorm.fit(asp) # fit log normal distribution
# plot distribution of aspect ratios
fig, ax = plt.subplots()
x = np.linspace(np.amin(asp),np.amax(asp),150)
y = stats.lognorm.pdf(x, asig, loc=amu, scale=ascale)
ax.plot(x,y,'-r', label='fit')
afreq, abins, art = ax.hist(asp, bins=20, density=True, label='density')
plt.legend()
plt.show()

# angles of main axis
omean, osig = stats.norm.fit(omega) # fit normal distribution
fig, ax = plt.subplots()
x = np.linspace(np.amin(omega),np.amax(omega),150)
y = stats.norm.pdf(x, scale=osig, loc=omean)
ax.plot(x,y,'-r',label='fit')
ofreq, bins, art = ax.hist(omega, bins=20, density=True, label='data')
plt.legend()
plt.show()

# define cutoff values
# cutoff deq
cut1_deq = 8.0
cut2_deq = 30.0

# cutoff asp
cut1_asp = 1.0
cut2_asp = 3.0

#cutoff omega
cut1_omg = 0.
cut2_omg = np.pi

# RVE box size 
lx = 100 # in x direction
ly = 100 # in y direction
lz = 100 # in z direction

# number of voxels
nx = 60 # in x direction
ny = 60 # in y direction
nz = 60 # in z direction

# specify RVE info
# type of grains either 'Elongated' or 'Equiaxed'
gtype = 'Elongated'
# unit of length
rveunit = 'um'
# structural periodicity True or False
periodicity = 'True'

# check the grain type and create the corresponding dict with statistical grain geometry information
ms_stats = {'Grain type': gtype, 
          'Equivalent diameter': {'std': dsig, 'mean': dmu, 'scale': dscale, 'cutoff_min': cut1_deq, 'cutoff_max': cut2_deq},
          'Aspect ratio': {'std':asig, 'mean': amu, 'scale': ascale, 'cutoff_min': cut1_asp, 'cutoff_max': cut2_asp}, 
          'Tilt angle': {'std': osig, 'mean': omean, "cutoff_min": cut1_omg, "cutoff_max": cut2_omg}, 
          'RVE': {'sideX': lx, 'sideY': ly, 'sideZ': lz, 'Nx': nx, 'Ny': ny, 'Nz': nz}, 
          'Simulation': {'periodicity': periodicity, 'output_units': rveunit}}

print('Analyzed microstructure with {} grains.'.format(ngrain))
print('Average grain size = {} {}, average aspect ratio = {}, average tilt angle = {}°'\
      .format(dmu.round(3), rveunit, amu.round(3), (omean*180/np.pi).round(3)))

# create RVE 
ms = Microstructure(descriptor=ms_stats, name=fname+'_RVE')
ms.create_stats(gs_data=deq, ar_data=asp)
ms.create_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
# estimate optimum kernel half-width and produce reduced set of orientations
orired, odfred, ero = eng.textureReconstruction(ms.particle_data['Number'],'orientation',ori, \
                      'grains', grains, nargout=3)
nbins=12
ms.output_stats()
orilist,ein,eout,mbin = eng.gb_textureReconstruction(grains, orired,\
                        matlab.double(ms.shared_area), nbins, nargout=4)
    
ori_rve = np.array(eng.Euler(orired))
#this should work, too, but produces an error
#psi = eng.calcKernel(eng.getfield(grains,'meanOrientation'))
#orired, odfred, ero = eng.textureReconstruction(ms.particle_data['Number'],'orientation',ori,'kernelShape',psi,nargout=3)
ms.output_abq('v')
ms.plot_3D(sliced=False)
ms.output_stats()
ms.plot_stats()
#ms.smoothen()
#ms.output_abq('s')
