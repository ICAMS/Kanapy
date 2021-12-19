import numpy as np
from kanapy import MTEX_DIR, MTEX_AVAIL

if MTEX_AVAIL:
    import matlab.engine
else:
    raise ModuleNotFoundError('Anaysis of EBSD maps is only possible with an existing MTEX installation in Matlab.')

eng = matlab.engine.start_matlab()
eng.addpath(MTEX_DIR,nargout=0)
eng.startup(nargout=0)

fname = 'ebsd_316L.ang'
matname = 'Iron fcc'

# read EBSD map and return the matlab.object of MTEX class EBSD
ebsd_full = eng.EBSD.load(fname, matname, 'interface', 'ang',
                          'convertSpatial2EulerReferenceFrame', 'silent')
# remove not indexed pixels
eng.workspace["ebsd_w"] = ebsd_full
ebsd = eng.eval("ebsd_w('indexed')")

# plot EBSD map
eng.plot(ebsd,eng.getfield(ebsd, 'orientations'))
eng.hold('on', nargout=0)

# analyze grain boundaries with MTEX function
grains_full = eng.calcGrains(ebsd,'boundary','tight','angle',5*(np.pi/180.0))
# filter out small grains
eng.workspace["grains_w"] = grains_full
grains = eng.eval("grains_w(grains_w.grainSize > 5)")

# plot grain boundaries into EBSD map
eng.plot(eng.getfield(grains,'boundary'),'linewidth',2.0,'micronbar','off')

# use MTEX function to analye grains and plot ellipses around grain centres
# calculate orientaion, long and short axes of ellipses
omega, ha, hb = eng.principalComponents(grains, nargout=3)
centres = eng.getfield(grains, 'centroid')  # evalute centres of grains
eng.plotEllipse(centres, ha, hb, omega, 'lineColor', 'r', 'linewidth', 2.0,
                nargout=0)
