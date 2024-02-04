"""
Read EBSD map and analyse microstructure w.r.t. grain shapes.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
February 2024
"""

import numpy as np
from kanapy import MTEX_DIR, MTEX_AVAIL

if MTEX_AVAIL:
    import matlab.engine
else:
    raise ModuleNotFoundError('Anaysis of EBSD maps is only possible with an existing MTEX installation in Matlab.')

eng = matlab.engine.start_matlab()
eng.addpath(MTEX_DIR, nargout=0)
eng.startup(nargout=0)

fname = 'ebsd_316L_500x500.ang'
fname = '/Users/alexander/Documents/Projekte/HybridWelding-GhazalMoeini/EBSD/AlSi10Mg_SLM_cast_SLM_100X.ctf'
vf_min = 0.03  # minimum volume fraction of phases to be considered
gs_min = 3  # minimum grains size to be considered

# read EBSD map and return the matlab.object of MTEX class EBSD
ebsd_full = eng.EBSD.load(fname,
                          'convertSpatial2EulerReferenceFrame', 'setting 2')
# remove not indexed pixels
eng.workspace["ebsd_w"] = ebsd_full
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
ms_data = []
for i in range(Nphase):
    data = dict()  # initialize data dictionary
    # generate phase-specific ebsd object in MTEX
    ebsd_h = eng.eval(f"ebsd('{i + 1}')")
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


    # analyze texture
    ori = eng.getfield(ebsd_h, 'orientations')  # get crystal orientations
    cs = eng.getfield(ebsd_h, 'CS')  # get crystal symmetry
    #psi = eng.deLaValleePoussinKernel('halfwidth', 5 * np.pi / 180.)
    #odf = eng.calcKernelODF(ori, 'kernel', psi)
    h = [eng.Miller(1, 0, 0, cs),
         eng.Miller(1, 1, 0, cs),
         eng.Miller(1, 1, 1, cs)]
    # plotting ODFs currently not supported in MATLAB engine:
    # eng.plotPDF(ori,h,'contourf', nargout=0)
    # passing of h as Matlab type "Miller" fails
    # using this workaround:
    eng.workspace["ori"] = ori
    eng.workspace["h"] = h
    eng.workspace["cs"] = cs
    eng.eval(f"plotPDF(ori,Miller(0,0,1,cs),'all')", nargout=0)

