"""
Read EBSD map, analyse grain statistics and generate a 3D synthetic 
microstructure in form of an RVE. Grain shape statistics and crystallographic
texture will be preserved in a statistical sense.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
November 2023
"""

import kanapy as knpy
import numpy as np
    
fname = 'ebsd_316L_500x500.ang'  # name of ang file to be imported
matname = 'Iron fcc'  # material name
matnumber = 4         # material number of austenite in CP UMAT
nvox = 30             # number of voxels per side
box_length = 50       # side length of generated RVE
periodic = False      # create RVE with periodic structure


if knpy.MTEX_AVAIL:
    # read EBSD map and evaluate statistics of microstructural features
    ebsd = knpy.EBSDmap(fname)
    # ebsd.showIPF()
    ms_data = ebsd.ms_data[0]
    gs_param = ms_data['gs_param']
    ar_param = ms_data['ar_param']
    om_param = ms_data['om_param']
    print('==== Grain size ====')
    print(f'mean equivalent diameter: {gs_param[2].round(3)}, ' +
          f'std. deviation: {gs_param[0].round(3)}, ' +
          f'offset: {gs_param[1].round(3)}')
    print('==== Aspect ratio ====')
    print(f'mean value: {ar_param[2].round(3)}, ' +
          f'std. deviation: {ar_param[0].round(3)}, ' +
          f'offset: {ar_param[1].round(3)}')
    print('==== Tilt angle ====')
    print(f'mean value: {(om_param[1] * 180 / np.pi).round(3)}, ' +
          f'std. deviation: {(om_param[0] * 180 / np.pi).round(3)}')
else:
    # MTEX is not available fall back to predefined values
    print('*** Warning: Anaysis of EBSD maps is only possible with an ' +
          'existing MTEX installation in Matlab.')
    print('\nWill continue with predefined settings for microstructure.\n')
    gs_param = np.array([1.06083584, 6.24824603, 13.9309554])
    ar_param = np.array([0.67525027, 0.76994992, 1.69901906])
    om_param = np.array([0.53941709, 1.44160447])

# create dictionary with statistical information obtained from EBSD map
# gs_param : [std deviation, offset of lognorm distrib., mean grain size]
# ar_param : [std deviation, offset of gamma distrib., mean aspect ration]
# om_param : [std deviation, mean tilt angle]
ms_stats = knpy.set_stats(gs_param, ar_param, om_param,
                          deq_min=8., deq_max=19., asp_min=0.95, asp_max=3.5,
                          omega_min=0., omega_max=2*np.pi, 
                          voxels=nvox, size=box_length,
                          periodicity=periodic,
                          VF=1.0, phasename=matname, phasenum=0)

# create and visualize synthetic RVE
ms = knpy.Microstructure(descriptor=ms_stats, name=fname+'_RVE')
if knpy.MTEX_AVAIL:
    ms.init_stats(gs_data=ms_data['gs_data'],
                  ar_data=ms_data['ar_data'])
else:
    ms.init_stats()
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels(sliced=True)
ms.generate_grains()
ms.plot_grains()
ms.res_data[0]['Unit_scale'] = 'um'
ms.plot_stats(gs_param=gs_param, ar_param=ar_param)

# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='poly')

if not knpy.MTEX_AVAIL:
    raise ModuleNotFoundError('Generation of grain orientation sets is only ' +
         'possible with an existing MTEX installation in Matlab.')
# get list of orientations for grains in RVE matching the ODF of the EBSD map
ori_rve = ebsd.calcORI(ms.Ngr, shared_area=ms.shared_area)

# export virtual EBSD map as ang file and analyse with MTEX
ang_rve = ms.output_ang(ori=ori_rve, cut='xy', pos='top', cs=ms_data['cs'],
                        matname=matname, plot=False)
ebsd_rve = knpy.EBSDmap(ang_rve, matname)

# write Abaqus input file for voxelated structure
ms.output_abq('v')
# write Euler angles of grains into Abaqus input file
knpy.writeAbaqusMat(matnumber, ori_rve)

# smoothen voxelated structure and write Abaqus .inp file
ms.smoothen()
ms.output_abq('s')
