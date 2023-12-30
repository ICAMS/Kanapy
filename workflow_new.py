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

import matplotlib
matplotlib.use('MacOSX')


fname = 'ebsd_316L_500x500.ang'  # name of ang file to be imported
matname = 'Iron fcc'  # material name
matnumber = 4         # material number of austenite in CP UMAT
nvox = 30             # number of voxels per side
box_length = 50       # side length of generated RVE
periodic = True      # create RVE with periodic structure

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
#ms.plot_stats_init()  # tested OK
ms.init_RVE()
ms.pack()
#ms.plot_ellipsoids()  # tested ??? in pyCharm
ms.voxelize()
#ms.plot_voxels(sliced=True)  # tested OK
ms.generate_grains()
#ms.plot_grains()  # tested OK
#ms.plot_stats(gs_param=gs_param, ar_param=ar_param)  # tested OK
#ms.smoothen()  # tested OK

# create grain orientations
if knpy.MTEX_AVAIL:
    ang = [0, 45, 0]    # Euler angles for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = knpy.createOriset(ms.Ngr, ang, omega)
    #ori_rnd = knpy.createOrisetRandom(ms.Ngr, Nbase=1000)  # tested OK

# output rve in different formats
ms.write_voxels(angles=ori_rve, script_name=__file__)
#ms.pckl()  # tested OK
#ms.write_stl()  # tested OK
#ms.write_stl()  # tested OK
#ms.write_ori(ori_rve)  # tested OK

# compare microstructure on three surfaces
# for voxelated and polygonalized grains
#ms.plot_slice(cut='xz', pos='top', data='poly')

#ms.output_abq('s')
