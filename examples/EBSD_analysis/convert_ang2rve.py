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

if not knpy.MTEX_AVAIL:
    raise ModuleNotFoundError('MTEX module not available. Run kanapy setupTexture before using this script.')

fname = 'ebsd_316L_500x500.ang'  # name of ang file to be imported
matname = 'Iron_fcc'  # material name
nvox = 30             # number of voxels per side
box_length = 50       # side length of generated RVE in micron
periodic = False      # create RVE with periodic structure

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname)
# ebsd.showIPF()
ms_data = ebsd.ms_data[0]  # analyse only data for majority phase with order parameter "0"
gs_param = ms_data['gs_param']  # [std deviation, offset of lognorm distrib., mean grain size]
ar_param = ms_data['ar_param']  # [std deviation, offset of gamma distrib., mean aspect ratio]
om_param = ms_data['om_param']  # [std deviation, mean tilt angle]
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

# create dictionary with statistical information obtained from EBSD map
ms_stats = knpy.set_stats(gs_param, ar_param, om_param,
                          deq_min=8., deq_max=19., asp_min=0.95, asp_max=3.5,
                          omega_min=0., omega_max=2*np.pi, 
                          voxels=nvox, size=box_length,
                          periodicity=periodic,
                          VF=1.0, phasename=matname, phasenum=0)

# create and visualize synthetic RVE
# create kanapy microstructure object
ms = knpy.Microstructure(descriptor=ms_stats, name=fname+'_RVE')
ms.init_RVE()  # initialize RVE including particle distribution and structured mesh
# plot initial statistics of equivalent grain diameter and aspect ratio together with data from EBSD map
ms.plot_stats_init(gs_data=ms_data['gs_data'], ar_data=ms_data['ar_data'])
ms.pack()   # perform particle simulation to distribute grain nuclei in RVE volume
ms.plot_ellipsoids()  # plot final configuration of particles
ms.voxelize()  # assign voxels to grains according to particle configuration
ms.plot_voxels(sliced=False)  # plot voxels colored according to grain number
ms.generate_grains()  # generate a polyhedral hull around each voxelized grain
ms.plot_grains()  # plot polyhedral grains
ms.plot_stats(gs_param=gs_param, ar_param=ar_param)  # compare final grain statistics with initial parameters

# generate and assign grains orientations
ms.generate_orientations(ebsd)

# output rve in voxel format
ms.write_voxels(script_name=__file__, mesh=False, system=False)
