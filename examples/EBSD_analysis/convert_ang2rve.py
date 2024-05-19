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
nvox = 30         # number of voxels per side
box_length = 50   # side length of generated RVE in micron
periodic = False  # create RVE with periodic structure

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname)
# ebsd.showIPF()
ms_data = ebsd.ms_data[0]  # analyse only data for majority phase with order parameter "0"
gs_param = ms_data['gs_param']  # lognorm distr of grain size: [std dev., location, scale]
ar_param = ms_data['ar_param']  # lognorm distr. of aspect ratios [std dev., loc., scale]
om_param = ms_data['om_param']  # normal distribution of tilt angles [std dev., mean]
matname = ms_data['name']  # material name
print('*** Statistical information on microstructure ***')
print(f'=== Phase: {matname} ===')
print('==== Grain size (equivalent grain diameter) ====')
print(f'scale: {gs_param[2].round(3)}, ' +
      f'location: {gs_param[1].round(3)}, ' +
      f'std. deviation: {gs_param[0].round(3)}')
print('==== Aspect ratio ====')
print(f'scale: {ar_param[2].round(3)}, ' +
      f'location: {ar_param[1].round(3)}, ' +
      f'std. deviation: {ar_param[0].round(3)}')
print('==== Tilt angle ====')
print(f'mean value: {(om_param[1] * 180 / np.pi).round(3)}, ' +
      f'std. deviation: {(om_param[0] * 180 / np.pi).round(3)}')

# create dictionary with statistical information obtained from EBSD map
ms_stats = knpy.set_stats(gs_param, ar_param, om_param,
                          deq_min=8.0, deq_max=19.0, asp_min=0.95, asp_max=3.5,
                          omega_min=0.0, omega_max=np.pi,
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

# compare final grain statistics with initial parameters
ms.plot_stats_init(show_res=True, gs_data=ms_data['gs_data'], ar_data=ms_data['ar_data'])

# generate and assign grains orientations
ms.generate_orientations(ebsd)
ms.plot_voxels(ori=True)

# output rve in voxel format
ms.write_voxels(file=f'{matname}_voxels.json', script_name=__file__, mesh=False, system=False)
