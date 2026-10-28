"""
Read an EBSD map, analyze grain statistics, and generate a complete 3D RVE.

This original Kanapy example runs EBSD analysis, descriptor construction,
initialization, packing, voxelization, orientation assignment, and voxel export.
It uses Kanapy installed in the active environment. The first retained phase
supplies the RVE statistics, following the original example's selection.

The grain graph is displayed first, followed by histograms and other EBSD plots.
Figures remain interactive and must be closed to continue. The voxel JSON is
saved directly beside this script. graph_to_rve.py is another complete EBSD to
RVE example and writes to its separate outputs/ directory.
Input and output paths resolve from this file, regardless of the working directory.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

September 2026
Updated: October 2026
"""

from pathlib import Path

import kanapy as knpy
import numpy as np
from orix.quaternion import Orientation

BASE_DIR = Path(__file__).resolve().parent
fname = str((BASE_DIR / '../fixtures/ebsd_316L_500x500.ang').resolve())
output_dir = BASE_DIR

nvox = 30         # number of voxels per side
box_length = 50   # RVE side length in micrometers
periodic = False  # create RVE with periodic structure

vf_min = 0.03  # minimum phase pixel fraction retained from the EBSD map
max_angle = 5.0  # reference setting; the original call below uses the library default
min_size = 10.0  # minimum grain size in pixels
connectivity = 8  # reference setting; not passed to EBSDmap in the original example

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname, vf_min=vf_min, gs_min=min_size,
                  show_plot=False, show_hist=False)

# show the graph before the measured distributions and remaining EBSD figures
ebsd.plot_graph_overlay(iphase=None)
ebsd.plot_histograms()
ebsd.plot_mo_map()
ebsd.plot_ipf_map()
ebsd.plot_segmentation()
ebsd.plot_pf()

# ebsd.showIPF()
ms_data = ebsd.ms_data[0]  # first retained phase; position 0 is not an original phase ID
gs_param = ms_data['gs_param']  # lognormal grain size parameters: [sigma, location, scale]
ar_param = ms_data['ar_param']  # lognormal aspect ratio parameters: [sigma, location, scale]
om_param = ms_data['om_param']  # von Mises [kappa, location], fitted to 2 * tilt - pi
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
print(f'von Mises kappa: {om_param[0].round(3)}, ' +
      f'location of 2 * tilt - pi (degrees): {np.degrees(om_param[1]).round(3)}')

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
ms.generate_orientations(ebsd, verbose=True, res_low=7, res_high=7, lim=0, hw_init=np.radians(7))
ms.plot_voxels(ori=True)

# plot pole figures from EBSD map and RVE in comparison
ori_r = Orientation.from_euler([o for o in ms.mesh.grain_ori_dict.values()])
ebsd.plot_pf()
knpy.plot_pole_figure(ori_r, ebsd.emap.phases[0])
ebsd.plot_pf_proj()
knpy.plot_pole_figure_proj(ori_r, ebsd.emap.phases[0])

# output rve in voxel format
ms.write_voxels(file=f'{matname}_voxels.json', path=output_dir,
                script_name=__file__, mesh=False, system=False)
