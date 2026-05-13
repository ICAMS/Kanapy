"""
Run the original EBSD analysis workflow with graph visualization enabled.

This script keeps the EBSD statistics and optional RVE-generation workflow from
the original example, but uses the local ANG file and the current 2D graph
construction settings. In addition to the standard EBSD maps, grain statistics,
and histogram plots, it displays the final microstructure graph as black node
centers and graph edges on the IPF map.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

repo_src = Path(__file__).resolve().parents[2] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

import kanapy as knpy

base_dir = Path(__file__).resolve().parent
fname = base_dir / "p558_250x_1.ang"  # cross_section

nvox = 30  # number of voxels per side
box_length = 50  # side length of generated RVE in micron
periodic = False  # create RVE with periodic structure

vf_min = 0.03  # minimum volume fraction of phases to be considered
max_angle = 5.0  # maximum misorientation angle within one grain in degrees
min_size = 20.0  # minimum grain size in pixels
connectivity = 4  # use edge sharing neighbors

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(str(fname), vf_min=vf_min, gs_min=min_size, max_angle=max_angle, show_plot=True,
                    connectivity=connectivity, show_grains=True, show_hist=True, show_graph=True)
# ebsd.showIPF()
ms_data = ebsd.ms_data[0]  # analyse only data for majority phase with order parameter "0"
gs_param = ms_data['gs_param']  # lognorm distr of grain size: [std dev., location, scale]
ar_param = ms_data['ar_param']  # lognorm distr. of aspect ratios [std dev., loc., scale]
om_param = ms_data['om_param']  # normal distribution of tilt angles [std dev., mean]
matname = ms_data['name']  # material name
print('*** Statistical information on microstructure ***')
print(f'=== Phase: {matname} ===')
print('==== Grain size (equivalent grain diameter) ====')
print(f'scale: {gs_param[2].round(3)} micron, '
      f'std. deviation: {gs_param[0].round(3)}')
print('==== Aspect ratio ====')
print(f'scale: {ar_param[2].round(3)}, '
      f'std. deviation: {ar_param[0].round(3)}')
print('==== Tilt angle ====')
print(f'most frequent value: {0.5*(om_param[1] + np.pi):.3f} rad, ' +
      f'std. deviation: {om_param[0]:.3f}')

""" Analyze microstructure graph """
G = ms_data['graph']

# plot components of major axis
va_ = []
for nn in G.nodes:
    va_.append(G.nodes[nn]['maj_ax'])
va_ = np.array(va_)
plt.hist(va_, bins=40, density=True, label=['x', 'y'])
plt.title('Components of major axis')
plt.legend()
plt.show()

# plot angles from arctan2
om1 = np.arctan2(np.abs(va_[:, 1]), va_[:, 0])
om2 = np.arccos(va_[:, 0])
plt.hist(np.c_[om1, om2], bins=40, density=True, label=['arctan2', 'arccos'])
plt.title('Tilt angle by arctan2 vs. acos')
plt.legend()
plt.show()

# compare area of hull and pixels
dar = ms_data['delta_x'] * ms_data['delta_y']
area = []
for nn in G.nodes:
    area.append(G.nodes[nn]['npix'] * dar)
eqd = 2.0 * (np.array(area) / np.pi) ** 0.5
plt.hist(np.c_[ms_data['gs_data'], eqd], bins=15, density=True, label=['hull', 'pix'])
plt.title('Grain diameters obtained from pixels and from convex hull')
plt.xlabel('Grain diameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()
