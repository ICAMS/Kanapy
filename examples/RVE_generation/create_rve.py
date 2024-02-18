"""
Create an RVE based on statistical information of a microstructure.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
January 2024
"""

import kanapy as knpy
from math import pi

texture = 'unimodal'  # chose texture type 'random' or 'unimodal'
matname = 'Simulanium'  # Material name
nvox = 30  # number of voxels in each Cartesian direction
size = 40  # size of RVE in micron
periodic = True  # create periodic (True) or non-periodic (False) RVE
# define statistical information on microstructure
ms_elong = {'Grain type': 'Elongated',
            'Equivalent diameter': {'sig': 1.0, 'scale': 12.0, 'loc': 4.0, 'cutoff_min': 8.0, 'cutoff_max': 18.0},
            'Aspect ratio': {'sig': 1.5, 'scale': 2.0, 'loc': 0.5, 'cutoff_min': 0.9, 'cutoff_max': 4.0},
            "Tilt angle": {"kappa": 1.0, "loc": 0.5*pi, "cutoff_min": 0.0, "cutoff_max": 2*pi},
            'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox},
            'Simulation': {'periodicity': str(periodic), 'output_units': 'um'}}

# create and visualize synthetic RVE
# create kanapy microstructure object
ms = knpy.Microstructure(descriptor=ms_elong, name=matname + '_' + texture + '_texture')
ms.init_RVE()  # initialize RVE including particle distribution
ms.plot_stats_init()  # plot initial statistics of equivalent grain diameter and aspect ratio
ms.pack()  # perform particle simulation to distribute grain nuclei in RVE volume
ms.plot_ellipsoids()  # plot final configuration of particles
ms.voxelize()  # create structured mesh and assign voxels to grains according to particles
ms.plot_voxels(sliced=True)  # plot voxels colored according to grain number
ms.generate_grains()  # generate a polyhedral hull around each voxelized grain
ms.plot_grains()  # plot polyhedral grains
ms.plot_stats()  # compared final grain statistics with initial parameters

# create grain orientations for Goss or random texture
# Note: Angles are given in degrees
if knpy.MTEX_AVAIL:
    ms.generate_orientations(texture, ang=[0, 45, 0], omega=7.5)
    print('Plotting grain colors according to IPF key.')
    ms.plot_voxels(ori=True)

# output rve in voxel format
ms.write_voxels(script_name=__file__, mesh=False, system=False)

