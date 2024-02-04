#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create RVE based on statistical microstructure information

@author: Alexander Hartmaier, ICAMS / Ruhr-University Bochum, Germany
January 2024

"""

import kanapy as knpy
from math import pi

if not knpy.MTEX_AVAIL:
    raise ModuleNotFoundError("The module MTEX is required for this example, please run 'kanapy setupTexture' first.")

texture = 'goss'  # chose texture type 'random', 'goss' or 'copper'
matname = 'Simulanium'  # Material name
nvox = 30  # number of voxels in each Cartesian direction
size = 40  # size of RVE in micron
periodic = False  # create periodic (True) or non-periodic (False) RVE
# define statistical information on microstructure
ms_elong = {'Grain type': 'Elongated',
            'Equivalent diameter': {'sig': 1.0, 'scale': 12.0, 'loc': 4.0, 'cutoff_min': 8.0, 'cutoff_max': 18.0},
            'Aspect ratio': {'sig': 0.5, 'scale': 2.0, 'loc': 0.8, 'cutoff_min': 1.0, 'cutoff_max': 3.0},
            "Tilt angle": {"kappa": 1.0, "loc": 0.5*pi, "cutoff_min": 0.0, "cutoff_max": 2*pi},
            'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox},
            'Simulation': {'periodicity': str(periodic), 'output_units': 'um'}}

# create and visualize synthetic RVE
# create kanapy microstructure object
ms = knpy.Microstructure(descriptor=ms_elong, name=matname + '_' + texture + '_texture')
ms.init_RVE()  # initialize RVE including particle distribution and structured mesh
ms.plot_stats_init()  # plot initial statistics of equivalent grain diameter and aspect ratio
ms.pack()  # perform particle simulation to distribute grain nuclei in RVE volume
ms.plot_ellipsoids()  # plot final configuration of particles
ms.voxelize()  # assign voxels to grains according to particle configuration
ms.plot_voxels(sliced=False)  # plot voxels colored according to grain number
ms.generate_grains()  # generate a polyhedral hull around each voxelized grain
ms.plot_grains()  # plot polyhedral grains
ms.plot_stats()  # compare final grain statistics with initial parameters

# Test cases for misorientation distribution (MDF)
"""adapted from
Miodownik, M., et al. "On boundary misorientation distribution functions
and how to incorporate them into three-dimensional models of microstructural
evolution." Acta materialia 47.9 (1999): 2661-2668.
https://doi.org/10.1016/S1359-6454(99)00137-8
"""
mdf_freq = {
    "High": [0.0013303016, 0.208758929, 0.3783092708, 0.7575794912, 1.6903069613,
             2.5798481069, 5.0380640643, 10.4289690569, 21.892113657, 21.0,
             22.1246762077, 13.9000439533],
    "Low": [4.5317, 18.6383, 25, 20.755, 12.5984, 7.2646, 4.2648, 3.0372, 2.5,
            1, 0.31, 0.1],
    "Random": [0.1, 0.67, 1.9, 3.65, 5.8, 8.8, 11.5, 15.5, 20, 16.7, 11.18, 4.2]
}
# mdf_bins = np.linspace(62.8 / 12, 62.8, 12)

# generate grain orientations
"""
Different textures can be chosen and assigned to the RVE geometry that has
been defined above.
Texture is defined by the orientation of the ideal component in Euler space
ang and a kernel half-width omega. Kernel used here is deLaValleePoussin.
The function createOriset will first create an artificial EBSD by sampling
a large number of discrete orientations from the ODF defined by ang and
omega. Then a reduction method is applied to reconstruct this initial ODF
with a smaller number of discrete orientations Ngr. The reduced set of
orientations is returned by the function.
For more information on the method see:
https://doi.org/10.1107/S1600576719017138
"""
if texture == 'goss':
    desc = 'unimodal'
    ang = [0, 45, 0]  # Euler angles (in degrees) for unimodal Goss texture
    omega = 7.5  # kernel half-width in degree
    Nbase = 2000  # number of orientations from which set is sub-sampled
elif texture == 'copper':
    desc = 'unimodal'
    ang = [90, 35, 45]  # Euler angles (in degrees) for unimodal copper texture
    omega = 7.5
    Nbase = 2000
elif texture == 'random':
    desc = 'random'
    ang = None  # for Random texture, no ODF recreation is necessary
    omega = None
    Nbase = 2000
else:
    raise ValueError('Texture not defined. Use "goss", "copper" or "random"')

# create grain orientations for Goss or random texture
ms.generate_orientations(desc, ang=ang, omega=omega, Nbase=Nbase,
                         hist=mdf_freq['Low'],
                         shared_area=ms.geometry['GBarea'])

# output rve in voxel format
ms.write_voxels(script_name=__file__, mesh=False, system=False)
