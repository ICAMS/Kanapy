"""
Create an RVE based on statistical information of a microstructure.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
January 2024
"""

import kanapy as knpy
import logging
logging.basicConfig(level=20)

texture = 'unimodal'  # 'random' or 'unimodal'
matname = 'Simulanium_fcc'
Nv = 30
size = 40
periodic = True
ms_elong = {'Grain type': 'Elongated',
          'Equivalent diameter': {'std': 1.0, 'mean': 12.0, 'offs': 4.0, 'cutoff_min': 8.0, 'cutoff_max': 18.0},
          'Aspect ratio': {'std':1.5, 'mean': 2.0, 'offs': 0.8, 'cutoff_min': 1.0, 'cutoff_max': 4.0},
          'Tilt angle': {'std': 15., 'mean': 90., "cutoff_min": 0.0, "cutoff_max": 180.0},
          'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv},
          'Simulation': {'periodicity': str(periodic), 'output_units': 'um'}}

# create and visualize synthetic RVE
ms = knpy.Microstructure(descriptor=ms_elong, name=matname+texture+'_texture_RVE')
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels(sliced=True)
ms.generate_grains()
ms.plot_grains()
ms.plot_stats()

# create grain orientations for Goss or random texture
ms.generate_orientations(texture, ang=[0, 45, 0], omega=7.5)

# output rve in voxel format
ms.write_voxels(script_name=__file__)
