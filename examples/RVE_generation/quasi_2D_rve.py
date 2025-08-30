"""
Create an RVE based on statistical information of a microstructure
and export quasi-2D slice of voxels at surface.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
January 2024
"""
import kanapy as knpy
import numpy as np

texture = 'random'  # chose texture type 'random' or 'unimodal'
matname = 'Simulanium'  # Material name
nvox = 30  # number of voxels in each Cartesian direction
size = 40  # size of RVE in micron
periodic = True  # create periodic (True) or non-periodic (False) RVE
# define statistical information on microstructure
ms_elong = {'Grain type': 'Elongated',
            'Equivalent diameter': {'sig': 1.0, 'scale': 12.0, 'loc': 4.0, 'cutoff_min': 8.0, 'cutoff_max': 18.0},
            'Aspect ratio': {'sig': 1.5, 'scale': 2.0, 'loc': 0.8, 'cutoff_min': 1.0, 'cutoff_max': 4.0},
            "Tilt angle": {"kappa": 1.0, "loc": 0.5*np.pi, "cutoff_min": 0.0, "cutoff_max": 2*np.pi},
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
ms.plot_voxels(sliced=True)  # plot voxels colored according to grain number
ms.generate_grains()  # generate a polyhedral hull around each voxelized grain
ms.plot_grains()  # plot polyhedral grains
ms.plot_stats_init(show_res=True)  # compare final grain statistics with initial parameters

# create grain orientations for Goss or random texture
ms.generate_orientations(texture, ang=[0, 45, 0], omega=7.5)

# output rve in voxel format such that it can be re-imported again
ms.write_voxels(script_name=__file__, mesh=False, system=False)

# re-entry point. If _voxels.json file exists, all steps up to here can be skipped
"""ms = knpy.import_voxels('Simulanium_unimodal_texture_voxels.json')"""

# export a slice of the surface with z=0 as Abaqus input deck
vox_dict = dict()
gr_dict = dict()
gr_ori_dict = dict()
zmax = ms.rve.size[2] / ms.rve.dim[2]  # voxel dimension in z-direction
for iv, ctr in ms.mesh.vox_center_dict.items():
    # select voxels on surface slice with z-coordinate less than voxel size
    if ctr[2] < zmax:
        vox_dict[iv] = ms.mesh.voxel_dict[iv]
        # find grain in which voxel iv is located
        for igr, gr_vox in ms.mesh.grain_dict.items():
            if iv in gr_vox:
                if igr not in gr_dict.keys():
                    gr_dict[igr] = []  # create empty voxel list for new grain
                    #gr_ori_dict[igr] = ms.mesh.grain_ori_dict[igr]  # save orientation of new grain
                gr_dict[igr].append(iv)  # add voxel to grain dictionary
                break
# get max. node number in voxels
# NOTE: This only works for slices with z=0, otherwise nodes must be re-defined!
max_nn = 0
for iv, nodes in vox_dict.items():
    max_nn = np.maximum(max_nn, np.max(nodes))

# write abq..._geom.inp (voxels as brick elements and element sets for grains)
# and abq..._mat.inp (material definitions with Euler angles
fgeom = ms.write_abq(nodes=ms.mesh.nodes[0:max_nn], voxel_dict=vox_dict,
                     grain_dict=gr_dict,
                     ialloy = 4)  # identifier of material in ICAMS crystal plasticity UMAT
