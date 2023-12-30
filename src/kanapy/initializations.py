import json
import os
import sys

import numpy as np
import warnings
import itertools
from scipy.stats import lognorm, norm
from collections import defaultdict


class RVE_creator(object):
    r"""
    Creates an RVE based on user-defined statistics

    :param inputFile: User-defined statistics file for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Standard deviation for ellipsoid equivalent diameter (Log-normal distribution)
                * Mean value of ellipsoid equivalent diameter (Log-normal distribution)
                * Minimum and Maximum cut-offs for ellipsoid equivalent diameters
                * Mean value for aspect ratio
                * Mean value for ellipsoid tilt angles (Normal distribution)
                * Standard deviation for ellipsoid tilt angles (Normal distribution)
                * Side dimension of the RVE
                * Discretization along the RVE sides

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle.
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as periodicity and output unit scale (:math:`mm`
                  or :math:`\mu m`) for ABAQUS .inp file.

    Generates ellipsoid size distribution (Log-normal) based on user-defined statistics


    :param inputFile: User-defined statistics file for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Standard deviation for ellipsoid equivalent diameter (Normal distribution)
                * Mean value of ellipsoid equivalent diameter (Normal distribution)
                * Minimum and Maximum cut-offs for ellipsoid equivalent diameters
                * Mean value for aspect ratio
                * Mean value for ellipsoid tilt angles (Normal distribution)
                * Standard deviation for ellipsoid tilt angles (Normal distribution)
                * Side dimension of the RVE
                * Discretization along the RVE sides

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle.
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as periodicity and output unit scale (:math:`mm`
                  or :math:`\mu m`) for ABAQUS .inp file.

    """
    def __init__(self, stats_dicts, nsteps=1000):
        """
        Create an RVE object.

        Parameters
        ----------
        stats_dict
        nsteps

        Attributes
        ----------
        self.nphases = len(stats_dicts)  # number of phases in RVE
        self.packing_steps = nsteps  # max number of steps for packing simulation
        self.size = None  #  tuple of lengths along Cartesian axes
        self.dim = None  # tuple of number of voxels along Cartesian axes
        self.periodic = None  # Boolean for periodicity of RVE
        self.units = None  # Units of RVE dimensions, either "mm" or "um" (micron)
        self.nparticles = []  # list of particle numbers for each phase
        particle_data = []  # list of cits for statistical particle data for each grains
        phase_names = []  # list of names of phases
        phase_vf = []  # list of volume fractions of phases
        """

        def gen_data_basic(pdict):
            # Compute the Log-normal PDF & CDF.
            if offs is None:
                frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
            else:
                frozen_lognorm = lognorm(s=sd, loc=offs, scale=mean)

            xaxis = np.linspace(0.1,200,1000)
            ycdf = frozen_lognorm.cdf(xaxis)

            # Get the mean value for each pair of neighboring points as centers of bins
            xaxis = np.vstack([xaxis[1:], xaxis[:-1]]).mean(axis=0)

            # Based on the cutoff specified, get the restricted distribution
            index_array = np.where((xaxis >= dia_cutoff_min) & (xaxis <= dia_cutoff_max))
            eq_Dia = xaxis[index_array]          # Selected diameters within the cutoff

            # Compute the number fractions and extract them based on the cut-off
            number_fraction = np.ediff1d(ycdf)  # better use lognorm.pdf
            numFra_Dia = number_fraction[index_array]

            # Volume of each ellipsoid
            volume_array = (4/3)*np.pi*(0.5*eq_Dia)**3

            # Volume fraction for each ellipsoid
            individualK = np.multiply(numFra_Dia, volume_array)
            K = individualK/np.sum(individualK)

            # Total number of ellipsoids for packing density 65%
            num = np.divide(K*phase_vf[-1]*np.prod(self.size), volume_array)*0.65
            num = np.rint(num).astype(int)       # Round to the nearest integer
            totalEllipsoids = int(np.sum(num))

            # Duplicate the diameter values
            eq_Dia = np.repeat(eq_Dia, num)  # better calculate num first

            # Raise value error in case the RVE side length is too small to fit grains inside.
            if len(eq_Dia) == 0:
                raise ValueError(
                    'RVE volume too small to fit grains inside, please increase the RVE side length (or) decrease the mean size for diameters!')

            # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain at least 3 voxels
            voxel_size = np.divide(self.size, self.dim)

            # raise value error if voxel sizes along the 3 directions are not equal
            dif1 = np.abs(voxel_size[0] - voxel_size[1])
            dif2 = np.abs(voxel_size[1] - voxel_size[2])
            dif3 = np.abs(voxel_size[2] - voxel_size[0])
            if (dif1 > 1.e-5) or (dif2 > 1.e-5) or (dif3 > 1.e-5):
                raise ValueError('Voxels are not cubic, voxel sizes must be identical in all directions.')

            # raise value error in case the grains are not voxelated well
            if voxel_size[0] >= np.amin(eq_Dia) / 3.:
                print(" ")
                print(f"    Grains with minimum size {np.amin(eq_Dia)} will not be voxelated well!")
                print(f"    Voxel size is {voxel_size[0]}")
                print("    Consider increasing the voxel numbers (OR) decreasing the RVE side lengths\n")
                if voxel_size[0] > np.amin(eq_Dia):
                    raise ValueError(' Voxel size larger than minimum grain size.')

            # raise warning if large grain occur in periodic box
            if np.amax(eq_Dia) >= self.size[0] * 0.5 and self.periodic:
                print("\n")
                print("    Periodic box with grains larger the half of box width.")
                print("    Check grain polygons carefully.")

            print(f'    Analyzed statistical data for phase {phase_names[-1]} ({ip})')
            print(f'    Total number of particles  = {totalEllipsoids}')
            pdict['Number'] = totalEllipsoids
            pdict['Equivalent_diameter'] = list(eq_Dia)
            return pdict

        def gen_data_elong(pdict):
            # Tilt angle statistics
            # Sample from Normal distribution: It takes mean and std of normal distribution
            tilt_angle = []
            num = pdict['Number']
            while num > 0:
                tilt = norm.rvs(scale=std_ori, loc=mean_ori, size=num)
                index_array = np.where((tilt >= ori_cutoff_min) & (tilt <= ori_cutoff_max))
                TA = tilt[index_array].tolist()
                tilt_angle.extend(TA)
                num = pdict['Number'] - len(tilt_angle)

            # Aspect ratio statistics
            # Sample from lognormal or gamma distribution:
            # it takes mean, std and scale of the underlying normal distribution
            finalAR = []
            num = pdict['Number']
            while num > 0:
                # ar = np.random.lognormal(mean_AR, sd_AR, num)
                if offs_AR is None:
                    ar = lognorm.rvs(sd_AR, scale=np.exp(mean_AR), size=num)
                else:
                    ar = lognorm.rvs(sd_AR, loc=offs_AR, scale=mean_AR, size=num)
                index_array = np.where((ar >= ar_cutoff_min) & (ar <= ar_cutoff_max))
                AR = ar[index_array].tolist()
                finalAR.extend(AR)
                num = pdict['Number'] - len(finalAR)
            finalAR = np.array(finalAR)

            # Calculate the major, minor axes lengths for particles using:
            # (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b
            minDia = pdict['Equivalent_diameter'] / finalAR ** (1 / 3)  # Minor axis length
            majDia = minDia * finalAR  # Major axis length
            minDia2 = minDia.copy()  # Minor2 axis length (assuming rotational shape)

            # Add data to dictionary to store the data generated
            pdict['Major_diameter'] = list(majDia)
            pdict['Minor_diameter1'] = list(minDia)
            pdict['Minor_diameter2'] = list(minDia2)
            pdict['Tilt angle'] = list(tilt_angle)
            return pdict

        print('Creating an RVE based on user defined statistics')
        # Extract grain diameter statistics info
        self.nphases = len(stats_dicts)  # number of phases in RVE
        self.packing_steps = nsteps  # max number of steps for packing simulation
        self.size = None  #  tuple of lengths along Cartesian axes
        self.dim = None  # tuple of number of voxels along Cartesian axes
        self.periodic = None  # Boolean for periodicity of RVE
        self.units = None  # Units of RVE dimensions, either "mm" or "um" (micron)
        self.nparticles = []  # List of article numbers for each phase
        particle_data = []  # list of cits for statistical particle data for each grains
        phase_names = []  # list of names of phases
        phase_vf = []  # list of volume fractions of phases

        # extract data from descriptors of individual phases
        for ip, stats in enumerate(stats_dicts):
            # Extract RVE side lengths and voxel numbers, must be provided for phase 0
            if 'RVE' in stats.keys():
                size = (stats["RVE"]["sideX"],
                        stats["RVE"]["sideY"],
                        stats["RVE"]["sideZ"])
                nvox = (int(stats["RVE"]["Nx"]),
                        int(stats["RVE"]["Ny"]),
                        int(stats["RVE"]["Nz"]))
                if self.size is None:
                    self.size = size
                else:
                    if self.size != size:
                        warnings.warn(f'Conflicting RVE sizes in descriptors: {self.size}, {size}.' +
                                      'Using first value.')
                if self.dim is None:
                    self.dim = nvox
                else:
                    if self.dim != nvox:
                        warnings.warn(f'Conflicting RVE voxel dimensions in descriptors: {self.dim}, {nvox}.' +
                                      'Using first value.')
            elif ip == 0:
                raise ValueError('RVE properties must be specified in descriptors for first phase.')

            # Extract other simulation attributes, must be specified for phase 0
            if "Simulation" in stats.keys():
                periodic = bool(stats["Simulation"]["periodicity"])
                if self.periodic is None:
                    self.periodic = periodic
                elif self.periodic != periodic:
                    warnings.warn(f'Inconsistent values for periodicity. Using periodicity: {self.periodic}.')
                units = str(stats["Simulation"]["output_units"])
                # Raise ValueError if units are not specified as 'mm' or 'um'
                if units != 'mm' and units != 'um':
                    raise ValueError('Output units can only be "mm" or "um"!')
                if self.units is None:
                    self.units = units
                elif self.units != units:
                    warnings.warn(f'Inconsistent values for units. Using units: {self.units}.')
            elif ip == 0:
                raise ValueError('Simulation attributes must be specified in descriptors for first phase.')

            # Extract phase information
            if "Phase" in stats.keys():
                phase_names.append(stats["Phase"]["Name"])
                phase_vf.append(stats["Phase"]["Volume fraction"])
            else:
                phase_names.append(f'Phase_{ip}')
                phase_vf.append(1. - np.sum(phase_vf))  # volume fraction can only be unspecified for last phase
            if np.sum(phase_vf) > 1.:
                raise ValueError(f"Sum of all phase fractions exceeds 1: {phase_vf}")

            # Extract grains shape attributes
            if stats["Grain type"] not in ["Elongated", "Equiaxed"]:
                raise ValueError('The value for "Grain type" must be either "Equiaxed" or "Elongated".')

            # Attributes for equivalent diameter
            sd = stats["Equivalent diameter"]["std"]
            mean = stats["Equivalent diameter"]["mean"]
            if "offs" in stats["Equivalent diameter"]:
                offs = stats["Equivalent diameter"]["offs"]
            else:
                offs = None
            dia_cutoff_min = stats["Equivalent diameter"]["cutoff_min"]
            dia_cutoff_max = stats["Equivalent diameter"]["cutoff_max"]
            # generate dict for particle data
            pdict = gen_data_basic(dict({'Type': stats["Grain type"], 'Phase': ip}))

            # Additional attributes for elongated grains
            if stats["Grain type"] == "Elongated":
                # Extract mean grain aspect ratio value info from dict
                sd_AR = stats["Aspect ratio"]["std"]
                mean_AR = stats["Aspect ratio"]["mean"]
                if "offs" in stats["Aspect ratio"]:
                    offs_AR = stats["Aspect ratio"]["offs"]
                else:
                    offs_AR = None
                ar_cutoff_min = stats["Aspect ratio"]["cutoff_min"]
                ar_cutoff_max = stats["Aspect ratio"]["cutoff_max"]

                # Extract grain tilt angle statistics info from dict
                std_ori = stats["Tilt angle"]["std"]
                mean_ori = stats["Tilt angle"]["mean"]
                ori_cutoff_min = stats["Tilt angle"]["cutoff_min"]
                ori_cutoff_max = stats["Tilt angle"]["cutoff_max"]

                # Add attributes for elongated particle to dictionary
                pdict = gen_data_elong(pdict)
            particle_data.append(pdict)
            self.nparticles.append(pdict['Number'])
        print('  RVE characteristics:')
        print(f'    RVE side lengths (X, Y, Z) = {self.size} ({self.units})')
        print(f'    Number of voxels (X, Y, Z) = {self.dim}')
        print(f'    Voxel resolution (X, Y, Z) = {np.divide(self.size, self.dim).round(4)}' +
              f'({self.units})')
        print(f'    Total number of voxels     = {np.prod(self.dim)}\n')
        self.phase_names = phase_names
        self.phase_vf = phase_vf
        self.particle_data = particle_data
        return


class mesh_creator(object):
    def __init__(self, dim):
        """
        Create a mesh object.

        Parameters
        ----------
        dim: 3-tuple for dimensions of mesh (numbers of voxels in each Cartesian direction)

        Attributes
        ----------
        self.dim = dim  # dimension tuple: number of voxels in each Cartesian direction
        self.nvox = np.prod(dim)  # number of voxels
        self.phases = np.zeros(dim, dtype=np.int8)  # field data with phase numbers
        self.grains = np.zeros(dim, dtype=np.int16)  # field data with grain numbers
        self.voxel_dict = dict()  # stores nodes assigned to each voxel (key; voxel number:int)
        self.grain_dict = dict()  # stored voxels assigned to each grain (key: grain number:int
        self.nodes = None  # array of nodal positions
        self.nodes_smooth = None  # array of nodal positions after smoothening grain boundaries
        """

        if not (type(dim) == tuple and len(dim) == 3):
            raise ValueError(f"Dimension dim must be a 3-tuple, not {type(dim)}, {dim}")
        self.dim = dim  # dimension tuple: number of voxels in each Cartesian direction
        self.nvox = np.prod(dim)  # number of voxels
        self.grains = None
        self.phases = None
        self.grain_dict = dict()  # stored voxels assigned to each grain (key: grain number:int)
        self.voxel_dict = defaultdict(list)  # dictionary to store list of node ids for each element
        self.vox_center_dict = dict()  # dictionary to store center of each voxel as 3-tupel
        self.nodes = None  # array of nodal positions
        self.nodes_smooth = None  # array of nodal positions after smoothening grain boundaries
        return

    def get_ind(self, addr):
        """
        Return the index in an array from an generic address.

        Parameters
        ----------
        addr

        Returns
        -------

        """
        if addr is None:
            ind = None
        elif len(addr) == 0:
            ind = addr
        elif len(addr) == 3:
            ind = addr[0]*self.dim[1]*self.dim[2] + addr[1]*self.dim[1] + addr[2]
        else:
            raise ValueError(f"Address must be a single int or a 3-tuple, not {type(addr)}, {addr}.")
        return ind

    def create_voxels(self, sim_box):
        """
        Generates voxels inside the defined RVE (Simulation box)

        :param sim_box: Simulation box representing RVE dimensions
        :type sim_box: :obj:`entities.Cuboid`
        :param voxNums: Number of voxels along the RVE sides X, Y & Z
        :type voxNums: tuple of int

        :returns: * Node list containing coordinates.
                  * Element dictionary containing element IDs and nodal connectivities.
                  * Voxel dictionary containing voxel ID and center coordinates.
        :rtype: Tuple of Python dictionaries.
        """
        print('    Generating voxels inside RVE')
        # generate nodes of all voxels from RVE side dimensions
        lim_minX, lim_maxX = sim_box.left, sim_box.right
        lim_minY, lim_maxY = sim_box.top, sim_box.bottom
        lim_minZ, lim_maxZ = sim_box.front, sim_box.back  # define the cuboidal RVE limits

        # generate points within these limits
        pointsX = np.linspace(lim_minX, lim_maxX, num=self.dim[0] + 1, endpoint=True)
        pointsY = np.linspace(lim_minY, lim_maxY, num=self.dim[1] + 1, endpoint=True)
        pointsZ = np.linspace(lim_minZ, lim_maxZ, num=self.dim[2] + 1, endpoint=True)

        # duplicate neighbouring points
        pointsX_dup = [(first, second) for first, second in zip(pointsX, pointsX[1:])]
        pointsY_dup = [(first, second) for first, second in zip(pointsY, pointsY[1:])]
        pointsZ_dup = [(first, second) for first, second in zip(pointsZ, pointsZ[1:])]

        verticesDict = {}  # dictionary to store vertices
        node_count = 0
        elmt_count = 0
        # loop over the duplicate pairs
        for (mk, nk), (mj, nj), (mi, ni) in itertools.product(pointsX_dup, pointsY_dup, pointsZ_dup):

            # Find the center of each voxel and update the center dictionary
            elmt_count += 1
            self.vox_center_dict[elmt_count] = (0.5 * (mi + ni), 0.5 * (mj + nj), 0.5 * (mk + nk))

            # group the 8 nodes of an element and update node & element dictionary accordingly
            # C3D8 element connectivity is maintained by this list (DON'T change this order)
            vertices = [(ni, mj, nk), (ni, mj, mk), (mi, mj, mk), (mi, mj, nk),
                        (ni, nj, nk), (ni, nj, mk), (mi, nj, mk), (mi, nj, nk)]

            for coo in vertices:
                if coo not in verticesDict.keys():
                    node_count += 1
                    verticesDict[coo] = node_count
                    self.voxel_dict[elmt_count].append(node_count)
                else:
                    self.voxel_dict[elmt_count].append(verticesDict[coo])

        # nodal positions array
        self.nodes = np.zeros((node_count, 3))
        print('### create voxels', node_count, self.nodes.shape)
        for pos, i in verticesDict.items():
            # print('***   ', i, pos)
            self.nodes[i - 1, :] = pos
        return


def set_stats(grains, ar=None, omega=None, deq_min=None, deq_max=None,
              asp_min=None, asp_max=None, omega_min=None, omega_max=None,
              size=100, voxels=60, gtype='Elongated', rveunit='um',
              periodicity=False, VF=None, phasename=None, phasenum=None,
              save_file=False):
    '''
    grains = [std deviation, offset , mean grain sizeof lognorm distrib.]
    ar = [std deviation, offset , mean aspect ratio of gamma distrib.]
    omega = [std deviation, mean tilt angle]
    '''

    # type of grains either 'Elongated' or 'Equiaxed'
    if not (gtype == 'Elongated' or gtype == 'Equiaxed'):
        raise ValueError('Wrong grain type given in set_stats: {}'
                         .format(gtype))
    if gtype == 'Elongated' and (ar is None or omega is None):
        raise ValueError('If elliptical grains are specified, aspect ratio ' +
                         '(ar) and orientation (omega) need to be given.')
    if gtype == 'Equiaxed' and (ar is not None or omega is not None):
        warnings.warn('Equiaxed grains have been specified, but aspect ratio' +
                      ' (ar) and orientation (omega) have been provided. ' +
                      'Will change grain type to "Elongated".')
        gtype = 'Elongated'

    # define cutoff values
    # cutoff deq
    if deq_min is None:
        deq_min = 1.3 * grains[1]  # 316L: 8
    if deq_max is None:
        deq_max = grains[1] + grains[2] + 6. * grains[0]  # 316L: 30
    if gtype == 'Elongated':
        # cutoff asp
        if asp_min is None:
            asp_min = np.maximum(1., ar[1])  # 316L: 1
        if asp_max is None:
            asp_max = ar[2] + ar[0]  # 316L: 3
        # cutoff omega
        if omega_min is None:
            omega_min = omega[1] - 2 * omega[0]
        if omega_max is None:
            omega_max = omega[1] + 2 * omega[0]

    # RVE box size
    lx = ly = lz = size  # size of box in each direction
    # number of voxels
    nx = ny = nz = voxels  # number of voxels in each direction
    # specify RVE info
    pbc = 'True' if periodicity else 'False'

    # check grain type
    # create the corresponding dict with statistical grain geometry information
    ms_stats = {'Grain type': gtype,
                'Equivalent diameter':
                    {'std': grains[0], 'mean': grains[2], 'offs': grains[1],
                     'cutoff_min': deq_min, 'cutoff_max': deq_max},
                'RVE':
                    {'sideX': lx, 'sideY': ly, 'sideZ': lz,
                     'Nx': nx, 'Ny': ny, 'Nz': nz},
                'Simulation': {'periodicity': pbc,
                               'output_units': rveunit},
                'Phase': {'Name': phasename,
                          'Number': phasenum,
                          'Volume fraction': VF}}
    if gtype == 'Elongated':
        ms_stats['Aspect ratio'] = {'std': ar[0], 'mean': ar[2], 'offs': ar[1],
                                    'cutoff_min': asp_min, 'cutoff_max': asp_max}
        ms_stats['Tilt angle'] = {'std': omega[0], 'mean': omega[1],
                                  'cutoff_min': omega_min, 'cutoff_max': omega_max}
    if save_file:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'  # Folder to store the json files
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(json_dir + '/stat_info.json', 'w') as outfile:
            json.dump(ms_stats, outfile, indent=2)

    return ms_stats
