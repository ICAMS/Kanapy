""" Module defining class Microstructure that contains the necessary
methods and attributes to analyze experimental microstructures in form
of EBSD maps to generate statistical descriptors for 3D microstructures, and 
to create synthetic RVE that fulfill the requires statistical microstructure
descriptors.

The methods of the class Microstructure for an API that can be used to generate
Python workflows.

Authors: Alexander Hartmaier, Golsa Tolooei Eshlghi, Abhishek Biswas
Institution: ICAMS, Ruhr University Bochum

"""
import os
import json
import itertools
import warnings
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean
from kanapy.input_output import particleStatGenerator, RVEcreator, \
    plot_output_stats, export2abaqus, l1_error_est
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_voxels_3D, plot_ellipsoids_3D, \
    plot_polygons_3D


class Microstructure:
    '''Define class for synthetic microstructures'''

    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.particle_data = None
        self.RVE_data = None
        self.simulation_data = None
        self.particles = None
        self.simbox = None
        self.nodes_s = None
        self.nodes_v = None
        self.voxels = None
        self.grain_facesDict = None
        self.elmtSetDict = None
        self.res_data = None
        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an input file name')

            # Open the user input statistics file and read the data
            try:
                with open(file) as json_file:
                    self.descriptor = json.load(json_file)
            except:
                raise FileNotFoundError("File: '{}' does not exist in the current working directory!\n".format(file))
        else:
            if type(descriptor) is not list:
                self.descriptor = [descriptor]
            else:
                self.descriptor = descriptor
            if file is not None:
                print('WARNING: Input parameter (descriptor) and file are given. Only descriptor will be used.')

    """
    --------        Routines for user interface        --------
    """

    def init_RVE(self, descriptor=None, nsteps=1000, save_files=False):
        """ Creates RVE based on the data provided in the input file."""
        if descriptor is None:
            descriptor = self.descriptor
        if type(descriptor) is not list:
            descriptor = [descriptor]
        self.nphase = len(descriptor)
        for des in descriptor:
            particle_data, RVE_data, simulation_data = \
                RVEcreator(des, nsteps=nsteps, save_files=save_files)
            Ngr = particle_data['Number']
            if des == descriptor[0]:
                self.Ngr = Ngr
                self.particle_data = particle_data
                self.RVE_data = RVE_data
                self.simulation_data = simulation_data
            else:
                if des['Grain type'] == 'Elongated':
                    self.Ngr += Ngr
                    """This needs to be stored in a phase-specific way"""
                    self.particle_data['Equivalent_diameter'] = self.particle_data['Equivalent_diameter'] + \
                                                                particle_data['Equivalent_diameter']
                    self.particle_data['Major_diameter'] = self.particle_data['Major_diameter'] + particle_data[
                        'Major_diameter']
                    self.particle_data['Minor_diameter1'] = self.particle_data['Minor_diameter1'] + particle_data[
                        'Minor_diameter1']
                    self.particle_data['Minor_diameter2'] = self.particle_data['Minor_diameter2'] + particle_data[
                        'Minor_diameter2']
                    self.particle_data['Number'] = self.particle_data['Number'] + particle_data['Number']
                    self.particle_data['Phase name'] = self.particle_data['Phase name'] + particle_data['Phase name']
                    self.particle_data['Phase number'] = self.particle_data['Phase number'] + particle_data[
                        'Phase number']
                    self.particle_data['Tilt angle'] = self.particle_data['Tilt angle'] + particle_data['Tilt angle']
                elif des['Grain type'] == 'Equiaxed':
                    self.Ngr = self.Ngr + Ngr
                    self.particle_data['Equivalent_diameter'] = self.particle_data['Equivalent_diameter'] + \
                                                                particle_data['Equivalent_diameter']
                    self.particle_data['Number'] = self.particle_data['Number'] + particle_data['Number']
                    self.particle_data['Phase name'] = self.particle_data['Phase name'] + particle_data['Phase name']
                    self.particle_data['Phase number'] = self.particle_data['Phase number'] + particle_data[
                        'Phase number']
            # if both Equiaxed and Elongated grains are present at the same time, it should be adjusted.

    def init_stats(self, descriptor=None, gs_data=None, ar_data=None,
                   save_files=False):
        """ Generates particle statistics based on the data provided in the 
        input file."""
        if descriptor is None:
            descriptor = self.descriptor
        if type(descriptor) is not list:
            descriptor = [descriptor]

        for des in descriptor:
            particleStatGenerator(des, gs_data=gs_data, ar_data=ar_data,
                                  save_files=save_files)

    def pack(self, particle_data=None, RVE_data=None, simulation_data=None,
             k_rep=0.0, k_att=0.0, save_files=False):

        """ Packs the particles into a simulation box."""
        if particle_data is None:
            particle_data = self.particle_data
            if particle_data is None:
                raise ValueError('No particle_data in pack. Run create_RVE first.')
        if RVE_data is None:
            RVE_data = self.RVE_data
        if simulation_data is None:
            simulation_data = self.simulation_data
        self.particles, self.simbox = \
            packingRoutine(particle_data, RVE_data, simulation_data,
                           k_rep=k_rep, k_att=k_att, save_files=save_files)

    def voxelize(self, particle_data=None, RVE_data=None, particles=None,
                 simbox=None, dual_phase=False, save_files=False):
        """ Generates the RVE by assigning voxels to grains."""
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if particles is None:
            particles = self.particles
            if particles is None:
                raise ValueError('No particles in voxelize. Run pack first.')
        if simbox is None:
            simbox = self.simbox
        self.nodes_v, self.elmtDict, self.elmtSetDict, \
        self.vox_centerDict, self.voxels, self.voxels_phase = \
            voxelizationRoutine(particle_data, RVE_data, particles, simbox,
                                save_files=save_files, dual_phase=dual_phase)
        Ngr = len(self.elmtSetDict.keys())
        if self.Ngr != Ngr:
            warnings.warn(f'Number of grains has changed from {self.Ngr} to {Ngr} during voxelization.')
            self.Ngr = Ngr

    def smoothen(self, nodes_v=None, elmtDict=None, elmtSetDict=None,
                 save_files=False):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        if nodes_v is None:
            nodes_v = self.nodes_v
            if nodes_v is None:
                raise ValueError('No nodes_v in smoothen. Run voxelize first.')
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        self.nodes_s, self.grain_facesDict = \
            smoothingRoutine(nodes_v, elmtDict, elmtSetDict,
                             save_files=save_files)

    def analyze_RVE(self, save_files=False, dual_phase=False):
        """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
        grain volumes and shared grain boundary surface areas info are written out as well."""

        if self.nodes_v is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if self.particle_data is None:
            raise ValueError('No particles created yet. Run init_RVE, pack and voxelize first.')

        self.grain_facesDict, self.shared_area = self.calcPolygons(dual_phase=dual_phase)  # updates RVE_data 
        self.res_data = self.get_stats(dual_phase=dual_phase)

    """
    --------     Plotting routines          --------
    """

    def plot_ellipsoids(self, cmap='prism', dual_phase=False):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        plot_ellipsoids_3D(self.particles, cmap=cmap, dual_phase=dual_phase)

    def plot_voxels(self, sliced=True, dual_phase=False, cmap='prism'):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.voxels is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        plot_voxels_3D(self.voxels, self.voxels_phase, Ngr=self.particle_data['Number'],
                       sliced=sliced, dual_phase=dual_phase, cmap=cmap)

    def plot_polygons(self, rve=None, cmap='prism', alpha=0.4,
                      ec=[0.5, 0.5, 0.5, 0.1], dual_phase=False):
        """ Plot polygonalized microstructure"""
        if rve is None:
            if 'Grains' in self.RVE_data.keys():
                rve = self.RVE_data
            else:
                raise ValueError('No polygons for grains defined. Run analyse_RVE first')
        plot_polygons_3D(rve, cmap=cmap, alpha=alpha, ec=ec, dual_phase=dual_phase)

    def plot_stats(self, data=None, gs_data=None, gs_param=None,
                   ar_data=None, ar_param=None, dual_phase=False, save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical 
        comparison."""
        if data is None:
            data = self.res_data
            if data is None:
                raise ValueError('No microstructure data created yet. Run analyse_RVE first.')
        else:
            data = [data]
        for dat in data:
            plot_output_stats(dat, gs_data=gs_data, gs_param=gs_param,
                              ar_data=ar_data, ar_param=ar_param,
                              save_files=save_files)

    def plot_slice(self, cut='xy', data=None, pos=None, fname=None,
                   dual_phase=False, save_files=False):
        """
        Plot a slice through the microstructure.
        
        If polygonalized microstructure is available, it will be used as data 
        basis, otherwise or if data='voxels' the voxelized microstructure 
        will be plotted.
        
        This subroutine calls the output_ang function with plotting active 
        and writing of ang file deactivated.

        Parameters
        ----------
        cut : str, optional
            Define cutting plane of slice as 'xy', 'xz' or 'yz'. The default is 'xy'.
        data : str, optional
            Define data basis for plotting as 'voxels' or 'poly'. The default is None.
        pos : str or float
            Position in which slice is taken, either as absolute value, or as 
            one of 'top', 'bottom', 'left', 'right'. The default is None.
        fname : str, optional
            Filename of PDF file. The default is None.
        save_files : bool, optional
            Indicate if figure file is saved and PDF. The default is False.

        Returns
        -------
        None.

        """
        self.output_ang(cut=cut, data=data, plot=True, save_files=False,
                        pos=pos, fname=fname, dual_phase=dual_phase, save_plot=save_files)

    """
    --------        Output/Export routines        --------
    """

    def output_abq(self, nodes=None, name=None, simulation_data=None,
                   elmtDict=None, elmtSetDict=None, faces=None):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""
        if simulation_data is None:
            simulation_data = self.simulation_data
        if nodes is None:
            if self.nodes_s is not None and self.grain_facesDict is not None:
                print('\nWarning: No information about nodes is given, will write smoothened structure')
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            elif self.nodes_v is not None:
                print('\nWarning: No information about nodes is given, will write voxelized structure')
                nodes = self.nodes_v
                faces = None
                ntag = 'voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        elif nodes == 'smooth' or nodes == 's':
            if self.nodes_s is not None and self.grain_facesDict is not None:
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            else:
                raise ValueError('No information about smoothed microstructure. Run smoothen first.')
        elif nodes == 'voxels' or nodes == 'v':
            if self.nodes_v is not None:
                nodes = self.nodes_v
                faces = None
                ntag = 'voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')

        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        if simulation_data is None:
            raise ValueError('No simulation data exists. Run create_RVE, pack and voxelize first.')
        if name is None:
            cwd = os.getcwd()
            name = cwd + '/kanapy_{0}grains_{1}.inp'.format(len(elmtSetDict), ntag)
            if os.path.exists(name):
                os.remove(name)  # remove old file if it exists
        export2abaqus(nodes, name, simulation_data, elmtSetDict, elmtDict, grain_facesDict=faces)
        return name

    # def output_neper(self, timestep=None):
    def output_neper(self):
        """ Writes out particle position and weights files required for
        tessellation in Neper."""
        # write_position_weights(timestep)
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        print('')
        print('Writing position and weights files for NEPER', end="")
        par_dict = dict()

        for pa in self.particles:
            x, y, z = pa.x, pa.y, pa.z
            a = pa.a
            par_dict[pa] = [x, y, z, a]

        with open('sphere_positions.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0} {1} {2}\n'.format(value[0], value[1], value[2]))

        with open('sphere_weights.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0}\n'.format(value[3]))
        print('---->DONE!\n')

    def output_ang(self, ori=None, cut='xy', data=None, plot=True, cs=None,
                   pos=None, fname=None, matname='XXXX', save_files=True,
                   dual_phase=False, save_plot=False):
        """
        Convert orientation information of microstructure into a .ang file,
        mimicking an EBSD map.
        If polygonalized microstructure is available, it will be used as data 
        basis, otherwise or if data='voxels' the voxelized microstructure 
        will be exported.
        If no orientations are provided, each grain will get a random 
        Euler angle.
        Values in ANG file:
        phi1 Phi phi2 X Y imageQuality confidenseIndex phase semSignal Fit(/mad)
        Output of ang file can be deactivated if called for plotting of slice.


        Parameters
        ----------
        ori : (self.Ngr,)-array, optional
            Euler angles of grains. The default is None.
        cut : str, optional
            Define cutting plane of slice as 'xy', 'xz' or 'yz'. The default is 'xy'.
        data : str, optional
            Define data basis for plotting as 'voxels' or 'poly'. The default is None.
        plot : bool, optional
            Indicate if slice is plotted. The default is True.
        pos : str or float
            Position in which slice is taken, either as absolute value, or as 
            one of 'top', 'bottom', 'left', 'right'. The default is None.
        cs : str, Optional
            Crystal symmetry. Default is None
        fname : str, optional
            Filename of ang file. The default is None.
        matname : str, optional
            Name of the material to be written in ang file. The default is 'XXXX'
        save_files : bool, optional
            Indicate if ang file is saved, The default is True.

        Returns
        -------
        fname : str
            Name of ang file.

        """
        cut = cut.lower()
        if cut == 'xy':
            sizeX = self.RVE_data['RVE_sizeX']
            sizeY = self.RVE_data['RVE_sizeY']
            sx = self.RVE_data['Voxel_resolutionX']
            sy = self.RVE_data['Voxel_resolutionY']
            sz = self.RVE_data['Voxel_resolutionZ']
            ix = np.arange(self.RVE_data['Voxel_numberX'])
            iy = np.arange(self.RVE_data['Voxel_numberY'])
            if pos is None or pos == 'top' or pos == 'right':
                iz = self.RVE_data['Voxel_numberZ'] - 1
            elif pos == 'bottom' or pos == 'left':
                iz = 0
            elif type(pos) == float or type(pos) == int:
                iz = int(pos / sz)
            else:
                raise ValueError('"pos" must be either float or "top", "bottom", "left" or "right"')
            if pos is None:
                pos = int(iz * sz)
            xl = r'x ($\mu$m)'
            yl = r'y ($\mu$m)'
            title = r'XY slice at z={} $\mu$m'.format(round(iz * sz, 1))
        elif cut == 'xz':
            sizeX = self.RVE_data['RVE_sizeX']
            sizeY = self.RVE_data['RVE_sizeZ']
            sx = self.RVE_data['Voxel_resolutionX']
            sy = self.RVE_data['Voxel_resolutionZ']
            sz = self.RVE_data['Voxel_resolutionY']
            ix = np.arange(self.RVE_data['Voxel_numberX'])
            iy = np.arange(self.RVE_data['Voxel_numberZ'])
            if pos is None or pos == 'top' or pos == 'right':
                iz = self.RVE_data['Voxel_numberY'] - 1
            elif pos == 'bottom' or pos == 'left':
                iz = 0
            elif type(pos) == float or type(pos) == int:
                iz = int(pos / sy)
            else:
                raise ValueError('"pos" must be either float or "top", "bottom", "left" or "right"')
            if pos is None:
                pos = int(iz * sz)
            xl = r'x ($\mu$m)'
            yl = r'z ($\mu$m)'
            title = r'XZ slice at y={} $\mu$m'.format(round(iz * sz, 1))
        elif cut == 'yz':
            sizeX = self.RVE_data['RVE_sizeY']
            sizeY = self.RVE_data['RVE_sizeZ']
            sx = self.RVE_data['Voxel_resolutionY']
            sy = self.RVE_data['Voxel_resolutionZ']
            sz = self.RVE_data['Voxel_resolutionX']
            ix = np.arange(self.RVE_data['Voxel_numberY'])
            iy = np.arange(self.RVE_data['Voxel_numberZ'])
            if pos is None or pos == 'top' or pos == 'right':
                iz = self.RVE_data['Voxel_numberX'] - 1
            elif pos == 'bottom' or pos == 'left':
                iz = 0
            elif type(pos) == float or type(pos) == int:
                iz = int(pos / sx)
            else:
                raise ValueError('"pos" must be either float or "top", "bottom", "left" or "right"')
            if pos is None:
                pos = int(iz * sz)
            xl = r'y ($\mu$m)'
            yl = r'z ($\mu$m)'
            title = r'YZ slice at x={} $\mu$m'.format(round(iz * sz, 1))
        else:
            raise ValueError('"cut" must bei either "xy", "xz" or "yz".')
        # ANG file header
        head = ['# TEM_PIXperUM          1.000000\n',
                '# x-star                0.000000\n',
                '# y-star                0.000000\n',
                '# z-star                0.000000\n',
                '# WorkingDistance       0.000000\n',
                '#\n',
                '# Phase 0\n',
                '# MaterialName  	{}\n'.format(matname),
                '# Formula\n',
                '# Info\n',
                '# Symmetry              m-3m\n',
                '# LatticeConstants       4.050 4.050 4.050  90.000  90.000  90.000\n',
                '# NumberFamilies        0\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# ElasticConstants 	0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n',
                '# Categories0 0 0 0 0\n',
                '# \n',
                '# GRID: SqrGrid\n',
                '# XSTEP: {}\n'.format(round(sx, 6)),
                '# YSTEP: {}\n'.format(round(sy, 6)),
                '# NCOLS_ODD: {}\n'.format(ix),
                '# NCOLS_EVEN: {}\n'.format(ix),
                '# NROWS: {}\n'.format(iy),
                '#\n',
                '# OPERATOR: 	Administrator\n',
                '#\n',
                '# SAMPLEID:\n',
                '#\n',
                '# SCANID:\n',
                '#\n'
                ]

        # determine whether polygons or voxels shall be exported
        if data is None:
            if 'Grains' in self.RVE_data.keys():
                data = 'poly'
            elif self.voxels is None:
                raise ValueError('Neither polygons nor voxels for grains are present.\
                                 \nRun voxelize and analyze_RVE first')
            else:
                data = 'voxels'
        elif data != 'voxels' and data != 'poly':
            raise ValueError('"data" must be either "voxels" or "poly".')

        if data == 'voxels':
            title += ' (Voxels)'
            if cut == 'xy':
                g_slice = np.array(self.voxels[:, :, iz], dtype=int)
            elif cut == 'xz':
                g_slice = np.array(self.voxels[:, iz, :], dtype=int)
            else:
                g_slice = np.array(self.voxels[iz, :, :], dtype=int)
            if dual_phase == True:
                if cut == 'xy':
                    g_slice_phase = np.array(self.voxels_phase[:, :, iz], dtype=int)
                elif cut == 'xz':
                    g_slice_phase = np.array(self.voxels_phase[:, iz, :], dtype=int)
                else:
                    g_slice_phase = np.array(self.voxels_phase[iz, :, :], dtype=int)
        else:
            title += ' (Polygons)'
            xv, yv = np.meshgrid(ix * sx, iy * sy, indexing='ij')
            grain_slice = np.ones(len(ix) * len(iy), dtype=int)
            if cut == 'xy':
                mesh_slice = np.array([xv.flatten(), yv.flatten(), grain_slice * iz * sz]).T
            elif cut == 'xz':
                mesh_slice = np.array([xv.flatten(), grain_slice * iz * sz, yv.flatten()]).T
            else:
                mesh_slice = np.array([grain_slice * iz * sz, xv.flatten(), yv.flatten()]).T
            grain_slice = np.zeros(len(ix) * len(iy), dtype=int)
            for igr in self.RVE_data['Grains'].keys():
                pts = self.RVE_data['Grains'][igr]['Points']
                try:
                    tri = Delaunay(pts)
                    i = tri.find_simplex(mesh_slice)
                    ind = np.nonzero(i >= 0)[0]
                    grain_slice[ind] = igr
                    if self.RVE_data['Periodic']:
                        # add periodic images of grain to slice
                        cb = np.array([self.RVE_data['RVE_sizeX'], self.RVE_data['RVE_sizeY'],
                                       self.RVE_data['RVE_sizeZ']]) * 0.5
                        sp = self.RVE_data['Grains'][igr]['Center'] - cb
                        plist = []
                        for j, split in enumerate(self.RVE_data['Grains'][igr]['Split']):
                            if split:
                                # store copy of points for each direction in 
                                # which grain is split
                                plist.append(deepcopy(pts))
                                for ppbc in plist:
                                    # shift grains to all image positions
                                    if sp[j] > 0.:
                                        ppbc[:, j] -= 2 * cb[j]
                                    else:
                                        ppbc[:, j] += 2 * cb[j]
                                    tri = Delaunay(ppbc)
                                    i = tri.find_simplex(mesh_slice)
                                    ind = np.nonzero(i >= 0)[0]
                                    grain_slice[ind] = igr
                                    if j == 2 and len(plist) == 3:
                                        # if split grain is in corner, 
                                        # cover last image position
                                        if sp[0] > 0.:
                                            ppbc[:, 0] -= 2 * cb[0]
                                        else:
                                            ppbc[:, 0] += 2 * cb[0]
                                        tri = Delaunay(ppbc)
                                        i = tri.find_simplex(mesh_slice)
                                        ind = np.nonzero(i >= 0)[0]
                                        grain_slice[ind] = igr
                except:
                    warnings.warn('Grain #{} has no convex hull (Nvertices: {})' \
                                  .format(igr, len(pts)))
            if np.any(grain_slice == 0):
                ind = np.nonzero(grain_slice == 0)[0]
                warnings.warn('Incomplete slicing for {} pixels in {} slice at {}.' \
                              .format(len(ind), cut, pos))
            g_slice = grain_slice.reshape(xv.shape)

        if save_files:
            if ori is None:
                ori = np.zeros((self.Ngr, 3))
                ori[:, 0] = np.random.rand(self.Ngr) * 2 * np.pi
                ori[:, 1] = np.random.rand(self.Ngr) * 0.5 * np.pi
                ori[:, 2] = np.random.rand(self.Ngr) * 0.5 * np.pi
            # write data to ang file
            fname = '{0}_slice_{1}_{2}.ang'.format(cut.upper(), pos, data)
            with open(fname, 'w') as f:
                f.writelines(head)
                for j in iy:
                    for i in ix:
                        p1 = ori[g_slice[j, i] - 1, 0]
                        P = ori[g_slice[j, i] - 1, 1]
                        p2 = ori[g_slice[j, i] - 1, 2]
                        f.write('  {0}  {1}  {2}  {3}  {4}   0.0  0.000  0   1  0.000\n' \
                                .format(round(p1, 5), round(P, 5), round(p2, 5),
                                        round(sizeX - i * sx, 5), round(sizeY - j * sy, 5)))
        if plot:
            # plot grains on slice
            # cmap = plt.cm.get_cmap('gist_rainbow')
            cmap = plt.cm.get_cmap('prism')
            fig, ax = plt.subplots(1)
            ax.grid(False)
            ax.imshow(g_slice, cmap=cmap, interpolation='none',
                      extent=[0, sizeX, 0, sizeY])
            ax.set(xlabel=xl, ylabel=yl)
            ax.set_title(title)
            if save_plot:
                plt.savefig(fname[:-4] + '.pdf', format='pdf', dpi=300)
            plt.show()

            if dual_phase == True:
                fig, ax = plt.subplots(1)
                ax.grid(False)
                ax.imshow(g_slice_phase, cmap=cmap, interpolation='none',
                          extent=[0, sizeX, 0, sizeY])
                ax.set(xlabel=xl, ylabel=yl)
                ax.set_title(title)
                if save_plot:
                    plt.savefig(fname[:-4] + '.pdf', format='pdf', dpi=300)
                plt.show()
        return fname

    def write_stl(self, file=None):
        """ Write triangles of convex polyhedra forming grains in form of STL
        files in the format:
        solid name
          facet normal n1 n2 n3
            outer loop
              vertex p1x p1y p1z
              vertex p2x p2y p2z
              vertex p3x p3y p3z
            endloop
          endfacet
        endsolid name

        Returns
        -------
        None.
        """
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains.stl'.format(self.Ngr)
            else:
                file = self.name + '.stl'
        with open(file, 'w') as f:
            f.write("solid {}\n".format(self.name));
            for ft in self.RVE_data['Facets']:
                pts = self.RVE_data['Points'][ft]
                nv = np.cross(pts[1] - pts[0], pts[2] - pts[0])  # facet normal
                if np.linalg.norm(nv) < 1.e-5:
                    warnings.warning(f'Acute facet detected. Facet: {ft}')
                    nv = np.cross(pts[1] - pts[0], pts[2] - pts[1])
                    if np.linalg.norm(nv) < 1.e-5:
                        warnings.warning(f'Irregular facet detected. Facet: {ft}')
                nv /= np.linalg.norm(nv)
                f.write(" facet normal {} {} {}\n"
                        .format(nv[0], nv[1], nv[2]))
                f.write(" outer loop\n")
                f.write("   vertex {} {} {}\n"
                        .format(pts[0, 0], pts[0, 1], pts[0, 2]))
                f.write("   vertex {} {} {}\n"
                        .format(pts[1, 0], pts[1, 1], pts[1, 2]))
                f.write("   vertex {} {} {}\n"
                        .format(pts[2, 0], pts[2, 1], pts[2, 2]))
                f.write("  endloop\n")
                f.write(" endfacet\n")
            f.write("endsolid\n")
            return

    def write_centers(self, file=None, grains=None):
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_centroid.csv'.format(self.Ngr)
            else:
                file = self.name + '_centroid.csv'
        if grains is None:
            grains = self.RVE_data['Grains']
        with open(file, 'w') as f:
            for gr in grains.values():
                # if polyhedral grains has no simplices, center should not be written!!!
                ctr = gr['Center']
                f.write('{}, {}, {}\n'.format(ctr[0], ctr[1], ctr[2]))
        return

    def write_ori(self, angles, file=None):
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_ori.csv'.format(self.Ngr)
            else:
                file = self.name + '_ori.csv'
        with open(file, 'w') as f:
            for ori in angles:
                f.write('{}, {}, {}\n'.format(ori[0], ori[1], ori[2]))
        return

    """
    --------        Supporting Routines         -------
    """

    def calcPolygons(self, tol=1.e-3, dual_phase=False):
        """
        Evaluates the grain volume and the grain boundary shared surface area 
        between neighbouring grains in voxelized microstrcuture. Generate 
        vertices as contact points between 3 or more grains. Generate 
        polyhedral convex hull for vertices.

        Parameters
        ----------
        tol : TYPE, optional
            DESCRIPTION. The default is 1.e-3.

        Returns
        -------
        grain_facesDict : TYPE
            DESCRIPTION.
        gbDict : TYPE
            DESCRIPTION.
        shared_area : TYPE
            DESCRIPTION.
            
        ISSUES: for periodic structures, large grains with segments in both 
        halves of the box and touching one boundary are split wrongly

        """

        def facet_on_boundary(conn):
            """
            Check if given voxel facet lies on any RVE boundary

            Parameters
            ----------
            conn

            Returns
            -------
            face : (3,2)-array of bool
            """
            n1 = self.nodes_v[conn[0] - 1, :]
            n2 = self.nodes_v[conn[1] - 1, :]
            n3 = self.nodes_v[conn[2] - 1, :]
            n4 = self.nodes_v[conn[3] - 1, :]
            face = np.zeros((3,2), dtype=bool)
            for i in range(3):
                h1 = np.abs(n1[i] - RVE_min[i]) < tol
                h2 = np.abs(n2[i] - RVE_min[i]) < tol
                h3 = np.abs(n3[i] - RVE_min[i]) < tol
                h4 = np.abs(n4[i] - RVE_min[i]) < tol
                face[i, 0] = h1 and h2 and h3 and h4
                h1 = np.abs(n1[i] - RVE_max[i]) < tol
                h2 = np.abs(n2[i] - RVE_max[i]) < tol
                h3 = np.abs(n3[i] - RVE_max[i]) < tol
                h4 = np.abs(n4[i] - RVE_max[i]) < tol
                face[i, 1] = h1 and h2 and h3 and h4
            return face

        def check_neigh(nodes, grains, margin):
            ''' Check if close neighbors of new vertices are already in list
            # nodes: list of nodes identified as new vertices
            # grains: set of grains containing the nodes in elist
            # margin: radius in which vertices will be united'''

            # create set of all vertices of all involved grains
            vset = set()
            for gid in grains:
                vset.update(vert[gid])
            # loop over all combinations
            for i, nid1 in enumerate(nodes):
                npos1 = self.nodes_v[nid1 - 1]
                sur1 = [np.abs(npos1[0] - RVE_min[0]) < tol,
                        np.abs(npos1[1] - RVE_min[1]) < tol,
                        np.abs(npos1[2] - RVE_min[2]) < tol,
                        np.abs(npos1[0] - RVE_max[0]) < tol,
                        np.abs(npos1[1] - RVE_max[1]) < tol,
                        np.abs(npos1[2] - RVE_max[2]) < tol]
                for nid2 in vset:
                    npos2 = self.nodes_v[nid2 - 1]
                    sur2 = [np.abs(npos2[0] - RVE_min[0]) < tol,
                            np.abs(npos2[1] - RVE_min[1]) < tol,
                            np.abs(npos2[2] - RVE_min[2]) < tol,
                            np.abs(npos2[0] - RVE_max[0]) < tol,
                            np.abs(npos2[1] - RVE_max[1]) < tol,
                            np.abs(npos2[2] - RVE_max[2]) < tol]
                    d = np.linalg.norm(npos1 - npos2)
                    if d < margin:
                        if (not np.any(sur1)) or (sur1 == sur2):
                            nodes[i] = nid2
                        break
            return nodes

        def get_voxel(pos):
            """
            Get voxel associated with position vector pos.

            Parameters
            ----------
            pos : TYPE
                DESCRIPTION.

            Returns
            -------
            v0 : TYPE
                DESCRIPTION.
            v1 : TYPE
                DESCRIPTION.
            v2 : TYPE
                DESCRIPTION.

            """
            v0 = np.minimum(int(pos[0] / self.RVE_data['Voxel_resolutionX']),
                            self.RVE_data['Voxel_numberX'] - 1)
            v1 = np.minimum(int(pos[1] / self.RVE_data['Voxel_resolutionY']),
                            self.RVE_data['Voxel_numberY'] - 1)
            v2 = np.minimum(int(pos[2] / self.RVE_data['Voxel_resolutionZ']),
                            self.RVE_data['Voxel_numberZ'] - 1)
            return (v0, v1, v2)

        def tet_in_grain(tet, vertices):
            """

            Parameters
            ----------
            tet
            vertices

            Returns
            -------

            """
            return self.RVE_data['Vertices'][tet[0]] in vertices and \
                   self.RVE_data['Vertices'][tet[1]] in vertices and \
                   self.RVE_data['Vertices'][tet[2]] in vertices and \
                   self.RVE_data['Vertices'][tet[3]] in vertices

        def vox_in_tet(vox_, tet_):
            """
            Determine whether centre of voxel lies within tetrahedron

            Parameters
            ----------
            vox_
            tet_

            Returns
            -------
            contained : bool
                Voxel lies in tetrahedron
            """

            v_pos = self.vox_centerDict[vox_]
            contained = True
            for node in tet_:
                n_pos = self.RVE_data['Points'][node]
                hh = set(tet_)
                hh.remove(node)
                ind_ = list(hh)
                f_pos = self.RVE_data['Points'][ind_]
                ctr_ = np.mean(f_pos, axis=0)
                normal = np.cross(f_pos[1, :] - f_pos[0, :], f_pos[2, :] - f_pos[0, :])
                hn = np.linalg.norm(normal)
                if hn > 1.e-5:
                    normal /= hn
                dist_to_vox = np.dot(v_pos - ctr_, normal)
                dist_to_node = np.dot(n_pos - ctr_, normal)
                if np.sign(dist_to_vox * dist_to_node) < 0. or \
                        np.abs(dist_to_vox) > np.abs(dist_to_node):
                    contained = False
                    break
            return contained

        # define constants
        periodic = self.RVE_data['Periodic']
        voxel_size = self.RVE_data['Voxel_resolutionX']
        RVE_min = np.amin(self.nodes_v, axis=0)
        if np.any(RVE_min > 1.e-3) or np.any(RVE_min < -1.e-3):
            raise ValueError('Irregular RVE geometry: RVE_min = {}'.format(RVE_min))
        RVE_max = np.amax(self.nodes_v, axis=0)
        Ng = np.amax(list(self.elmtSetDict.keys()))  # highest grain number

        # create dicts for GB facets, including fake facets at surfaces
        grain_facesDict = dict()  # {Grain: faces}
        for i in range(1, Ng + 7):
            grain_facesDict[i] = dict()

        # genrate following objects:
        # outer_faces: {face_id's of outer voxel faces}  (potential GB facets)
        # face_nodes: {face_id: list with 4 nodes}
        # grain_facesDict: {grain_id: {face_id: list with 4 nodes}}
        for gid, elset in self.elmtSetDict.items():
            outer_faces = set()
            face_nodes = dict()
            nodeConn = [self.elmtDict[el] for el in elset]  # Nodal connectivity of a voxel

            # For each voxel, re-create its 6 faces
            for nc in nodeConn:
                faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                         [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                         [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]
                # Sort in ascending order
                sorted_faces = [sorted(fc) for fc in faces]
                # create unique face ids by joining node id's
                face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]
                # Create face_nodes = {face_id: nodal connectivity} dictionary
                for enum, fid in enumerate(face_ids):
                    if fid not in face_nodes.keys():
                        face_nodes[fid] = faces[enum]
                # Identify outer faces that occur only once and store in outer_faces
                for fid in face_ids:
                    if fid not in outer_faces:
                        outer_faces.add(fid)
                    else:
                        outer_faces.remove(fid)

            # Update grain_faces_dict= {grain_id: {face_id: facet nodes}
            for of in outer_faces:
                # add face nodes to grain faces dictionary
                conn = face_nodes[of]
                grain_facesDict[gid][of] = conn  # list of four nodes
                # if voxel faces lies on RVE boundary add also to fake grains
                fob = facet_on_boundary(conn)
                for i in range(3):
                    for j in range(2):
                        if fob[i, j]:
                            grain_facesDict[Ng + 1 + 2*i + j][of] = conn

        # Find all combinations of grains to check for common area
        # analyse grain_facesDict and create object:
        # gbDict: {f{gid1}_{gid2}: list with 4 nodes shared by grains #gid1 and #gid2}
        # shared_area: [[gid1, gid2, GB area]]
        shared_area = []  # GB area
        gbDict = dict()  # voxel factes on GB
        # Find the shared area and generate gbDict for all pairs of neighboring grains
        combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))
        for cb in combis:
            finter = set(grain_facesDict[cb[0]]).intersection(set(grain_facesDict[cb[1]]))
            if finter:
                ind = set()
                [ind.update(grain_facesDict[cb[0]][key]) for key in finter]
                key = 'f{}_{}'.format(cb[0], cb[1])
                gbDict[key] = ind
                if cb[0] <= Ng and cb[1] <= Ng:
                    # grain facet is not on boundary
                    try:
                        hull = ConvexHull(self.nodes_v[list(ind), :])
                        shared_area.append([cb[0], cb[1], hull.area])
                    except:
                        sh_area = len(finter) * (voxel_size ** 2)
                        shared_area.append([cb[0], cb[1], sh_area])

        # analyse gbDict to find intersection lines of GB's
        # (triple or quadruple lines) -> edges
        # vertices are end points of edges, represented by
        # nodes in voxelized microstructure
        # created objects:
        # vert: {grain_id: [node_numbers of vertices]}
        # grains_of_vert: {node_number: [grain_id's connected to vertex]}

        # for periodic structures vertices at surfaces should be mirrored!!!

        vert = dict()
        grains_of_vert = dict()
        for i in grain_facesDict.keys():
            vert[i] = set()
        for key0 in gbDict.keys():
            klist = list(gbDict.keys())
            # select grains with list positions before the current one
            while key0 != klist.pop(0):
                pass
            for key1 in klist:
                finter = gbDict[key0].intersection(gbDict[key1])
                if finter:
                    if len(finter) == 1:
                        # only one node in intersection of GBs
                        elist = list(finter)
                    else:
                        # mulitple nodes in intersection 
                        # identify end points of triple or quadruple line 
                        edge = np.array(list(finter), dtype=int)
                        rn = self.nodes_v[edge - 1]
                        dmax = 0.
                        for j, rp0 in enumerate(rn):
                            for k, rp1 in enumerate(rn[j + 1:, :]):
                                d = np.sqrt(np.dot(rp0 - rp1, rp0 - rp1))
                                if d > dmax:
                                    elist = [edge[j], edge[k + j + 1]]
                                    dmax = d
                    # select all involved grains
                    gr_set = set()
                    j = key0.index('_')
                    gr_set.add(int(key0[1:j]))
                    gr_set.add(int(key0[j + 1:]))
                    j = key1.index('_')
                    gr_set.add(int(key1[1:j]))
                    gr_set.add(int(key1[j + 1:]))
                    # check if any neighboring nodes are already in list of
                    # vertices. If yes, replace new vertex with existing one
                    newlist = check_neigh(elist, gr_set, margin=2 * voxel_size)
                    # update grains with new vertex
                    for j in gr_set:
                        vert[j].update(newlist)
                    for nv in newlist:
                        for j in range(1, 7):
                            # discard fake grains at RVE boundary in grain list
                            gr_set.discard(self.Ngr + j)
                        if nv in grains_of_vert.keys():
                            grains_of_vert[nv].update(gr_set)
                        else:
                            grains_of_vert[nv] = gr_set

        # Store grain-based information and do Delaunay tesselation
        # Sort grains w.r.t number of vertices
        num_vert = [len(vert[igr]) for igr in self.elmtSetDict.keys()]
        glist = np.array(list(self.elmtSetDict.keys()), dtype=int)
        glist = list(glist[np.argsort(num_vert)])
        assert len(glist) == self.Ngr

        # re-sort by keeping neighborhood relations
        grain_sequence = [glist.pop()]  # start sequence with grain with most vertices
        while len(glist) > 0:
            igr = grain_sequence[-1]
            neigh = set()
            for gb in shared_area:
                if igr == gb[0]:
                    neigh.add(gb[1])
                elif igr == gb[1]:
                    neigh.add(gb[0])
            # remove grains already in list from neighbor set
            neigh.difference_update(set(grain_sequence))
            if len(neigh) == 0:
                # continue with next grain in list
                grain_sequence.append(glist.pop())
            else:
                # continue with neighboring grain that has most vertices
                ind = [glist.index(i) for i in neigh]
                if len(ind) == 0:
                    grain_sequence.append(glist.pop())
                else:
                    grain_sequence.append(glist.pop(np.amax(ind)))
        if len(grain_sequence) != self.Ngr or len(glist) > 0:
            raise ValueError(f'Grain list incomplete: {grain_sequence}, remaining elements: {glist}')

        # initialize dictionary for grain information
        grains = dict()
        vertices = np.array([], dtype=int)
        for step, igr in enumerate(grain_sequence):
            add_vert = vert[igr].difference(set(vertices))
            grains[igr] = dict()
            grains[igr]['Vertices'] = np.array(list(vert[igr]), dtype=int) - 1
            grains[igr]['Points'] = self.nodes_v[grains[igr]['Vertices']]
            center = np.mean(grains[igr]['Points'], axis=0)
            grains[igr]['Center'] = center
            # initialize values to be incrementally updated later
            grains[igr]['Simplices'] = []
            grains[igr]['Volume'] = 0.
            grains[igr]['Area'] = 0.

            # Construct incremental Delaunay tesselation of
            # structure given by vertices
            vlist = np.array(list(add_vert), dtype=int) - 1
            vertices = np.append(vertices, list(add_vert))
            if step == 0:
                tetra = Delaunay(self.nodes_v[vlist], incremental=True)
            else:
                try:
                    tetra.add_points(self.nodes_v[vlist])
                except:
                    warnings.warn(
                        f'Incremental Delaunay tesselation failed for grain {step + 1}. Restarting Delaunay process from there')
                    vlist = np.array(vertices, dtype=int) - 1
                    tetra = Delaunay(self.nodes_v[vlist], incremental=True)

        tetra.close()
        # store global result of tesselation
        self.RVE_data['Vertices'] = np.array(vertices, dtype=int) - 1
        self.RVE_data['Points'] = tetra.points
        self.RVE_data['Simplices'] = tetra.simplices

        # assign simplices (tetrahedra) to grains
        Ntet = len(tetra.simplices)
        print('\nGenerated Delaunay tesselation of grain vertices.')
        print(f'Assigning {Ntet} tetrahedra to grains ...')
        tet_to_grain = np.zeros(Ntet, dtype=int)
        for i, tet in tqdm(enumerate(tetra.simplices)):
            ctr = np.mean(tetra.points[tet], axis=0)
            igr = self.voxels[get_voxel(ctr)]
            # test if all vertices of tet belong to that grain
            if not tet_in_grain(tet, grains[igr]['Vertices']):
                # print(f'Grain {igr}: Tetra = {tet}, missing vertices')
                # try to find a neighboring grain with all vertices of tet
                neigh_list = set()
                for hv in tet:
                    neigh_list.update(grains_of_vert[vertices[hv]])
                    #    self.voxels[get_voxel(tetra.points[hv])])
                # print(f'### Neighboring grains: {neigh_list}')
                match_found = False
                for jgr in neigh_list:
                    if tet_in_grain(tet, grains[jgr]['Vertices']):
                        # print(f'*** Grain {jgr} contains all vertices')
                        igr = jgr
                        match_found = True
                        break
                if not match_found:
                    # get a majority vote. BETTER: split up tet
                    # count all voxels in tet
                    neigh_list.add(igr)
                    neigh_list = list(neigh_list)
                    num_vox = []
                    # print(f'nodes in tet: {tet}')
                    for ngr in neigh_list:
                        nv = 0
                        for vox in self.elmtSetDict[ngr]:
                            if vox_in_tet(vox, tet):
                                nv += 1
                        num_vox.append(nv)
                    igr = neigh_list[np.argmax(num_vox)]
                    # print(f'+++ Majority vote grain {igr}')
                    # print(f'Neighbors: {neigh_list}')
                    # print(f'Counts: {num_vox}')

            tet_to_grain[i] = igr
            # Update grain volume with tet volume
            dv = tetra.points[tet[3]]
            vmat = [tetra.points[tet[0]] - dv,
                    tetra.points[tet[1]] - dv,
                    tetra.points[tet[2]] - dv]
            grains[igr]['Volume'] += np.abs(np.linalg.det(vmat))/6.

        # Keep only facets at boundary or between different grains
        facet_keys = set()
        for i, tet in enumerate(tetra.simplices):
            igr = tet_to_grain[i]
            for j, neigh in enumerate(tetra.neighbors[i, :]):
                if neigh == -1 or tet_to_grain[neigh] != igr:
                    ft = []
                    for k in range(4):
                        if k != j:
                            ft.append(tet[k])
                    ft = sorted(ft)
                    facet_keys.add(f'{ft[0]}_{ft[1]}_{ft[2]}')
                    grains[igr]['Simplices'].append(ft)
                    # Update grain surface area
                    cv = tetra.points[ft[2]]
                    avec = np.cross(tetra.points[ft[0]] - cv,
                                    tetra.points[ft[1]] - cv)
                    grains[igr]['Area'] += np.linalg.norm(avec)

        facets = []
        for key in facet_keys:
            hh = key.split('_')
            facets.append([int(hh[0]), int(hh[1]), int(hh[2])])
        self.RVE_data['Facets'] = np.array(facets)

        for igr in self.elmtSetDict.keys():
            # Find the euclidean distance to all surface points from the center
            dists = [euclidean(grains[igr]['Center'], pt) for pt in
                     self.nodes_v[grains[igr]['Vertices']]]
            if len(dists) == 0:
                warnings.warn(f'Grain {igr} with few vertices: {grains[igr]["Vertices"]}')
                dists = [0.]
            grains[igr]['eqDia'] = 2.0 * (3 * grains[igr]['Volume']
                                          / (4 * np.pi)) ** (1 / 3)
            grains[igr]['majDia'] = 2.0 * np.amax(dists)
            grains[igr]['minDia'] = 2.0 * np.amin(dists)
            if dual_phase:
                grains[igr]['PhaseID'] = self.particle_data['Phase number'][igr - 1]
                grains[igr]['PhaseName'] = self.particle_data['Phase name'][igr - 1]

        self.RVE_data['Grains'] = grains
        self.RVE_data['GBnodes'] = gbDict
        self.RVE_data['GBarea'] = shared_area
        print('Finished generating polyhedral hulls for grains.')
        vref = self.RVE_data['RVE_sizeX'] * \
               self.RVE_data['RVE_sizeY'] * \
               self.RVE_data['RVE_sizeZ']
        vtot = 0.
        vtot_vox = 0.
        vunit = self.RVE_data['Voxel_resolutionX'] * \
                self.RVE_data['Voxel_resolutionY'] * \
                self.RVE_data['Voxel_resolutionZ']
        vol_mae = 0.
        for igr, grain in self.RVE_data['Grains'].items():
            vg = grain['Volume']
            vtot += vg
            vvox = np.count_nonzero(self.voxels == igr)*vunit
            vtot_vox += vvox
            vol_mae += np.abs(1. - vg / vvox)
            #print(f'igr: {igr}, vol={vg}, vox={vvox}')
        vol_mae /= self.Ngr
        if np.abs(vtot - vref) > 1.e-5:
            warnings.warn(f'Inconsistent volume of polyhedral grains: {vtot}, Reference volume: {vref}')
        print(f'Mean absolute error of polyhedral vs. voxel volume of grains: {vol_mae}')

        return grain_facesDict, shared_area

    def get_stats(self, dual_phase=False):
        """
        Compare the geometries of particles used for packing and the resulting 
        grains.

        Parameters
        ----------
        save_files : bool, optional
            Indicate if output is written to file. The default is False.

        Returns
        -------
        output_data : dict
            Statistical information about particle and grain geometries.

        """
        # Analyse geometry of particles used for packing algorithm
        par_eqDia = np.array(self.particle_data['Equivalent_diameter'])
        if self.particle_data['Type'] == 'Elongated':
            par_majDia = np.array(self.particle_data['Major_diameter'])
            par_minDia = np.array(self.particle_data['Minor_diameter1'])

        # Analyse grain geometries        
        grain_eqDia = np.zeros(self.Ngr)
        grain_majDia = np.zeros(self.Ngr)
        grain_minDia = np.zeros(self.Ngr)
        grain_PhaseID = np.zeros(self.Ngr)
        grain_PhaseName = np.zeros(self.Ngr).astype(str)
        for i, igr in enumerate(self.RVE_data['Grains'].keys()):
            grain_eqDia[i] = self.RVE_data['Grains'][igr]['eqDia']
            if self.particle_data['Type'] == 'Elongated':
                grain_minDia[i] = self.RVE_data['Grains'][igr]['minDia']
                grain_majDia[i] = self.RVE_data['Grains'][igr]['majDia']
            if dual_phase:
                grain_PhaseID[i] = self.RVE_data['Grains'][igr]['PhaseID']
                grain_PhaseName[i] = self.RVE_data['Grains'][igr]['PhaseName']

        output_data_list = []
        indexPhase = []
        if dual_phase:
            for iph in range(self.nphase):
                indexPhase.append(grain_PhaseID == iph)
            # indexPhase = [grain_PhaseID == 0, grain_PhaseID == 1]
            # Compute the L1-error between particle and grain geometries for phase 0
            for index in indexPhase:
                if self.particle_data['Type'] == 'Elongated':
                    kwargs = {
                        'Ellipsoids': {'Equivalent': {'Particles': par_eqDia[index], 'Grains': grain_eqDia[index]},
                                       'Major diameter': {'Particles': par_majDia[index],
                                                          'Grains': grain_majDia[index]},
                                       'Minor diameter': {'Particles': par_minDia[index],
                                                          'Grains': grain_minDia[index]}}}
                else:
                    kwargs = {'Spheres': {'Equivalent': {'Particles': par_eqDia[index], 'Grains': grain_eqDia[index]}}}

                error = l1_error_est(**kwargs)
                print('\n    L1 error between particle and grain geometries: {}' \
                      .format(round(error, 5)))

                # Create dictionaries to store the data generated
                output_data = {'Number_of_particles/grains': int(len(par_eqDia[index])),
                               'Grain type': self.particle_data['Type'],
                               'Unit_scale': self.RVE_data['Units'],
                               'L1-error': error,
                               'Particle_Equivalent_diameter': par_eqDia[index],
                               'Grain_Equivalent_diameter': grain_eqDia[index]}
                if self.particle_data['Type'] == 'Elongated':
                    output_data['Particle_Major_diameter'] = par_majDia[index]
                    output_data['Particle_Minor_diameter'] = par_minDia[index]
                    output_data['Grain_Major_diameter'] = grain_majDia[index]
                    output_data['Grain_Minor_diameter'] = grain_minDia[index]
                    output_data['PhaseID'] = grain_PhaseID[index]
                    output_data['PhaseName'] = grain_PhaseName[index]
                output_data_list.append(output_data)

        else:
            # Compute the L1-error between particle and grain geometries
            if self.particle_data['Type'] == 'Elongated':
                kwargs = {'Ellipsoids': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia},
                                         'Major diameter': {'Particles': par_majDia, 'Grains': grain_majDia},
                                         'Minor diameter': {'Particles': par_minDia, 'Grains': grain_minDia}}}
            else:
                kwargs = {'Spheres': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia}}}

            error = l1_error_est(**kwargs)
            print('\n    L1 error between particle and grain geometries: {}' \
                  .format(round(error, 5)))

            # Create dictionaries to store the data generated
            output_data = {'Number_of_particles/grains': int(len(par_eqDia)),
                           'Grain type': self.particle_data['Type'],
                           'Unit_scale': self.RVE_data['Units'],
                           'L1-error': error,
                           'Particle_Equivalent_diameter': par_eqDia,
                           'Grain_Equivalent_diameter': grain_eqDia}
            if self.particle_data['Type'] == 'Elongated':
                output_data['Particle_Major_diameter'] = par_majDia
                output_data['Particle_Minor_diameter'] = par_minDia
                output_data['Grain_Major_diameter'] = grain_majDia
                output_data['Grain_Minor_diameter'] = grain_minDia
            output_data_list.append(output_data)

        return output_data_list
