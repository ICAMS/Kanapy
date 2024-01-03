""" Module defining class Microstructure that contains the necessary
methods and attributes to analyze experimental microstructures in form
of EBSD maps to generate statistical descriptors for 3D microstructures, and 
to create synthetic RVE that fulfill the requires statistical microstructure
descriptors.

The methods of the class Microstructure for an API that can be used to generate
Python workflows.

Authors: Alexander Hartmaier, Golsa Tolooei Eshlghi, Abhishek Biswas
Institution: ICAMS, Ruhr University Bochum

os.path.normpath(p)
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from kanapy.grains import calc_polygons, get_stats
from kanapy.entities import Simulation_Box
from kanapy.input_output import export2abaqus
from kanapy.initializations import RVE_creator, mesh_creator
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.import_EBSD import EBSDmap, createOrisetRandom, createOriset
from kanapy.plotting import plot_init_stats, plot_voxels_3D, plot_ellipsoids_3D, \
    plot_polygons_3D, plot_output_stats
from kanapy.util import log_level

"""
Class grain(number)
Attributes:
•	phase
•	voxels
•	facets
•	vertices
•	points
•	center
•	simplices
•	particle
•	eq_dia
•	maj_dia
•	min_dia
•	volume
•	area

Class geometry()
Attributes:
•	vertices
•	points
•	simplices
•	facets
•	GBnodes
•	GBarea

Class particle(number)
Attributes:
•	equiv_dia
•	maj_dia
•	min_dia
•	phase
•	…

Class phase(number)
Attributes:
•	particles (list of numbers)
•	grains (list of numbers)
"""


class Microstructure(object):
    """Define class for synthetic microstructures"""

    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.ngrains = None
        self.nparticles = None
        self.nphases = None
        self.porosity = None
        self.rve = None
        self.particles = None
        self.geometry = None
        self.simbox = None
        self.mesh = None
        self.res_data = None
        self.from_voxels = False

        logging.basicConfig(level=log_level)
        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an input file name')

            # Open the user input statistics file and read the data
            try:
                with open(os.path.normpath(file)) as json_file:
                    self.descriptor = json.load(json_file)
            except:
                raise FileNotFoundError("File: '{}' does not exist in the current working directory!\n".format(file))
        elif descriptor == 'from_voxels':
            self.from_voxels = True
        else:
            if type(descriptor) is not list:
                self.descriptor = [descriptor]
                self.nphases = 1
            else:
                self.descriptor = descriptor
                self.nphases = len(self.descriptor)
                if self.nphases > 2:
                    raise ValueError(f'Kanapy currently only supports 2 phases, but Nphase={self.nphases}')
            if file is not None:
                logging.warning(
                    'WARNING: Input parameter (descriptor) and file are given. Only descriptor will be used.')
        return

    """
    --------        Routines for user interface        --------
    """

    def init_RVE(self, descriptor=None, nsteps=1000, porosity=None):
        """
        Creates particle distribution inside simulation box (RVE) based on
        the data provided in the input file.

        Parameters
        ----------
        descriptor
        nsteps
        porosity

        Returns
        -------

        """
        if descriptor is None:
            descriptor = self.descriptor
        if type(descriptor) is not list:
            descriptor = [descriptor]
        elif porosity is not None:
            descriptor = descriptor[0:1]  # second phase will be treated as gaps in first phase
            self.porosity = porosity

        # initialize RVE, including mesh dimensions and particle distribution
        self.rve = RVE_creator(descriptor, nsteps=nsteps)
        self.nparticles = self.rve.nparticles

        # store geometry in simbox object
        self.simbox = Simulation_Box(self.rve.size)

        # initialize voxel structure (= mesh)
        self.mesh = mesh_creator(self.rve.dim)
        self.mesh.nphases = self.nphases
        self.mesh.create_voxels(self.simbox)

    def pack(self, particle_data=None,
             k_rep=0.0, k_att=0.0, vf=None, save_files=False):

        """ Packs the particles into a simulation box."""
        if particle_data is None:
            particle_data = self.rve.particle_data
            if particle_data is None:
                raise ValueError('No particle_data in pack. Run create_RVE first.')
        self.particles, self.simbox = \
            packingRoutine(particle_data, self.rve.periodic,
                           self.rve.packing_steps, self.simbox,
                           k_rep=k_rep, k_att=k_att, vf=vf,
                           save_files=save_files)

    def voxelize(self, particles=None):
        """ Generates the RVE by assigning voxels to grains."""
        if particles is None:
            particles = self.particles
            if particles is None:
                raise ValueError('No particles in voxelize. Run pack first.')

        self.mesh = \
            voxelizationRoutine(particles, self.mesh, porosity=self.porosity)
        if np.any(self.nparticles != self.mesh.ngrains_phase):
            logging.info(f'Number of grains per phase changed from {self.nparticles} to' +
                         f'{self.mesh.ngrains_phase} during voxelization.')
        self.ngrains = self.mesh.ngrains_phase
        self.Ngr = np.sum(self.mesh.ngrains_phase, dtype=int)  # legacy notation

    def smoothen(self, nodes_v=None, voxel_dict=None, grain_dict=None):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        if nodes_v is None:
            nodes_v = self.mesh.nodes
            if nodes_v is None:
                raise ValueError('No nodes_v in smoothen. Run voxelize first.')
        if voxel_dict is None:
            voxel_dict = self.mesh.voxel_dict
        if grain_dict is None:
            grain_dict = self.mesh.grain_dict
        self.mesh.nodes_smooth, grain_facesDict = \
            smoothingRoutine(nodes_v, voxel_dict, grain_dict)
        self.geometry['GBfaces'] = grain_facesDict

    def generate_grains(self):
        """ Writes out the particle- and grain diameter attributes for 
        statistical comparison. Final RVE grain volumes and shared grain
        boundary surface areas info are written out as well."""

        if self.mesh.nodes is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')

        self.geometry = \
            calc_polygons(self.rve, self.mesh)  # updates RVE_data
        self.res_data = \
            get_stats(self.rve.particle_data, self.geometry, self.rve.units, self.rve.nphases)


    def generate_orientations(self, input, ang=None, omega=None, Nbase=2000):
        """
        Calculates the orientations of grains to give a desired crystallographic texture.

        Parameters
        ----------
        input
        ang
        omega
        Nbase

        Returns
        -------

        """
        from kanapy import MTEX_AVAIL

        if not MTEX_AVAIL:
            raise ModuleNotFoundError('MTEX not installed. Run "kanapy setupTexture" first to use this function.')
        if self.mesh.grains is None:
            raise ValueError('Grain geometry is not defined. Run voxelize first.')
        if self.geometry is None:
            logging.warning('Grain boundary areas are not defined and cannot be considered in orientation assignment.')
            gba = None
        else:
            gba = self.geometry['GBarea']

        ori_dict = dict()
        for ip, ngr in enumerate(self.ngrains):
            if type(input) is EBSDmap:
                ori_rve = input.calcORI(ngr, iphase=ip, shared_area=gba)
            elif type(input) is str:
                if input.lower() in ['random', 'rnd']:
                    ori_rve = createOrisetRandom(ngr, Nbase=Nbase)
                elif input.lower() in ['unimodal', 'uni_mod', 'uni_modal']:
                    if ang is None or omega is None:
                        raise ValueError('To generate orientation sets of type "unimodal" angle "ang" and kernel' +
                                         'halfwidth "omega" are required.')
                    ori_rve = createOriset(ngr, ang, omega)
            else:
                raise ValueError('Argument to generate grain orientation must be either of type EBSDmap or ' +
                                 '"random" or "unimodal"')
            for i, igr in enumerate(self.mesh.grain_dict.keys()):
                if self.mesh.grain_phase_dict[igr] == ip:
                    ori_dict[igr] = ori_rve[i, :]
        self.mesh.grain_ori_dict = ori_dict
        return


    """
    --------     Plotting methods          --------
    """

    def plot_ellipsoids(self, cmap='prism', dual_phase=False):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        plot_ellipsoids_3D(self.particles, cmap=cmap, dual_phase=dual_phase)

    def plot_voxels(self, sliced=True, dual_phase=False, porous=None,
                    cmap='prism'):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.mesh.grains is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        if porous is None:
            porous = bool(self.porosity)
        plot_voxels_3D(self.mesh.grains, phases=self.mesh.phases, Ngr=np.sum(self.ngrains),
                       sliced=sliced, dual_phase=dual_phase, porous=porous,
                       cmap=cmap)

    def plot_grains(self, geometry=None, cmap='prism', alpha=0.4,
                    ec=[0.5, 0.5, 0.5, 0.1], dual_phase=False):
        """ Plot polygonalized microstructure"""
        if geometry is None:
            geometry = self.geometry
        if geometry is None:
            raise ValueError('No polygons for grains defined. Run analyse_RVE first')
        plot_polygons_3D(geometry, cmap=cmap, alpha=alpha, ec=ec,
                         dual_phase=dual_phase)

    def plot_stats(self, data=None, gs_data=None, gs_param=None,
                   ar_data=None, ar_param=None,
                   save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical 
        comparison."""
        if data is None:
            data = self.res_data
            if data is None:
                raise ValueError('No microstructure data created yet. Run analyse_RVE first.')
        elif type(data) != list:
            data = [data]
        plot_output_stats(data, gs_data=gs_data, gs_param=gs_param,
                          ar_data=ar_data, ar_param=ar_param,
                          save_files=save_files)

    def plot_stats_init(self, descriptor=None, gs_data=None, ar_data=None,
                        porous=False, save_files=False):
        """ Plots initial statistical microstructure descriptors ."""
        if descriptor is None:
            descriptor = self.descriptor
        if type(descriptor) is not list:
            descriptor = [descriptor]
        if porous:
            descriptor = descriptor[0:1]
        if type(gs_data) is not list:
            gs_data = [gs_data] * len(descriptor)
        if type(ar_data) is not list:
            ar_data = [ar_data] * len(descriptor)

        for i, des in enumerate(descriptor):
            plot_init_stats(des, gs_data=gs_data[i], ar_data=ar_data[i],
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
                        pos=pos, fname=fname, dual_phase=dual_phase,
                        save_plot=save_files)

    """
    --------        Output/Export methods        --------
    """

    def output_abq(self, nodes=None, name=None,
                   elmtDict=None, elmtSetDict=None, faces=None,
                   dual_phase=False, thermal=False, units=None):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""
        if nodes is None:
            if self.nodes_s is not None and self.grain_facesDict is not None:
                logging.warning('\nWarning: No information about nodes is given, will write smoothened structure')
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            elif self.nodes_v is not None:
                logging.warning('\nWarning: No information about nodes is given, will write voxelized structure')
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
        if units is None:
            units = self.rve['units']
        elif (not units == 'mm') and (not units == 'um'):
            raise ValueError(f'Units must be either "mm" or "um", not {units}.')
        if dual_phase:
            nct = '2phases'
            elmt_list = self.mesh.phases.reshape(self.mesh.nvox, order='F')
            ind1 = np.nonzero(elmt_list == 0)[0]
            ind2 = np.nonzero(elmt_list == 1)[0]
            elmtSetDict = {
                1: ind1 + 1,
                2: ind2 + 1}
        else:
            nct = '{0}grains'.format(len(elmtSetDict))
            if elmtSetDict is None:
                elmtSetDict = self.grain_dict
        if name is None:
            cwd = os.getcwd()
            name = os.path.normpath(cwd + f'/kanapy_{nct}_{ntag}.inp')
            if os.path.exists(name):
                os.remove(name)  # remove old file if it exists
        export2abaqus(nodes, name, elmtSetDict, elmtDict,
                      units=units, grain_facesDict=faces,
                      dual_phase=dual_phase, thermal=thermal)
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
            sizeX = self.rve['size'][0]
            sizeY = self.rve['size'][1]
            (sx, sy, sz) = np.divide(self.rve['size'], self.rve['dim'])
            ix = np.arange(self.rve['dim'][0])
            iy = np.arange(self.rve['dim'][0])
            if pos is None or pos == 'top' or pos == 'right':
                iz = self.rve['dim'][2] - 1
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
                                 \nRun voxelize and generate_grains first')
            else:
                data = 'voxels'
        elif data != 'voxels' and data != 'poly':
            raise ValueError('"data" must be either "voxels" or "poly".')

        if data == 'voxels':
            title += ' (Voxels)'
            if cut == 'xy':
                g_slice = np.array(self.mesh.grains[:, :, iz], dtype=int)
            elif cut == 'xz':
                g_slice = np.array(self.mesh.grains[:, iz, :], dtype=int)
            else:
                g_slice = np.array(self.mesh.grains[iz, :, :], dtype=int)
            if dual_phase == True:
                if cut == 'xy':
                    g_slice_phase = np.array(self.mesh.phases[:, :, iz], dtype=int)
                elif cut == 'xz':
                    g_slice_phase = np.array(self.mesh.phases[:, iz, :], dtype=int)
                else:
                    g_slice_phase = np.array(self.mesh.phases[iz, :, :], dtype=int)
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
            for igr in self.geometry['Grains'].keys():
                pts = self.geometry['Grains'][igr]['Points']
                try:
                    tri = Delaunay(pts)
                    i = tri.find_simplex(mesh_slice)
                    ind = np.nonzero(i >= 0)[0]
                    grain_slice[ind] = igr
                except:
                    logging.error('Grain #{} has no convex hull (Nvertices: {})' \
                                  .format(igr, len(pts)))
            if np.any(grain_slice == 0):
                ind = np.nonzero(grain_slice == 0)[0]
                logging.error('Incomplete slicing for {} pixels in {} slice at {}.' \
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
        files in the format
        '
        solid name
          facet normal n1 n2 n3
            outer loop
              vertex p1x p1y p1z
              vertex p2x p2y p2z
              vertex p3x p3y p3z
            endloop
          endfacet
        endsolid name
        '

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
            f.write("solid {}\n".format(self.name))
            for ft in self.geometry['Facets']:
                pts = self.geometry['Points'][ft]
                nv = np.cross(pts[1] - pts[0], pts[2] - pts[0])  # facet normal
                if np.linalg.norm(nv) < 1.e-5:
                    logging.warning(f'Acute facet detected. Facet: {ft}')
                    nv = np.cross(pts[1] - pts[0], pts[2] - pts[1])
                    if np.linalg.norm(nv) < 1.e-5:
                        logging.warning(f'Irregular facet detected. Facet: {ft}')
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
            grains = self.geometry['Grains']
        with open(file, 'w') as f:
            for gr in grains.values():
                # if polyhedral grain has no simplices, center should not be written!!!
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

    def write_voxels(self, angles=None, script_name=None, file=None, path='./', mesh=True,
                     source=None, system=True):
        """
        Write voxel structure into JSON file.

        Parameters
        ----------
        angles
        script_name
        file
        path
        mesh
        source
        system

        Returns
        -------

        """

        import platform
        import getpass
        from datetime import date
        from pkg_resources import get_distribution
        from json import dump

        if script_name is None:
            script_name = __file__
        if path[-1] != '/':
            path += '/'
        if file is None:
            if self.name == 'Microstructure':
                file = path + 'px_{}grains_voxels.json'.format(self.Ngr)
            else:
                file = path + self.name + '_voxels.json'
        file = os.path.normpath(file)
        print(f'Writing voxel information of microstructure to {file}.')
        # metadata
        today = str(date.today())  # date
        owner = getpass.getuser()  # username
        sys_info = platform.uname()  # system information
        # output dict
        structure = {
            "Info": {
                "Owner": owner,
                "Institution": "ICAMS, Ruhr University Bochum, Germany",
                "Date": today,
                "Description": "Voxels of microstructure",
                "Method": "Synthetic microstructure generator Kanapy",
            },
            "Model": {
                "Creator": "kanapy",
                "Version": get_distribution('kanapy').version,
                "Repository": "https://github.com/ICAMS/Kanapy.git",
                "Input": source,
                "Script": script_name,
                "Material": self.name,
                "Phase_names": self.rve.phase_names,
                "Size": self.rve.size,
                "Periodicity": str(self.rve.periodic),
                "Units": {
                    'Length': self.rve.units,
                },
            },
            "Data": {
                "Description": 'Grain numbers per voxel',
                "Type": 'int',
                "Shape": self.rve.dim,
                "Order": 'C',
                "Values": [int(val) for val in self.mesh.grains.flatten()],
            },
            "Grains": {
                "Description": "Grain-related data",
                "Orientation": "Euler-Bunge angle",
                "Phase": "Phase number"
            },
        }
        if system:
            structure["Info"]["System"] = {
                "sysname": sys_info[0],
                "nodename": sys_info[1],
                "release": sys_info[2],
                "version": sys_info[3],
                "machine": sys_info[4],
            }
        for igr in self.mesh.grain_dict.keys():
            structure["Grains"][igr] = {
                "Phase": self.mesh.grain_phase_dict[igr]
            }
        if angles is None:
            if self.mesh.grain_ori_dict is None:
                logging.info('No angles for grains are given. Writing only geometry of RVE.')
            else:
                for igr in self.mesh.grain_dict.keys():
                    structure["Grains"][igr]["Orientation"] = list(self.mesh.grain_ori_dict[igr])
        else:
            for i, igr in enumerate(self.mesh.grain_dict.keys()):
                structure["Grains"][igr]["Orientation"] = list(angles[i, :])
        if mesh:
            structure['Mesh'] = {
                "Nodes": {
                    "Description": 'Nodal coordinates',
                    "Type": 'float',
                    "Shape": self.mesh.nodes.shape,
                    "Values": [list(val) for val in self.mesh.nodes],
                },
                "Voxels": {
                    "Description": 'Node list per voxel',
                    "Type": 'int',
                    "Shape": (len(self.mesh.voxel_dict.keys()), 8),
                    "Values": [val for val in self.mesh.voxel_dict.values()],
                }
            }

        # write file
        with open(file, 'w') as fp:
            dump(structure, fp, indent=2)
        return

    def pckl(self, file=None, path='./'):
        """Write microstructure into pickle file. Usefull for to store complex structures.


        Parameters
        ----------
        file : string (optional, default: None)
            File name for pickled microstructure. The default is None, in which case
            the filename will be the microstructure name + '.pckl'.
        path : string
            Path to location for pickles

        Returns
        -------
        None.

        """
        import pickle

        if path[-1] != '/':
            path += '/'
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_microstructure.pckl'.format(self.Ngr)
            else:
                file = self.name + '_microstructure.pckl'
        file = os.path.normpath(path + file)
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return

    """
    --------        legacy methods        --------
    """

    def init_stats(self, descriptor=None, gs_data=None, ar_data=None, porous=False, save_files=False):
        """ Legacy function for plot_stats_init."""
        logging.warning('This legacy function is depracted, please use "plot_stats_init()".')
        self.plot_stats_init(descriptor, gs_data=gs_data, ar_data=ar_data, porous=porous, save_files=save_files)
