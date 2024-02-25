""" Module defining class Microstructure that contains the necessary
methods and attributes to analyze experimental microstructures in form
of EBSD maps to generate statistical descriptors for 3D microstructures, and 
to create synthetic RVE that fulfill the required statistical microstructure
descriptors.

The methods of the class Microstructure for an API that can be used to generate
Python workflows.

Authors: Alexander Hartmaier, Golsa Tolooei Eshlghi, Abhishek Biswas
Institution: ICAMS, Ruhr University Bochum
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from kanapy.grains import calc_polygons, get_stats
from kanapy.entities import Simulation_Box
from kanapy.input_output import export2abaqus, writeAbaqusMat, read_dump
from kanapy.initializations import RVE_creator, mesh_creator
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_init_stats, plot_voxels_3D, plot_ellipsoids_3D, \
    plot_polygons_3D, plot_output_stats, plot_particles_3D


class Microstructure(object):
    """Define class for synthetic microstructures

    Attributes:
        name : str
            Name of microstructure
        nphases : int
            Number of phases in microstructure
        ngrains : ndarray
            Array of grain number in each phase
        Ngr : int
            Total number of grains summed over all phases
        nparticles : list
            List with number of particles in each phase
        descriptor : list
            List of dictionaries describing the microstructure of each phase;
            Dict Keys: "Grains type", "Equivalent diameter", "Aspect ratio", "Tilt Angle", "RVE", "Simulation"
        precipit : None or float
            Indicates microstructure with precipitates/pores/particles in continuous matrix. If type is float,
             it gives volume the fraction of that precipitate phase
        from_voxels : bool
            Indicates whether microstructure object is imported from voxel file, not generated from particle simulation
        particles : list
            List of particle objects of class entities containing information object particle geometries
        rve : object of class RVE_creator
            Contains information about the RVE
            Attributes: dim, size, nparticles, periodic, units, packing_steps, particle_data,
            phase_names, phase_vf, ialloy
        simbox : Object of class Simulation_Box
            Contains information about geometry of simulation box for particle simulation
        mesh : object of class mesh_creator
            Attributes: dim, grain_dict, grain_ori_dict, grain_phase_dict, grains, ngrains_phase. nodes, nodes_smooth,
                nvox, phases, prec_vf_voxels, vox_center_dict, voxel_dict
        geometry : dict
            Dictionary of grain geometries;
            Dict keys: "Vertices", "Points", "Simplices", "Facets", "Grains", "GBnodes", GBarea" "GBfaces"
            "Grains" : dictionary with key grain_number  
            Keys:"Vertices", "Points", "Center", "Simplices", "Volume", "Area", "Phase", "eqDia", "majDia", "minDia"
        res_data : list
            List of dictionaries containing effective microstructure descriptors for each phase in RVE
            Dict keys: "Number", "Unit scale", "Grain_type",
            "Grain_Equivalent_diameter", "Grain_Major_diameter", "Grain_Minor_diameter",
            "Particle_Equivalent_diameter", "Particle_Major_diameter", "Particle_Minor_diameter",
            "L1-error"
    """

    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.nphases = None
        self.ngrains = None
        self.nparticles = None
        self.precipit = None
        self.rve = None
        self.particles = None
        self.geometry = None
        self.simbox = None
        self.mesh = None
        self.res_data = None
        self.from_voxels = False
        self.ialloy = None

        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an data file name')

            # Open the user data statistics file and read the data
            try:
                with open(os.path.normpath(file)) as json_file:
                    self.descriptor = json.load(json_file)
            except Exception as e:
                logging.error(f'An unexpected exception occurred: {e}')
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
                    logging.warning(f'Kanapy is only tested for 2 phases, use at own risk for {self.nphases} phases')
            if file is not None:
                logging.warning(
                    'WARNING: Input parameter (descriptor) and file are given. Only descriptor will be used.')
        if self.nphases == 1 and 'Phase' in self.descriptor[0].keys():
            vf = self.descriptor[0]['Phase']['Volume fraction']
            if vf < 1.0:
                # consider precipitates/pores/particles with volume fraction fill_factor in a matrix
                # precipitates will be phase 0, matrix phase will get number 1 and be assigned to grain with ID 0
                self.precipit = vf
                self.nphases = 2
                logging.info(f'Only one phase with volume fraction {vf} is given.')
                logging.info('Will consider a sparse distribution in a matrix phase with phase number 1, ' +
                             'which will be assigned to grain with ID 0.')
        return

    """
    --------        Routines for user interface        --------
    """

    def init_RVE(self, descriptor=None, nsteps=1000):
        """
        Creates particle distribution inside simulation box (RVE) based on
        the data provided in the data file.

        Parameters
        ----------
        descriptor
        nsteps

        Returns
        -------

        """
        if descriptor is None:
            descriptor = self.descriptor
        if type(descriptor) is not list:
            descriptor = [descriptor]

        # initialize RVE, including mesh dimensions and particle distribution
        self.rve = RVE_creator(descriptor, nsteps=nsteps)
        if self.precipit is not None:
            self.rve.phase_names.append('Matrix')
            self.rve.phase_vf.append(1.0 - self.precipit)
        self.nparticles = self.rve.nparticles
        # store geometry in simbox object
        self.simbox = Simulation_Box(self.rve.size)

    def pack(self, particle_data=None,
             k_rep=0.0, k_att=0.0, fill_factor=None,
             poly=None, save_files=False):

        """ Packs the particles into a simulation box."""
        if particle_data is None:
            particle_data = self.rve.particle_data
            if particle_data is None:
                raise ValueError('No particle_data in pack. Run create_RVE first.')
        if fill_factor is None and self.precipit is not None:
            fill_factor = 1.0  # pack to full volume fraction defined in particles
            print(f'Sparse particles (precipitates/pores): '
                  f'Packing up to particle volume fraction of {(100*self.precipit):.1f}%.')
            if self.precipit > 0.65:
                print('Overlap of particles will occur since volume fraction > 65%')
        self.particles, self.simbox = \
            packingRoutine(particle_data, self.rve.periodic,
                           self.rve.packing_steps, self.simbox,
                           k_rep=k_rep, k_att=k_att, fill_factor=fill_factor,
                           poly=poly, save_files=save_files)

    def voxelize(self, particles=None, dim=None):
        """ Generates the RVE by assigning voxels to grains."""
        if particles is None:
            particles = self.particles
            if particles is None:
                raise ValueError('No particles in voxelize. Run pack first.')
        if dim is None:
            dim = self.rve.dim
        else:
            if len(dim) != 3 or type(dim) is not tuple:
                raise ValueError(f'"dim" must be a 3-tuple of the voxel numbers in each direction, not {dim}.')
            self.rve.dim = dim

        # initialize voxel structure (= mesh)
        self.mesh = mesh_creator(dim)
        self.mesh.nphases = self.nphases
        self.mesh.create_voxels(self.simbox)

        self.mesh = \
            voxelizationRoutine(particles, self.mesh, self.nphases, prec_vf=self.precipit)
        if np.any(self.nparticles != self.mesh.ngrains_phase):
            logging.info(f'Number of grains per phase changed from {self.nparticles} to ' +
                         f'{list(self.mesh.ngrains_phase)} during voxelization.')
        self.ngrains = self.mesh.ngrains_phase
        self.Ngr = np.sum(self.mesh.ngrains_phase, dtype=int)
        # extract volume fractions from voxelized grains
        if self.nphases > 1:
            vox_count = np.zeros(self.nphases, dtype=int)
            for igr, ip in self.mesh.grain_phase_dict.items():
                vox_count[ip] += len(self.mesh.grain_dict[igr])
            print('Volume fractions of phases in voxel structure:')
            vt = 0.
            for ip in range(self.nphases):
                vf = 100.0*vox_count[ip]/self.mesh.nvox
                vt += vf
                print(f'{ip}: {self.rve.phase_names[ip]} ({vf.round(1)}%)')
            if not np.isclose(vt, 1.0):
                logging.warning(f'Volume fractions of phases in voxels do not add up to 1. Value: {vt}')

        # remove grain information if it already exists to avoid inconsistencies
        if self.geometry is not None:
            logging.info('Removing polyhedral grain geometries and statistical data after re-meshing.')
            self.geometry = None
            self.res_data = None

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

        if self.mesh is None or self.mesh.grains is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if self.precipit and 0 in self.mesh.grain_dict.keys():
            # in case of precipit, remove irregular grain 0 from analysis
            empty_vox = self.mesh.grain_dict.pop(0)
            grain_store = self.mesh.grain_phase_dict.pop(0)
            nphases = self.nphases - 1
        else:
            empty_vox = None
            grain_store = None
            nphases = self.nphases

        self.geometry: dict = \
            calc_polygons(self.rve, self.mesh)  # updates RVE_data
        # verify that geometry['Grains'] and mesh.grain_dict are consistent
        """if np.any(self.geometry['Ngrains'] != self.ngrains):
            logging.warning(f'Only facets for {self.geometry["Ngrains"]} created, but {self.Ngr} grains in voxels.')
            for igr in self.mesh.grain_dict.keys():
                if igr not in self.geometry['Grains'].keys():
                    logging.warning(f'Grain: {igr} not in geometry. Be aware when creating GB textures.')"""
        # verify that geometry['GBarea'] is consistent with geometry['Grains']
        gba = self.geometry['GBarea']
        ind = []
        igr = []
        for i, gblist in enumerate(gba):
            if not gblist[0] in self.geometry['Grains'].keys():
                ind.append(i)
                igr.append(gblist[0])
                continue
            if not gblist[1] in self.geometry['Grains'].keys():
                ind.append(i)
                igr.append(gblist[1])
        if len(ind) > 0:
            logging.warning(f'{len(ind)} grains are not represented in polyhedral geometry.')
            #logging.warning('Consider increasing the number of voxels, as grains appear to be very irregular.')
            """ind.reverse()
            igr.reverse()
            for j, i in enumerate(ind):
                logging.warning(f'Removing {gba[i]} from GBarea as grain {igr[j]} does not exist.')
                gba.pop(i)
            self.geometry['GBarea'] = gba"""
        # extract volume fractions from polyhedral grains
        if self.nphases > 1:
            ph_vol = np.zeros(self.nphases)
            for igr, grd in self.geometry['Grains'].items():
                ip = grd['Phase']
                ph_vol[ip] += grd['Volume']
            print('Volume fractions of phases in polyhedral geometry:')
            for ip in range(self.nphases):
                vf = 100.0*ph_vol[ip]/np.prod(self.rve.size)
                print(f'{ip}: {self.rve.phase_names[ip]} ({vf.round(1)}%)')
        # analyse grains w.r.t. statistical descriptors
        self.res_data = \
            get_stats(self.rve.particle_data, self.geometry, self.rve.units,
                      nphases, self.mesh.ngrains_phase)
        if empty_vox is not None:
            # add removed grain again
            self.mesh.grain_dict[0] = empty_vox
            self.mesh.grain_phase_dict[0] = grain_store

    def generate_orientations(self, data, ang=None, omega=None, Nbase=5000,
                              hist=None, shared_area=None):
        """
        Calculates the orientations of grains to give a desired crystallographic texture.

        Parameters
        ----------
        data
        ang
        omega
        Nbase
        hist
        shared_area

        Returns
        -------

        """
        from kanapy import MTEX_AVAIL
        try:
            from kanapy.textures import EBSDmap, createOrisetRandom, createOriset
        except Exception as e:
            logging.error(f'An unexpected exception occurred: {e}')
            raise ImportError('Matlab or MTEX might not be installed properly. ' +
                              'Run "kanapy setupTexture" first to use this function.')

        if not MTEX_AVAIL:
            raise ModuleNotFoundError('MTEX not installed. Run "kanapy setupTexture" first to use this function.')
        if self.mesh.grains is None:
            raise ValueError('Grain geometry is not defined. Run voxelize first.')
        if shared_area is None:
            if hist is None:
                gba = None
            else:
                if self.geometry is None:
                    raise ValueError('If histogram for GB texture is provided, GB areas must be defined.\n' +
                                     'Run generate_grains() first, to calculate GB areas.')
                gba = self.geometry['GBarea']
        else:
            if shared_area == 0:
                gba = None
            else:
                gba = shared_area

        ori_dict = dict()
        for ip, ngr in enumerate(self.ngrains):
            if type(data) is EBSDmap:
                ori_rve = data.calcORI(ngr, iphase=ip, shared_area=gba)
            elif type(data) is str:
                if data.lower() in ['random', 'rnd']:
                    ori_rve = createOrisetRandom(ngr, Nbase=Nbase, hist=hist, shared_area=gba)
                elif data.lower() in ['unimodal', 'uni_mod', 'uni_modal']:
                    if ang is None or omega is None:
                        raise ValueError('To generate orientation sets of type "unimodal" angle "ang" and kernel' +
                                         'halfwidth "omega" are required.')
                    ori_rve = createOriset(ngr, ang, omega, hist=hist, shared_area=gba)
            else:
                raise ValueError('Argument to generate grain orientation must be either of type EBSDmap or ' +
                                 '"random" or "unimodal"')
            for i, igr in enumerate(self.mesh.grain_dict.keys()):
                if self.mesh.grain_phase_dict[igr] == ip:
                    ind = i - ip*self.ngrains[0]
                    ori_dict[igr] = ori_rve[ind, :]
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

    def plot_particles(self, cmap='prism', dual_phase=False, plot_hull=True):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        if self.particles[0].inner is None:
            raise ValueError('Ellipsoids without inner polygon cannot be plotted.')
        plot_particles_3D(self.particles, cmap=cmap,
                          dual_phase=dual_phase, plot_hull=plot_hull)

    def plot_voxels(self, sliced=True, dual_phase=False, cmap='prism',
                    ori=None):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.mesh.grains is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        elif dual_phase:
            data = self.mesh.phases
        else:
            data = self.mesh.grains
        if ori is not None:
            try:
                from kanapy.textures import get_ipf_colors
                if type(ori) is bool and ori:
                    ori = np.array([val for val in self.mesh.grain_ori_dict.values()])
                """Create possibility to pass actual CS to MTEX"""
                clist = get_ipf_colors(ori)
            except Exception as e:
                logging.error(f'An unexpected exception occurred: {e}')
                logging.error('Orientations can only be plotted if MTEX is available.')
                clist = None
        else:
            clist = None

        plot_voxels_3D(data, Ngr=np.sum(self.ngrains), sliced=sliced,
                       dual_phase=dual_phase, cmap=cmap, clist=clist)

    def plot_grains(self, geometry=None, cmap='prism', alpha=0.4,
                    ec=[0.5, 0.5, 0.5, 0.1], dual_phase=False):
        """ Plot polygonalized microstructure"""
        if geometry is None:
            geometry = self.geometry
        if geometry is None:
            raise ValueError('No polygons for grains defined. Run generate_grains() first')
        plot_polygons_3D(geometry, cmap=cmap, alpha=alpha, ec=ec,
                         dual_phase=dual_phase)

    def plot_stats(self, data=None,
                   gs_data=None, gs_param=None,
                   ar_data=None, ar_param=None,
                   particles=True,
                   save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical 
        comparison."""
        if data is None:
            data = self.res_data
            if data is None:
                raise ValueError('No microstructure data created yet. Run generate_grains() first.')
        elif type(data) is not list:
            data = [data]
        if gs_data is None or type(gs_data) is not list:
            gs_data = [gs_data] * len(data)
        if gs_param is None or type(gs_param) is not list:
            gs_param = [gs_param] * len(data)
        if ar_data is None or type(ar_data) is not list:
            ar_data = [ar_data] * len(data)
        if ar_param is None or type(ar_param) is not list:
            ar_param = [ar_param] * len(data)
        """
        gs_data = [ebsd.ms_data[i]['gs_data'] for i in range(Nphases)]
        gs_param = [ebsd.ms_data[i]['gs_param'] for i in range(Nphases)]
        ar_data = [ebsd.ms_data[i]['ar_data'] for i in range(Nphases)]
        ar_param = [ebsd.ms_data[i]['ar_param'] for i in range(Nphases)]
        """

        for i, ds in enumerate(data):
            print(f'Plotting input & output statistics for phase {i}')
            plot_output_stats(ds, gs_data=gs_data[i], gs_param=gs_param[i],
                              ar_data=ar_data[i], ar_param=ar_param[i],
                              plot_particles=particles,
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
    --------        Import/Export methods        --------
    """

    def write_abq(self, nodes=None, file=None, path='./', voxel_dict=None, grain_dict=None,
                  dual_phase=False, thermal=False, units=None,
                  ialloy=None, nsdv=200):
        """
        Writes out the Abaqus deck (.inp file) for the generated RVE. The parameter nodes should be
        a string indicating if voxel ("v") or smoothened ("s") mesh should be written. It can also
        provide an array of nodal positions. If dual_phase is true, the
        generated deck contains plain material definitions for each phase. Material parameters must
        be specified by the user. If ialloy is provided, the generated deck material definitions
        for each grain. For dual phase structures to be used with crystal plasticity, ialloy
        can be a list with all required material definitions. If the list ialloy is shorted than the
        number of phases in the RVE, plain material definitions for the remaining phases will
        be included in the input deck.

        Parameters
        ----------
        nodes
        file
        path
        voxel_dict
        grain_dict
        dual_phase
        thermal
        units
        ialloy
        nsdv

        Returns
        -------

        """
        if nodes is None:
            if self.mesh.nodes_smooth is not None and 'GBarea' in self.geometry.keys():
                logging.warning('\nWarning: No argument "nodes" is given, will write smoothened structure')
                nodes = self.mesh.nodes_smooth
                faces = self.geometry['GBarea']
                ntag = '_smooth'
            elif self.mesh.nodes is not None:
                logging.warning('\nWarning: No argument "nodes" is given, will write voxelized structure')
                nodes = self.mesh.nodes
                faces = None
                ntag = '_voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        elif type(nodes) is not str:
            faces = None
            ntag = '_voxels'
        elif nodes.lower() in ['smooth', 's']:
            if self.mesh.nodes_smooth is not None and 'GBarea' in self.geometry.keys():
                nodes = self.mesh.nodes_smooth
                faces = self.geometry['GBarea']  # use tet elements for smoothened structure
                ntag = '_smooth'
            else:
                raise ValueError('No information about smoothed microstructure. Run smoothen first.')
        elif nodes.lower() in ['voxels', 'v', 'voxel']:
            if self.mesh.nodes is not None:
                nodes = self.mesh.nodes
                faces = None  # use brick elements for voxel structure
                ntag = '_voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        else:
            raise ValueError('Wrong value for parameter "nodes". Must be either "smooth" ' +
                             f'or "voxels", not {nodes}')
        if voxel_dict is None:
            voxel_dict = self.mesh.voxel_dict
        if units is None:
            units = self.rve.units
        elif (not units == 'mm') and (not units == 'um'):
            raise ValueError(f'Units must be either "mm" or "um", not {units}.')
        if dual_phase:
            nct = 'abq_dual_phase'
            if grain_dict is None:
                grain_dict = dict()
                for i in range(self.nphases):
                    grain_dict[i] = list()
                for igr, ip in self.mesh.grain_phase_dict.items():
                    grain_dict[ip] = np.concatenate(
                        [grain_dict[ip], self.mesh.grain_dict[igr]])
        else:
            if grain_dict is None:
                grain_dict = self.mesh.grain_dict
            nct = f'abq_px_{len(grain_dict)}'
        if ialloy is None:
            ialloy = self.rve.ialloy
        if type(ialloy) is list and len(ialloy) > self.nphases:
            raise ValueError('List of values in ialloy is larger than number of phases in RVE.' +
                             f'({len(ialloy)} > {self.nphases})')
        if self.nphases > 1:
            grpd = self.mesh.grain_phase_dict
        else:
            grpd = None
        if file is None:
            if self.name == 'Microstructure':
                file = nct + ntag + '_geom.inp'
            else:
                file = self.name + ntag + '_geom.inp'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        export2abaqus(nodes, file, grain_dict, voxel_dict,
                      units=units, gb_area=faces,
                      dual_phase=dual_phase,
                      ialloy=ialloy, grain_phase_dict=grpd,
                      thermal=thermal)
        # if orientation exists and ialloy is defined also write material file with Euler angles
        if not (self.mesh.grain_ori_dict is None or ialloy is None):
            writeAbaqusMat(ialloy, self.mesh.grain_ori_dict,
                           file=file[0:-8] + 'mat.inp',
                           grain_phase_dict=grpd, nsdv=nsdv)
        return file

    def write_abq_ori(self, ialloy=None, ori=None, file=None, path='./', nsdv=200):
        if ialloy is None:
            ialloy = self.rve.ialloy
            if ialloy is None:
                raise ValueError('Value of material number in ICAMS CP-UMAT (ialloy) not defined.')
        if ori is None:
            ori = self.mesh.grain_ori_dict
            if ori is None:
                raise ValueError('No orientations present. Run "generate_oriantations" first.')
        if file is None:
            if self.name == 'Microstructure':
                file = f'abq_px_{self.Ngr}_mat.inp'
            else:
                file = self.name + '_mat.inp'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        writeAbaqusMat(ialloy, ori, file=file, nsdv=nsdv)

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
        ori : (self.Ngr, )-array, optional
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
        if type(ori) is dict:
            ori = np.array([val for val in ori.values()])
        cut = cut.lower()
        if type(pos) is str:
            pos = pos.lower()
        botlist = ['bottom', 'bot', 'left', 'b', 'l']
        toplist = ['top', 'right', 't', 'r']
        if cut == 'xy':
            sizeX = self.rve.size[0]
            sizeY = self.rve.size[1]
            (sx, sy, sz) = np.divide(self.rve.size, self.rve.dim)
            ix = np.arange(self.rve.dim[0])
            iy = np.arange(self.rve.dim[1])
            if pos is None or pos in toplist:
                iz = self.rve.dim[2] - 1
            elif pos in botlist:
                iz = 0
            elif type(pos) is float or type(pos) is int:
                iz = int(pos / sz)
            else:
                raise ValueError('"pos" must be either float or "top", "bottom", "left" or "right"')
            if pos is None:
                pos = int(iz * sz)
            xl = r'x ($\mu$m)'
            yl = r'y ($\mu$m)'
            title = r'XY slice at z={} $\mu$m'.format(round(iz * sz, 1))
        elif cut == 'xz':
            sizeX = self.rve.size[0]
            sizeY = self.rve.size[2]
            vox_res = np.divide(self.rve.size, self.rve.dim)
            sx = vox_res[0]
            sy = vox_res[1]
            sz = vox_res[2]
            ix = np.arange(self.rve.dim[0])
            iy = np.arange(self.rve.dim[2])
            if pos is None or pos in toplist:
                iz = self.rve.dim[1] - 1
            elif pos in botlist:
                iz = 0
            elif type(pos) is float or type(pos) is int:
                iz = int(pos / sy)
            else:
                raise ValueError('"pos" must be either float or "top", "bottom", "left" or "right"')
            if pos is None:
                pos = int(iz * sz)
            xl = r'x ($\mu$m)'
            yl = r'z ($\mu$m)'
            title = r'XZ slice at y={} $\mu$m'.format(round(iz * sz, 1))
        elif cut == 'yz':
            sizeX = self.rve.size[1]
            sizeY = self.rve.size[2]
            vox_res = np.divide(self.rve.size, self.rve.dim)
            sx = vox_res[0]
            sy = vox_res[1]
            sz = vox_res[2]
            ix = np.arange(self.rve.dim[1])
            iy = np.arange(self.rve.dim[2])
            if pos is None or pos in toplist:
                iz = self.rve.dim[0] - 1
            elif pos in botlist:
                iz = 0
            elif type(pos) is float or type(pos) is int:
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
            if 'Grains' in self.geometry.keys():
                data = 'poly'
            elif self.mesh.voxels is None:
                raise ValueError('Neither polygons nor voxels for grains are present.\
                                 \nRun voxelize and generate_grains first.')
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
            if dual_phase:
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
                except Exception as e:
                    logging.error(f'An unexpected exception occurred: {e}')
                    logging.error('Grain #{} has no convex hull (Nvertices: {})'
                                  .format(igr, len(pts)))
            if np.any(grain_slice == 0):
                ind = np.nonzero(grain_slice == 0)[0]
                logging.error('Incomplete slicing for {} pixels in {} slice at {}.'
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
                        f.write('  {0}  {1}  {2}  {3}  {4}   0.0  0.000  0   1  0.000\n'
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

            if dual_phase:
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

    def write_stl(self, data='grains', file=None, path='./'):
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
        def write_facet(nv, pts, ft):
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
        def write_grains():
            for ft in self.geometry['Facets']:
                pts = self.geometry['Points'][ft]
                nv = np.cross(pts[1] - pts[0], pts[2] - pts[0])  # facet normal
                write_facet(nv, pts, ft)

        def write_particles():
            for pa in self.particles:
                for ft in pa.inner.convex_hull:
                    pts = pa.inner.points[ft]
                    nv = np.cross(pts[1] - pts[0], pts[2] - pts[0])  # facet normal
                    write_facet(nv, pts, ft)

        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains.stl'.format(self.Ngr)
            else:
                file = self.name + '.stl'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        with open(file, 'w') as f:
            f.write("solid {}\n".format(self.name))
            if data in ['particles', 'pa', 'p']:
                if self.particles[0].inner is None:
                    logging.error("Particles don't contain inner polyhedron, cannot write STL file.")
                else:
                    for pa in self.particles:
                        pa.sync_poly()
                    write_particles()
            else:
                write_grains()
            f.write("endsolid\n")
        return

    def write_centers(self, file=None, path='./', grains=None):
        """Write grain center positions into CSV file."""
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_centroid.csv'.format(self.Ngr)
            else:
                file = self.name + '_centroid.csv'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        if grains is None:
            grains = self.geometry['Grains']
        with open(file, 'w') as f:
            for gr in grains.values():
                # if polyhedral grain has no simplices, center should not be written!!!
                ctr = gr['Center']
                f.write('{}, {}, {}\n'.format(ctr[0], ctr[1], ctr[2]))
        return

    def write_ori(self, angles=None, file=None, path='./'):
        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_ori.csv'.format(self.Ngr)
            else:
                file = self.name + '_ori.csv'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        if angles is None:
            if self.mesh.grain_ori_dict is None:
                raise ValueError('No grain orientations given or stored.')
            angles = [val for val in self.mesh.grain_ori_dict.values()]

        with open(file, 'w') as f:
            for ori in angles:
                f.write('{}, {}, {}\n'.format(ori[0], ori[1], ori[2]))
        return

    def write_voxels(self, angles=None, script_name=None, file=None, path='./',
                     mesh=True, source=None, system=False):
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

        if script_name is None:
            script_name = __file__
        if file is None:
            if self.name == 'Microstructure':
                file = f'px_{self.Ngr}grains_voxels.json'
            else:
                file = self.name + '_voxels.json'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
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
                for igr in self.mesh.grain_ori_dict.keys():
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
            json.dump(structure, fp)
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

        if file is None:
            if self.name == 'Microstructure':
                file = 'px_{}grains_microstructure.pckl'.format(self.Ngr)
            else:
                file = self.name + '_microstructure.pckl'
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return

    def import_particles(self, file, path='./'):
        path = os.path.normpath(path)
        file = os.path.join(path, file)
        self.simbox, self.particles = read_dump(file)


    """
    --------        legacy methods        --------
    """

    def init_stats(self, descriptor=None, gs_data=None, ar_data=None, porous=False, save_files=False):
        """ Legacy function for plot_stats_init."""
        logging.warning('"init_stats" is a legacy function and will be depracted, please use "plot_stats_init()".')
        self.plot_stats_init(descriptor, gs_data=gs_data, ar_data=ar_data, save_files=save_files)

    def output_abq(self, nodes=None, name=None,
                   voxel_dict=None, grain_dict=None, faces=None,
                   dual_phase=False, thermal=False, units=None):
        """ Legacy function for write_abq."""
        logging.warning('"output_abq" is a legacy function and will be depracted, please use "write_abq()".')
        if faces is not None:
            logging.warning('Parameter "faces" will be determined automatically.')
        self.write_abq(nodes=nodes, file=name, voxel_dict=voxel_dict, grain_dict=grain_dict,
                       dual_phase=dual_phase, thermal=thermal, units=units)
