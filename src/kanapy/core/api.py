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
import sys
import json
import logging
import platform
import hashlib
import orix

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Dict, Any, List, Optional, Union
from importlib.metadata import version as pkg_version
from datetime import datetime

from .grains import calc_polygons
from .entities import Simulation_Box
from .input_output import export2abaqus, writeAbaqusMat, read_dump
from .initializations import RVE_creator, mesh_creator
from .packing import packingRoutine
from .voxelization import voxelizationRoutine
from .smoothingGB import smoothingRoutine
from .rve_stats import get_stats_vox, get_stats_part, get_stats_poly
from .plotting import plot_init_stats, plot_voxels_3D, plot_ellipsoids_3D, \
    plot_polygons_3D, plot_output_stats, plot_particles_3D


class Microstructure(object):
    """
    Define a class for creating and managing synthetic microstructures

    This class provides tools to define, generate, and analyze synthetic
    microstructures composed of one or multiple phases. It integrates particle
    packing, voxelization, and grain geometry generation for RVE (Representative
    Volume Element) modeling.

    Attributes
    ----------
    name : str
        Name of the microstructure.
    nphases : int or None
        Number of phases in the microstructure.
    ngrains : ndarray or None
        Array of grain counts in each phase.
    nparticles : list or None
        List of the number of particles in each phase.
    descriptor : list of dict or None
        List of dictionaries describing the microstructure of each phase.
        Defined when input `descriptor` or `file` is provided.
        Dictionary keys typically include:
        "Grains type", "Equivalent diameter", "Aspect ratio",
        "Tilt Angle", "RVE", and "Simulation".
    precipit : None or float
        Indicates the presence of precipitates, pores, or secondary particles
        in a continuous matrix. If float, specifies their volume fraction.
    from_voxels : bool
        Indicates whether the microstructure object is imported from a voxel file
        rather than generated from a particle simulation.
    particles : list or None
        List of particle objects containing geometric information.
    rve : RVE_creator or None
        Object containing information about the Representative Volume Element (RVE),
        including mesh dimensions, particle count, periodicity, and phase volume fractions.
    simbox : Simulation_Box or None
        Object defining the geometric boundaries of the RVE simulation domain.
    mesh : mesh_creator or None
        Object storing voxelized mesh data including grain assignments,
        voxel connectivity, and smoothed node coordinates.
    geometry : dict or None
        Dictionary of grain geometries. Keys may include:
        "Vertices", "Points", "Simplices", "Facets", "Grains",
        "GBnodes", "GBarea", and "GBfaces".
    rve_stats : list of dict or None
        List containing statistical information for different RVE representations,
        including particles, voxels, and polyhedral grains.
    rve_stats_labels : list of str or None
        Labels corresponding to the types of RVEs analyzed, e.g., "Particles",
        "Voxels", "Grains".
    vf_vox : ndarray or None
        Array containing the phase volume fractions obtained from voxelized structures.

    Notes
    -----
    - The class can initialize from either a statistical descriptor or a JSON data file.
    - When only one phase is specified with a volume fraction < 1.0,
      a matrix phase is automatically added with a complementary fraction (1 - vf).
    - Kanapy is tested for up to two phases; using more may yield unpredictable results.
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
        self.rve_stats = None
        self.rve_stats_labels = None
        self.from_voxels = False
        self.ialloy = None
        self.vf_vox = None

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
        Initialize the Representative Volume Element (RVE) of the microstructure

        This method creates the voxel- or particle-based RVE using the provided
        phase descriptor(s). It sets up the mesh dimensions, particle distribution,
        and simulation box geometry. The RVE object is constructed using the
        `RVE_creator` class, and key attributes such as `nparticles` and `simbox`
        are updated accordingly.

        Parameters
        ----------
        descriptor : list of dict, dict, or None, optional
            Description of the microstructure phases. Each dictionary specifies
            parameters such as grain type, equivalent diameter, aspect ratio,
            and tilt angle. If `None`, the class attribute `self.descriptor` is used.
        nsteps : int, optional
            Number of optimization or relaxation steps for RVE generation.
            Default is 1000.

        Notes
        -----
        - Assumes that `RVE_creator` and `Simulation_Box` are available and correctly configured.
        - If `self.precipit` is defined, an additional 'Matrix' phase is automatically
          added to preserve total volume fraction normalization.

        Attributes Updated
        ------------------
        rve : RVE_creator
            The created RVE object containing voxel/particle information, size,
            periodicity, and phase volume fractions.
        nparticles : list
            Number of particles per phase after RVE generation.
        simbox : Simulation_Box
            Object containing the geometric boundaries of the RVE domain.
        precipit : float or None
            If not None, a 'Matrix' phase is appended to `rve.phase_names` and its
            volume fraction is adjusted as (1.0 - precipit).

        Returns
        -------
        None
            Updates class attributes with the initialized RVE and simulation box.
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
             poly=None, save_files=False, verbose=False):
        """
        Pack particles into the simulation box according to the RVE settings

        Parameters
        ----------
        particle_data : array-like or None, optional
            Particle information. If None, uses self.rve.particle_data. Raises
            ValueError if no particle data is available.
        k_rep : float, optional
            Repulsion coefficient between particles. Default is 0.0.
        k_att : float, optional
            Attraction coefficient between particles. Default is 0.0.
        fill_factor : float or None, optional
            Fraction of simulation box to fill. Defaults to 1.0 if self.precipit is set.
        poly : optional
            Additional packing options for polyhedral particles.
        save_files : bool, optional
            If True, saves packed particle data to files.
        verbose : bool, optional
            If True, prints progress and warnings during packing.

        Returns
        -------
        particles : list
            List of packed particle objects with positions and geometrical information.
        simbox : Simulation_Box
            Updated simulation box reflecting particle packing.

        Notes
        -----
        If self.precipit > 0.65, particle overlaps may occur. Requires self.rve
        and self.simbox to be initialized before calling.
        """
        if particle_data is None:
            particle_data = self.rve.particle_data
            if particle_data is None:
                raise ValueError('No particle_data in pack. Run create_RVE first.')
        if fill_factor is None and self.precipit is not None:
            fill_factor = 1.0  # pack to full volume fraction defined in particles
            print(f'Sparse particles (precipitates/pores): '
                  f'Packing up to particle volume fraction of {(100 * self.precipit):.1f}%.')
            if self.precipit > 0.65:
                print('Overlap of particles will occur since volume fraction > 65%')
        self.particles, self.simbox = \
            packingRoutine(particle_data, self.rve.periodic,
                           self.rve.packing_steps, self.simbox,
                           k_rep=k_rep, k_att=k_att, fill_factor=fill_factor,
                           poly=poly, save_files=save_files, verbose=verbose)

    def voxelize(self, particles=None, dim=None):
        """
        Generate the RVE by assigning voxels to grains

        Parameters
        ----------
        particles : list or None, optional
            List of particle objects to voxelize. If None, uses `self.particles`.
            Raises ValueError if no particles are available.
        dim : tuple of int, optional
            3-tuple specifying the number of voxels in each spatial direction.
            If None, uses `self.rve.dim`. Raises ValueError if not a 3-tuple.

        Notes
        -----
        - Requires `self.particles` to be initialized by `pack`.
        - Updates `self.mesh`, `self.ngrains`, `self.Ngr`, and `self.vf_vox`.
        - Logs warnings if phase volume fractions do not sum to 1.
        - Removes polyhedral grain geometries (`self.geometry`) after re-meshing to avoid inconsistencies.

        Returns
        -------
        None
            The voxelized mesh is stored in `self.mesh`. Phase voxel fractions are
            stored in `self.vf_vox` and printed to the console.

        Raises
        ------
        ValueError
            - If `particles` is None and `self.particles` is not available.
            - If `dim` is not a 3-tuple when specified.
        """
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
            self.vf_vox = np.zeros(self.nphases)
            vox_count = np.zeros(self.nphases, dtype=int)
            for igr, ip in self.mesh.grain_phase_dict.items():
                vox_count[ip] += len(self.mesh.grain_dict[igr])
            print('Volume fractions of phases in voxel structure:')
            vt = 0.
            for ip in range(self.nphases):
                vf_act = vox_count[ip] / self.mesh.nvox
                self.vf_vox[ip] = vf_act
                vt += vf_act
                print(f'{ip}: {self.rve.phase_names[ip]} ({(vf_act * 100):.3f}%)')
            if not np.isclose(vt, 1.0):
                logging.warning(f'Volume fractions of phases in voxels do not add up to 1. Value: {vt}')
        else:
            self.vf_vox = np.ones(1)

        # remove grain information if it already exists to avoid inconsistencies
        if self.geometry is not None:
            logging.info('Removing polyhedral grain geometries and statistical data after re-meshing.')
            self.geometry = None

    def smoothen(self, nodes_v=None, voxel_dict=None, grain_dict=None):
        """
        Generate smoothed grain boundaries from a voxelated mesh

        Parameters
        ----------
        nodes_v : array-like or None, optional
            Mesh node coordinates. If None, uses `self.mesh.nodes`. Raises
            ValueError if no nodes are available.
        voxel_dict : dict or None, optional
            Dictionary of voxels in the mesh. If None, uses `self.mesh.voxel_dict`.
        grain_dict : dict or None, optional
            Dictionary mapping grains to their voxels. If None, uses `self.mesh.grain_dict`.

        Notes
        -----
        - Requires `self.mesh` to be initialized by `voxelize`.
        - Updates `self.mesh.nodes_smooth` with smoothed node coordinates.
        - Adds 'GBfaces' entry to `self.geometry` if it exists.

        Returns
        -------
        None
            The smoothed nodes are stored in `self.mesh.nodes_smooth`. Grain boundary
            faces are optionally stored in `self.geometry['GBfaces']`.

        Raises
        ------
        ValueError
            - If `nodes_v` is None and `self.mesh.nodes` is not available.
        """
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
        if isinstance(self.geometry, dict):
            self.geometry['GBfaces'] = grain_facesDict

    def generate_grains(self):
        """
        Calculate and store polyhedral grain geometry, including particle- and grain-diameter attributes,
        for statistical comparison

        Notes
        -----
        - Requires `self.mesh` to be initialized by `voxelize`.
        - Updates `self.geometry` with polyhedral grain volumes and shared grain boundary (GB) areas.
        - If `self.precipit` is True, irregular grain 0 is temporarily removed from analysis.
        - Logs warnings if grains are not represented in the polyhedral geometry.
        - Prints volume fractions of each phase in the polyhedral geometry.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The calculated grain geometry is stored in `self.geometry`. Phase volume fractions
            are printed to the console.

        Raises
        ------
        ValueError
            - If `self.mesh` or `self.mesh.grains` is None (i.e., voxelized microstructure not available).
        """

        if self.mesh is None or self.mesh.grains is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if self.precipit and 0 in self.mesh.grain_dict.keys():
            # in case of precipit, remove irregular grain 0 from analysis
            empty_vox = self.mesh.grain_dict.pop(0)
            grain_store = self.mesh.grain_phase_dict.pop(0)
        else:
            empty_vox = None
            grain_store = None

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
            # logging.warning('Consider increasing the number of voxels, as grains appear to be very irregular.')
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
                vf = 100.0 * ph_vol[ip] / np.prod(self.rve.size)
                print(f'{ip}: {self.rve.phase_names[ip]} ({vf.round(1)}%)')
        if empty_vox is not None:
            # add removed grain again
            self.mesh.grain_dict[0] = empty_vox
            self.mesh.grain_phase_dict[0] = grain_store

    def generate_orientations(self, data, ang=None, omega=None, Nbase=5000,
                              hist=None, shared_area=None, iphase=None, verbose=False):
        """
        Generate orientations for grains in a representative volume element (RVE)
        to achieve a desired crystallographic texture

        This method assigns grain orientations to achieve a specified texture. The
        input can be an EBSDmap object or a string defining the type of orientation set
        (random or unimodal). For unimodal textures, `ang` and `omega` must be specified.
        Generated orientations are stored in `self.mesh.grain_ori_dict`, and
        `self.mesh.texture` is updated accordingly.

        Parameters
        ----------
        data : EBSDmap or str
            Input source for orientations. Can be an `EBSDmap` object or a string
            specifying the type of orientation set:
            - 'random' or 'rnd' : generate a random orientation set
            - 'unimodal', 'uni_mod', or 'uni_modal' : generate a unimodal orientation set
        ang : float, optional
            Orientation angle for unimodal texture (required if `data` is unimodal).
        omega : float, optional
            Kernel halfwidth for unimodal texture (required if `data` is unimodal).
        Nbase : int, default=5000
            Number of base orientations used in random or unimodal generation.
        hist : array_like, optional
            Histogram for the grain orientations, used to weight orientations.
        shared_area : array_like or float, optional
            Shared grain boundary area for weighted orientation generation.
        iphase : int, optional
            Phase index for which orientations are generated. If None, all phases are processed.
        verbose : bool, default=False
            If True, prints additional information during orientation generation.

        Returns
        -------
        None
            The generated orientations are stored in `self.mesh.grain_ori_dict` and
            `self.mesh.texture` is updated to reflect the type of generated texture.

        Raises
        ------
        ValueError
            - If grain geometry is not defined (i.e., `self.mesh.grains` is None).
            - If `data` is unimodal but `ang` or `omega` are not provided.
            - If `data` is neither `EBSDmap` nor a recognized string option.
            - If histogram is provided but GB areas are not defined (and `shared_area` is None).

        Examples
        --------
        >>> # Generate random orientations for all grains
        >>> rve.generate_orientations('random')

        >>> # Generate unimodal orientations with specified angle and halfwidth
        >>> rve.generate_orientations('unimodal', ang=30.0, omega=10.0)

        >>> # Generate orientations for a specific phase with histogram weighting
        >>> rve.generate_orientations('random', hist=hist_array, iphase=1)

        >>> # Generate orientations using an EBSDmap object
        >>> rve.generate_orientations(ebsd_map_obj)
        """
        from kanapy import __backend__
        if __backend__ == 'mtex':
            from kanapy_mtex.texture import EBSDmap, createOrisetRandom, createOriset
            logging.info('Using MTEX library to read EBSD maps and generate orientations.')
            MTEX = True
        else:
            from kanapy.texture import EBSDmap, createOrisetRandom, createOriset
            logging.info('Using ORIX library to read EBSD maps and generate orientations.')
            MTEX = False

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
            if isinstance(data, EBSDmap):
                if iphase is None or iphase == ip:
                    if gba is not None and not MTEX:
                        logging.warning('Shared are is currently only available in kanapy-mtex.\n'
                                        'This option will be ignored.')
                        gba = None
                    ori_rve = data.calcORI(ngr, iphase=ip, shared_area=gba)
                    self.mesh.texture = "ODF"
            elif isinstance(data, str):
                if data.lower() in ['random', 'rnd']:
                    ori_rve = createOrisetRandom(ngr, Nbase=Nbase, hist=hist, shared_area=gba)
                    self.mesh.texture = "Random"
                elif data.lower() in ['unimodal', 'uni_mod', 'uni_modal']:
                    if ang is None or omega is None:
                        raise ValueError('To generate orientation sets of type "unimodal" angle "ang" and kernel' +
                                         'halfwidth "omega" are required.')
                    ori_rve = createOriset(ngr, ang, omega, hist=hist, shared_area=gba, verbose=verbose)
                    self.mesh.texture = "Unimodal"
            else:
                self.mesh.texture = None
                raise ValueError('Argument to generate grain orientation must be either of type EBSDmap or ' +
                                 '"random" or "unimodal"')

            for i, igr in enumerate(self.mesh.grain_dict.keys()):
                if self.mesh.grain_phase_dict[igr] == ip:
                    if iphase is None or iphase == ip:
                        ind = i - ip * self.ngrains[0]
                        ori_dict[igr] = ori_rve[ind, :]
        self.mesh.grain_ori_dict = ori_dict
        return

    """
    --------     Plotting methods          --------
    """

    def plot_ellipsoids(self, cmap='prism', dual_phase=None, phases=False):
        """
        Generate a 3D plot of particles in the RVE

        This function visualizes ellipsoidal particles in the Representative Volume Element (RVE).
        Particles can be colored according to phase, and a custom colormap can be used.
        Note that `dual_phase` is deprecated; use `phases` instead.

        Parameters
        ----------
        cmap : str, optional
            Colormap used for plotting particles. Default is 'prism'.
        dual_phase : bool or None, optional
            Deprecated parameter for indicating dual-phase visualization.
            Use `phases` instead. Default is None.
        phases : bool, optional
            If True, color particles according to their phase. Default is False.

        Notes
        -----
        - Requires `self.particles` to be initialized by `pack`.
        - Automatically calculates aspect ratios of the RVE for proper 3D plotting.
        - Prints a warning if `dual_phase` is used and maps it to `phases`.

        Returns
        -------
        None
            Displays a 3D plot of ellipsoidal particles using `plot_ellipsoids_3D`.

        Raises
        ------
        ValueError
            - If `self.particles` is None.

        Examples
        --------
        >>> # Simple 3D plot with default colormap
        >>> rve.plot_ellipsoids()

        >>> # Color particles according to phase
        >>> rve.plot_ellipsoids(phases=True)

        >>> # Use a custom colormap
        >>> rve.plot_ellipsoids(cmap='viridis')
        """
        if dual_phase is not None:
            print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
            phases = dual_phase
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        hmin = min(self.rve.size)
        asp_arr = [int(self.rve.size[0] / hmin),
                   int(self.rve.size[1] / hmin),
                   int(self.rve.size[2] / hmin)]
        plot_ellipsoids_3D(self.particles, cmap=cmap, phases=phases, asp_arr=asp_arr)

    def plot_particles(self, cmap='prism', dual_phase=None, phases=False, plot_hull=True):
        """
        Generate a 3D plot of particles in the RVE

        This function visualizes particles in the Representative Volume Element (RVE) in 3D.
        Particles can be colored according to phase, plotted with their convex hulls, and
        displayed using a custom colormap. Note that `dual_phase` is deprecated; use `phases`
        instead.

        Parameters
        ----------
        cmap : str, optional
            Colormap used for plotting particles. Default is 'prism'.
        dual_phase : bool or None, optional
            Deprecated parameter for indicating dual-phase visualization.
            Use `phases` instead. Default is None.
        phases : bool, optional
            If True, color particles according to their phase. Default is False.
        plot_hull : bool, optional
            If True, plot the convex hull (inner polygon) of each particle. Default is True.

        Notes
        -----
        - Requires `self.particles` to be initialized by `pack`.
        - Raises a ValueError if `self.particles` is None or if particles lack inner polygons.
        - Automatically calculates aspect ratios of the RVE for proper 3D plotting.
        - Prints a warning if `dual_phase` is used and maps it to `phases`.

        Returns
        -------
        None
            Displays a 3D plot of particles using `plot_particles_3D`.

        Raises
        ------
        ValueError
            - If `self.particles` is None.
            - If particles do not have inner polygons.

        Examples
        --------
        >>> # Simple 3D plot with default colormap and convex hulls
        >>> rve.plot_particles()

        >>> # Color particles according to phase
        >>> rve.plot_particles(phases=True)

        >>> # Plot particles without convex hulls and use a custom colormap
        >>> rve.plot_particles(cmap='viridis', plot_hull=False)
        """
        if dual_phase is not None:
            print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
            phases = dual_phase
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        if self.particles[0].inner is None:
            raise ValueError('Ellipsoids without inner polygon cannot be plotted.')
        hmin = min(self.rve.size)
        asp_arr = [int(self.rve.size[0] / hmin),
                   int(self.rve.size[1] / hmin),
                   int(self.rve.size[2] / hmin)]
        plot_particles_3D(self.particles, cmap=cmap,
                          phases=phases, plot_hull=plot_hull, asp_arr=asp_arr)

    def plot_voxels(self, sliced=False, dual_phase=None, phases=False, cmap='prism', ori=None,
                    color_key=0, silent=False):
        """
        Generate a 3D visualization of the voxelized RVE structure

        This function visualizes the voxel-based microstructure of the RVE. It supports
        coloring by phase, grain ID, or crystallographic orientation, and can optionally
        render a sliced view of the 3D voxel mesh.

        Parameters
        ----------
        sliced : bool, optional
            If True, generates a sliced view of the voxel mesh to visualize the internal structure.
            Default is False.
        dual_phase : bool or None, optional
            Deprecated parameter for dual-phase visualization. Use `phases` instead.
            Default is None.
        phases : bool, optional
            If True, color voxels by phase instead of grain ID. Default is False.
        cmap : str, optional
            Name of the matplotlib colormap used for rendering. Default is 'prism'.
        ori : array-like, bool, or None, optional
            Array of grain orientations, or True to use `self.mesh.grain_ori_dict` for coloring
            via inverse pole figure (IPF) mapping. Default is None.
        color_key : int, optional
            Selects the color mapping for orientations:
            - 0: iphHSVKey
            - 1: BungeColorKey
            - 2: ipfHKLKey
            Default is 0.
        silent : bool, optional
            If True, suppresses figure display and returns the matplotlib figure object instead.
            Default is False.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            Returns the figure object if `silent=True`. Otherwise, displays the 3D plot and returns None.

        Raises
        ------
        ValueError
            If `self.mesh.grains` is None, indicating that voxelization has not been performed.

        Notes
        -----
        - Requires that the voxel mesh (`self.mesh`) has been generated by `voxelize()`.
        - If `ori` is provided, orientation-based coloring is applied using `get_ipf_colors()`.
        - Prints a warning if `dual_phase` is used, since it is deprecated and replaced by `phases`.
        - Automatically computes aspect ratios of the RVE to ensure accurate 3D visualization.

        Examples
        --------
        >>> # Simple 3D plot of voxelized RVE
        >>> rve.plot_voxels()

        >>> # Plot a sliced view of the voxel mesh
        >>> rve.plot_voxels(sliced=True)

        >>> # Color voxels by phase and use a custom colormap
        >>> rve.plot_voxels(phases=True, cmap='viridis')

        >>> # Use orientation-based coloring via IPF mapping
        >>> rve.plot_voxels(ori=True, color_key=2)

        >>> # Suppress figure display and get the matplotlib Figure object
        >>> fig = rve.plot_voxels(silent=True)
        """
        if dual_phase is not None:
            print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
            phases = dual_phase
        if self.mesh.grains is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        elif phases:
            data = self.mesh.phases
        else:
            data = self.mesh.grains
        if ori is not None:
            from  kanapy import __backend__
            if __backend__ == "mtex":
                from kanapy_mtex.texture import get_ipf_colors
            else:
                from kanapy.texture import get_ipf_colors
            if isinstance(ori, bool) and ori:
                ori = np.array([val for val in self.mesh.grain_ori_dict.values()])
            clist = get_ipf_colors(ori, color_key)
        else:
            clist = None
        hmin = min(self.rve.size)
        asp_arr = [int(self.rve.size[0] / hmin),
                   int(self.rve.size[1] / hmin),
                   int(self.rve.size[2] / hmin)]
        fig = plot_voxels_3D(data, sliced=sliced,
                             phases=phases, cmap=cmap, clist=clist,
                             silent=silent, asp_arr=asp_arr)
        if silent:
            return fig

    def plot_grains(self, geometry=None, cmap='prism', alpha=0.4,
                    ec=None, dual_phase=None, phases=False):
        """
        Plot the polygonalized microstructure of the RVE in 3D

        This function visualizes the polygonal (polyhedral) grain geometry of the
        microstructure, typically after `generate_grains()` has been executed.
        Users can customize the colormap, transparency, and edge color. It also
        supports coloring grains by phase.

        Parameters
        ----------
        geometry : dict or None, optional
            Dictionary containing the polygonal grain geometries. If None, uses
            `self.geometry`. Raises ValueError if geometry is unavailable.
        cmap : str, optional
            Matplotlib colormap name for rendering grain colors. Default is 'prism'.
        alpha : float, optional
            Transparency level of the grain surfaces (0 = fully transparent,
            1 = fully opaque). Default is 0.4.
        ec : list or None, optional
            Edge color specified as an RGBA list, e.g. `[0.5, 0.5, 0.5, 0.1]`.
            Default is `[0.5, 0.5, 0.5, 0.1]`.
        dual_phase : bool or None, optional
            Deprecated parameter for dual-phase coloring. Use `phases` instead.
            Default is None.
        phases : bool, optional
            If True, color grains by phase rather than by grain ID. Default is False.

        Raises
        ------
        ValueError
            If no polygonal geometry is available (i.e., `self.geometry` is None).

        Notes
        -----
        - Requires polygonal grain data generated by `generate_grains()`.
        - Automatically computes the aspect ratio of the RVE for correct 3D scaling.
        - The parameter `dual_phase` is deprecated; prefer using `phases` instead.
        - Visualization is handled by `plot_polygons_3D()`.

        Examples
        --------
        >>> # Simple 3D plot of polygonal grains with default settings
        >>> rve.plot_grains()

        >>> # Color grains by phase
        >>> rve.plot_grains(phases=True)

        >>> # Use a custom colormap and set transparency
        >>> rve.plot_grains(cmap='viridis', alpha=0.6)

        >>> # Specify edge color
        >>> rve.plot_grains(ec=[0.2, 0.2, 0.2, 0.3])
        """
        if ec is None:
            ec = [0.5, 0.5, 0.5, 0.1]
        if dual_phase is not None:
            print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
            phases = dual_phase
        if geometry is None:
            geometry = self.geometry
        if geometry is None:
            raise ValueError('No polygons for grains defined. Run generate_grains() first')
        hmin = min(self.rve.size)
        asp_arr = [int(self.rve.size[0] / hmin),
                   int(self.rve.size[1] / hmin),
                   int(self.rve.size[2] / hmin)]
        plot_polygons_3D(geometry, cmap=cmap, alpha=alpha, ec=ec,
                         phases=phases, asp_arr=asp_arr)

    def plot_stats(self, data=None,
                   gs_data=None, gs_param=None,
                   ar_data=None, ar_param=None,
                   dual_phase=None, phases=False,
                   save_files=False,
                   show_all=False, verbose=False,
                   silent=False, enhanced_plot=False):
        """
        Plot particle, voxel, and grain diameter statistics for comparison

        This method analyzes the microstructure at different representation levels
        (particles, voxels, and polyhedral grains) and plots corresponding statistical
        distributions. It can optionally handle multiphase materials, save plots,
        and display detailed information about geometric parameters.

        Parameters
        ----------
        data : str or None, optional
            Specifies which type of data to analyze and plot:
            - `'p'` : particles
            - `'v'` : voxels
            - `'g'` : grains
            If None, all available data types are analyzed.
        gs_data : list or array-like, optional
            Grain size data for comparison with simulation results.
        gs_param : list or array-like, optional
            Parameters for fitting grain size distributions.
        ar_data : list or array-like, optional
            Aspect ratio data for comparison with simulation results.
        ar_param : list or array-like, optional
            Parameters for fitting aspect ratio distributions.
        dual_phase : bool or None, optional
            Deprecated. Use `phases` instead.
        phases : bool, optional
            If True, perform separate statistical analysis for each phase.
        save_files : bool, optional
            If True, save generated plots and statistical results to files.
        show_all : bool, optional
            If True, display all generated plots interactively.
        verbose : bool, optional
            If True, print detailed numerical results during analysis.
        silent : bool, optional
            If True, suppresses console output and returns figures directly.
        enhanced_plot : bool, optional
            If True, use enhanced plot styling (automatically enabled when `silent=True`).

        Returns
        -------
        flist : list of matplotlib.figure.Figure or None
            List of generated figure objects if `silent=True`, otherwise None.

        Raises
        ------
        ValueError
            If required data (particles, voxels, or grains) are missing for the selected mode.

        Notes
        -----
        - The method computes and compares geometric parameters such as:
          principal axis lengths (a, b, c), aspect ratios, rotation axes,
          and equivalent diameters.
        - The results are stored in `self.rve_stats` and labeled in `self.rve_stats_labels`.
        - For multiphase structures, statistics are calculated per phase.

        Examples
        --------
        >>> # Plot all statistics with default settings
        >>> rve.plot_stats()

        >>> # Plot only particle statistics and show all plots interactively
        >>> rve.plot_stats(data='p', show_all=True)

        >>> # Perform multiphase statistical analysis and save figures
        >>> rve.plot_stats(phases=True, save_files=True)

        >>> # Suppress console output and get figure objects
        >>> figs = rve.plot_stats(silent=True)
        """
        if dual_phase is not None:
            print('Use of "dual_phase" is depracted. Use parameter "phases" instead.')
            phases = dual_phase
        if silent:
            verbose = False
            show_all = False
            enhanced_plot = True
        ax_max = np.prod(self.rve.size) ** (1 / 3)
        if phases:
            nphases = self.nphases
            if self.precipit and 0 in self.mesh.grain_dict.keys():
                # in case of precipit, remove irregular grain 0 from analysis
                nphases -= 1
        else:
            nphases = 1
        if not (isinstance(gs_data, list) and len(gs_data) == nphases):
            gs_data = [gs_data] * nphases
        if not (isinstance(gs_param, list) and len(gs_param) == nphases):
            gs_param = [gs_param] * nphases
        if not (isinstance(ar_data, list) and len(ar_data) == nphases):
            ar_data = [ar_data] * nphases
        if not (isinstance(ar_param, list) and len(ar_param) == len(gs_param)):
            ar_param = [ar_param] * nphases
        """
        gs_data = [ebsd.ms_data[i]['gs_data'] for i in range(nphases)]
        gs_param = [ebsd.ms_data[i]['gs_param'] for i in range(nphases)]
        ar_data = [ebsd.ms_data[i]['ar_data'] for i in range(nphases)]
        ar_param = [ebsd.ms_data[i]['ar_param'] for i in range(nphases)]
        """
        iphase = None
        flist = []
        for ip in range(nphases):
            stats_list = []
            labels = []
            if phases:
                iphase = ip
                print(f'Plotting statistical information for phase {ip}')

            # Analyze and plot particles statistics
            if (data is None and self.particles is not None) or \
                    (type(data) is str and 'p' in data.lower()):
                if self.particles is None:
                    logging.error('Particle statistics requested, but no particles defined. '
                                  'Run "pack()" first.')
                    return
                part_stats = get_stats_part(self.particles, iphase=iphase, ax_max=ax_max,
                                            show_plot=show_all,
                                            verbose=verbose, save_files=save_files)
                stats_list.append(part_stats)
                labels.append("Partcls")

            # Analyze and plot statistics of voxel structure in RVE
            if (data is None and self.mesh is not None) or \
                    (type(data) is str and 'v' in data.lower()):
                if self.mesh is None:
                    logging.error('Voxel statistics requested, but no voxel mesh defined. '
                                  'Run "voxelize()" first.')
                    return
                vox_stats = get_stats_vox(self.mesh, iphase=iphase, ax_max=ax_max,
                                          show_plot=show_all,
                                          verbose=verbose, save_files=save_files)
                stats_list.append(vox_stats)
                labels.append('Voxels')

            # Analyze and plot statistics of polyhedral grains in RVE
            if (data is None and self.geometry is not None) or \
                    (type(data) is str and 'g' in data.lower()):
                if self.geometry is None:
                    logging.error('Geometry statistics requested, but no polyhedral grains defined. '
                                  'Run "generate_grains()" first.')
                    return
                grain_stats = get_stats_poly(self.geometry['Grains'], iphase=iphase, ax_max=ax_max,
                                             show_plot=show_all, phase_dict=self.mesh.grain_phase_dict,
                                             verbose=verbose, save_files=save_files)
                stats_list.append(grain_stats)
                labels.append('Grains')

            if phases:
                print(f'\nStatistical microstructure parameters of phase {iphase} in RVE')
                print('-------------------------------------------------------')
            else:
                print('\nStatistical microstructure parameters of RVE')
                print('--------------------------------------------')
            print(f'Type\t| a (µm) \t| b (µm) \t| c (µm) \t| std.dev\t| rot.axis\t| asp.ratio\t| std.dev\t|'
                  f' equ.dia. (µm)\t| std.dev')
            for i, sd in enumerate(stats_list):
                av_std = np.mean([sd['a_sig'], sd['b_sig'], sd['c_sig']])
                print(f'{labels[i]}\t|  {sd["a_scale"]:.3f}\t|  {sd["b_scale"]:.3f}\t|  {sd["c_scale"]:.3f}\t|  '
                      f'{av_std:.4f}\t|     {sd["ind_rot"]}   \t|  {sd["ar_scale"]:.3f}\t|  {sd["ar_sig"]:.4f}\t|     '
                      f'{sd["eqd_scale"]:.3f} \t|  {sd["eqd_sig"]:.4f}')
            self.rve_stats = stats_list
            self.rve_stats_labels = labels
            fig = plot_output_stats(stats_list, labels, iphase=iphase,
                                    gs_data=gs_data[ip], gs_param=gs_param[ip],
                                    ar_data=ar_data[ip], ar_param=ar_param[ip],
                                    save_files=save_files, silent=silent,
                                    enhanced_plot=enhanced_plot)
            flist.append(fig)
        if silent:
            return flist

    def plot_stats_init(self, descriptor=None, gs_data=None, ar_data=None,
                        porous=False,
                        get_res=False, show_res=False,
                        save_files=False, silent=False, return_descriptors=False):
        """
        Plot initial statistical microstructure descriptors of RVE and optionally return computed descriptors

        This method analyzes the initial microstructure defined by input descriptors
        (or the class attribute `self.descriptor`) and plots the distributions of
        equivalent diameters and aspect ratios. For elongated grains, it prints a
        summary of input and output statistics. Optionally, statistical descriptors
        can be returned for further processing.

        Parameters
        ----------
        descriptor : list of dict, dict, or None, optional
            Microstructure phase descriptor(s). If None, uses `self.descriptor`.
        gs_data : list or array-like, optional
            Grain size data for comparison with initial statistics.
        ar_data : list or array-like, optional
            Aspect ratio data for comparison with initial statistics.
        porous : bool, optional
            If True, only the first phase is considered (e.g., for porous structures).
        get_res : bool, optional
            If True, computes statistical descriptors from the voxelized structure.
        show_res : bool, optional
            If True, prints detailed statistical results to the console.
        save_files : bool, optional
            If True, saves generated plots to files.
        silent : bool, optional
            If True, suppresses console output and returns figures directly.
        return_descriptors : bool, optional
            If True, returns computed statistical descriptors along with figures.

        Returns
        -------
        flist : list of matplotlib.figure.Figure
            List of generated figure objects.
        descs : list of dict
            Optional list of computed statistical descriptors if `return_descriptors=True`.

        Notes
        -----
        - Requires voxelized microstructure (`self.mesh`) for computing descriptors.
        - Computes statistics for equivalent diameter, aspect ratio, principal axes,
          and rotation axes when applicable.
        - Can handle multiphase or single-phase microstructures.

        Examples
        --------
        >>> # Plot initial statistics with default settings
        >>> rve.plot_stats_init()

        >>> # Plot only the first phase (porous) and save figures
        >>> rve.plot_stats_init(porous=True, save_files=True)

        >>> # Compute descriptors and get figures and descriptor objects
        >>> figs, descs = rve.plot_stats_init(get_res=True, return_descriptors=True)

        >>> # Suppress console output and return figure objects
        >>> figs = rve.plot_stats_init(silent=True)
        """
        def analyze_voxels(ip, des):
            """
            Compute voxel-based statistical descriptors for a given phase of the RVE

            This function calculates equivalent diameter, aspect ratio, principal axes, and rotation axis
            statistics from the voxelized microstructure. It prints input vs output statistics if the phase
            is elongated and returns formatted arrays for grain size, aspect ratio, and a dictionary of descriptors

            Parameters
            ----------
            ip : int
                Index of the phase to analyze
            des : dict
                Descriptor dictionary for the phase containing grain type and target statistics

            Returns
            -------
            gsp : list
                Summary of equivalent diameter statistics for plotting
            arp : list
                Summary of aspect ratio statistics for plotting
            statistical_descriptors : dict
                Detailed statistics including mean, std, min, max of eqd, ar, axes, and rotation axis
            """
            if self.mesh is None:
                raise ValueError('show_res is True, but no voxels have been defined. Run voxelize first.')
            vox_stats = get_stats_vox(self.mesh, iphase=ip, show_plot=show_res)
            gsp = [vox_stats['eqd_sig'], 0.0, vox_stats['eqd_scale'],
                   min(vox_stats['eqd']), max(vox_stats['eqd'])]
            arp = [vox_stats['ar_sig'], 0.0, vox_stats['ar_scale'],
                   min(vox_stats['ar']), max(vox_stats['ar'])]
            if 'Grain type' in des.keys() and des['Grain type'] == 'Elongated':
                if nel > 1:
                    print(f'\nStatistical microstructure parameters of phase {ip} in RVE')
                    print('-------------------------------------------------------')
                else:
                    print('\nStatistical microstructure parameters of RVE')
                    print('--------------------------------------------')
                print(f'Type\t| a (µm) \t| b (µm) \t| c (µm) \t| std.dev\t| rot.axis\t| asp.ratio\t| std.dev\t|'
                      f' equ.dia. (µm)\t| std.dev')

                av_std = np.mean([vox_stats['a_sig'], vox_stats['b_sig'], vox_stats['c_sig']])
                print(f'Input\t|  -      \t|  -      \t|  -      \t|   -      \t|     -   \t|  '
                      f'{des["Aspect ratio"]["scale"]:.3f}\t|  {des["Aspect ratio"]["sig"]:.4f}\t|     '
                      f'{des["Equivalent diameter"]["scale"]:.3f} \t|  {des["Equivalent diameter"]["sig"]:.4f}')
                print(f'Output\t|  {vox_stats["a_scale"]:.3f}\t|  {vox_stats["b_scale"]:.3f}\t|  '
                      f'{vox_stats["c_scale"]:.3f}\t|  {av_std:.4f}\t|     {vox_stats["ind_rot"]}   \t|  '
                      f'{vox_stats["ar_scale"]:.3f}\t|  {vox_stats["ar_sig"]:.4f}\t|     '
                      f'{vox_stats["eqd_scale"]:.3f} \t|  {vox_stats["eqd_sig"]:.4f}')
                statistical_descriptors = {
                    'eqd': {
                        'mean': float(vox_stats['eqd_scale']),
                        'std':  float(vox_stats['eqd_sig']),
                        'min':  float(min(vox_stats['eqd'])),
                        'max':  float(max(vox_stats['eqd'])),
                    },
                    'ar': {
                        'mean': float(vox_stats['ar_scale']),
                        'std':  float(vox_stats['ar_sig']),
                        'min':  float(min(vox_stats['ar'])),
                        'max':  float(max(vox_stats['ar'])),
                    },
                    'axes': {
                        'a': float(vox_stats['a_scale']),
                        'b': float(vox_stats['b_scale']),
                        'c': float(vox_stats['c_scale']),
                        'a_std': float(vox_stats.get('a_sig', np.nan)),
                        'b_std': float(vox_stats.get('b_sig', np.nan)),
                        'c_std': float(vox_stats.get('c_sig', np.nan)),
                    },
                    'rotation_axis': vox_stats.get('ind_rot', None),
                }
            return gsp, arp, statistical_descriptors

        if show_res: get_res = True
        if silent: show_res = False
        if descriptor is None: descriptor = self.descriptor
        if not isinstance(descriptor, list): descriptor = [descriptor]
        if porous: descriptor = descriptor[0:1]
        nel = len(descriptor)

        if not (isinstance(gs_data, list) and len(gs_data) == nel): gs_data = [gs_data] * nel
        if not (isinstance(ar_data, list) and len(ar_data) == nel): ar_data = [ar_data] * nel

        flist, descs = [] , []
        for ip, des in enumerate(descriptor):
            gsp = arp = None
            statistical_descriptors = None
            if get_res:
                gsp, arp, statistical_descriptors = analyze_voxels(ip, des)
            fig = plot_init_stats(des, gs_data=gs_data[ip], ar_data=ar_data[ip],
                                  gs_param=gsp, ar_param=arp,
                                  save_files=save_files, silent=silent)
            flist.append(fig)
            if return_descriptors:
                descs.append({'phase': ip, **(statistical_descriptors or {})})

        if return_descriptors or silent: return flist, descs

    def plot_slice(self, cut='xy', data=None, pos=None, fname=None,
                   dual_phase=False, save_files=False):
        """
        Plot a 2D slice through the microstructure.

        The function visualizes a cross-section of the microstructure. If a polygonalized
        microstructure is available, it will be used as the plotting basis; otherwise,
        or if `data='voxels'`, the voxelized microstructure will be plotted. This method
        internally calls `output_ang` with plotting enabled and file writing disabled.

        Parameters
        ----------
        cut : str, optional
            The cutting plane of the slice. Options are 'xy', 'xz', or 'yz'.
            Default is 'xy'.
        data : str, optional
            Data basis for plotting. Options are 'voxels' or 'poly'. Default is None.
        pos : str or float, optional
            Position of the slice, either as an absolute value or as one of
            'top', 'bottom', 'left', 'right'. Default is None.
        fname : str, optional
            Filename to save the figure as a PDF. Default is None.
        dual_phase : bool, optional
            If True, enable dual-phase visualization. Default is False.
        save_files : bool, optional
            If True, the figure will be saved to disk. Default is False.

        Returns
        -------
        None

        Examples
        --------
        >>> micro.plot_slice(cut='xz', data='poly', pos='top', save_files=True)
        """
        self.output_ang(cut=cut, data=data, plot=True, save_files=False,
                        pos=pos, fname=fname, dual_phase=dual_phase,
                        save_plot=save_files)

    """
    --------        Import/Export methods        --------
    """

    def write_abq(self, nodes=None, file=None, path='./', voxel_dict=None, grain_dict=None,
                  dual_phase=False, thermal=False, units=None, ialloy=None, nsdv=200, crystal_plasticity=False,
                  phase_props=None,
                  *,
                  boundary_conditions: Optional[Dict[str, Any]] = None):
        """
        Write the Abaqus input deck (.inp) for the generated RVE

        This method generates an Abaqus input file for the current RVE. It supports
        voxelized or smoothened meshes, dual-phase materials, crystal plasticity,
        thermal analysis, custom material properties, and optional boundary conditions.
        Material definitions and mesh data are automatically handled based on provided
        arguments or class attributes.

        Parameters
        ----------
        nodes : str or array-like, optional
            Defines the mesh to write:
            - 'voxels', 'v' : use voxelized mesh
            - 'smooth', 's' : use smoothened mesh
            - array-like : explicit nodal coordinates
            Default is None, automatically selecting available mesh.
        file : str, optional
            Filename for the Abaqus input deck. Default is auto-generated.
        path : str, optional
            Directory path to save the input deck. Default is './'.
        voxel_dict : dict, optional
            Dictionary with voxel information. Default is `self.mesh.voxel_dict`.
        grain_dict : dict, optional
            Dictionary mapping grain IDs to nodes. Default is `self.mesh.grain_dict`.
        dual_phase : bool, optional
            If True, generate input for dual-phase materials. Default is False.
        thermal : bool, optional
            If True, include thermal material definitions. Default is False.
        units : str, optional
            Units for the model, 'mm' or 'um'. Default is `self.rve.units`.
        ialloy : list or object, optional
            Material definitions for each phase. Default is `self.rve.ialloy`.
        nsdv : int, optional
            Number of state variables per integration point for crystal plasticity. Default is 200.
        crystal_plasticity : bool, optional
            If True, enable crystal plasticity material definitions. Default is False.
        phase_props : dict, optional
            Additional phase-specific material properties.
        boundary_conditions : dict, optional
            Dictionary specifying boundary conditions to write. Default is None.

        Returns
        -------
        file : str
            Full path to the generated Abaqus input file.

        Raises
        ------
        ValueError
            - If no voxelized or smoothened mesh is available when required.
            - If invalid `nodes` argument is provided.
            - If units are not 'mm' or 'um'.
            - If the list `ialloy` is longer than the number of phases in the RVE.
            - If periodic boundary conditions are requested but the RVE is non-periodic.

        Notes
        -----
        - Automatically selects voxel or smoothened mesh if `nodes` is None.
        - Handles dual-phase structures and crystal plasticity input.
        - Writes additional material files if orientations are available and `ialloy` is provided.
        - Visualization or mesh generation must be performed before calling this function.

        Examples
        --------
        >>> # Write voxelized RVE input deck with default settings
        >>> abq_file = rve.write_abq(nodes='voxels')

        >>> # Write smoothened RVE deck for dual-phase material
        >>> abq_file = rve.write_abq(nodes='smooth', dual_phase=True, ialloy=alloy_list)

        >>> # Include boundary conditions
        >>> bc = {'ux': [0, 0.1], 'uy': [0, 0.1]}
        >>> abq_file = rve.write_abq(nodes='voxels', boundary_conditions=bc)
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
        periodic = self.rve.periodic

        if not periodic and periodicBC:
            raise ValueError("Periodic boundary conditions cannot be applied to a non-periodic RVE.")

        export2abaqus(nodes, file, grain_dict, voxel_dict,
                      units=units, gb_area=faces,
                      dual_phase=dual_phase,
                      ialloy=ialloy, grain_phase_dict=grpd,
                      thermal=thermal,
                      crystal_plasticity = crystal_plasticity,
                      phase_props = phase_props, boundary_conditions = boundary_conditions)

        # if orientations exist and ialloy is defined also write material file with Euler angles
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
                raise ValueError('No orientations present. Run "generate_orientations" first.')
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

    def write_stl(self, data='grains', file=None, path='./',
                  phases=False, phase_num=None):
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

        def write_phases(ip):
            for grain in self.geometry['Grains'].values():
                if grain['Phase'] == ip:
                    for ft in grain['Simplices']:
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
                if phases:
                    if phase_num is None:
                        raise ValueError('Phase-specific output requested, but no phase number specified.')
                    write_phases(phase_num)
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
        from kanapy import __version__

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
                "Version": __version__,
                "Repository": "https://github.com/ICAMS/Kanapy.git",
                "Input": source,
                "Script": script_name,
                "Material": self.name,
                "Phase_names": self.rve.phase_names,
                "Size": [int(val) for val in self.rve.size],
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
            structure["Grains"][int(igr)] = {
                "Phase": int(self.mesh.grain_phase_dict[igr])
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
                    "Values": [int(val) for val in self.mesh.voxel_dict.values()],
                }
            }

        # write file
        with open(file, 'w') as fp:
            json.dump(structure, fp)
        return

    def write_data(self,
                         user_metadata: Optional[Dict[str, Any]] = None,
                         boundary_condition: Optional[Dict[str, Any]] = None,
                         phases: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                         interactive: bool = True,
                         structured: bool = True,
                         ialloy: int = 1,
                         length_unit: str = 'µm') -> dict:

        """
        Generate a JSON file containing User-, System-, and Job-Specific Elements.

        - `user_metadata`: prefilled metadata fields (if interactive=False).
        - `boundary_condition`: separate BC dict (mechanical_BC or thermal_BC).
        - `interactive`: if True and inputs missing, prompt user.
        - `structured`: whether mesh is structured.
        """


        # interpret length_unit argument
        if length_unit == 'µm':
            length_scale = 1.0
        elif length_unit == 'mm':
            length_scale = 1e-3
        elif length_unit == 'm':
            length_scale = 1e-6
        else:
            raise ValueError("length_unit must be 'µm' or 'mm'")

        # Material library definitions (pulled from mod_alloys.f)
        material_library = {
            1: {  # Aluminum
                'ialloy': 1,
                'material_identifier': 'Aluminum',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 247000.0, 'C12': 147000.0, 'C44': 125000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 1e-6,  # shrt0
                    'initial_critical_resolved_shear_stress': 20.0,  # crss0
                    'saturated_slip_resistance': 1500.0,  # crsss
                    'strain_rate_sensitivity_exponent': 20.0,  # pwfl
                    'reference_hardening_rate': 60.0,  # hdrt0
                    'hardening_exponent': 2.25  # pwhd
                }
            },
            2: {  # Copper
                'ialloy': 2,
                'material_identifier': 'Copper',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 170000.0, 'C12': 124000.0, 'C44': 75000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 0.001,  # shrt0
                    'initial_critical_resolved_shear_stress': 16.0,  # crss0
                    'saturated_slip_resistance': 148.0,  # crsss
                    'strain_rate_sensitivity_exponent': 83.0,  # pwfl
                    'reference_hardening_rate': 250.0,  # hdrt0
                    'hardening_exponent': 2.25  # pwhd
                }
            },
            3: {  # Ferrite
                'ialloy': 3,
                'material_identifier': 'Ferrite',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 230000.0, 'C12': 135000.0, 'C44': 116000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 1e-6,
                    'initial_critical_resolved_shear_stress': 25.0,
                    'saturated_slip_resistance': 1600.0,
                    'strain_rate_sensitivity_exponent': 18.0,
                    'reference_hardening_rate': 70.0,
                    'hardening_exponent': 2.0
                }
            },
            4: {  # Austenite
                'ialloy': 4,
                'material_identifier': 'Austenite',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 190000.0, 'C12': 130000.0, 'C44': 115000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 1e-6,
                    'initial_critical_resolved_shear_stress': 15.0,
                    'saturated_slip_resistance': 1400.0,
                    'strain_rate_sensitivity_exponent': 22.0,
                    'reference_hardening_rate': 65.0,
                    'hardening_exponent': 2.2
                }
            },
            5: {  # Superalloy
                'ialloy': 5,
                'material_identifier': 'Superalloy',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 260000.0, 'C12': 150000.0, 'C44': 120000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 1e-6,
                    'initial_critical_resolved_shear_stress': 30.0,
                    'saturated_slip_resistance': 1700.0,
                    'strain_rate_sensitivity_exponent': 25.0,
                    'reference_hardening_rate': 80.0,
                    'hardening_exponent': 2.8
                }
            },
            6: {  # Nickel
                'ialloy': 6,
                'material_identifier': 'Nickel',
                'elastic_model_name': 'Anisotropic Elasticity',
                'elastic_parameters': {
                    'C11': 246000.0, 'C12': 147000.0, 'C44': 124000.0
                },
                'plastic_model_name': 'Crystal Plasticity',
                'plastic_parameters': {
                    'reference_shear_rate': 1e-6,
                    'initial_critical_resolved_shear_stress': 18.0,
                    'saturated_slip_resistance': 1450.0,
                    'strain_rate_sensitivity_exponent': 20.0,
                    'reference_hardening_rate': 75.0,
                    'hardening_exponent': 2.4
                }
            }
        }

        # Define required fields
        required_fields = [
            'identifier', 'title',
            'creator', 'creator_ORCID', 'creator_affiliation', 'creator_institute', 'creator_group',
            'contributor', 'contributor_ORCID', 'contributor_affiliation', 'contributor_institute', 'contributor_group',
            'date', 'shared_with', 'description', 'rights', 'rights_holder','funder_name', 'fund_identifier',
            'publisher', 'relation', 'keywords'
        ]

        def prompt_list(field_name: str) -> List[str]:
            vals = input(f"Enter comma-separated {field_name}: ").strip()
            return [v.strip() for v in vals.split(',')] if vals else []

        # Gather metadata
        if interactive and user_metadata is None:
            use: Dict[str, Any] = {}
            # identifier
            ident = input("Identifier (leave blank to auto-generate): ").strip()
            if not ident:
                now = datetime.utcnow().isoformat()
                ident = hashlib.sha256(now.encode()).hexdigest()[:8]
            use['identifier'] = ident
            use['title'] = input("Title: ").strip()
            # creator fields
            use['creator'] = prompt_list('creator names (e.g. Last, First)')
            use['creator_ORCID'] = prompt_list('creator ORCID(s)')
            use['creator_affiliation'] = prompt_list('creator affiliations')
            use['creator_institute'] = prompt_list('creator institutes')
            use['creator_group'] = prompt_list('creator groups')
            # contributor fields
            use['contributor'] = prompt_list('contributor names')
            use['contributor_ORCID'] = prompt_list('contributor ORCID(s)')
            use['contributor_affiliation'] = prompt_list('contributor affiliations')
            use['contributor_institute'] = prompt_list('contributor institutes')
            use['contributor_group'] = prompt_list('contributor groups')

            use['date'] = input("Date (YYYY-MM-DD): ").strip() or datetime.utcnow().strftime('%Y-%m-%d')
            # shared_with
            shared: List[Dict[str, str]] = []
            print("Enter shared_with access entries. Valid types: c, u, g, all. Blank to stop.")
            while True:
                atype = input("  access_type: ").strip()
                if not atype:
                    break
                shared.append({'access_type': atype})
            use['shared_with'] = shared
            use['description'] = input("Description: ").strip()
            use['rights'] = input("Rights (e.g. Creative Commons Attribution 4.0 International): ").strip()
            use['rights_holder'] = prompt_list('rights_holder')
            # other fields
            use['funder_name'] = input("Funder name: ").strip()
            use['fund_identifier'] = input("Fund identifier: ").strip()
            use['publisher'] = input("Publisher: ").strip()
            use['relation'] = prompt_list('relation (DOI or URL)')
            use['keywords'] = prompt_list('keywords')
        else:
            if not user_metadata:
                raise ValueError("user_metadata must be provided when interactive is False.")
            use = user_metadata.copy()
            # Validate presence of required fields
            missing = [f for f in required_fields if f not in use]
            if missing:
                raise ValueError(f"Missing required metadata fields: {', '.join(missing)}")

        ig = {
            'RVE_size': [float(v) * length_scale for v in self.rve.size],
            'RVE_continuity': self.rve.periodic,
            'discretization_type': 'Structured' if structured else 'Unstructured',
            'discretization_unit_size': [(float(s) / float(d)) * length_scale for s, d in zip(self.rve.size, self.rve.dim)],
            'discretization_count': int(self.mesh.nvox),
            'Origin': {
                'software': 'kanapy',
                'software_version': pkg_version('kanapy'),
                'system': platform.system(),
                'system_version': platform.version()
            }
        }

        # ─── Show vertex‐diagram for BC reference ────────────────────────────────
        try:
            # locate the package root, two levels up from this file
            script_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
            img_path = os.path.join(project_root, 'docs', 'figs', 'RVE', 'Vertices.png')
            if os.path.exists(img_path):
                # first try with Pillow
                try:
                    from PIL import Image
                    Image.open(img_path).show()
                except ImportError:
                    # fallback to the system browser/viewer
                    import webbrowser
                    webbrowser.open(f'file://{img_path}')
        except (FileNotFoundError, OSError) as e:
            logging.warning(f"[Warning] Unable to open Vertices.png: {e}")

        # Boundary conditions
        if boundary_condition:
            # Ensure 'type' is present
            if 'mechanical_BC' in boundary_condition:
                # Determine mechanical vs thermal by key presence
                mech_list = boundary_condition['mechanical_BC']
                # Normalize to list
                if not isinstance(mech_list, list):
                    mech_list = [mech_list]
                job_bc = {'mechanical_BC': mech_list}
            elif 'thermal_BC' in boundary_condition:
                th_list = boundary_condition['thermal_BC']
                if not isinstance(th_list, list):
                    th_list = [th_list]
                job_bc = {'thermal_BC': th_list}
            else:
                # fallback if wrong keys provided
                if interactive:
                    # ask user which type to populate
                    bc_choice = input("Boundary condition key not found. Enter 'mechanical' or 'thermal': ").strip()
                    if bc_choice == 'mechanical':
                        job_bc = {'mechanical_BC': []}
                    else:
                        job_bc = {'thermal_BC': []}
                else:
                    raise ValueError(
                        "boundary_condition dict must include 'mechanical_BC' or 'thermal_BC' key when interactive=False.")

        elif interactive: # Interactive entry of multiple mechanical BCs
            mech_entries: List[Dict[str, Any]] = []
            bc_type = input("Boundary condition type ('mechanical' or 'thermal'): ").strip()
            if bc_type == 'mechanical':
                while True:
                    print("Define a mechanical BC (leave vertex_list blank to stop):")
                    vertex_list = prompt_list('vertex_list')
                    if not vertex_list:
                        break
                    constraints = input("Constraints xyz (e.g. 'free,fixed,loaded'): ").split(',')
                    loading_type = input("Loading type (force/displacement/stress/strain/none): ").strip()
                    loading_mode = input("Loading mode (static/cyclic/monotonic/intermittent): ").strip()
                    # count how many of those constraints are “loaded”:
                    num = sum(1 for c in constraints if c.strip().lower() == 'loaded')
                    loads = []
                    for i in range(num):
                        print(f"Load entry #{i + 1}:")
                        mag = float(input("  magnitude: "))
                        freq = float(input("  frequency: "))
                        dur = float(input("  duration: "))
                        R = float(input("  R: "))
                        loads.append({'magnitude': mag, 'frequency': freq, 'duration': dur, 'R': R})
                    mech_entries.append({
                        'vertex_list': vertex_list,
                        'constraints': constraints,
                        'loading_type': loading_type,
                        'loading_mode': loading_mode,
                        'applied_load': loads
                    })
                job_bc = {'mechanical_BC': mech_entries}
            else:
                job_bc = {'thermal_BC': []}
        else:
            job_bc = {}


        # Phase data
        phase_list = []
        if phases:  # user provided a dict or list of dicts
            if isinstance(phases, dict):
                phase_list = [phases]
            elif isinstance(phases, list):
                phase_list = phases
            else:
                raise TypeError("`phases` must be a dict or list of dicts.")
        else:  # fallback: use ialloy + material_library
            if not ialloy or ialloy not in material_library:
                if interactive:
                    ialloy = int(input(f"Choose ialloy from {list(material_library.keys())}: "))
                else:
                    raise ValueError(
                        f"No phases provided and invalid ialloy. "
                        f"Valid ialloy values: {list(material_library.keys())}"
                    )

            for idx in range(self.nphases):
                mat = material_library[ialloy]
                pe = mat['elastic_parameters']
                pp = mat['plastic_parameters']

                phase_entry = {
                    "id": idx,
                    "phase_identifier": self.rve.phase_names[idx],
                    "volume_fraction": self.rve.phase_vf[idx],
                    "lattice_structure": None,
                    "constitutive_model": {
                        "$schema": "http://json-schema.org/draft-04/schema#",
                        "elastic_model_name": mat['elastic_model_name'],
                        "elastic_parameters": pe,
                        "plastic_model_name": mat['plastic_model_name'],
                        "plastic_parameters": pp
                    },
                    "orientation": {
                        "euler_angles": {
                            "Phi1": None,
                            "Phi": None,
                            "Phi2": None,
                        },
                        "grain_count": self.mesh.ngrains_phase[idx],
                        "orientation_identifier": None,
                        "texture_type": None,
                        "software": "orix",
                        "software_version": orix.__version__,
                    }
                }
                phase_list.append(phase_entry)

        # ─── pull Mesh + RVE into locals ─────────────────────────────────────────
        grain_phase_dict = getattr(self.mesh, 'grain_phase_dict', {}) or {} # {gid: phase_id}
        grain_ori_dict = getattr(self.mesh, 'grain_ori_dict', None)  # can be None ({gid: [euler…]})
        vox_center_dict = getattr(self.mesh, 'vox_center_dict', {}) or {} # {vid: (x,y,z)}
        grain_to_voxels = getattr(self.mesh, 'grain_dict', {}) or {} # {gid: [vid,…]}
        rve_size = getattr(self.rve, 'size', [0, 0, 0]) or [0, 0, 0] # e.g. [20.0,20.0,20.0]
        rve_dim = getattr(self.rve, 'dim', [1, 1, 1]) or [1, 1, 1]  # # e.g. [10,10,10] (avoid /0)

        # Is orientation available?
        include_orientation = isinstance(grain_ori_dict, dict) and len(grain_ori_dict) > 0
        # ─── compute one‐voxel volume ─────────────────────────────────────────────
        unit_sizes = []
        for s, d in zip(rve_size, rve_dim):
            try:
                unit_sizes.append((float(s) / float(d)) * length_scale if float(d) != 0 else 0.0)
            except Exception:
                unit_sizes.append(0.0)

        voxel_volume = (unit_sizes[0] if len(unit_sizes) > 0 else 0.0) \
                       * (unit_sizes[1] if len(unit_sizes) > 1 else 0.0) \
                       * (unit_sizes[2] if len(unit_sizes) > 2 else 0.0)

        # grid size per axis
        Nx, Ny, Nz = int(rve_dim[0]), int(rve_dim[1]), int(rve_dim[2])
        # infer origin from centers if available: origin = min(center) - 0.5*unit
        if len(vox_center_dict) > 0:
            xs = [float(c[0])* length_scale for c in vox_center_dict.values()]
            ys = [float(c[1])* length_scale for c in vox_center_dict.values()]
            zs = [float(c[2])* length_scale for c in vox_center_dict.values()]
            ox = (min(xs) if xs else 0.0) - 0.5 * unit_sizes[0]
            oy = (min(ys) if ys else 0.0) - 0.5 * unit_sizes[1]
            oz = (min(zs) if zs else 0.0) - 0.5 * unit_sizes[2]
        else:
            ox = oy = oz = 0.0
        origin = [ox, oy, oz]

        # ─── precompute voxel→grain lookup ───────────────────────────────────────
        voxel_to_grain = {
            vid: gid
            for gid, vids in grain_to_voxels.items()
            for vid in vids
        }
        # ─── build time-0 grid dict  ────────────────────────────────────────────
        grid_t0 = {
            "status": "regular",
            "grid_size": [float(v) * length_scale for v in self.rve.size],
            "grid_spacing": [(float(s) / float(d)) * length_scale for s, d in zip(self.rve.size, self.rve.dim)],
        }
        # ─── build time‐0 grain dict ────────────────────────────────────────────
        grains_t0 = []
        for gid in grain_phase_dict.keys():
            entry = {
                "grain_id": gid,
                "phase_id": grain_phase_dict.get(gid),
                "grain_volume": len(grain_to_voxels.get(gid, [])) * voxel_volume,
            }
            if include_orientation:
                ori = grain_ori_dict.get(gid)
                if ori is not None:
                    entry["orientation"] = list(ori)
            grains_t0.append(entry)
        grains_t0_sorted = sorted(grains_t0, key=lambda d: int(d["grain_id"]))

        # ─── Build time‐0 voxel dictionary ────────────────────────────────────────
        def _coord_to_index_1based(c, o, d):
            # i = round((c - o)/d + 0.5), robust to tiny float noise
            return int(round((float(c) - float(o)) / float(d) + 0.5))

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        voxels_t0 = []
        for vid, gid in voxel_to_grain.items():
            cx, cy, cz = vox_center_dict.get(vid, (0.0, 0.0, 0.0))
            centroid = [float(cx) * length_scale,
                        float(cy) * length_scale,
                        float(cz) * length_scale]

            # Compute 1-based voxel indices from centers
            ix = _coord_to_index_1based(centroid[0], origin[0], unit_sizes[0])
            iy = _coord_to_index_1based(centroid[1], origin[1], unit_sizes[1])
            iz = _coord_to_index_1based(centroid[2], origin[2], unit_sizes[2])

            # Clamp to valid range [1..N*]
            ix = _clamp(ix, 1, Nx)
            iy = _clamp(iy, 1, Ny)
            iz = _clamp(iz, 1, Nz)


            entry = {
                "voxel_id": vid,
                "grain_id": gid,
                "centroid_coordinates": centroid, # scaled for output
                "voxel_index": [ix, iy, iz], # 1-based indices
                "voxel_volume": voxel_volume,
            }
            if include_orientation:
                ori = grain_ori_dict.get(gid)
                if ori is not None:
                    entry["orientation"] = list(ori)
            voxels_t0.append(entry)
        voxels_t0_sorted = sorted(voxels_t0, key=lambda d: int(d["voxel_id"]))

        # ─── wrap under the time‐step keys ────────────────────────────────────────
        time_steps = [
            {   "time"  : 0               ,
                "grid"  : grid_t0         ,
                "grains": grains_t0_sorted,
                "voxels": voxels_t0_sorted,
            },
            # …etc…
        ]



        # Assemble final structure with placeholders
        data = {
            **use,                   # expand user-specific dict entries directly
            "software": "",
            "software_version": "",
            "system": "",
            "system_version": "",
            "processor_specifications": "",
            "input_path": "",
            "results_path": "",
            **ig,                    # expand initial geometry dict entries directly
            "global_temperature": 298,
            **job_bc,                 # expand boundary condition dict entries directly
            "phases":phase_list,
            "microstructure":  time_steps  # Time-level storage: time-frame data of voxels and grains
        }

        return data




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
