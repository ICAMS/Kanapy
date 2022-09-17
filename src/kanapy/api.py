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
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from kanapy.grains import calcPolygons, get_stats
from kanapy.input_output import export2abaqus
from kanapy.initializations import particleStatGenerator, RVEcreator
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_voxels_3D, plot_ellipsoids_3D, \
    plot_polygons_3D, plot_output_stats


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
        elif descriptor == 'from_voxels':
            pass
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
            particle_data, phases, RVE_data, simulation_data = \
                RVEcreator(des, nsteps=nsteps, save_files=save_files)
            Ngr = particle_data['Number']
            if des == descriptor[0]:
                self.Ngr = Ngr
                self.particle_data = particle_data
                self.phases = phases
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
                    self.phases['Phase name'] = self.phases['Phase name'] + phases['Phase name']
                    self.phases['Phase number'] = self.phases['Phase number'] + phases['Phase number']
                    self.particle_data['Tilt angle'] = self.particle_data['Tilt angle'] + particle_data['Tilt angle']
                elif des['Grain type'] == 'Equiaxed':
                    self.Ngr = self.Ngr + Ngr
                    self.particle_data['Equivalent_diameter'] = self.particle_data['Equivalent_diameter'] + \
                                                                particle_data['Equivalent_diameter']
                    self.particle_data['Number'] = self.particle_data['Number'] + particle_data['Number']
                    self.phases['Phase name'] = self.phases['Phase name'] + phases['Phase name']
                    self.phases['Phase number'] = self.phases['Phase number'] + phases['Phase number']
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
            packingRoutine(particle_data, self.phases, RVE_data, simulation_data,
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

    def generate_grains(self, save_files=False, dual_phase=False):
        """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
        grain volumes and shared grain boundary surface areas info are written out as well."""

        if self.nodes_v is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if self.phases is None:
            raise ValueError('No phases defined.')

        self.grain_facesDict, self.shared_area = calcPolygons(self.RVE_data, self.nodes_v, self.elmtSetDict,
                                                              self.elmtDict, self.Ngr, self.voxels, self.phases,
                                                              self.vox_centerDict,
                                                              dual_phase=dual_phase)  # updates RVE_data
        if self.particle_data is not None:
            self.res_data = \
                get_stats(self.particle_data, self.Ngr, self.RVE_data, self.nphase, dual_phase=dual_phase)

    """
    --------     Plotting methods          --------
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
        plot_voxels_3D(self.voxels, self.voxels_phase, Ngr=self.Ngr,
                       sliced=sliced, dual_phase=dual_phase, cmap=cmap)

    def plot_grains(self, rve=None, cmap='prism', alpha=0.4,
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
        elif type(data) != list:
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
    --------        Output/Export methods        --------
    """

    def output_abq(self, nodes=None, name=None, RVE_data=None,
                   elmtDict=None, elmtSetDict=None, faces=None):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""
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
        if RVE_data is None:
            RVE_data = self.RVE_data
        if name is None:
            cwd = os.getcwd()
            name = cwd + '/kanapy_{0}grains_{1}.inp'.format(len(elmtSetDict), ntag)
            if os.path.exists(name):
                os.remove(name)  # remove old file if it exists
        export2abaqus(nodes, name, RVE_data, elmtSetDict, elmtDict, grain_facesDict=faces)
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

    def write_voxels(self, sname, file=None, path='./', mesh=True, source=None):
        """
        Write voxel structure into JSON file.

        Parameters
        ----------
        sname
        file
        path
        mesh
        source

        Returns
        -------

        """

        import platform
        import getpass
        from datetime import date
        from pkg_resources import get_distribution
        from json import dump

        if path[-1] != '/':
            path += '/'
        if file is None:
            if self.name == 'Microstructure':
                file = path + 'px_{}grains_voxels.json'.format(self.Ngr)
            else:
                file = path + self.name + '_voxels.json'
        # metadata
        today = str(date.today())  # date
        owner = getpass.getuser()  # username
        sys_info = platform.uname()  # system information
        structure = {
            "Info": {
                "Owner": owner,
                "Institution": "ICAMS, Ruhr University Bochum, Germany",
                "Date": today,
                "Description": "Voxels of microstructure",
                "Method": "Synthetic microstructure generator Kanapy",
                "System": {
                    "sysname": sys_info[0],
                    "nodename": sys_info[1],
                    "release": sys_info[2],
                    "version": sys_info[3],
                    "machine": sys_info[4]},
            },
            "Model": {
                "Creator": "kanapy",
                "Version": get_distribution('kanapy').version,
                "Repository": "https://github.com/ICAMS/Kanapy.git",
                "Input": source,
                "Script": sname,
            },
            "Data": {
                "Class" : 'phase_numbers',
                "Type"  : 'int',
                "Shape" : self.voxels.shape,
                "Order" : 'C',
                "Values": [val.item() for val in self.voxels.flatten()],  # item() converts numpy-int64 to python int
                "Geometry" : (self.RVE_data['RVE_sizeX'],
                              self.RVE_data['RVE_sizeY'],
                              self.RVE_data['RVE_sizeZ']),
                "Units": {
                    'Length': self.RVE_data['Units'],
                    },
                "Periodicity": self.RVE_data['Periodic'],
            }
        }
        if mesh:
            nout = []
            for pos in self.nodes_v:
                nout.append([val.item() for val in pos])
            structure['Mesh'] = {
                "Nodes" : {
                    "Class" : 'coordinates',
                    "Type"  : 'float',
                    "Shape" : self.nodes_v.shape,
                    "Values"  : nout,
                },
                "Voxels" : {
                    "Class" : 'node_list',
                    "Type"  : 'int',
                    "Shape" : (len(self.elmtDict.keys()), 8),
                    "Values"  : [val for val in self.elmtDict.values()],
                }
            }
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
                file = path + 'px_{}grains_microstructure.pckl'.format(self.Ngr)
            else:
                file = path + self.name + '_microstructure.pckl'
        with open(path + file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return
