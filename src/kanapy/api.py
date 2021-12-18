# -*- coding: utf-8 -*-
import os
import json
from kanapy.input_output import particleStatGenerator, RVEcreator, \
    extract_volume_sharedGBarea, write_output_stat, plot_output_stats,\
    export2abaqus
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_microstructure_3D, plot_ellipsoids

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
            self.descriptor = descriptor
            if file is not None:
                print('WARNING: Input parameter (descriptor) and file are given. Only descriptor will be used.')
    
    def create_RVE(self, descriptor=None, save_files=False):    
        """ Creates RVE based on the data provided in the input file."""
        if descriptor is None:
            descriptor = self.descriptor  
        self.particle_data, self.RVE_data, self.simulation_data = \
            RVEcreator(descriptor, save_files=save_files)
            
    def create_stats(self, descriptor=None, gs_data=None, ar_data=None, save_files=False):    
        """ Generates particle statistics based on the data provided in the input file."""
        if descriptor is None:
            descriptor = self.descriptor  
        particleStatGenerator(descriptor, gs_data=gs_data, ar_data=ar_data, save_files=save_files)
        
    def pack(self, particle_data=None, RVE_data=None, simulation_data=None, 
             save_files=False):
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
            packingRoutine(particle_data, RVE_data, simulation_data, save_files=save_files)
    
    def plot_ellipsoids(self, cmap='prism', test=False):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        plot_ellipsoids(self.particles, cmap=cmap, test=test)
        
    def voxelize(self, particle_data=None, RVE_data=None, particles=None, 
                 simbox=None, save_files=False):
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
        self.nodes_v, self.elmtDict, self.elmtSetDict, self.vox_centerDict, self.voxels = \
            voxelizationRoutine(particle_data, RVE_data, particles, simbox, save_files=save_files)

    def smoothen(self, nodes_v=None, elmtDict=None, elmtSetDict=None, save_files=False):
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
            smoothingRoutine(nodes_v, elmtDict, elmtSetDict, save_files=save_files)
        
            
    def plot_3D(self, sliced=True, dual_phase=False, cmap='prism', test=False):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.voxels is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        plot_microstructure_3D(self.voxels, Ngr=self.particle_data['Number'], sliced=sliced, dual_phase=dual_phase, \
                               cmap=cmap, test=test)

    def output_stats(self, nodes_v=None, elmtDict=None, elmtSetDict=None, \
                     particle_data=None, RVE_data=None, simulation_data=None, save_files=False):
        """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
        grain volumes and shared grain boundary surface areas info are written out as well."""
        if nodes_v is None:
            nodes_v = self.nodes_v
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if simulation_data is None:
            simulation_data = self.simulation_data
            
        if nodes_v is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if particle_data is None:
            raise ValueError('No particles created yet. Run create_RVE, pack and voxelize first.')
            
        self.res_data = write_output_stat(nodes_v, elmtDict, elmtSetDict, particle_data, RVE_data, \
                          simulation_data, save_files=save_files)
        self.gv_sorted_values, self.shared_area = \
            extract_volume_sharedGBarea(elmtDict, elmtSetDict, RVE_data, save_files=save_files)
        
    def plot_stats(self, data=None, gs_data=None, gs_param=None, 
                          ar_data=None, ar_param=None, save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical comparison."""   
        if data is None:
            data = self.res_data
        if data is None:
            raise ValueError('No microstructure data created yet. Run output_stats first.')
        plot_output_stats(data, gs_data=gs_data, gs_param=gs_param,
                          ar_data=ar_data, ar_param=ar_param,
                          save_files=save_files)

    def output_abq(self, nodes=None, name=None, simulation_data=None, elmtDict=None, elmtSetDict=None, faces=None):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        #write_abaqus_inp()
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
        elif nodes=='smooth' or nodes=='s':
            if self.nodes_s is not None and self.grain_facesDict is not None:
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            else:
                raise ValueError('No information about smoothed microstructure. Run smoothen first.')
        elif nodes=='voxels' or nodes=='v':
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
            name = cwd + '/kanapy_{0}grains_{1}.inp'.format(len(elmtSetDict),ntag)
            if os.path.exists(name):
                os.remove(name)                  # remove old file if it exists
        export2abaqus(nodes, name, simulation_data, elmtSetDict, elmtDict, grain_facesDict=faces)

    #def output_neper(self, timestep=None):
    def output_neper(self):
        """ Writes out particle position and weights files required for tessellation in Neper."""
        #write_position_weights(timestep)
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        print('')
        print('Writing position and weights files for NEPER', end="")
        par_dict = dict()
        
        for pa in self.particles:
            x, y, z = pa.x, pa.y, pa.z
            a, b, c = pa.a, pa.b, pa.c
            par_dict[pa] = [x, y, z, a]
            
        with open('sphere_positions.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0} {1} {2}\n'.format(value[0], value[1], value[2]))
    
        with open('sphere_weights.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0}\n'.format(value[3]))
        print('---->DONE!\n') 
