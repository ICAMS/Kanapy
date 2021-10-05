# -*- coding: utf-8 -*-
import json
from kanapy.input_output import particleStatGenerator, RVEcreator, \
    write_abaqus_inp, write_position_weights, extract_volume_sharedGBarea, \
    write_output_stat, plot_output_stats
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_microstructure_3D

class Microstructure:
    '''Define class for synthetic microstructures'''
    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.particle_data = None
        self.RVE_data = None
        self.simulation_data = None
        self.particles = None
        self.simbox = None
        self.allNodes = None
        self.nodeDict = None
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
            
    def create_stats(self, descriptor=None, save_files=False):    
        """ Generates particle statistics based on the data provided in the input file."""
        if descriptor is None:
            descriptor = self.descriptor  
        particleStatGenerator(descriptor, save_files=save_files)
        
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
        self.nodeDict, self.elmtDict, self.elmtSetDict, self.vox_centerDict = \
            voxelizationRoutine(particle_data, RVE_data, particles, simbox, save_files=save_files)

    def smoothen(self, nodeDict=None, elmtDict=None, elmtSetDict=None, save_files=False):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        if nodeDict is None:
            nodeDict = self.nodeDict
            if nodeDict is None:
                raise ValueError('No nodeDict in smoothen. Run voxelize first.')
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        self.allNodes, self.grain_facesDict = \
            smoothingRoutine(nodeDict, elmtDict, elmtSetDict, save_files=save_files)
            
    def plot_3D(self, sliced=True, dual_phase=False, cmap='prism', test=False):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.elmtSetDict is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        plot_microstructure_3D(ms=self,sliced=sliced, dual_phase=dual_phase, \
                               cmap=cmap, test=test)

    def output_stats(self, nodeDict=None, elmtDict=None, elmtSetDict=None, \
                     particle_data=None, RVE_data=None, simulation_data=None, save_files=False):
        """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
        grain volumes and shared grain boundary surface areas info are written out as well."""
        if nodeDict is None:
            nodeDict = self.nodeDict
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
            
        if nodeDict is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if particle_data is None:
            raise ValueError('No particles created yet. Run create_RVE, pack and voxelize first.')
            
        self.res_data = write_output_stat(nodeDict, elmtDict, elmtSetDict, particle_data, RVE_data, \
                          simulation_data, save_files=save_files)
        self.gv_sorted_values, self.shared_area = \
            extract_volume_sharedGBarea(nodeDict, elmtDict, elmtSetDict, RVE_data, save_files=save_files)
        
    def plot_stats(self, data=None, save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical comparison."""   
        if data is None:
            data = self.res_data
        plot_output_stats(data, save_files=save_files)

    # the following subroutines are not yet adapted as API
    # futher subroutines for visualization are required
    def output_abq(self):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        write_abaqus_inp()

    def output_neper(self, timestep=None):
        """ Writes out particle position and weights files required for tessellation in Neper."""
        write_position_weights(timestep)
