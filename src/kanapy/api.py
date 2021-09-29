# -*- coding: utf-8 -*-
import json
from kanapy.input_output import particleStatGenerator, RVEcreator, \
    write_abaqus_inp, write_position_weights
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine

class Microstructure:
    '''Define class for synthetic microstructures'''
    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.particle_data = None
        self.RVE_data = None
        self.simulation_data = None
        self.allNodes = None
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
        
    def pack(self, particle_data=None, RVE_data=None, simulation_data=None):
        """ Packs the particles into a simulation box."""
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if simulation_data is None:
            simulation_data = self.simulation_data
        self.particles, self.simbox = \
            packingRoutine(particle_data, RVE_data, simulation_data)
        
    def voxelize(self, particle_data=None, RVE_data=None, particles=None, simbox=None):
        """ Generates the RVE by assigning voxels to grains."""   
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if particles is None:
            particles = self.particles
        if simbox is None:
            simbox = self.simbox
        self.nodeDict, self.elmtDict, self.elmtSetDict = \
            voxelizationRoutine(particle_data, RVE_data, particles, simbox)

    def smoothen(self, nodeDict=None, elmtDict=None, elmtSetDict=None, save_files=False):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        if nodeDict is None:
            nodeDict = self.nodeDict
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
            
        self.allNodes, self.grain_facesDict = \
            smoothingRoutine(nodeDict, elmtDict, elmtSetDict, save_files) 
    
    # the following subroutines are not yet adapted as API
    # futher subroutines for visualization are required
    def abq_output(self):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        write_abaqus_inp()
        
    def neperoutput(ctx, timestep=None):
        """ Writes out particle position and weights files required for tessellation in Neper."""
        write_position_weights(timestep)
