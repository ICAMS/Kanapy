# -*- coding: utf-8 -*-
import json
from kanapy.input_output import particleStatGenerator, RVEcreator, write_abaqus_inp 
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine

class Microstructure:
    '''Define class for synthetic microstructures'''
    def __init__(self, name='Microstructure'):
        self.name = name
        
    def pname(self):
        print(self.name)
    
    def create_RVE(self, descriptor=None, file=None):    
        """ Creates RVE based on the data provided in the input file."""
        
        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an input file name')
                 
            # Open the user input statistics file and read the data
            try:
                with open(file) as json_file:  
                     descriptor = json.load(json_file)
            except:
                raise FileNotFoundError("File: '{}' does not exist in the current working directory!\n".format(file))

        self.particle_data, self.RVE_data, self.simulation_data = RVEcreator(descriptor)
            
    def create_stats(self, descriptor=None, file=None):    
        """ Generates particle statistics based on the data provided in the input file."""
                    
        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an input file name')
            # Open the user input statistics file and read the data
            try:
                with open(file) as json_file:  
                     descriptor = json.load(json_file)
            except:
                raise FileNotFoundError("File: '{}' does not exist in the current working directory!\n".format(file))  
        particleStatGenerator(descriptor)
        
    def pack(self, pd=None, rd=None, sd=None):
        """ Packs the particles into a simulation box."""
        if pd is None:
            pd = self.particle_data
        if rd is None:
            rd = self.RVE_data
        if sd is None:
            sd = self.simulation_data
        self.particles, self.simbox = packingRoutine(pd, rd, sd)
        
    def voxelize(self, pd=None, rd=None, kana=None, sb=None):
        """ Generates the RVE by assigning voxels to grains."""   
        if pd is None:
            pd = self.particle_data
        if rd is None:
            rd = self.RVE_data
        if kana is None:
            kana = self.particles
        if sb is None:
            sb = self.simbox
        self.nodeDict, self.elmtDict, self.elmtSetDict = voxelizationRoutine(pd, rd, kana, sb)

    def smoothen(self):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        smoothingRoutine()    
            
    def abq_output(self):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        write_abaqus_inp()