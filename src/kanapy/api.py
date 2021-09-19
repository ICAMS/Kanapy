# -*- coding: utf-8 -*-
import json
from kanapy.input_output import particleStatGenerator, RVEcreator
from kanapy.packing import packingRoutine

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
        
    def pack(self):
        packingRoutine(self.particle_data, self.RVE_data, self.simulation_data)

ms = Microstructure()
ms.create_RVE(file='../../examples/ellipsoid_packing/stat_input.json')
ms.pack()