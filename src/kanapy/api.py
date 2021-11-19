# -*- coding: utf-8 -*-
import os
import json
from kanapy.input_output import particleStatGenerator, RVEcreator, \
    write_abaqus_inp, write_position_weights, extract_volume_sharedGBarea, \
    write_output_stat, plot_output_stats
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
    
    def plot_ellipsoids(self, cmap='prism', test=False):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        plot_ellipsoids(ms=self, cmap=cmap, test=test)
        
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
    def output_abq(self, simulation_data=None, nodeDict=None, elmtDict=None, elmtSetDict=None,):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        #write_abaqus_inp()
        if simulation_data is None:
            simulation_data = self.simulation_data
        if nodeDict is None:
            nodeDict = self.nodeDict
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
            
        if nodeDict is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if simulation_data is None:
            raise ValueError('No simulation data exists. Run create_RVE, pack and voxelize first.')
 
        print('')
        print('Writing ABAQUS (.inp) file', end="")

        cwd = os.getcwd()

        # Factor used to generate nodal cordinates in 'mm' or 'um' scale
        if simulation_data['Output units'] == 'mm':
            scale = 'mm'
            divideBy = 1000
        elif simulation_data['Output units'] == 'um':
            scale = 'um'
            divideBy = 1
                    
        abaqus_file = cwd + '/kanapy_{0}grains.inp'.format(len(elmtSetDict))
        if os.path.exists(abaqus_file):
            os.remove(abaqus_file)                  # remove old file if it exists

        with open(abaqus_file, 'w') as f:
            f.write('** Input file generated by kanapy\n')
            f.write('** Nodal coordinates scale in {0}\n'.format(scale))
            f.write('*HEADING\n')
            f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
            f.write('**\n')
            f.write('** PARTS\n')
            f.write('**\n')
            f.write('*Part, name=PART-1\n')
            f.write('*Node\n')

            # Create nodes
            for k, v in nodeDict.items():
                # Write out coordinates in 'mm' or 'um'
                f.write('{0}, {1}, {2}, {3}\n'.format(k, v[0]/divideBy, v[1]/divideBy, v[2]/divideBy))

            # Create Elements
            f.write('*ELEMENT, TYPE=C3D8\n')
            for k, v in elmtDict.items():
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                    k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))

            # Create element sets
            for k, v in elmtSetDict.items():
                f.write('*ELSET, ELSET=Grain{0}_set\n'.format(k))
                for enum, el in enumerate(v, 1):
                    if enum % 16 != 0:
                        if enum == len(v):
                            f.write('%d\n' % el)
                        else:
                            f.write('%d, ' % el)
                    else:
                        if enum == len(v):
                            f.write('%d\n' % el)
                        else:
                            f.write('%d\n' % el)

            # Create sections
            for k, v in elmtSetDict.items():
                f.write(
                    '*Solid Section, elset=Grain{0}_set, material=Grain{1}_mat\n'.format(k, k))
            f.write('*End Part\n')
            f.write('**\n')
            f.write('**\n')
            f.write('** ASSEMBLY\n')
            f.write('**\n')
            f.write('*Assembly, name=Assembly\n')
            f.write('**\n')
            f.write('*Instance, name=PART-1-1, part=PART-1\n')
            f.write('*End Instance\n')
            f.write('*End Assembly\n')
        print('---->DONE!\n') 


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
