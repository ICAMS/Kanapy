import os
import sys

import kanapy
from kanapy.packing import packingRoutine
from kanapy.input_output import write_position_weights
from kanapy.input_output import particleStatGenerator
from kanapy.voxelization import voxelizationRoutine
from kanapy.input_output import write_abaqus_inp, write_output_stat


def main():
    """
    The complete process consists of 3 stages: 

    * Particle data generation based on user defined statitics.
    * Particle packing routine.
    * Writing output files.

    Individual stages can be run by commenting out the other stages.  
    """

    inputFile = os.getcwd() + '/stat_input.txt'    
    particleStatGenerator(inputFile)                    # Generate data for particle simulation
    packingRoutine()                                    # Particle packing simulation    
    write_position_weights(800)                         # Write out position and weight files for tessellation.    
    
    voxelizationRoutine(800)                            # RVE voxelization (Meshing)    
    write_abaqus_inp()                                  # Write out Abaqus input (.inp) file
    write_output_stat()                                 # Compare input and output statistics    
    return


if __name__ == '__main__':
    main()
