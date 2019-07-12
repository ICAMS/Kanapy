import os
import sys
sys.path.append('../..')

import kanapy
from src.kanapy.input_output import particleStatGenerator 
from src.kanapy.packing import packingRoutine
from src.kanapy.voxelization import voxelizationRoutine
from src.kanapy.input_output import write_abaqus_inp, write_output_stat

def main():
    """
    The complete process consists of 4 stages: 

    * Particle data generation based on user defined statitics.
    * Particle packing routine.
    * RVE voxelization routine.
    * Writing output files.

    Individual stages can be run by commenting out the remaining stages.  
    """

    inputFile = os.getcwd() + '/stat_input.txt'
    particleStatGenerator(inputFile)                    # Generate data for particle simulation
    packingRoutine()                                    # Particle packing simulation    
    voxelizationRoutine(800)                            # RVE voxelization (Meshing)    
    write_abaqus_inp()                                  # Write out Abaqus input (.inp) file
    write_output_stat()                                 # Compare input and output statistics
    return


if __name__ == '__main__':
    main()
