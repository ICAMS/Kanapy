import os
import sys

import kanapy
from kanapy.packing import packingRoutine
from kanapy.input_output import write_position_weights
from kanapy.input_output import particleStatGenerator


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
    return


if __name__ == '__main__':
    main()
