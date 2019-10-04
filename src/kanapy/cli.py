# -*- coding: utf-8 -*-
import os
import click

from kanapy.input_output import particleStatGenerator, write_position_weights
from kanapy.input_output import write_abaqus_inp, write_output_stat
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine

@click.group()
@click.pass_context
def main(ctx):    
    pass    
   
@main.command()
@click.option('--filename', help='Input statistics file name in the current directory.')
@click.pass_context
def statgenerate(ctx, filename: str):    
    """ Generates particle statistics based on the data provided in the input file."""

    if filename == None:
        raise ValueError('Please provide the name of the input file available in the current directory!')
            
    cwd = os.getcwd()
    particleStatGenerator(cwd + '/' + filename)           

        
@main.command()
@click.pass_context
def pack(ctx):
    """ Packs the particles into a simulation box."""
    packingRoutine()


@main.command()
@click.option('--timestep', help='Time step for voxelization.')
@click.pass_context
def voxelize(ctx, timestep: int):
    """ Generates the RVE by assigning voxels to grains.""" 

    if timestep == None:
        raise ValueError('Please provide the timestep value for voxelization!')
    voxelizationRoutine(timestep)


@main.command()
@click.pass_context
def abaqusoutput(ctx):
    """ Writes out the Abaqus (.inp) file for the generated RVE."""    
    write_abaqus_inp()
        
        
@main.command()
@click.pass_context
def outputstats(ctx):
    """ Writes out the particle- and grain diameter attributes for statistical comparison."""
    write_output_stat()
        
                
@main.command()
@click.option('--timestep', help='Time step for which Neper input files will be generated.')
@click.pass_context
def neperoutput(ctx, timestep: int):
    """ Writes out particle position and weights files required for tessellation in Neper."""

    if timestep == None:
        raise ValueError('Please provide an timestep value for generating ouput!')
    write_position_weights(timestep)

                
def start():
    main(obj={})

    
if __name__ == '__main__':
    start()
