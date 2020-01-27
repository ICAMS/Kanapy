# -*- coding: utf-8 -*-
import os, sys
import shutil, json
import click

from kanapy.util import ROOT_DIR 
from kanapy.input_output import particleStatGenerator, write_position_weights
from kanapy.input_output import write_abaqus_inp, write_output_stat
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.analyze_texture import textureReduction

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


@main.command()
@click.pass_context
def setupTexture(ctx):    
    """ Stores the user provided MATLAB & MTEX paths for texture analysis."""
    setPaths()                    

    
def setPaths():
    ''' Requests user input for MATLAB & MTEX installation paths'''
    
    # For MATLAB executable
    status1 = input('Is MATLAB installed in this system (yes/no): ')
    
    if status1 == 'yes' or status1 == 'y' or status1 == 'Y' or status1 == 'YES':
        print('Searching your system for MATLAB ...')
        MATLAB = shutil.which("matlab")        

        if MATLAB:
            decision1 = input('Found MATLAB in {0}, continue (yes/no): '.format(MATLAB))
            
            if decision1 == 'yes' or decision1 == 'y' or decision1 == 'Y' or decision1 == 'YES':
                userpath1 = MATLAB
            elif decision1 == 'no' or decision1 == 'n' or decision1 == 'N' or decision1 == 'NO':
                userpath1 = input('Please provide the path to MATLAB executable: ')
            else:
                click.echo('Invalid entry!, Run: kanapy setuptexture again', err=True)
                sys.exit(0) 
                            
        elif not MATLAB:
            print('No MATLAB executable found!')            
            userpath1 = input('Please provide the path to MATLAB executable: ')

        # For MTEX installation path
        print('\n')
        status2 = input('Is MTEX installed in this system (yes/no): ')

        if status2 == 'yes' or status2 == 'y' or status2 == 'Y' or status2 == 'YES':                    
            userpath2 = input('Please provide the path to MTEX installation: ')                                         
        elif status2 == 'no' or status2 == 'n' or status2 == 'N' or status2 == 'NO':
            print("Kanapy's texture analysis code requires MTEX. Please install it from: https://mtex-toolbox.github.io/download.")
            userpath2 = False
        else:
            click.echo('Invalid entry!, Run: kanapy setuptexture again', err=True)
            sys.exit(0)  
        
                     
    elif status1 == 'no' or status1 == 'n' or status1 == 'N' or status1 == 'NO':
        print("Kanapy's texture analysis code requires MATLAB. Please install it.")
        userpath1 = False
    else:
        click.echo('Invalid entry!, Run: kanapy setuptexture again', err=True)
        sys.exit(0)        
        
    # Create a file in the 'src/kanapy' folder that stores the paths
    if userpath1 and userpath2:        
        
        pathDict = {'MATLABpath': '{}'.format(userpath1), 'MTEXpath': '{}'.format(userpath2)}                
        path_path = ROOT_DIR+'/PATHS.json'
        
        if os.path.exists(path_path):
            os.remove(path_path)

        with open(path_path,'w') as outfile:
            json.dump(pathDict, outfile, indent=2)                
        
        print('\n')
        print('Kanapy is now configured for texture analysis!\n')


@main.command()
@click.option('--ebsd', default=None, help='EBSD (.mat) file name located in the current directory.')
@click.option('--grains', default=None, help='Grains (.mat) file name located in the current directory.')
@click.option('--kernel', default=None, help='Optimum kernel shape factor as float (in radians).')
@click.option('--fit_mad', default='no', help='Fit Misorientation Angle Distribution (yes/no).')
@click.pass_context
def reducetexture(ctx, ebsd: str, grains: str, kernel: float, fit_mad: bool):
    
    if ebsd==None:
        click.echo('Please provide some EBSD inputs for texture reduction!') 
        click.echo('For more info, run: kanapy reducetexture --help\n', err=True)
        sys.exit(0)
    else:
        cwd = os.getcwd()
        arg_dict = {}           
        if ebsd != None:
            if not os.path.exists(cwd + '/{}'.format(ebsd)):
                click.echo('Mentioned file: {} does not exist in the current working directory!\n'.format(ebsd), err=True)
                sys.exit(0)
            else:
                arg_dict['ebsdMatFile'] = cwd + '/{}'.format(ebsd)

        if grains != None:
            if not os.path.exists(cwd + '/{}'.format(grains)):
                click.echo('Mentioned file: {} does not exist in the current working directory!\n'.format(grains), err=True)
                sys.exit(0)
            else:        
                arg_dict['grainsMatFile'] = cwd + '/{}'.format(grains)
                
        if kernel != None:
            arg_dict['kernelShape'] = kernel        
            
        if fit_mad == 'yes' or fit_mad == 'y' or fit_mad == 'Y' or fit_mad == 'YES': 
            arg_dict['MisAngDist'] = fit_mad            
            textureReduction(arg_dict)      
                  
        elif fit_mad == 'no' or fit_mad == 'n' or fit_mad == 'N' or fit_mad == 'NO': 
            textureReduction(arg_dict)
        else:
            click.echo('Invalid entry! Run: kanapy reducetexture --help\n', err=True)
            sys.exit(0)
        
    
def start():
    main(obj={})

    
if __name__ == '__main__':
    start()
