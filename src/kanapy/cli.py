# -*- coding: utf-8 -*-
import csv
import itertools
import os
import re
import sys
import shutil
import json
import click
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean

from kanapy.grains import l1_error_est
from kanapy.util import MAIN_DIR, WORK_DIR
from kanapy.input_output import read_dump, export2abaqus
from kanapy.initializations import particleStatGenerator, RVEcreator
from kanapy.plotting import plot_output_stats
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.textures import textureReduction
from kanapy.smoothingGB import smoothingRoutine
from numpy import asarray

@click.group()
@click.pass_context
def main(ctx):    
    pass    


@main.command(name='autoComplete')
@click.pass_context
def autocomplete(ctx):    
    """ Kanapy bash auto completion.""" 
       
    click.echo('')  
    os.system("echo '# For KANAPY bash autocompletion' >> ~/.bashrc")
    os.system("echo '. {}' >> ~/.bashrc".format(MAIN_DIR+'/src/kanapy/kanapy-complete.sh'))  


@main.command(name='runTests')
@click.option('-no_texture', default=False)
@click.pass_context
def tests(ctx, no_texture: bool):    
    """ Runs unittests built within kanapy."""  
    shutil.rmtree(WORK_DIR + '/tests', ignore_errors=True)
    os.makedirs(WORK_DIR + '/tests')
    os.system(f'cp {MAIN_DIR}/tests/unitTest_ODF_MDF_reconstruction.m {WORK_DIR}/tests')
    os.system(f'cp {MAIN_DIR}/tests/shared_surfaceArea.csv {WORK_DIR}/tests')
    click.echo('')
    if no_texture:
        t1 = "{0}/tests/test_collide_detect_react.py".format(MAIN_DIR)
        t2 = "{0}/tests/test_entities.py".format(MAIN_DIR)
        t3 = "{0}/tests/test_entities.py".format(MAIN_DIR)
        t4 = "{0}/tests/test_entities.py".format(MAIN_DIR)
        t5 = "{0}/tests/test_entities.py".format(MAIN_DIR)
        os.system(f"pytest {t1} {t2} {t3} {t4} {t5} -v")
    else:
        os.system("pytest {0}/tests/ -v".format(MAIN_DIR))
    click.echo('')    
        
    
@main.command(name='genDocs')
@click.pass_context
def docs(ctx):    
    """ Generates a HTML-based reference documentation."""
    
    click.echo('')
    os.system("make -C {0}/docs/ clean && make -C {0}/docs/ html".format(MAIN_DIR))      
    click.echo('')
    click.echo("The HTML documentation can be found at '/path/to/your/kanapy/docs/index.html'")
    click.echo('')
    
       
@main.command(name='genStats')
@click.option('-f', default=None, help='Input statistics file name in the current directory.')
@click.pass_context
def createStats(ctx, f: str):    
    """ Generates particle statistics based on the data provided in the input file."""
                
    if f == None:
        click.echo('')
        click.echo('Please provide the name of the input file available in the current directory', err=True)
        click.echo('For more info. run: kanapy statgenerate --help\n', err=True)
        sys.exit(0)         
    else:
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/{}'.format(f)):
            click.echo('')
            click.echo("Mentioned file: '{}' does not exist in the current working directory!\n".format(f), err=True)
            sys.exit(0)     
        # Open the user input statistics file and read the data
        try:                
            with open(cwd + '/' + f) as json_file:  
                 stats_dict = json.load(json_file)                   
                 
        except FileNotFoundError:
            print('Input file not found, make sure "stat_input.json" file is present in the working directory!')
            raise FileNotFoundError  
        particleStatGenerator(stats_dict, save_files=True)           


@main.command(name='genRVE')
@click.option('-f', default=None, help='Input statistics file name in the current directory.')
@click.pass_context
def createRVE(ctx, f: str):    
    """ Creates RVE based on the data provided in the input file."""
                
    if f == None:
        click.echo('')
        click.echo('Please provide the name of the input file available in the current directory', err=True)
        click.echo('For more info. run: kanapy statgenerate --help\n', err=True)
        sys.exit(0)         
    else:
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/{}'.format(f)):
            click.echo('')
            click.echo("Mentioned file: '{}' does not exist in the current working directory!\n".format(f), err=True)
            sys.exit(0)
        # Open the user input statistics file and read the data
        try:
            with open(cwd + '/' + f) as json_file:  
                stats_dict = json.load(json_file)                               
                 
        except FileNotFoundError:
            print('Input file not found, make sure "stat_input.json" file is present in the working directory!')
            raise FileNotFoundError
        RVEcreator(stats_dict, save_files=True)   
                

@main.command(name='readGrains')
@click.option('-f', default=None, help='Input file name in the current directory.')
@click.option('-periodic', default='True', help='RVE periodicity status.')
@click.option('-units', default='mm', help='Output unit format.')
@click.pass_context
def readGrains(ctx, f: str, periodic: str, units: str):    
    ''' Generates particles based on the grain data provided in the input file.'''
    
    if f == None:
        click.echo('')
        click.echo('Please provide the name of the input file available in the current directory', err=True)
        click.echo('For more info. run: kanapy readGrains --help\n', err=True)
        sys.exit(0)    
    elif ((periodic!='True') and (periodic!='False')):
        click.echo('')
        click.echo('Invalid entry!, Run: kanapy readGrains again', err=True)
        click.echo('For more info. run: kanapy readGrains --help\n', err=True)
        sys.exit(0)                
    elif ((units!='mm') and (units!='um')):
        click.echo('')
        click.echo('Invalid entry!, Run: kanapy readGrains again', err=True)
        click.echo('For more info. run: kanapy readGrains --help\n', err=True)
        sys.exit(0)                            
    else:
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/{}'.format(f)):
            click.echo('')
            click.echo("Mentioned file: '{}' does not exist in the current working directory!\n".format(f), err=True)
            sys.exit(0)          
        particleCreator(cwd + '/' + f, periodic=periodic, units=units)         
        
        
@main.command()
@click.pass_context
def pack(ctx):
    """ Packs the particles into a simulation box."""
    try:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files
    
        try:
            # Load the dictionaries from json files
            with open(json_dir + '/particle_data.json') as json_file:
                particle_data = json.load(json_file)
    
            with open(json_dir + '/RVE_data.json') as json_file:
                RVE_data = json.load(json_file)
    
            with open(json_dir + '/simulation_data.json') as json_file:
                simulation_data = json.load(json_file)
    
        except:
            raise FileNotFoundError('Json files not found, make sure "RVE creator" command is executed first!')
        phases = {
            'Phase name': ['Simulanium'] * particle_data['Number'],
            'Phase number': [0] * particle_data['Number'],
        }
        packingRoutine(particle_data, phases, RVE_data, simulation_data, save_files=True)
    except KeyboardInterrupt:
        sys.exit(0)

@main.command()
@click.pass_context
def voxelize(ctx):
    """ Generates the RVE by assigning voxels to grains."""
    try:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files
    
        try:
            with open(json_dir + '/RVE_data.json') as json_file:
                RVE_data = json.load(json_file)
    
            with open(json_dir + '/particle_data.json') as json_file:  
                particle_data = json.load(json_file)
            
        except FileNotFoundError:
            raise FileNotFoundError('Json file not found, make sure "RVE_data.json" file exists!')
        
        # Read the required dump file
        if particle_data['Type'] == 'Equiaxed':
            #filename = cwd + '/dump_files/particle.{0}.dump'.format(800)    
            filename = cwd + '/dump_files/particle.{0}.dump'.format(700)        
        else:
            filename = cwd + '/dump_files/particle.{0}.dump'.format(500) 
    
        sim_box, Ellipsoids = read_dump(filename)  
        voxelizationRoutine(particle_data, RVE_data, Ellipsoids, sim_box, save_files=True)
    except KeyboardInterrupt:
        sys.exit(0)


@main.command()
@click.pass_context
def smoothen(ctx):
    """ Generates smoothed grain boundary from a voxelated mesh."""    
    try:
        print('')
        print('Starting Grain boundary smoothing')
            
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'
        
        try:                
            with open(json_dir + '/nodes_v.csv', 'r') as f:
                hh = f.read()
            hx = hh.split('\n')
            hs = []
            for hy in hx[0:-1]:
                hs.append(hy.split(', '))
            nodes_v = asarray(hs, dtype=float)
                    
            with open(json_dir + '/elmtDict.json') as json_file:
                elmtDict = {int(k):v for k,v in json.load(json_file).items()}

            with open(json_dir + '/elmtSetDict.json') as json_file:    
                elmtSetDict = {int(k):v for k,v in json.load(json_file).items()}
            
        except FileNotFoundError:
            print('Json files not found, make sure "nodes_v.json", "elmtDict.json" and "elmtSetDict.json" files exist!')
            raise FileNotFoundError

        smoothingRoutine(nodes_v, elmtDict, elmtSetDict, save_files=True)    
    except KeyboardInterrupt:
        sys.exit(0)
    
    return  

@main.command(name='abaqusOutput')
@click.pass_context
def abaqusoutput(ctx):
    """ Writes out the Abaqus (.inp) file for the voxelized RVE."""
    try:
        print('\nStarting Abaqus export for voxelized structure')
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files   
            
        try:
            with open(json_dir + '/simulation_data.json') as json_file:  
                simulation_data = json.load(json_file)     
        
            with open(json_dir + '/nodes_v.csv', 'r') as f:
                hh = f.read()
            hx = hh.split('\n')
            hs = []
            for hy in hx[0:-1]:
                hs.append(hy.split(', '))
            nodes_v = asarray(hs, dtype=float)
    
            with open(json_dir + '/elmtDict.json') as json_file:
                elmtDict = json.load(json_file)
    
            with open(json_dir + '/elmtSetDict.json') as json_file:
                elmtSetDict = json.load(json_file)
    
        except FileNotFoundError:
            raise FileNotFoundError('Json file not found, make sure "kanapy voxelize" command is executed first!')

        name = cwd + '/kanapy_{0}grains_voxels.inp'.format(len(elmtSetDict))
        if os.path.exists(name):
            os.remove(name)                  # remove old file if it exists
        export2abaqus(nodes_v, name, simulation_data, elmtSetDict, elmtDict, grain_facesDict=None)
    except KeyboardInterrupt:
        sys.exit(0)
        
@main.command(name='abaqusOutput-smooth')
@click.pass_context
def abaqusoutput_smooth(ctx):
    """ Writes out the Abaqus (.inp) file for the smoothened RVE."""
    try:
        print('\nStarting Abaqus export for smoothened structure')
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files   
            
        try:
            with open(json_dir + '/simulation_data.json') as json_file:  
                simulation_data = json.load(json_file)     
        
            with open(json_dir + '/nodes_s.csv', 'r') as f:
                hh = f.read()
            hx = hh.split('\n')
            hs = []
            for hy in hx[0:-1]:
                hs.append(hy.split(', '))
            nodes_v = asarray(hs, dtype=float)
    
            with open(json_dir + '/elmtDict.json') as json_file:
                elmtDict = json.load(json_file)
    
            with open(json_dir + '/elmtSetDict.json') as json_file:
                elmtSetDict = json.load(json_file)
                
            with open(json_dir + '/grain_facesDict.json') as json_file:
                grain_facesDict = json.load(json_file)
    
        except FileNotFoundError:
            raise FileNotFoundError('Json file not found, make sure "kanapy smoothen" command is executed first!')

        name = cwd + '/kanapy_{0}grains_smooth.inp'.format(len(elmtSetDict))
        if os.path.exists(name):
            os.remove(name)                  # remove old file if it exists
        export2abaqus(nodes_v, name, simulation_data, elmtSetDict, elmtDict, grain_facesDict=grain_facesDict)
    except KeyboardInterrupt:
        sys.exit(0)
        
@main.command(name='outputStats')
@click.pass_context
def outputstats(ctx):
    """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
    grain volumes and shared grain boundary surface areas info are written out as well.
    
    .. note:: Particle information is read from (.json) file generated by :meth:`kanapy.input_output.particleStatGenerator`.
              RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.
    """
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files
    
    try:
        with open(json_dir + '/nodes_v.csv', 'r') as f:
            hh = f.read()
        hx = hh.split('\n')
        hs = []
        for hy in hx[0:-1]:
            hs.append(hy.split(', '))
        nodes_v = asarray(hs, dtype=float)

        with open(json_dir + '/elmtDict.json') as json_file:
            inpDict = json.load(json_file)
            elmtDict =dict([int(a), x] for a, x in inpDict.items())

        with open(json_dir + '/elmtSetDict.json') as json_file:
            inpDict = json.load(json_file)
            elmtSetDict = dict([int(a), x] for a, x in inpDict.items())

        with open(json_dir + '/particle_data.json') as json_file:  
            particle_data = json.load(json_file)
        
        with open(json_dir + '/RVE_data.json') as json_file:  
            RVE_data = json.load(json_file)

        with open(json_dir + '/simulation_data.json') as json_file:  
            simulation_data = json.load(json_file)    
          
    except FileNotFoundError:
        print('Json file not found, make sure "Input statistics, Packing, & Voxelization" commands are executed first!')
        raise FileNotFoundError

    write_output_stat(nodes_v, elmtDict, elmtSetDict, particle_data, RVE_data, \
                      simulation_data, save_files=True)
    extract_volume_sharedGBarea(elmtDict, elmtSetDict, RVE_data, save_files=True)


@main.command(name='plotStats')
@click.pass_context
def plotstats(ctx):
    """ Plots the particle- and grain diameter attributes for statistical comparison.
    
    .. note:: Particle information is read from (.json) file generated by :meth:`kanapy.input_output.particleStatGenerator`.
    """   
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    try:
        with open(json_dir + '/output_statistics.json') as json_file:
            data_dict = json.load(json_file) 
          
    except FileNotFoundError:
        print('Json file not found, make sure "Input statistics, Packing, Voxelization & Output Statistics" commands are executed first!')
        raise FileNotFoundError
    plot_output_stats(data_dict, save_files=True)

                
@main.command(name='neperOutput')
@click.option('-timestep', help='Time step for which Neper input files will be generated.')
@click.pass_context
def neperoutput(ctx, timestep: int):
    """ Writes out particle position and weights files required for tessellation in Neper."""

    if timestep == None:
        click.echo('')    
        click.echo('Please provide the timestep value for generating ouput!', err=True)
        click.echo('For more info. run: kanapy neperoutput --help\n', err=True)
        sys.exit(0)                
    write_position_weights(timestep)


@main.command(name='setupTexture')
@click.pass_context
def setupTexture(ctx):    
    """ Stores the user provided MATLAB & MTEX paths for texture analysis."""
    setPaths()                    


def chkVersion(matlab):
    ''' Read the version of Matlab'''
    output = os.popen('{} -r quit -nojvm | grep "R20[0-9][0-9][ab]"'.format(matlab)).read()     
        
    try:                                  # Find the matlab version available in the system
        version = output.split()[0]
        version = int(version[1:-1])
    except:                               # Set NONE if MATLAB installation is corrupt
        version == None    
    return version
    
        
def setPaths():
    ''' Requests user input for MATLAB & MTEX installation paths'''
    if not os.path.exists(WORK_DIR):
        raise FileNotFoundError('Package not properly installed, working directory is missing.')
    with open(WORK_DIR + '/PATHS.json') as json_file:
        path_dict = json.load(json_file)
        
    # For MATLAB executable
    click.echo('')
    status1 = input('Is MATLAB installed in this system (yes/no): ')
    
    if status1 == 'yes' or status1 == 'y' or status1 == 'Y' or status1 == 'YES':
        click.echo('Searching your system for MATLAB ...')
        MATLAB = shutil.which("matlab")        

        if MATLAB:
            decision1 = input('Found MATLAB in {0}, continue (yes/no): '.format(MATLAB))
            
            if decision1 == 'yes' or decision1 == 'y' or decision1 == 'Y' or decision1 == 'YES':                

                version = chkVersion(MATLAB)        # Get the MATLAB version
                if version is None:
                    click.echo('')
                    click.echo('MATLAB installation: {} is corrupted!\n'.format(MATLAB), err=True)
                    sys.exit(0)
                elif version < 2015:
                    click.echo('')
                    click.echo('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above\n', err=True)
                    sys.exit(0)
                else:
                    userpath1 = MATLAB

            elif decision1 == 'no' or decision1 == 'n' or decision1 == 'N' or decision1 == 'NO':
                userinput = input('Please provide the path to MATLAB executable: ')
                
                version = chkVersion(userinput)
                if version is None:
                    click.echo('')
                    click.echo('MATLAB installation: {} is corrupted!\n'.format(userinput), err=True)
                    sys.exit(0)
                elif version < 2015:
                    click.echo('')
                    click.echo('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above\n', err=True)
                    sys.exit(0)
                else:
                    userpath1 = userinput
                                    
            else:
                click.echo('Invalid entry!, Run: kanapy setuptexture again', err=True)
                sys.exit(0) 
                            
        elif not MATLAB:
            print('No MATLAB executable found!')            
            userinput = input('Please provide the path to MATLAB executable: ')
            
            version = chkVersion(userinput)
            if version is None:
                click.echo('')
                click.echo('MATLAB installation: {} is corrupted!\n'.format(userinput), err=True)
                sys.exit(0)
            elif version < 2015:
                click.echo('')
                click.echo('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above\n', err=True)
                sys.exit(0)
            else:
                userpath1 = userinput
                     
    elif status1 == 'no' or status1 == 'n' or status1 == 'N' or status1 == 'NO':
        click.echo("Kanapy's texture analysis code requires MATLAB. Please install it.")
        click.echo('')
        userpath1 = False
    else:
        click.echo('Invalid entry!, Run: kanapy setuptexture again', err=True)
        sys.exit(0)        
        
    # Create a file in the 'src/kanapy' folder that stores the paths
    if userpath1:        
        
        path_dict['MATLABpath'] = userpath1
        path_path = WORK_DIR + '/PATHS.json'
        
        if os.path.exists(path_path):
            os.remove(path_path)

        with open(path_path,'w') as outfile:
            json.dump(path_dict, outfile, indent=2)                
        
        os.chdir('{}extern/engines/python'.format(userpath1[0:-10])) # remove bin/matlab from matlab path
        os.system('python -m pip install .')
        path = os.path.abspath(__file__)[0:-7] # remove /cli.py from kanapy path
        os.chdir(path)
        os.system('python init_engine.py')
        click.echo('')
        click.echo('Kanapy is now configured for texture analysis!\n')
        # store paths in Python API?


@main.command(name='reduceODF')
@click.option('-ebsd', default=None, help='EBSD (.mat) file name located in the current directory.')
@click.option('-grains', default=None, help='Grains (.mat) file name located in the current directory.')
@click.option('-kernel', default=None, help='Optimum kernel shape factor as float (in radians).')
@click.option('-fit_mad', default='no', help='Fit Misorientation Angle Distribution (yes/no).')
@click.pass_context
def reducetexture(ctx, ebsd: str, grains: str, kernel: float, fit_mad: bool):
    """ Texture reduction algorithm with optional Misorientation angle fitting."""
    
    if ebsd==None:
        click.echo('')
        click.echo('Please provide some EBSD inputs for texture reduction!') 
        click.echo('For more info, run: kanapy reducetexture --help\n', err=True)
        sys.exit(0)
    else:
        cwd = os.getcwd()
        arg_dict = {}           
        if ebsd is not None:
            if not os.path.exists(cwd + '/{}'.format(ebsd)):
                click.echo('')
                click.echo("Mentioned file: '{}' does not exist in the current working directory!\n".format(ebsd), err=True)
                sys.exit(0)
            else:
                arg_dict['ebsdMatFile'] = cwd + '/{}'.format(ebsd)

        if grains is not None:
            if not os.path.exists(cwd + '/{}'.format(grains)):
                click.echo('')
                click.echo("Mentioned file: '{}' does not exist in the current working directory!\n".format(grains), err=True)
                sys.exit(0)
            else:        
                arg_dict['grainsMatFile'] = cwd + '/{}'.format(grains)
                
        if kernel is not None:
            arg_dict['kernelShape'] = kernel        
            
        if fit_mad == 'yes' or fit_mad == 'y' or fit_mad == 'Y' or fit_mad == 'YES': 
            arg_dict['MisAngDist'] = fit_mad            
            textureReduction(arg_dict)      
                  
        elif fit_mad == 'no' or fit_mad == 'n' or fit_mad == 'N' or fit_mad == 'NO': 
            textureReduction(arg_dict)
        else:
            click.echo('')
            click.echo('Invalid entry! Run: kanapy reducetexture --help\n', err=True)
            sys.exit(0)
        
    
def start():
    main(obj={})

    
if __name__ == '__main__':
    start()

# ************************************************************
# Functions for RVE handling that are only used in CLI version
# ************************************************************

def particleCreator(inputFile, periodic='True', units="mm", output=False):
    r"""
    Generates ellipsoid particles based on user-defined inputs.

    :param inputFile: User-defined grain informationfile for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Grain major diameter (:math:`\mu m`)
                * Grain minor diameter (:math:`\mu m`)
                * Grain's major axis tilt angle (degrees) with respect to the +ve X-axis (horizontal axis)

              2. Other user defined inputs: Periodicity & output units format (:math:`mm` or :math:`\mu m`).
                 Default values: periodicity=True & units= :math:`\mu m`.

              3. Particle, RVE and simulation data are written as JSON files in a folder in the current
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle.
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as total number of timesteps, periodicity and Output unit scale (:math:`mm`
                  or :math:`\mu m`) for ABAQUS .inp file.

    """
    print('')
    print('------------------------------------------------------------------------')
    print('Welcome to KANAPY - A synthetic polycrystalline microstructure generator')
    print('------------------------------------------------------------------------')

    print('Generating particles based on user defined grains')

    # Open the user input grain file and read the data
    try:
        input_data = np.loadtxt(inputFile, delimiter=',')
    except FileNotFoundError:
        print('Input file not found, make sure {0} file is present in the working directory!'.format(inputFile))
        raise FileNotFoundError

    # User defined major, minor axes lengths using: (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b
    majDia = input_data[:,0]                          # Major axis length
    minDia = input_data[:,1]                          # Minor axis length
    minDia2 = minDia.copy()                           # Minor2 axis length (assuming spheroid)
    tilt_angle = input_data[:,2]                      # Tilt angle

    # Volume of each ellipsoid
    volume_array = (4/3)*np.pi*(majDia*minDia*minDia2)*(1/8)

    # Equivalent diameter of each ellipsoid
    eq_Dia = (majDia*minDia*minDia2)**(1/3)

    # RVE size: RVE volume = sum(ellipsoidal volume)
    RVEvol = (np.sum(volume_array))

    # Determine the RVE side lengths
    dia_max = np.amax(majDia)
    RVEsizeY = 1.1*dia_max                 # The Y-side length should accomodate the Biggest dimension of the biggest ellipsoid
    RVEsizeX = round((RVEvol/ RVEsizeY)**0.5, 4)
    RVEsizeZ = RVEsizeX

    # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain atleast 3 voxels
    voxel_size = 1.1*(np.amin(minDia) / 3.)
    Nx = int(round(RVEsizeX / voxel_size))         # Number of voxel/RVE side
    Ny = int(round(RVEsizeY / voxel_size))
    Nz = int(round(RVEsizeZ / voxel_size))

    # Re-calculate the voxel resolution
    voxel_sizeX = RVEsizeX / Nx
    voxel_sizeY = RVEsizeY / Ny
    voxel_sizeZ = RVEsizeZ / Nz

    totalEllipsoids = len(majDia)
    print('    Total number of grains        = {}'.format(totalEllipsoids))
    print('    RVE side lengths (X, Y, Z)    = {0}, {1}, {2}'.format(RVEsizeX, RVEsizeY, RVEsizeZ))
    print('    Number of voxels (X, Y, Z)    = {0}, {1}, {2}'.format(Nx, Ny, Nz))
    print('    Voxel resolution (X, Y, Z)    = {0:.4f}, {1:.4f}, {2:.4f}'.format(voxel_sizeX, voxel_sizeY, voxel_sizeZ))
    print('    Total number of voxels (C3D8) = {}\n'.format(Nx*Ny*Nz))

    # Create dictionaries to store the data generated
    particle_data = {'Type': 'Elongated', 'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia), 'Major_diameter': list(majDia),
                     'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Tilt angle': list(tilt_angle)}

    RVE_data = {'RVE_sizeX': RVEsizeX, 'RVE_sizeY': RVEsizeY, 'RVE_sizeZ': RVEsizeZ,
                'Voxel_numberX': Nx, 'Voxel_numberY': Ny, 'Voxel_numberZ': Nz,
                'Voxel_resolutionX': voxel_sizeX,'Voxel_resolutionY': voxel_sizeY, 'Voxel_resolutionZ': voxel_sizeZ}

    simulation_data = {'Time steps': 1000, 'Periodicity': "{}".format(periodic), 'Output units': units}

    # Dump the Dictionaries as json files
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/particle_data.json', 'w') as outfile:
        json.dump(particle_data, outfile, indent=2)

    with open(json_dir + '/RVE_data.json', 'w') as outfile:
        json.dump(RVE_data, outfile, indent=2)

    with open(json_dir + '/simulation_data.json', 'w') as outfile:
        json.dump(simulation_data, outfile, indent=2)

    return


def extract_volume_sharedGBarea(elmtDict, elmtSetDict, RVE_data, save_files=False):
    r"""
    Evaluates the grain volume and the grain boundary shared surface area between neighbouring grains
    and writes them to 'grainVolumes.csv' & 'shared_surfaceArea.csv' files.

    --- WARNING --- This function is only used in CLI version of kanapy and
                    will no longer be updated.
                    The API version uses kanapy/api/calcPolygons which
                    offers more functionality.

    .. note:: 1. RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.

              2. The grain volumes written to the 'grainVolumes.csv' file are sorted in ascending order of grain IDs. And the values are written
                 in either :math:`mm` or :math:`\mu m` scale, as requested by the user in the input file.

              3. The shared surface area written to the 'shared_surfaceArea.csv' file are in either :math:`mm` or :math:`\mu m` scale,
                 as requested by the user in the input file.
    """
    print('')
    print('Evaluating grain volumes.')
    print('Evaluating shared Grain Boundary surface area between grains.')

    voxel_size = RVE_data['Voxel_resolutionX']

    grain_vol = {}
    # For each grain find its volume and output it
    for gid, elset in elmtSetDict.items():
        # Convert to
        gvol = len(elset) * (voxel_size**3)
        grain_vol[gid] = gvol

    # Sort the grain volumes in ascending order of grain IDs
    gv_sorted_keys = sorted(grain_vol, key=grain_vol.get)
    gv_sorted_values = [[grain_vol[gk]] for gk in gv_sorted_keys]
    # gv_sorted_values = [[gk,gv] for gk,gv in grain_vol.items()]

    # For each grain find its outer face ids
    grain_facesDict = dict()
    for gid, elset in elmtSetDict.items():
        outer_faces = set()
        nodeConn = [elmtDict[el] for el in elset]        # For each voxel/element get node connectivity
        # create the 6 faces of the voxel
        for nc in nodeConn:
            faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                     [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                     [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]

            # Sort each list in ascending order
            sorted_faces = [sorted(fc) for fc in faces]

            # create face ids by joining node id's
            face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]

            # Update the set to include only the outer face id's
            for fid in face_ids:
                if fid not in outer_faces:
                    outer_faces.add(fid)
                else:
                    outer_faces.remove(fid)
        grain_facesDict[gid] = outer_faces

    # Find all combination of grains to check for common area
    combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))

    # Find the shared area
    shared_area = []
    for cb in combis:
        finter = grain_facesDict[cb[0]].intersection(grain_facesDict[cb[1]])
        if finter:
            sh_area = len(finter) * (voxel_size**2)
            shared_area.append([cb[0], cb[1], sh_area])
        else:
            continue

    if save_files:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files
        print("Writing grain volumes info. to file ('grainVolumes.csv')\n", end="")

        # Write out grain volumes to a file
        with open(json_dir + '/grainVolumes.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(gv_sorted_values)
        print("Writing shared GB surface area info. to file ('shared_surfaceArea.csv')", end="")

        # Write out shared grain boundary area to a file
        with open(json_dir + '/shared_surfaceArea.csv', "w", newline="") as f:
            f.write('GrainA, GrainB, SharedArea\n')
            writer = csv.writer(f)
            writer.writerows(shared_area)

    print('---->DONE!\n')
    return gv_sorted_values, shared_area, grain_facesDict


def write_output_stat(nodes_v, elmtDict, elmtSetDict, particle_data, RVE_data,\
                      simulation_data, save_files=False):
    r"""
    Evaluates particle- and output RVE grain statistics with respect to Major,
    Minor & Equivalent diameters for comparison
    and writes them to 'output_statistics.json' file.

    WARNING: This subroutine is only used by kanapy CLI and will no longer be
             maintained.
             Kanapy API uses api.calcPolygens and api.get_stats, which offer
             more functionality.

    .. note:: The particle and grain diameter values are written in either
              :math:`mm` or :math:`\mu m` scale,
              as requested by the user in the input file.
    """
    print('')
    print('Comparing input & output statistics')

    # Extract from dictionaries
    par_eqDia = particle_data['Equivalent_diameter']
    voxel_size = RVE_data['Voxel_resolutionX']
    RVE_sizeX, RVE_sizeY, RVE_sizeZ = RVE_data['RVE_sizeX'], RVE_data['RVE_sizeY'], RVE_data['RVE_sizeZ']

    if particle_data['Type'] == 'Elongated':
        par_majDia = particle_data['Major_diameter']
        par_minDia = particle_data['Minor_diameter1']

    if simulation_data['Periodicity'] == 'True':
        periodic = True
    elif simulation_data['Periodicity'] == 'False':
        periodic = False

    # Factor used to generate particle and grains diameters in 'mm' or 'um' scale
    if simulation_data['Output units'] == 'mm':
        scale = 'mm'
        divideBy = 1000
    elif simulation_data['Output units'] == 'um':
        scale = 'um'
        divideBy = 1

    # Check if Equiaxed or elongated particles
    if particle_data['Type'] == 'Equiaxed':          # Equiaxed grains (spherical particles)

        # Find each grain's equivalent diameter
        grain_eqDia = []
        for k, v in elmtSetDict.items():
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)

        # write out the particle and grain equivalent diameters to files
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        # Compute the L1-error
        kwargs = {'Spheres': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia}}}
        error = l1_error_est(**kwargs)

        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)),
                       'Grain type': particle_data['Type'],
                       'Unit_scale': scale,
                       'L1-error':error,
                       'Particle_Equivalent_diameter': par_eqDia,
                       'Grain_Equivalent_diameter': grain_eqDia}

        if save_files:
            print("Writing particle & grain equivalent, major & minor diameter to file ('output_statistics.json')")
            cwd = os.getcwd()
            json_dir = cwd + '/json_files'          # Folder to store the json files
            with open(json_dir + '/output_statistics.json', 'w') as outfile:
                json.dump(output_data, outfile, indent=2)

    else:                                               # Elongated grains (ellipsoidal particles)

        grain_eqDia, grain_majDia, grain_minDia = [], [], []
        # Find all the nodal coordinates belonging to the grain
        grain_node = {}
        for k, v in elmtSetDict.items():
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)

            # All nodes belonging to grain
            nodeset = set()
            for el in v:
                nodes = elmtDict[el]
                for n in nodes:
                    if n not in nodeset:
                        nodeset.add(n)
            # Get the coordinates as an array
            points = [nodes_v[n-1] for n in nodeset]
            points = np.asarray(points)
            grain_node[k] = points

        if periodic:
            # If periodic, find the grains whose perodic halves have to be shifted
            shiftRight, shiftTop, shiftBack = [], [], []
            for key, value in grain_node.items():

                # Find all nodes on left, Right, Top, Bottom, Front & Back faces
                nodeLS, nodeRS = set(), set()
                nodeTS, nodeBS = set(), set()
                nodeFS, nodeBaS = set(), set()
                for enum, coord in enumerate(value):

                    if abs(0.0000 - coord[0]) <= 0.00000001:       # nodes on Left face
                        nodeLS.add(enum)
                    elif abs(RVE_sizeX - coord[0]) <= 0.00000001:    # nodes on Right face
                        nodeRS.add(enum)

                    if abs(0.0000 - coord[1]) <= 0.00000001:       # nodes on Bottom face
                        nodeBS.add(enum)
                    elif abs(RVE_sizeY - coord[1]) <= 0.00000001:    # nodes on Top face
                        nodeTS.add(enum)

                    if abs(0.0000 - coord[2]) <= 0.00000001:       # nodes on Front face
                        nodeFS.add(enum)
                    elif abs(RVE_sizeZ - coord[2]) <= 0.00000001:    # nodes on Back face
                        nodeBaS.add(enum)

                if len(nodeLS) != 0 and len(nodeRS) != 0:   # grain is periodic, has faces on both Left & Right sides
                    shiftRight.append(key)                  # left set has to be moved to right side
                if len(nodeBS) != 0 and len(nodeTS) != 0:   # grain is periodic, has faces on both Top & Bottom sides
                    shiftTop.append(key)                    # bottom set has to be moved to Top side
                if len(nodeFS) != 0 and len(nodeBaS) != 0:  # grain is periodic, has faces on both Front & Back sides
                    shiftBack.append(key)                   # front set has to be moved to Back side

            # For each grain that has to be shifted, pad along x, y, z respectively
            for grain in shiftRight:
                pts = grain_node[grain]
                # Pad the nodes on the left side by RVE x-dimension
                for enum, val in enumerate(pts[:, 0]):
                    if val>=0.0 and val<=RVE_sizeX/2.:
                        pts[enum, 0] += RVE_sizeX

            for grain in shiftBack:
                pts = grain_node[grain]
                # Pad the nodes on the front side by RVE z-dimension
                for enum, val in enumerate(pts[:, 2]):
                    if val>=0.0 and val<=RVE_sizeZ/2.:
                        pts[enum, 2] += RVE_sizeZ

            for grain in shiftTop:
                pts = grain_node[grain]
                # Pad the nodes on the bottom side by RVE y-dimension
                for enum, val in enumerate(pts[:, 1]):
                    if val>=0.0 and val<=RVE_sizeY/2.:
                        pts[enum, 1] += RVE_sizeY

        # For periodic & Non-periodic: create the convex hull and find the major & minor diameters
        for grain, points in grain_node.items():
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]

            # Find the approximate center of the grain using extreme surface points
            xmin, xmax = np.amin(points[:, 0]), np.amax(points[:, 0])
            ymin, ymax = np.amin(points[:, 1]), np.amax(points[:, 1])
            zmin, zmax = np.amin(points[:, 2]), np.amax(points[:, 2])
            center = np.array([xmin + (xmax-xmin)/2.0, ymin + (ymax-ymin)/2.0, zmin + (zmax-zmin)/2.0])

            # Find the euclidean distance to all surface points from the center
            dists = [euclidean(center, pt) for pt in hull_pts]
            a2 = 2.0*np.amax(dists)
            b2 = 2.0*np.amin(dists)

            # Calculate ellipsoid dimensions using eigen values
            #ellPoints = points.T
            #eigvals, eigvecs = np.linalg.eig(np.cov(ellPoints))
            #eigvals = np.sort(eigvals)
            #a2, b2 = eigvals[-1], eigvals[-2]

            grain_majDia.append(a2)                 # update the major diameter list
            grain_minDia.append(b2)                 # update the minor diameter list

        # write out the particle and grain equivalent, major, minor diameters to file
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        par_majDia = list(np.array(par_majDia)/divideBy)
        grain_majDia = list(np.array(grain_majDia)/divideBy)

        par_minDia = list(np.array(par_minDia)/divideBy)
        grain_minDia = list(np.array(grain_minDia)/divideBy)

        # Compute the L1-error
        kwargs = {'Ellipsoids': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia},
                                'Major diameter': {'Particles': par_majDia, 'Grains': grain_majDia},
                                'Minor diameter': {'Particles': par_minDia, 'Grains': grain_minDia}}}
        error = l1_error_est(**kwargs)

        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)),
                       'Grain type': particle_data['Type'],
                       'Unit_scale': scale,
                       'L1-error': error,
                       'Particle_Equivalent_diameter': par_eqDia,
                       'Particle_Major_diameter': par_majDia,
                       'Particle_Minor_diameter': par_minDia,
                       'Grain_Equivalent_diameter': grain_eqDia,
                       'Grain_Major_diameter': grain_majDia,
                       'Grain_Minor_diameter': grain_minDia}

        if save_files:
            print("Writing particle & grain equivalent, major & minor diameter to file ('output_statistics.json')")
            cwd = os.getcwd()
            json_dir = cwd + '/json_files'          # Folder to store the json files
            with open(json_dir + '/output_statistics.json', 'w') as outfile:
                json.dump(output_data, outfile, indent=2)

    print('---->DONE!')
    return output_data


def write_position_weights(file_num):
    r"""
    Reads the (.dump) file to extract information and ouputs the position and weight files for tessellation.

    :param file_num: Simulation time step for which position and weights output.
    :type file_num: int

    .. note:: 1. Applicable only to spherical particles.
              2. The generated 'sphere_positions.txt' and 'sphere_weights.txt' files can be inputted
                 into NEPER for tessellation and meshing.
              3. The values of positions and weights are written in :math:`\mu m` scale only.
    """
    print('')
    print('Writing position and weights files for NEPER', end="")
    cwd = os.getcwd()
    dump_file = cwd + '/dump_files/particle.{0}.dump'.format(file_num)

    try:
        with open(dump_file, 'r+') as fd:
            lookup = "ITEM: NUMBER OF ATOMS"
            lookup2 = "ITEM: BOX BOUNDS ff ff ff"
            for num, lines in enumerate(fd, 1):
                if lookup in lines:
                    number_particles = int(next(fd))
                    par_line_num = num + 7

                if lookup2 in lines:
                    valuesX = re.findall(r'\S+', next(fd))
                    RVE_minX, RVE_maxX = list(map(float, valuesX))

                    valuesY = re.findall(r'\S+', next(fd))
                    RVE_minY, RVE_maxY = list(map(float, valuesY))

                    valuesZ = re.findall(r'\S+', next(fd))
                    RVE_minZ, RVE_maxZ = list(map(float, valuesZ))


    except FileNotFoundError:
        print('    .dump file not found, make sure "Packing" command is executed first!')
        raise FileNotFoundError

    par_dict = dict()
    with open(dump_file, "r") as f:
        count = 0
        for num, lines in enumerate(f, 1):
            if num >= par_line_num:

                values = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                if '_' in values[0]:
                    # Duplicates exists (ignore them when writing position
                    # and weight files)
                    continue
                else:
                    count += 1
                    iden = count
                    par_dict[iden] = [values[1], values[2],
                                      values[3], values[4]]

    with open('sphere_positions.txt', 'w') as fd:
        for key, value in par_dict.items():
            fd.write('{0} {1} {2}\n'.format(value[0], value[1], value[2]))

    with open('sphere_weights.txt', 'w') as fd:
        for key, value in par_dict.items():
            fd.write('{0}\n'.format(value[3]))
    print('---->DONE!\n')
    return
