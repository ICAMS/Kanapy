# -*- coding: utf-8 -*-
import os, sys
import shutil, json
import click

from kanapy.util import ROOT_DIR, MAIN_DIR 
from kanapy.input_output import particleStatGenerator, particleCreator, RVEcreator, \
    write_position_weights, write_output_stat, plot_output_stats, \
    extract_volume_sharedGBarea, read_dump, export2abaqus
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.analyze_texture import textureReduction
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
    os.system("echo '. {}' >> ~/.bashrc".format(ROOT_DIR+'/kanapy-complete.sh'))  


@main.command(name='runTests')
@click.pass_context
def tests(ctx):    
    """ Runs unittests built within kanapy."""    
    
    click.echo('')
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
        click.echo('For more info. run: kanapy readgrains --help\n', err=True)
        sys.exit(0)    
    elif ((periodic!='True') and (periodic!='False')):
        click.echo('')
        click.echo('Invalid entry!, Run: kanapy readgrains again', err=True)
        click.echo('For more info. run: kanapy readgrains --help\n', err=True)
        sys.exit(0)                
    elif ((units!='mm') and (units!='um')):
        click.echo('')
        click.echo('Invalid entry!, Run: kanapy readgrains again', err=True)
        click.echo('For more info. run: kanapy readgrains --help\n', err=True)
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
        packingRoutine(particle_data, RVE_data, simulation_data, save_files=True)
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
            filename = cwd + '/dump_files/particle.{0}.dump'.format(800)            
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
                    
        
        # For MTEX installation path
        userpath2 = MAIN_DIR+'/libs/mtex/'
                     
    elif status1 == 'no' or status1 == 'n' or status1 == 'N' or status1 == 'NO':
        click.echo("Kanapy's texture analysis code requires MATLAB. Please install it.")
        click.echo('')
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
        
        os.chdir('{}extern/engines/python'.format(userpath1[0:-10])) # remove bin/matlab from matlab path
        os.system('python setup.py install')
        path = os.path.abspath(__file__)[0:-7] # remove /cli.py from kanapy path
        os.chdir(path)
        os.system('python setup_mtex.py')
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
