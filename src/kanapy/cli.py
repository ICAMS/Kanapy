# -*- coding: utf-8 -*-
import os
import shutil
import json
import click
from kanapy.util import MAIN_DIR, WORK_DIR, ROOT_DIR


@click.group()
@click.pass_context
def main(ctx):    
    pass


@main.command(name='runTests')
@click.option('-no_texture', default=False)
@click.pass_context
def tests(ctx, no_texture: bool):
    """ Runs unittests built within kanapy."""
    click.echo('')
    if no_texture:
        t1 = "{0}/tests/test_collide_detect_react.py".format(MAIN_DIR)
        t2 = "{0}/tests/test_entities.py".format(MAIN_DIR)
        t3 = "{0}/tests/test_input_output.py".format(MAIN_DIR)
        t4 = "{0}/tests/test_packing.py".format(MAIN_DIR)
        t5 = "{0}/tests/test_voxelization.py".format(MAIN_DIR)
        os.system(f"pytest {t1} {t2} {t3} {t4} {t5} -v")
    else:
        os.system("pytest {0}/tests/ -v".format(MAIN_DIR))
    cwd = os.getcwd()
    shutil.rmtree(os.path.normpath(cwd + "/dump_files"))
    click.echo('')    
        
    
@main.command(name='genDocs')
@click.pass_context
def docs(ctx):    
    """ Generates an HTML-based reference documentation."""
    
    click.echo('')
    os.system("make -C {0}/docs/ clean && make -C {0}/docs/ html".format(MAIN_DIR))      
    click.echo('')
    click.echo("The HTML documentation can be found at '/path/to/your/kanapy/docs/index.html'")
    click.echo('')


@main.command(name='copyExamples')
@click.pass_context
def docs(ctx):
    """ Copies examples to local filesystem."""

    click.echo('')
    dst = os.path.join(os.getcwd(), 'examples')
    epath = os.path.join(MAIN_DIR, 'examples')
    if os.path.exists(dst):
        raise IsADirectoryError('{0} already exists'.format(dst))
    shutil.copytree(epath, dst)
    click.echo(f'Copied examples to "{dst}".')


@main.command(name='setupTexture')
@click.option('-admin', default=False)
@click.pass_context
def setupTexture(ctx, admin: bool):
    """ Stores the user provided MATLAB & MTEX paths for texture analysis."""
    setPaths(admin)


def chkVersion(matlab):
    """ Read the version of Matlab"""
    ind = matlab.find('R20')
    if ind < 0:
        version = None 
    else:                 # Find the matlab version available in the system
        try: 
            version = int(matlab[ind+1:ind+5])
            click.echo(f'Detected Matlab version R{version}')
        except:
            version = None
    return version
    
        
def setPaths(admin):
    """ Requests user input for MATLAB & MTEX installation paths
    if admin==True: use paths in MAIN_DIR otherwise use WORK_DIR
    """
    if admin:
        path = ROOT_DIR
    else:
        if not os.path.exists(WORK_DIR):
            raise FileNotFoundError('Package not properly installed, working directory is missing.')
        path = WORK_DIR
    path_json = os.path.join(path, 'PATHS.json')
    with open(path_json) as json_file:
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
                    click.echo('MATLAB version is unknown, compatibility could not be verified.\n')
                    
                elif version < 2015:
                    click.echo('')
                    raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above')
                userpath1 = MATLAB

            elif decision1 == 'no' or decision1 == 'n' or decision1 == 'N' or decision1 == 'NO':
                userinput = input('Please provide the path to MATLAB executable: ')
                
                version = chkVersion(userinput)
                if version is None:
                    click.echo('')
                    click.echo('MATLAB version is unknown, compatibility could not be verified.\n')
                elif version < 2015:
                    click.echo('')
                    raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above')
                userpath1 = userinput
                                    
            else:
                raise ValueError('Invalid entry!, Run: kanapy setuptexture again.')
                            
        else:
            print('No MATLAB executable found!')            
            userinput = input('Please provide the path to MATLAB executable: ')
            
            version = chkVersion(userinput)
            if version is None:
                click.echo('')
                click.echo('MATLAB version is unknown, compatibility could not be verified.\n')
            elif version < 2015:
                raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above.')
            userpath1 = userinput
                     
    elif status1 == 'no' or status1 == 'n' or status1 == 'N' or status1 == 'NO':
        click.echo("Kanapy's texture module requires MATLAB. Please install it.")
        click.echo('')
        userpath1 = False
    else:
        raise ValueError('Invalid entry!, Run: kanapy setupTexture again')
        
    # Create a file in ".kanapy" folder that stores the paths
    if userpath1:
        path_dict['MATLABpath'] = os.path.normpath(userpath1)
        if os.path.exists(path_json):
            os.remove(path_json)
        with open(path_json, 'w') as outfile:
            json.dump(path_dict, outfile, indent=2)                
        
        # check if Matlab Engine library is already installed
        try:
            import matlab.engine
            click.echo('Using existing matlab.engine. Please update if required.')
        except:
            # if not, install matlab engine
            click.echo('Installing matlab.engine...')
            ind = userpath1.find('bin')
            path = os.path.join(userpath1[0:ind], 'extern', 'engines', 'python')
            os.chdir(path)  # remove bin/matlab from matlab path
            res = os.system('python -m pip install .')
            if res != 0:
                click.echo('\n Error in installing matlab.engine')
                click.echo('Please contact system administrator to run "> python -m pip install ."')
                click.echo(f'in directory {path}')
                raise ModuleNotFoundError()
        
        # initalize matlab engine and MTEX for kanapy
        path = os.path.abspath(__file__)[0:-7]  # remove /cli.py from kanapy path
        os.chdir(path)
        os.system('python init_engine.py')
        click.echo('')
        click.echo('Kanapy is now configured for texture analysis!\n')

    
def start():
    main(obj={})

    
if __name__ == '__main__':
    start()
