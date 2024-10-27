# -*- coding: utf-8 -*-
import os
import shutil
import click
from kanapy.util import MAIN_DIR


@click.group()
@click.pass_context
def main(ctx):    
    pass


@main.command(name='gui')
@click.pass_context
def gui(ctx):
    """ Start Kanapy's GUI """
    import matplotlib.pyplot as plt
    import tkinter as tk
    import tkinter.font as tkFont
    from tkinter import ttk
    from kanapy.gui import particle_rve, cuboid_rve

    app = tk.Tk()
    app.title("RVE_Generation")
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    plt.rcParams['figure.dpi'] = screen_height / 19  # height stats_plot: 9, height voxel_plot: 6, margin: 4
    window_width = int(screen_width * 0.6)
    window_height = int(screen_height * 0.8)
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = 0  # int((screen_height / 2) - (window_height / 2))
    app.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    notebook = ttk.Notebook(app)
    notebook.pack(fill='both', expand=True)
    style = ttk.Style(app)
    default_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
    style.configure('TNotebook.Tab', font=('Helvetica', '12', "bold"))
    style.configure('TButton', font=default_font)

    """ Start main loop """
    prve = particle_rve(app, notebook)  # First tab: Particle-based grains
    crve = cuboid_rve(app, notebook)  # Second tab: Cuboid grains
    #erve = ebsd_rve()  # Third tab: EBSDbased RVE
    app.mainloop()


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
    shutil.rmtree(os.path.join(cwd, "dump_files"))
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
@click.pass_context
def setupTexture(ctx):
    """ Stores the user provided MATLAB & MTEX paths for texture analysis."""
    setPaths()


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
    
        
def setPaths():
    """ Starts matlab engine, after installation if required, and initializes MTEX.
    """

    """
    Legacy version   
    # For MATLAB executable
    click.echo('')
    status1 = input('Is MATLAB installed in this system (yes/no): ')
    str_yes = ['yes', 'y', 'Y', 'Yes', 'YES']
    str_no = ['no', 'n', 'N', 'No', 'NO']
    
    if status1 in str_yes:
        click.echo('Searching your system for MATLAB ...')
        MATLAB = shutil.which("matlab")        

        if MATLAB:
            decision1 = input('Found MATLAB in {0}, continue (yes/no): '.format(MATLAB))
            
            if decision1 in str_yes:                

                version = chkVersion(MATLAB)        # Get the MATLAB version
                if version is None:
                    click.echo('')
                    click.echo('MATLAB version is unknown, compatibility could not be verified.\n')
                    
                elif version < 2015:
                    click.echo('')
                    raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2015a and above')
                userpath1 = MATLAB

            elif decision1 in str_no:
                userinput = input('Please provide the path to MATLAB executable: ')
                
                version = chkVersion(userinput)
                if version is None:
                    click.echo('')
                    click.echo('MATLAB version is unknown, compatibility could not be verified.\n')
                elif version < 2024:
                    click.echo('')
                    raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2024a and above')
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
                raise ModuleNotFoundError('Sorry!, Kanapy is compatible with MATLAB versions 2024a and above.')
            userpath1 = userinput
                     
    elif status1 in str_no:
        click.echo("Kanapy's texture module requires MATLAB. Please install it.")
        click.echo('')
        userpath1 = False
    else:
        raise ValueError('Invalid entry!, Run: kanapy setupTexture again')
        
    # Create a file that stores the paths
    if userpath1:"""
    # check if Matlab Engine library is already installed
    try:
        import matlab.engine
        click.echo('Using existing matlab.engine. Please update if required.')
    except:
        # if not, install matlab engine
        click.echo('Installing matlab.engine...')
        # ind = userpath1.find('bin')  # remove bin/matlab from matlab path
        # path = os.path.join(userpath1[0:ind], 'extern', 'engines', 'python')  # complete path to matlab engine
        # os.chdir(path)
        res = os.system('python -m pip install matlabengine==24.1.2')
        if res != 0:
            click.echo('\n Error in installing matlab.engine. This feature requires Matlab 2024a or above.')
            # click.echo('Please contact system administrator to run "> python -m pip install ."')
            # click.echo(f'in directory {path}')
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
