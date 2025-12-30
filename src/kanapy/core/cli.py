# -*- coding: utf-8 -*-
import os
import shutil
import click
import zipfile
import requests
import webbrowser
from io import BytesIO
from importlib.metadata import version

    
@click.group()
@click.version_option(version=version('kanapy'))
@click.pass_context
def main(ctx):    
    pass

@main.command(name='gui')
@click.pass_context
def gui(ctx):
    """
    Start Kanapy's graphical user interface (experimental alpha version)

    This function initializes and launches the Kanapy GUI using Tkinter.
    It creates a main application window with multiple tabs for different
    RVE generation modes, such as particle-based and cuboid-based RVEs.

    Parameters
    ----------
    ctx : object
        Execution context (reserved for CLI or application integration)

    Notes
    -----
    - This GUI version is experimental and intended for testing only
    - Requires `matplotlib` and `tkinter` libraries
    """
    import matplotlib.pyplot as plt
    import tkinter as tk
    import tkinter.font as tkFont
    from tkinter import ttk
    from .gui import particle_rve, cuboid_rve

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
@click.option('--no_texture', default=False)
@click.pass_context
def tests(ctx, no_texture: bool):
    """
    Run Kanapy's internal unittests

    Executes the built-in test suite for Kanapy using pytest.
    Depending on the `no_texture` flag, it either runs a subset of tests
    or all available tests in the `tests` directory.

    Parameters
    ----------
    ctx : object
        Execution context (reserved for CLI or internal use)
    no_texture : bool
        If True, run tests excluding texture-related modules;
        if False, run the full test suite

    Notes
    -----
    - Must be executed from the root directory of the Kanapy installation
    - Temporary dump files generated during testing are automatically removed
    """
    click.echo('Will only work in root directory of kanapy installation.')
    cwd = os.getcwd()
    if no_texture:
        #t1 = "{0}/tests/test_collide_detect_react.py".format(cwd)
        t2 = "{0}/tests/test_entities.py".format(cwd)
        t3 = "{0}/tests/test_input_output.py".format(cwd)
        t4 = "{0}/tests/test_packing.py".format(cwd)
        t5 = "{0}/tests/test_voxelization.py".format(cwd)
        os.system(f"pytest {t2} {t3} {t4} {t5} -v")
    else:
        os.system("pytest {0}/tests/ -v".format(cwd))
    shutil.rmtree(os.path.join(cwd, "dump_files"))
    click.echo('')
        
    
@main.command(name='readDocs')
@click.pass_context
def docs(ctx):
    """
    Open the Kanapy documentation webpage

    Launches the default web browser and navigates to the official
    Kanapy documentation page hosted on GitHub Pages.

    Parameters
    ----------
    ctx : object
        Execution context (reserved for CLI or internal use)
    """
    webbrowser.open("https://icams.github.io/Kanapy/")


@main.command(name='copyExamples')
@click.pass_context
def download_subdir(ctx):
    """
    Download example files from Kanapy's GitHub repository

    Fetches the latest Kanapy repository archive from GitHub,
    extracts the `examples` subdirectory, and saves it to the local
    working directory as `kanapy_examples`.

    Parameters
    ----------
    ctx : object
        Execution context (reserved for CLI or internal use)

    Notes
    -----
    - Requires an active internet connection
    - Downloads from the `master` branch of the Kanapy repository
    - Automatically creates the output directory if it does not exist
    """
    zip_url = f"https://github.com/ICAMS/kanapy/archive/refs/heads/master.zip"
    output_dir = os.path.join(os.getcwd(), 'kanapy_examples')
    subdir_prefix = "Kanapy-master/examples/"

    click.echo(f"Downloading ZIP from: {zip_url}")
    r = requests.get(zip_url)
    r.raise_for_status()

    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        members = [f for f in zf.namelist() if f.startswith(subdir_prefix)]
        if not members:
            click.echo(f"Error: Subdirectory 'examples' not found in branch 'master'.", err=True)
            return

        for member in members:
            rel_path = os.path.relpath(member, subdir_prefix)
            if not rel_path or member.endswith("/"):
                continue
            target_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as out_file:
                out_file.write(zf.read(member))

    click.echo(f"Extracted 'examples' from ICAMS/kanapy to '{output_dir}/'")
    

@main.command(name='setupMTEX')
@click.pass_context
def setup_mtex(ctx):
    """
    Start the MATLAB engine and initialize MTEX

    Launches the MATLAB engine from Python and sets up the
    MTEX toolbox environment for crystallographic computations.

    Parameters
    ----------
    ctx : object
        Execution context (reserved for CLI or internal use)

    Notes
    -----
    - Requires MATLAB and the MTEX toolbox to be installed
    - Calls `setPaths()` to configure MATLAB path settings
    """
    setPaths()


def chkVersion(matlab):
    """
    Read the installed MATLAB version from a version string

    Parses the given MATLAB version string to extract the release year
    (e.g., 'R2023a' → 2023). If the version cannot be determined, returns None.

    Parameters
    ----------
    matlab : str
        String containing MATLAB version information

    Returns
    -------
    int or None
        MATLAB release year if successfully parsed, otherwise None
    """
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
    """
    Start the MATLAB engine and initialize the MTEX environment

    Checks if the MATLAB engine for Python is installed and installs it if missing.
    Then initializes the MATLAB engine and MTEX toolbox for use with Kanapy.

    Raises
    ------
    ModuleNotFoundError
        If `kanapy-mtex` or MATLAB are not installed,
        or if the MATLAB engine installation fails

    Notes
    -----
    - Requires MATLAB 2025a or later
    - Automatically installs the `matlabengine` package if not found
    - Configures Kanapy for texture analysis by running `init_engine.py`
    """
    try:
        from kanapy_mtex import ROOT_DIR
    except:
        raise ModuleNotFoundError('This function in only evailable if kanapy-mtex and Matlab are installed.')
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
        res = os.system('python -m pip install matlabengine==25.1.2')
        if res != 0:
            raise ModuleNotFoundError('Error in installing matlab.engine. This feature requires Matlab 2025a or above.')
        
    # initalize matlab engine and MTEX for kanapy
    os.chdir(ROOT_DIR)
    os.system('python init_engine.py')
    click.echo('')
    click.echo('Kanapy is now configured for texture analysis!\n')

def start():
    main(obj={})

    
if __name__ == '__main__':
    start()
