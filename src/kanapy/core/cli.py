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
    """ Start Kanapy's GUI (experimental alpha-version). """
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
    """ Runs unittests built within kanapy."""
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
    """Open webpage with  Kanapy documentation."""
    webbrowser.open("https://icams.github.io/Kanapy/")


@main.command(name='copyExamples')
@click.pass_context
def download_subdir(ctx):
    """
    Download examples from Kanapy's GitHub repository.

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
    """ Starts Matlab engine and initializes MTEX."""
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
    try:
        from kanapy.mtex import ROOT_DIR
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
