"""
A Graphical User Interface for create_rve.py and cuboid_grains.py
Created on May 2024
@author: Ronak Shoghi
"""

import tkinter as tk
from tkinter import ttk
from math import pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import kanapy as knpy
import tkinter.font as tkFont
import numpy as np
import itertools
from kanapy.initializations import RVE_creator, mesh_creator
from kanapy.entities import Simulation_Box


def parse_entry(entry):
    return list(map(int, entry.strip().split(',')))


def add_label_and_entry(row, label_text, entry_var, entry_type="entry", bold=True, options=None):
    label_font = ("Helvetica", 12, "bold") if bold else ("Helvetica", 12)
    ttk.Label(main_frame1, text=label_text, font=label_font).grid(row=row, column=0, sticky='w')
    if entry_type == "entry":
        ttk.Entry(main_frame1, textvariable=entry_var, width=20, font=("Helvetica", 12)).grid(row=row, column=1,
                                                                                              sticky='e')
    elif entry_type == "checkbox":
        ttk.Checkbutton(main_frame1, variable=entry_var).grid(row=row, column=1, sticky='e')
    elif entry_type == "combobox" and options is not None:
        combobox = ttk.Combobox(main_frame1, textvariable=entry_var, values=options, state='readonly', width=19)
        combobox.grid(row=row, column=1, sticky='e')
        combobox.configure(font=("Helvetica", 12))
        combobox.current(0)


def display_plot(fig, plot_type):
    global stats_canvas, rve_canvas1

    if plot_type == "stats":
        if stats_canvas is not None:
            stats_canvas.get_tk_widget().destroy()
        stats_canvas = FigureCanvasTkAgg(fig, master=stats_plot_frame)
        stats_canvas.draw()
        stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    elif plot_type == "rve":
        if rve_canvas1 is not None:
            rve_canvas1.get_tk_widget().destroy()
        rve_canvas1 = FigureCanvasTkAgg(fig, master=rve_plot_frame1)
        rve_canvas1.draw()
        rve_canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    app.update_idletasks()
    width, height = app.winfo_reqwidth(), app.winfo_reqheight()
    app.geometry(f"{width}x{height}")


""" Functions for RVEs based on particle simulations """


def create_and_plot_stats():
    """Plot statistics of current microstructure descriptors
    Will erase global microstructure object if it exists."""
    global ms, ms_stats

    texture = texture_var1.get()
    matname = matname_var1.get()
    nvox = int(nvox_var1.get())
    size = int(size_var1.get())
    periodic = periodic_var1.get()

    ms_stats = {
        'Grain type': 'Elongated',
        'Equivalent diameter': {
            'sig': eq_diameter_sig.get(), 'scale': eq_diameter_scale.get(),
            'loc': eq_diameter_loc.get(), 'cutoff_min': eq_diameter_min.get(),
            'cutoff_max': eq_diameter_max.get()
        },
        'Aspect ratio': {
            'sig': aspect_ratio_sig.get(), 'scale': aspect_ratio_scale.get(),
            'loc': aspect_ratio_loc.get(), 'cutoff_min': aspect_ratio_min.get(),
            'cutoff_max': aspect_ratio_max.get()
        },
        "Tilt angle": {
            "kappa": tilt_angle_kappa.get(), "loc": tilt_angle_loc.get(),
            "cutoff_min": tilt_angle_min.get(), "cutoff_max": tilt_angle_max.get()
        },
        'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox},
        'Simulation': {'periodicity': str(periodic), 'output_units': 'mm'}
    }

    ms = knpy.Microstructure(descriptor=ms_stats, name=f"{matname}_{texture}_texture")
    ms.init_RVE()
    flist = ms.plot_stats_init(silent=True)
    for fig in flist:
        display_plot(fig, plot_type="stats")


def create_rve_and_plot():
    """Create and plot the RVE
    Will overwrite existing ms_stats and ms objects"""
    global ms, ms_stats

    texture = texture_var1.get()
    matname = matname_var1.get()
    nvox = int(nvox_var1.get())
    size = int(size_var1.get())
    periodic = periodic_var1.get()

    ms_stats = {
        'Grain type': 'Elongated',
        'Equivalent diameter': {
            'sig': eq_diameter_sig.get(), 'scale': eq_diameter_scale.get(),
            'loc': eq_diameter_loc.get(), 'cutoff_min': eq_diameter_min.get(),
            'cutoff_max': eq_diameter_max.get()
        },
        'Aspect ratio': {
            'sig': aspect_ratio_sig.get(), 'scale': aspect_ratio_scale.get(),
            'loc': aspect_ratio_loc.get(), 'cutoff_min': aspect_ratio_min.get(),
            'cutoff_max': aspect_ratio_max.get()
        },
        "Tilt angle": {
            "kappa": tilt_angle_kappa.get(), "loc": tilt_angle_loc.get(),
            "cutoff_min": tilt_angle_min.get(), "cutoff_max": tilt_angle_max.get()
        },
        'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox},
        'Simulation': {'periodicity': str(periodic), 'output_units': 'mm'}
    }

    ms = knpy.Microstructure(descriptor=ms_stats, name=f"{matname}_{texture}_texture")
    ms.init_RVE()
    ms.pack()
    ms.voxelize()
    fig = ms.plot_voxels(silent=True, sliced=False)
    display_plot(fig, plot_type="rve")
    flist = ms.plot_stats_init(silent=True, get_res=True)
    for fig in flist:
        display_plot(fig, plot_type="stats")


def compare_statistics():
    """Create plot of initial statistics together with final microstructure
    descriptors for particles and voxels"""
    global ms, ms_stats
    if ms_stats is None or ms is None or ms.mesh is None:
        print("No microstructure. Create RVE first.")
        return
    gs_param = [ms_stats['Equivalent diameter']['sig'],
                ms_stats['Equivalent diameter']['loc'],
                ms_stats['Equivalent diameter']['scale']]
    ar_param = [ms_stats['Aspect ratio']['sig'],
                ms_stats['Aspect ratio']['loc'],
                ms_stats['Aspect ratio']['scale']]
    flist = ms.plot_stats(silent=True, gs_param=[gs_param], ar_param=[ar_param], enhanced_plot=True)
    for fig in flist:
        display_plot(fig, plot_type="stats")


""" Functions for RVEs with cuboid grains """


def run_simulation(texture, matname, ngr, nv_gr, size, nphases, periodic):
    """Create and plot microstructure object with cuboid grains
    return figure axes
    """
    global ms
    dim = (ngr[0] * nv_gr[0], ngr[1] * nv_gr[1], ngr[2] * nv_gr[2])
    stats_dict = {
        'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
                'Nx': dim[0], 'Ny': dim[1], 'Nz': dim[2]},
        'Simulation': {'periodicity': periodic, 'output_units': 'um'},
        'Phase': {'Name': matname, 'Volume fraction': 1.0}
    }
    ms = knpy.Microstructure('from_voxels')
    ms.name = matname
    ms.Ngr = np.prod(ngr)
    ms.nphases = 1
    ms.descriptor = [stats_dict]
    ms.ngrains = [ms.Ngr]
    ms.rve = RVE_creator(ms.descriptor, from_voxels=True)
    ms.simbox = Simulation_Box(size)
    ms.mesh = mesh_creator(dim)
    ms.mesh.create_voxels(ms.simbox)

    grains = np.zeros(dim, dtype=int)
    grain_dict = {}
    grain_phase_dict = {}
    for ih in range(ngr[0]):
        for ik in range(ngr[1]):
            for il in range(ngr[2]):
                igr = il + ik * ngr[1] + ih * ngr[0] * ngr[1] + 1
                grain_dict[igr] = []
                grain_phase_dict[igr] = 0
                ind0 = np.arange(nv_gr[0], dtype=int) + ih * nv_gr[0]
                ind1 = np.arange(nv_gr[1], dtype=int) + ik * nv_gr[1]
                ind2 = np.arange(nv_gr[2], dtype=int) + il * nv_gr[2]
                ind_list = itertools.product(*[ind0, ind1, ind2])
                for ind in ind_list:
                    nv = np.ravel_multi_index(ind, dim, order='F')
                    grain_dict[igr].append(nv + 1)
                grains[ind0[0]:ind0[-1] + 1, ind1[0]:ind1[-1] + 1, ind2[0]:ind2[-1] + 1] = igr

    ms.mesh.grains = grains
    ms.mesh.grain_dict = grain_dict
    ms.mesh.phases = np.zeros(dim, dtype=int)
    ms.mesh.grain_phase_dict = grain_phase_dict
    ms.mesh.ngrains_phase = ms.ngrains

    print("Simulation completed with parameters:", texture, matname, ngr, nv_gr, size, nphases, periodic)
    fig = ms.plot_voxels(sliced= True, silent=True)
    ptag = 'pbc' if periodic else 'no_pbc'
    fname = ms.write_abq(nodes='v', file=f'abq{nv_gr[0]}_gr{ngr[0]}_{ptag}_geom.inp')
    return fig


def run_simulation_from_gui():
    """Create and plot RVE with cubic grains from tab 2"""
    texture = texture_var2.get()
    matname = matname_var2.get()
    ngr = parse_entry(ngr_var.get())
    nv_gr = parse_entry(nv_gr_var.get())
    size = parse_entry(size_var2.get())
    nphases = int(nphases_var.get())
    periodic = periodic_var2.get()
    ax = run_simulation(texture, matname, ngr, nv_gr, size, nphases, periodic)
    display_plot_cuboid(ax.figure)


def write_abaqus_input_file_from_gui():
    texture = texture_var2.get()
    matname = matname_var2.get()
    ngr = parse_entry(ngr_var.get())
    nv_gr = parse_entry(nv_gr_var.get())
    size = parse_entry(size_var2.get())
    nphases = int(nphases_var.get())
    periodic = periodic_var2.get()
    write_abaqus_input_file(texture, matname, ngr, nv_gr, size, nphases, periodic)
    print("Sucess!")


def create_orientation_from_gui():
    texture = texture_var2.get()
    matname = matname_var2.get()
    ngr = parse_entry(ngr_var.get())
    nv_gr = parse_entry(nv_gr_var.get())
    size = parse_entry(size_var2.get())
    nphases = int(nphases_var.get())
    periodic = periodic_var2.get()
    create_orientation(texture, matname, ngr, nv_gr, size, nphases, periodic)
    print("Sucess!")


def display_plot_cuboid(fig):
    for widget in rve_plot_frame2.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=rve_plot_frame2)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    app.update_idletasks()
    width, height = app.winfo_reqwidth(), app.winfo_reqheight()
    app.geometry(f"{width}x{height}")


def write_abaqus_input_file(texture, matname, ngr, nv_gr, size, nphases, periodic):
    dim = (ngr[0] * nv_gr[0], ngr[1] * nv_gr[1], ngr[2] * nv_gr[2])
    stats_dict = {
        'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
                'Nx': dim[0], 'Ny': dim[1], 'Nz': dim[2]},
        'Simulation': {'periodicity': periodic, 'output_units': 'um'},
        'Phase': {'Name': matname, 'Volume fraction': 1.0}
    }
    ms = knpy.Microstructure('from_voxels')
    ms.name = matname
    ms.Ngr = np.prod(ngr)
    ms.nphases = 1
    ms.descriptor = [stats_dict]
    ms.ngrains = [ms.Ngr]
    ms.rve = RVE_creator(ms.descriptor, from_voxels=True)
    ms.simbox = Simulation_Box(size)
    ms.mesh = mesh_creator(dim)
    ms.mesh.create_voxels(ms.simbox)

    grains = np.zeros(dim, dtype=int)
    grain_dict = {}
    grain_phase_dict = {}
    for ih in range(ngr[0]):
        for ik in range(ngr[1]):
            for il in range(ngr[2]):
                igr = il + ik * ngr[1] + ih * ngr[0] * ngr[1] + 1
                grain_dict[igr] = []
                grain_phase_dict[igr] = 0
                ind0 = np.arange(nv_gr[0], dtype=int) + ih * nv_gr[0]
                ind1 = np.arange(nv_gr[1], dtype=int) + ik * nv_gr[1]
                ind2 = np.arange(nv_gr[2], dtype=int) + il * nv_gr[2]
                ind_list = itertools.product(*[ind0, ind1, ind2])
                for ind in ind_list:
                    nv = np.ravel_multi_index(ind, dim, order='F')
                    grain_dict[igr].append(nv + 1)
                grains[ind0[0]:ind0[-1] + 1, ind1[0]:ind1[-1] + 1, ind2[0]:ind2[-1] + 1] = igr

    ms.mesh.grains = grains
    ms.mesh.grain_dict = grain_dict
    ms.mesh.phases = np.zeros(dim, dtype=int)
    ms.mesh.grain_phase_dict = grain_phase_dict
    ms.mesh.ngrains_phase = ms.ngrains
    ms.write_voxels(script_name=__file__, mesh=False, system=False)


def create_orientation(texture, matname, ngr, nv_gr, size, nphases, periodic):
    dim = (ngr[0] * nv_gr[0], ngr[1] * nv_gr[1], ngr[2] * nv_gr[2])
    stats_dict = {
        'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
                'Nx': dim[0], 'Ny': dim[1], 'Nz': dim[2]},
        'Simulation': {'periodicity': periodic, 'output_units': 'um'},
        'Phase': {'Name': matname, 'Volume fraction': 1.0}
    }
    ms = knpy.Microstructure('from_voxels')
    ms.name = matname
    ms.Ngr = np.prod(ngr)
    ms.nphases = 1
    ms.descriptor = [stats_dict]
    ms.ngrains = [ms.Ngr]
    ms.rve = RVE_creator(ms.descriptor, from_voxels=True)
    ms.simbox = Simulation_Box(size)
    ms.mesh = mesh_creator(dim)
    ms.mesh.create_voxels(ms.simbox)

    grains = np.zeros(dim, dtype=int)
    grain_dict = {}
    grain_phase_dict = {}
    for ih in range(ngr[0]):
        for ik in range(ngr[1]):
            for il in range(ngr[2]):
                igr = il + ik * ngr[1] + ih * ngr[0] * ngr[1] + 1
                grain_dict[igr] = []
                grain_phase_dict[igr] = 0
                ind0 = np.arange(nv_gr[0], dtype=int) + ih * nv_gr[0]
                ind1 = np.arange(nv_gr[1], dtype=int) + ik * nv_gr[1]
                ind2 = np.arange(nv_gr[2], dtype=int) + il * nv_gr[2]
                ind_list = itertools.product(*[ind0, ind1, ind2])
                for ind in ind_list:
                    nv = np.ravel_multi_index(ind, dim, order='F')
                    grain_dict[igr].append(nv + 1)
                grains[ind0[0]:ind0[-1] + 1, ind1[0]:ind1[-1] + 1, ind2[0]:ind2[-1] + 1] = igr

    ms.mesh.grains = grains
    ms.mesh.grain_dict = grain_dict
    ms.mesh.phases = np.zeros(dim, dtype=int)
    ms.mesh.grain_phase_dict = grain_phase_dict
    ms.mesh.ngrains_phase = ms.ngrains
    if knpy.MTEX_AVAIL:
        ms.generate_orientations(texture, ang=[0, 45, 0], omega=7.5)


""" Main code section """
# define global variables
stats_canvas = None
rve_canvas1 = None
ms = None
ms_stats = None

# initialize app and notebook
app = tk.Tk()
app.geometry("1200x800")
app.title("RVE_Generation")

notebook = ttk.Notebook(app)
notebook.pack(fill='both', expand=True)
style = ttk.Style(app)
default_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
style.configure('TNotebook.Tab', font=('Helvetica', '12', "bold"))
style.configure('TButton', font=default_font)

""" First tab: Particle-based grains """
# define standard parameters
texture_var1 = tk.StringVar(value="unimodal")
matname_var1 = tk.StringVar(value="Simulanium")
nvox_var1 = tk.IntVar(value=30)
size_var1 = tk.IntVar(value=30)
periodic_var1 = tk.BooleanVar(value=True)

eq_diameter_sig = tk.DoubleVar(value=0.7)
eq_diameter_scale = tk.DoubleVar(value=20.0)
eq_diameter_loc = tk.DoubleVar(value=0.0)
eq_diameter_min = tk.DoubleVar(value=9.0)
eq_diameter_max = tk.DoubleVar(value=18.0)

aspect_ratio_sig = tk.DoubleVar(value=0.9)
aspect_ratio_scale = tk.DoubleVar(value=2.5)
aspect_ratio_loc = tk.DoubleVar(value=0.0)
aspect_ratio_min = tk.DoubleVar(value=1.0)
aspect_ratio_max = tk.DoubleVar(value=4.0)

tilt_angle_kappa = tk.DoubleVar(value=1.0)
tilt_angle_loc = tk.DoubleVar(value=0.5 * pi)
tilt_angle_min = tk.DoubleVar(value=0.0)
tilt_angle_max = tk.DoubleVar(value=2 * pi)

# plot frames
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Create RVE")

main_frame1 = ttk.Frame(tab1)
main_frame1.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)

plot_frame1 = ttk.Frame(tab1)
plot_frame1.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
plot_frame1.rowconfigure(0, weight=1)
plot_frame1.columnconfigure(0, weight=1)

stats_plot_frame = ttk.Frame(plot_frame1)
stats_plot_frame.grid(row=0, column=0, sticky='nsew')

rve_plot_frame1 = ttk.Frame(plot_frame1)
rve_plot_frame1.grid(row=1, column=0, sticky='nsew')

# define labels and buttons
add_label_and_entry(0, "Texture", texture_var1, entry_type="combobox", options=["random", "unimodal"])
add_label_and_entry(1, "Material Name", matname_var1)
add_label_and_entry(2, "Number of Voxels", nvox_var1)
add_label_and_entry(3, "Size of RVE (in microns)", size_var1)
add_label_and_entry(4, "Periodic", periodic_var1, entry_type="checkbox")

ttk.Label(main_frame1, text="Equivalent Diameter Parameters", font=("Helvetica", 12, "bold")) \
    .grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky='w')
add_label_and_entry(6, "Sigma", eq_diameter_sig, bold=False)
add_label_and_entry(7, "Scale", eq_diameter_scale, bold=False)
add_label_and_entry(8, "Location", eq_diameter_loc, bold=False)
add_label_and_entry(9, "Min", eq_diameter_min, bold=False)
add_label_and_entry(10, "Max", eq_diameter_max, bold=False)

ttk.Label(main_frame1, text="Aspect Ratio Parameters", font=("Helvetica", 12, "bold")) \
    .grid(row=11, column=0, columnspan=2, pady=(10, 0), sticky='w')
add_label_and_entry(12, "Sigma", aspect_ratio_sig, bold=False)
add_label_and_entry(13, "Scale", aspect_ratio_scale, bold=False)
add_label_and_entry(14, "Location", aspect_ratio_loc, bold=False)
add_label_and_entry(15, "Min", aspect_ratio_min, bold=False)
add_label_and_entry(16, "Max", aspect_ratio_max, bold=False)

ttk.Label(main_frame1, text="Tilt Angle Parameters", font=("Helvetica", 12, "bold")) \
    .grid(row=17, column=0, columnspan=2, pady=(10, 0), sticky='w')
add_label_and_entry(18, "Kappa", tilt_angle_kappa, bold=False)
add_label_and_entry(19, "Location", tilt_angle_loc, bold=False)
add_label_and_entry(20, "Min", tilt_angle_min, bold=False)
add_label_and_entry(21, "Max", tilt_angle_max, bold=False)

button_frame1 = ttk.Frame(main_frame1)
button_frame1.grid(row=22, column=0, columnspan=2, pady=10, sticky='ew')

button_create_rve = ttk.Button(button_frame1, text="Create RVE", style='TButton', command=create_rve_and_plot)
button_create_rve.grid(row=0, column=1, padx=10)

button_statistics = ttk.Button(button_frame1, text="Statistics", style='TButton',
                               command=create_and_plot_stats)
button_statistics.grid(row=0, column=0, padx=10)

button_compare_statistics = ttk.Button(button_frame1, text="Compare Statistics", style='TButton',
                                       command=compare_statistics)
button_compare_statistics.grid(row=0, column=2, padx=10)

""" Second tab: Cuboid grains """
# define standard parameters
texture_var2 = tk.StringVar(value="random")
matname_var2 = tk.StringVar(value="Simulanium")
ngr_var = tk.StringVar(value="5,5,5")
nv_gr_var = tk.StringVar(value="3,3,3")
size_var2 = tk.StringVar(value="45,45,45")
nphases_var = tk.StringVar(value="1")
periodic_var2 = tk.BooleanVar(value=False)

# plot frames
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Cuboid Grains")

main_frame2 = ttk.Frame(tab2)
main_frame2.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)
plot_frame2 = ttk.Frame(tab2)
plot_frame2.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
plot_frame2.rowconfigure(0, weight=1)
plot_frame2.columnconfigure(0, weight=1)

rve_plot_frame2 = ttk.Frame(plot_frame2)
rve_plot_frame2.grid(row=0, column=0, sticky='nsew')

labels2 = ["Texture", "Material Name", "Number of Grains",
           "Number of Voxels", "Size of RVE (in micron)", "Number of Phases"]
entries2 = [ttk.Combobox(main_frame2, textvariable=texture_var2, values=['random', 'unimodal'], font=("Helvetica", 12),
                         width=20),
            tk.Entry(main_frame2, textvariable=matname_var2, font=("Helvetica", 12), width=20),
            tk.Entry(main_frame2, textvariable=ngr_var, font=("Helvetica", 12), width=20),
            tk.Entry(main_frame2, textvariable=nv_gr_var, font=("Helvetica", 12), width=20),
            tk.Entry(main_frame2, textvariable=size_var2, font=("Helvetica", 12), width=20),
            tk.Entry(main_frame2, textvariable=nphases_var, font=("Helvetica", 12), width=20)]

for i, (label, entry) in enumerate(zip(labels2, entries2)):
    tk.Label(main_frame2, text=label, font=("Helvetica", 12, "bold")).grid(row=i, column=0, sticky="w")
    entry.grid(row=i, column=1, sticky="ew")

button_frame2 = ttk.Frame(main_frame2)
button_frame2.grid(row=len(labels2), column=0, columnspan=2, pady=10, sticky='ew')
run_simulation_button = ttk.Button(button_frame2, text="Create RVE", style='TButton', command=run_simulation_from_gui)
run_simulation_button.grid(row=0, column=0, padx=10)
create_orientation_button = ttk.Button(button_frame2, text="Create Orientation", style='TButton',
                                       command=create_orientation_from_gui)
create_orientation_button.grid(row=0, column=1, padx=10)
write_files_button = ttk.Button(button_frame2, text="Write Abaqus Input Files", style='TButton',
                                command=write_abaqus_input_file_from_gui)
write_files_button.grid(row=0, column=2, padx=10)

""" Start main loop """
app.mainloop()
