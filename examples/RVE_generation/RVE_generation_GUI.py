#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Graphical User Interface for create_rve.py and cuboid_grains.py
Created on May 2024
@author: Ronak Shoghi, Alexander Hartmaier
"""
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import kanapy as knpy
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kanapy.initializations import RVE_creator, mesh_creator
from kanapy.entities import Simulation_Box

"""   ===== Global Subroutines =====   """


def self_closing_message(message, duration=4000):
    """
    Display a self-closing message box.

    :param message: The message to be displayed.
    :param duration: The time in milliseconds before the message box closes automatically.
    :return: A reference to the popup window.
    """
    popup = Toplevel()
    popup.title("Information")
    popup.geometry("300x100")

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    window_width = 300
    window_height = 100
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    popup.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    label = ttk.Label(popup, text=message, font=("Helvetica", 12), wraplength=250)
    label.pack(expand=True)
    popup.after(duration, popup.destroy)
    popup.update_idletasks()
    return


def add_label_and_entry(frame, row, label_text, entry_var, entry_type="entry", bold=True, options=None):
    label_font = ("Helvetica", 12, "bold") if bold else ("Helvetica", 12)
    ttk.Label(frame, text=label_text, font=label_font).grid(row=row, column=0, sticky='w')
    if entry_type == "entry":
        ttk.Entry(frame, textvariable=entry_var, width=20, font=("Helvetica", 12)) \
            .grid(row=row, column=1, sticky='e')
    elif entry_type == "checkbox":
        ttk.Checkbutton(frame, variable=entry_var).grid(row=row, column=1, sticky='e')
    elif entry_type == "combobox" and options is not None:
        combobox = ttk.Combobox(frame, textvariable=entry_var, values=options, state='readonly',
                                width=19)
        combobox.grid(row=row, column=1, sticky='e')
        combobox.configure(font=("Helvetica", 12))
        combobox.current(0)
    return


def close():
    global app
    app.quit()
    app.destroy()


def parse_entry(entry):
    return list(map(int, entry.strip().split(',')))


class particle_rve(object):
    """Class for RVEs based on particle simulations
    first tab
    """

    def __init__(self):
        # define standard parameters
        self.ms = None
        self.ms_stats = None
        self.stats_canvas1 = None
        self.rve_canvas1 = None
        self.texture_var1 = tk.StringVar(value="random")
        self.matname_var1 = tk.StringVar(value="Simulanium")
        self.ialloy = tk.IntVar(value=2)
        self.nvox_var1 = tk.IntVar(value=30)
        self.size_var1 = tk.IntVar(value=30)
        self.periodic_var1 = tk.BooleanVar(value=True)

        self.eq_diameter_sig = tk.DoubleVar(value=0.7)
        self.eq_diameter_scale = tk.DoubleVar(value=20.0)
        self.eq_diameter_loc = tk.DoubleVar(value=0.0)
        self.eq_diameter_min = tk.DoubleVar(value=9.0)
        self.eq_diameter_max = tk.DoubleVar(value=18.0)

        self.aspect_ratio_sig = tk.DoubleVar(value=0.9)
        self.aspect_ratio_scale = tk.DoubleVar(value=2.5)
        self.aspect_ratio_loc = tk.DoubleVar(value=0.0)
        self.aspect_ratio_min = tk.DoubleVar(value=1.0)
        self.aspect_ratio_max = tk.DoubleVar(value=4.0)

        self.tilt_angle_kappa = tk.DoubleVar(value=1.0)
        self.tilt_angle_loc = tk.DoubleVar(value=round(0.5*np.pi, 4))
        self.tilt_angle_min = tk.DoubleVar(value=0.0)
        self.tilt_angle_max = tk.DoubleVar(value=round(np.pi, 4))
        if self.texture_var1.get() == 'random':
            self.kernel_var1 = tk.StringVar(value="-")
            self.euler_var1 = tk.StringVar(value="-")
        else:
            self.kernel_var1 = tk.StringVar(value="7.5")
            self.euler_var1 = tk.StringVar(value="0.0, 45.0, 0.0")
        self.texture_var1.trace('w', self.update_kernel_var)
        self.texture_var1.trace('w', self.update_euler_var)

        # plot frames
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Standard RVE")
        main_frame1 = ttk.Frame(tab1)
        main_frame1.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)

        plot_frame1 = ttk.Frame(tab1)
        plot_frame1.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
        plot_frame1.rowconfigure(0, weight=1)
        plot_frame1.columnconfigure(0, weight=1)

        self.stats_plot_frame = ttk.Frame(plot_frame1)
        self.stats_plot_frame.grid(row=0, column=0, sticky='nsew')

        self.rve_plot_frame1 = ttk.Frame(plot_frame1)
        self.rve_plot_frame1.grid(row=1, column=0, sticky='nsew')

        # define labels and entries
        line_seq = np.linspace(0, 50, dtype=int)
        line = iter(line_seq)
        ttk.Label(main_frame1, text="General Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Material Name", self.matname_var1, bold=False)
        add_label_and_entry(main_frame1, next(line), "Material Number", self.ialloy, bold=False)
        add_label_and_entry(main_frame1, next(line), "Number of Voxels", self.nvox_var1, bold=False)
        add_label_and_entry(main_frame1, next(line), "Size of RVE (in micron)", self.size_var1, bold=False)
        add_label_and_entry(main_frame1, next(line), "Periodic", self.periodic_var1, entry_type="checkbox",
                            bold=False)

        ttk.Label(main_frame1, text="Equivalent Diameter Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Sigma", self.eq_diameter_sig, bold=False)
        add_label_and_entry(main_frame1, next(line), "Scale", self.eq_diameter_scale, bold=False)
        add_label_and_entry(main_frame1, next(line), "Location", self.eq_diameter_loc, bold=False)
        add_label_and_entry(main_frame1, next(line), "Min", self.eq_diameter_min, bold=False)
        add_label_and_entry(main_frame1, next(line), "Max", self.eq_diameter_max, bold=False)

        ttk.Label(main_frame1, text="Aspect Ratio Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Sigma", self.aspect_ratio_sig, bold=False)
        add_label_and_entry(main_frame1, next(line), "Scale", self.aspect_ratio_scale, bold=False)
        add_label_and_entry(main_frame1, next(line), "Location", self.aspect_ratio_loc, bold=False)
        add_label_and_entry(main_frame1, next(line), "Min", self.aspect_ratio_min, bold=False)
        add_label_and_entry(main_frame1, next(line), "Max", self.aspect_ratio_max, bold=False)

        ttk.Label(main_frame1, text="Tilt Angle Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Kappa", self.tilt_angle_kappa, bold=False)
        add_label_and_entry(main_frame1, next(line), "Location", self.tilt_angle_loc, bold=False)
        add_label_and_entry(main_frame1, next(line), "Min", self.tilt_angle_min, bold=False)
        add_label_and_entry(main_frame1, next(line), "Max", self.tilt_angle_max, bold=False)

        ttk.Label(main_frame1, text="Orientation Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Texture", self.texture_var1, entry_type="combobox",
                            options=["random", "unimodal"], bold=False)
        add_label_and_entry(main_frame1, next(line), "Kernel Half Width (degree)", self.kernel_var1,
                            bold=False)
        add_label_and_entry(main_frame1, next(line), "Euler Angles (degree)", self.euler_var1, bold=False)

        # create buttons
        button_frame1 = ttk.Frame(main_frame1)
        button_frame1.grid(row=next(line), column=0, columnspan=2, pady=10, sticky='ew')

        button_statistics = ttk.Button(button_frame1, text="Statistics", style='TButton',
                                       command=self.create_and_plot_stats)
        button_statistics.grid(row=0, column=0, padx=10)
        button_create_rve = ttk.Button(button_frame1, text="Create RVE", style='TButton',
                                       command=self.create_and_plot_rve)
        button_create_rve.grid(row=0, column=1, padx=10)

        button_create_ori = ttk.Button(button_frame1, text="Create Orientations", style='TButton',
                                       command=self.create_orientation)
        button_create_ori.grid(row=0, column=2, padx=10)
        write_files_button = ttk.Button(button_frame1, text="Write Abaqus Input", style='TButton',
                                        command=self.export_abq)
        write_files_button.grid(row=1, column=0, padx=10)
        button_exit1 = ttk.Button(button_frame1, text="Exit", style='TButton', command=close)
        button_exit1.grid(row=1, column=2, padx=10)

    def update_kernel_var(self, *args):
        self.kernel_var1.set("-" if self.texture_var1.get() == 'random' else "7.5")


    def update_euler_var(self, *args):
        self.euler_var1.set("-" if self.texture_var1.get() == 'random' else "0.0, 45.0, 0.0")


    def display_plot(self, fig, plot_type):
        """ Show either stats graph or RVE on canvas. """
        global app
        app.update_idletasks()
        width, height = app.winfo_reqwidth(), app.winfo_reqheight()
        app.geometry(f"{width}x{height}")
        if plot_type == "stats":
            if self.stats_canvas1 is not None:
                self.stats_canvas1.get_tk_widget().destroy()
            self.stats_canvas1 = FigureCanvasTkAgg(fig, master=self.stats_plot_frame)
            self.stats_canvas1.draw()
            self.stats_canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        elif plot_type == "rve":
            if self.rve_canvas1 is not None:
                self.rve_canvas1.get_tk_widget().destroy()
            self.rve_canvas1 = FigureCanvasTkAgg(fig, master=self.rve_plot_frame1)
            self.rve_canvas1.draw()
            self.rve_canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        app.update_idletasks()
        width, height = app.winfo_reqwidth(), app.winfo_reqheight()
        app.geometry(f"{width}x{height}")

    def create_and_plot_stats(self):
        """Plot statistics of current microstructure descriptors
        Will erase global microstructure object if it exists."""
        texture = self.texture_var1.get()
        matname = self.matname_var1.get()
        nvox = int(self.nvox_var1.get())
        size = int(self.size_var1.get())
        periodic = self.periodic_var1.get()

        ms_stats = {
            'Grain type': 'Elongated',
            'Equivalent diameter': {
                'sig': self.eq_diameter_sig.get(), 'scale': self.eq_diameter_scale.get(),
                'loc': self.eq_diameter_loc.get(), 'cutoff_min': self.eq_diameter_min.get(),
                'cutoff_max': self.eq_diameter_max.get()
            },
            'Aspect ratio': {
                'sig': self.aspect_ratio_sig.get(), 'scale': self.aspect_ratio_scale.get(),
                'loc': self.aspect_ratio_loc.get(), 'cutoff_min': self.aspect_ratio_min.get(),
                'cutoff_max': self.aspect_ratio_max.get()
            },
            "Tilt angle": {
                "kappa": self.tilt_angle_kappa.get(), "loc": self.tilt_angle_loc.get(),
                "cutoff_min": self.tilt_angle_min.get(), "cutoff_max": self.tilt_angle_max.get()
            },
            'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox,
                    'ialloy': int(self.ialloy.get())},
            'Simulation': {'periodicity': periodic, 'output_units': 'mm'}
        }

        self.ms = knpy.Microstructure(descriptor=ms_stats, name=f"{matname}_{texture}_texture")
        self.ms.init_RVE()
        flist = self.ms.plot_stats_init(silent=True)
        for fig in flist:
            self.display_plot(fig, plot_type="stats")

    def create_and_plot_rve(self):
        """Create and plot the RVE
        Will overwrite existing ms_stats and ms objects"""

        self_closing_message("The process has been started, please wait...")
        start_time = time.time()
        matname = self.matname_var1.get()
        nvox = int(self.nvox_var1.get())
        size = int(self.size_var1.get())
        periodic = self.periodic_var1.get()

        self.ms_stats = {
            'Grain type': 'Elongated',
            'Equivalent diameter': {
                'sig': self.eq_diameter_sig.get(), 'scale': self.eq_diameter_scale.get(),
                'loc': self.eq_diameter_loc.get(), 'cutoff_min': self.eq_diameter_min.get(),
                'cutoff_max': self.eq_diameter_max.get()
            },
            'Aspect ratio': {
                'sig': self.aspect_ratio_sig.get(), 'scale': self.aspect_ratio_scale.get(),
                'loc': self.aspect_ratio_loc.get(), 'cutoff_min': self.aspect_ratio_min.get(),
                'cutoff_max': self.aspect_ratio_max.get()
            },
            "Tilt angle": {
                "kappa": self.tilt_angle_kappa.get(), "loc": self.tilt_angle_loc.get(),
                "cutoff_min": self.tilt_angle_min.get(), "cutoff_max": self.tilt_angle_max.get()
            },
            'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': nvox, 'Ny': nvox, 'Nz': nvox,
                    'ialloy': int(self.ialloy.get())},
            'Simulation': {'periodicity': str(periodic), 'output_units': 'mm'}
        }

        self.ms = knpy.Microstructure(descriptor=self.ms_stats, name=f"{matname}")
        self.ms.init_RVE()
        self.ms.pack()
        self.ms.voxelize()
        fig = self.ms.plot_voxels(silent=True, sliced=False)
        self.display_plot(fig, plot_type="rve")
        flist = self.ms.plot_stats_init(silent=True, get_res=True)
        end_time = time.time()
        duration = end_time - start_time
        self_closing_message(f"Process completed in {duration:.2f} seconds")
        for fig in flist:
            self.display_plot(fig, plot_type="stats")

    def create_orientation(self):
        """A function to create the orientation """
        if not knpy.MTEX_AVAIL:
            self_closing_message("Generation of grain orientation requires MTEX module.")
        self_closing_message("The process has been started, please wait...")
        texture = self.texture_var1.get()
        matname = self.matname_var1.get()
        if texture == 'unimodal':
            omega = float(self.kernel_var1.get())
            ang_string = self.euler_var1.get()
            ang = [float(angle.strip()) for angle in ang_string.split(',')]
        else:
            omega = None
            ang = None
        if self.ms_stats is None or self.ms is None or self.ms.mesh is None:
            self_closing_message("Generating RVE and assigning orientations to grains.")
            self.create_and_plot_rve()

        start_time = time.time()
        self.ms.generate_orientations(texture, ang=ang, omega=omega)
        self.ms.write_voxels(file=f'{matname}_voxels.json', script_name=__file__, mesh=False,
                             system=False)
        fig = self.ms.plot_voxels(silent=True, sliced=False, ori=True)
        self.display_plot(fig, plot_type="rve")
        end_time = time.time()
        duration = end_time - start_time
        self_closing_message(f"Process completed in {duration:.2f} seconds, the Voxel file has been saved. ")

    def export_abq(self):
        if self.ms_stats is None or self.ms is None or self.ms.mesh is None:
            self_closing_message("Generating and exporting RVE without orientations.")
            self.create_and_plot_rve()
        self.ms.write_abq('v')


class cuboid_rve(object):
    """ Functions for RVEs with cuboid grains 
    second tab"""

    def __init__(self):
        # define standard parameters
        self.ms = None
        self.canvas = None
        self.texture_var2 = tk.StringVar(value="random")
        self.matname_var2 = tk.StringVar(value="Simulanium")
        self.ialloy = tk.IntVar(value=2)
        self.ngr_var = tk.StringVar(value="5, 5, 5")
        self.nv_gr_var = tk.StringVar(value="3, 3, 3")
        self.size_var2 = tk.StringVar(value="45, 45, 45")
        if self.texture_var2.get() == 'random':
            self.kernel_var2 = tk.StringVar(value="-")
            self.euler_var2 = tk.StringVar(value="-")
        else:
            self.kernel_var2 = tk.StringVar(value="7.5")
            self.euler_var2 = tk.StringVar(value="0.0, 45.0, 0.0")
        self.texture_var2.trace('w', self.update_kernel_var)
        self.texture_var2.trace('w', self.update_euler_var)

        # plot frames
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Cuboid Grains")
        main_frame2 = ttk.Frame(tab2)
        main_frame2.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)

        plot_frame2 = ttk.Frame(tab2)
        plot_frame2.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
        plot_frame2.rowconfigure(0, weight=1)
        plot_frame2.columnconfigure(0, weight=1)

        self.rve_plot_frame2 = ttk.Frame(plot_frame2)
        self.rve_plot_frame2.grid(row=0, column=0, sticky='nsew')

        # define labels and entries
        line_seq = np.linspace(0, 50, dtype=int)
        line = iter(line_seq)
        ttk.Label(main_frame2, text="General Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame2, next(line), "Material Name", self.matname_var2, bold=False)
        add_label_and_entry(main_frame2, next(line), "Material Number", self.ialloy, bold=False)
        add_label_and_entry(main_frame2, next(line), "Number of Grains", self.ngr_var, bold=False)
        add_label_and_entry(main_frame2, next(line), "Number of Voxels", self.nv_gr_var, bold=False)
        add_label_and_entry(main_frame2, next(line), "Size of RVE (in micron)", self.size_var2, bold=False)

        ttk.Label(main_frame2, text="Orientation Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame2, next(line), "Texture", self.texture_var2, entry_type="combobox",
                            options=["random", "unimodal"], bold=False)
        add_label_and_entry(main_frame2, next(line), "Kernel Half Width (degree)", self.kernel_var2,
                            bold=False)
        add_label_and_entry(main_frame2, next(line), "Euler Angles (degree)", self.euler_var2, bold=False)

        # add buttons
        button_frame2 = ttk.Frame(main_frame2)
        button_frame2.grid(row=next(line), column=0, columnspan=2, pady=10, sticky='ew')
        run_simulation_button = ttk.Button(button_frame2, text="Create RVE", style='TButton',
                                           command=self.create_cubes_and_plot)
        run_simulation_button.grid(row=0, column=0, padx=10)
        create_orientation_button = ttk.Button(button_frame2, text="Create Orientations", style='TButton',
                                               command=self.create_orientation)
        create_orientation_button.grid(row=0, column=1, padx=10)
        write_files_button = ttk.Button(button_frame2, text="Write Abaqus Input", style='TButton',
                                        command=self.export_abq)
        write_files_button.grid(row=1, column=0, padx=10)
        button_exit2 = ttk.Button(button_frame2, text="Exit", style='TButton', command=close)
        button_exit2.grid(row=1, column=1, padx=10)

    def update_kernel_var(self, *args):
        self.kernel_var2.set("-" if self.texture_var2.get() == 'random' else "7.5")

    def update_euler_var(self, *args):
        self.euler_var2.set("-" if self.texture_var2.get() == 'random' else "0.0, 45.0, 0.0")

    def display_cuboid(self, fig):
        global app
        app.update_idletasks()
        width, height = app.winfo_reqwidth(), app.winfo_reqheight()
        app.geometry(f"{width}x{height}")
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.rve_plot_frame2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        app.update_idletasks()
        width, height = app.winfo_reqwidth(), app.winfo_reqheight()
        app.geometry(f"{width}x{height}")

    def create_cubes_and_plot(self):
        """Create and plot microstructure object with cuboid grains
        return figure axes
        """
        matname = self.matname_var2.get()
        ngr = parse_entry(self.ngr_var.get())
        nv_gr = parse_entry(self.nv_gr_var.get())
        size = parse_entry(self.size_var2.get())
        dim = (ngr[0] * nv_gr[0], ngr[1] * nv_gr[1], ngr[2] * nv_gr[2])
        stats_dict = {
            'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
                    'Nx': dim[0], 'Ny': dim[1], 'Nz': dim[2],
                    'ialloy': int(self.ialloy.get())},
            'Simulation': {'periodicity': False, 'output_units': 'um'},
            'Phase': {'Name': matname, 'Volume fraction': 1.0}
        }
        self.ms = knpy.Microstructure('from_voxels')
        self.ms.name = matname
        self.ms.Ngr = np.prod(ngr)
        self.ms.nphases = 1
        self.ms.descriptor = [stats_dict]
        self.ms.ngrains = [self.ms.Ngr]
        self.ms.rve = RVE_creator(self.ms.descriptor, from_voxels=True)
        self.ms.simbox = Simulation_Box(size)
        self.ms.mesh = mesh_creator(dim)
        self.ms.mesh.create_voxels(self.ms.simbox)

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

        self.ms.mesh.grains = grains
        self.ms.mesh.grain_dict = grain_dict
        self.ms.mesh.phases = np.zeros(dim, dtype=int)
        self.ms.mesh.grain_phase_dict = grain_phase_dict
        self.ms.mesh.ngrains_phase = self.ms.ngrains
        fig = self.ms.plot_voxels(sliced=False, silent=True)
        self.display_cuboid(fig)
        return


    def create_orientation(self):
        """Create grain orientations according to texture descriptors"""
        if not knpy.MTEX_AVAIL:
            self_closing_message("Generation of grain orientation requires MTEX module.")
            return
        self_closing_message("The process has been started, please wait...")
        texture = self.texture_var2.get()
        matname = self.matname_var2.get()
        if texture == 'unimodal':
            omega = float(self.kernel_var2.get())
            ang_string = self.euler_var2.get()
            ang = [float(angle.strip()) for angle in ang_string.split(',')]
        else:
            omega = None
            ang = None
        start_time = time.time()
        if self.ms is None or self.ms.mesh is None:
            self_closing_message("Generating RVE with cuboidal grains to assign orientation to")
            self.create_cubes_and_plot()
        self.ms.generate_orientations(texture, ang=ang, omega=omega)
        self.ms.write_voxels(file=f'{matname}_voxels.json', script_name=__file__, mesh=False, system=False)
        fig = self.ms.plot_voxels(silent=True, sliced=False, ori=True)
        self.display_cuboid(fig)
        end_time = time.time()
        duration = end_time - start_time
        self_closing_message(f"Process completed in {duration:.2f} seconds, the Voxel file has been saved.")


    def export_abq(self):
        if self.ms is None:
            self_closing_message("Generating and exporting RVE with cuboidal grains w/o orientations.")
            self.create_cubes_and_plot()
        self.ms.write_abq('v')


""" Main code section """
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
prve = particle_rve()  # First tab: Particle-based grains
crve = cuboid_rve()  # Second tab: Cuboid grains
app.mainloop()
