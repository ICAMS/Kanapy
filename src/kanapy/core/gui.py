#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Graphical User Interface for create_rve.py, cuboid_grains.py and cpnvert_ang2rve.py
Created on May 2024
last Update Oct 2024
@author: Ronak Shoghi, Alexander Hartmaier
"""
import sys
import time
import itertools
import numpy as np
import tkinter as tk
from tkinter import ttk, Toplevel, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .api import Microstructure
from .initializations import RVE_creator, mesh_creator
from .input_output import import_stats, write_stats
from .entities import Simulation_Box

if 'kanapy_mtex' in sys.modules:
    from kanapy_mtex.texture import EBSDmap
else:
    from kanapy.texture import EBSDmap


def self_closing_message(message, duration=4000):
    """
    Display a temporary popup message box that closes automatically after a given duration

    Parameters
    ----------
    message : str
        The message text to display in the popup window
    duration : int, optional
        Time in milliseconds before the message box closes automatically (default is 4000)

    Returns
    -------
    None
        The function creates and destroys a popup window; no value is returned
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


def add_label_and_entry(frame, row, label_text, entry_var, entry_type="entry", bold=False,
                        options=None, col=0):
    """
    Add a label and an input widget (entry, checkbox, or combobox) to a given frame

    Parameters
    ----------
    frame : tkinter.Frame
        Parent frame where the widgets will be added
    row : int
        Row index in the frame grid layout
    label_text : str or None
        Text for the label; if None, no label is created
    entry_var : tkinter variable
        Variable bound to the input widget (e.g., StringVar, BooleanVar)
    entry_type : str, optional
        Type of input widget: "entry", "checkbox", or "combobox" (default is "entry")
    bold : bool, optional
        Whether to display the label text in bold (default is False)
    options : list, optional
        List of selectable options when entry_type is "combobox"
    col : int, optional
        Column index for placing the label and entry in the grid (default is 0)

    Returns
    -------
    None
        The function adds widgets directly to the frame
    """
    if label_text is not None:
        label_font = ("Helvetica", 12, "bold") if bold else ("Helvetica", 12)
        ttk.Label(frame, text=label_text, font=label_font).grid(row=row, column=col, sticky='w')
        col_entry = col + 1
    else:
        col_entry = col
    if entry_type == "entry":
        ttk.Entry(frame, textvariable=entry_var, width=20, font=("Helvetica", 12)) \
            .grid(row=row, column=col_entry, sticky='e')
    elif entry_type == "checkbox":
        ttk.Checkbutton(frame, variable=entry_var).grid(row=row, column=col_entry, sticky='e')
    elif entry_type == "combobox" and options is not None:
        combobox = ttk.Combobox(frame, textvariable=entry_var, values=options, state='readonly',
                                width=19)
        combobox.grid(row=row, column=col_entry, sticky='e')
        combobox.configure(font=("Helvetica", 12))
        combobox.current(0)
    return


def parse_entry(entry):
    """
    Convert a comma-separated string into a list of integers

    Parameters
    ----------
    entry : str
        String of comma-separated integer values

    Returns
    -------
    list of int
        List of integers parsed from the input string
    """
    return list(map(int, entry.strip().split(',')))


class particle_rve(object):
    """
    Class for generating Representative Volume Elements (RVEs) based on particle simulations
    first tab

    This class defines the GUI logic and default parameters for building RVEs using
    particle-based microstructure simulations. It manages various parameters such as
    particle size distribution, aspect ratio, texture settings, and periodic boundary conditions.

    Parameters
    ----------
    app : tkinter.Tk or tkinter.Frame
        Reference to the main Tkinter application instance.
    notebook : ttk.Notebook
        The notebook widget in which this tab (RVE setup) is placed.

    Attributes
    ----------
    app : tkinter.Tk or tkinter.Frame
        Main application reference.
    ms : object or None
        Placeholder for microstructure data.
    ms_stats : object or None
        Placeholder for microstructure statistical data.
    ebsd : object or None
        Placeholder for EBSD data (if used).
    stats_canvas1 : tkinter.Canvas or None
        Canvas for displaying statistical plots.
    rve_canvas1 : tkinter.Canvas or None
        Canvas for displaying the RVE visualization.

    texture_var1 : tk.StringVar
        Texture type, default "random".
    matname_var1 : tk.StringVar
        Material name, default "Simulanium".
    nphases : tk.IntVar
        Number of material phases, default 1.
    ialloy : tk.IntVar
        Alloy type indicator, default 2.
    nvox_var1 : tk.IntVar
        Number of voxels per dimension in the RVE, default 30.
    size_var1 : tk.IntVar
        Physical size of the RVE, default 30.
    periodic_var1 : tk.BooleanVar
        Whether to apply periodic boundary conditions, default True.
    volume_fraction : tk.DoubleVar
        Target total volume fraction, default 1.0.
    vf_act : tk.DoubleVar
        Actual volume fraction, initialized to 0.0.

    eq_diameter_sig : tk.DoubleVar
        Standard deviation of equivalent diameter distribution.
    eq_diameter_scale : tk.DoubleVar
        Scale parameter for equivalent diameter distribution.
    eq_diameter_loc : tk.DoubleVar
        Location parameter for equivalent diameter distribution.
    eq_diameter_min : tk.DoubleVar
        Minimum equivalent diameter.
    eq_diameter_max : tk.DoubleVar
        Maximum equivalent diameter.
    eqd_act_sig : tk.DoubleVar
        Active (fitted or adjusted) standard deviation for equivalent diameter.
    eqd_act_scale : tk.DoubleVar
        Active scale for equivalent diameter.
    eqd_act_loc : tk.DoubleVar
        Active location for equivalent diameter.

    aspect_ratio_sig : tk.DoubleVar
        Standard deviation for aspect ratio distribution.
    aspect_ratio_scale : tk.DoubleVar
        Scale parameter for aspect ratio distribution.
    aspect_ratio_loc : tk.DoubleVar
        Location parameter for aspect ratio distribution.
    aspect_ratio_min : tk.DoubleVar
        Minimum aspect ratio.
    aspect_ratio_max : tk.DoubleVar
        Maximum aspect ratio.
    ar_act_sig : tk.DoubleVar
        Active standard deviation for aspect ratio.
    ar_act_scale : tk.DoubleVar
        Active scale for aspect ratio.
    ar_act_loc : tk.DoubleVar
        Active location for aspect ratio.

    tilt_angle_kappa : tk.DoubleVar
        Concentration parameter for tilt angle distribution.
    tilt_angle_loc : tk.DoubleVar
        Mean (location) of tilt angle distribution in radians.
    tilt_angle_min : tk.DoubleVar
        Minimum tilt angle (0.0).
    tilt_angle_max : tk.DoubleVar
        Maximum tilt angle (π).

    kernel_var1 : tk.StringVar
        Orientation kernel width; depends on texture setting.
    euler_var1 : tk.StringVar
        Euler angles defining the texture orientation; depends on texture setting.
    """

    def __init__(self, app, notebook):
        # define standard parameters
        self.app = app
        self.ms = None
        self.ms_stats = None
        self.ebsd = None
        self.stats_canvas1 = None
        self.rve_canvas1 = None
        self.texture_var1 = tk.StringVar(value="random")
        self.matname_var1 = tk.StringVar(value="Simulanium")
        self.nphases = tk.IntVar(value=1)
        self.ialloy = tk.IntVar(value=2)
        self.nvox_var1 = tk.IntVar(value=30)
        self.size_var1 = tk.IntVar(value=30)
        self.periodic_var1 = tk.BooleanVar(value=True)
        self.volume_fraction = tk.DoubleVar(value=1.0)
        self.vf_act = tk.DoubleVar(value=0.0)

        self.eq_diameter_sig = tk.DoubleVar(value=0.7)
        self.eq_diameter_scale = tk.DoubleVar(value=20.0)
        self.eq_diameter_loc = tk.DoubleVar(value=0.0)
        self.eq_diameter_min = tk.DoubleVar(value=9.0)
        self.eq_diameter_max = tk.DoubleVar(value=18.0)
        self.eqd_act_sig = tk.DoubleVar(value=0.0)
        self.eqd_act_scale = tk.DoubleVar(value=0.0)
        self.eqd_act_loc = tk.DoubleVar(value=0.0)

        self.aspect_ratio_sig = tk.DoubleVar(value=0.9)
        self.aspect_ratio_scale = tk.DoubleVar(value=2.5)
        self.aspect_ratio_loc = tk.DoubleVar(value=0.0)
        self.aspect_ratio_min = tk.DoubleVar(value=1.0)
        self.aspect_ratio_max = tk.DoubleVar(value=4.0)
        self.ar_act_sig = tk.DoubleVar(value=0.0)
        self.ar_act_scale = tk.DoubleVar(value=0.0)
        self.ar_act_loc = tk.DoubleVar(value=0.0)

        self.tilt_angle_kappa = tk.DoubleVar(value=1.0)
        self.tilt_angle_loc = tk.DoubleVar(value=round(0.5 * np.pi, 4))
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
        ttk.Label(main_frame1, text="Simulation box", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Number of Voxels", self.nvox_var1)
        add_label_and_entry(main_frame1, next(line), "Size of RVE (in micron)", self.size_var1)
        add_label_and_entry(main_frame1, next(line), "Periodic", self.periodic_var1, entry_type="checkbox",
                            bold=False)
        ttk.Label(main_frame1, text="Phases", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        #add_label_and_entry(main_frame1, next(line), "Number of phases", self.nphases)
        add_label_and_entry(main_frame1, next(line), "Phase Name", self.matname_var1)
        add_label_and_entry(main_frame1, next(line), "Phase Number", self.ialloy)
        hl = next(line)
        ttk.Label(main_frame1, text="     target", font=("Helvetica", 12, "italic")) \
            .grid(row=hl, column=1, pady=(10, 0), sticky='w')
        ttk.Label(main_frame1, text="   actual", font=("Helvetica", 12, "italic")) \
            .grid(row=hl, column=2, pady=(10, 0), sticky='w')
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Volume fraction", self.volume_fraction)
        add_label_and_entry(main_frame1, hl, label_text=None, entry_var=self.vf_act, col=2)

        ttk.Label(main_frame1, text="Equivalent Diameter Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Sigma", self.eq_diameter_sig)
        add_label_and_entry(main_frame1, hl, None, self.eqd_act_sig, col=2)
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Scale", self.eq_diameter_scale)
        add_label_and_entry(main_frame1, hl, None, self.eqd_act_scale, col=2)
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Location", self.eq_diameter_loc)
        add_label_and_entry(main_frame1, hl, None, self.eqd_act_loc, col=2)
        add_label_and_entry(main_frame1, next(line), "Min", self.eq_diameter_min)
        add_label_and_entry(main_frame1, next(line), "Max", self.eq_diameter_max)

        ttk.Label(main_frame1, text="Aspect Ratio Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Sigma", self.aspect_ratio_sig)
        add_label_and_entry(main_frame1, hl, None, self.ar_act_sig, col=2)
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Scale", self.aspect_ratio_scale)
        add_label_and_entry(main_frame1, hl, None, self.ar_act_scale, col=2)
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Location", self.aspect_ratio_loc)
        add_label_and_entry(main_frame1, hl, None, self.ar_act_loc, col=2)
        add_label_and_entry(main_frame1, next(line), "Min", self.aspect_ratio_min)
        add_label_and_entry(main_frame1, next(line), "Max", self.aspect_ratio_max)

        ttk.Label(main_frame1, text="Tilt Angle Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Kappa", self.tilt_angle_kappa)
        #add_label_and_entry(main_frame1, hl, None, self.ta_act_kappa, col=2)
        hl = next(line)
        add_label_and_entry(main_frame1, hl, "Location", self.tilt_angle_loc)
        add_label_and_entry(main_frame1, next(line), "Min", self.tilt_angle_min)
        add_label_and_entry(main_frame1, next(line), "Max", self.tilt_angle_max)

        ttk.Label(main_frame1, text="Orientation Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame1, next(line), "Texture", self.texture_var1, entry_type="combobox",
                            options=["random", "unimodal", "EBSD-ODF"])
        add_label_and_entry(main_frame1, next(line), "Kernel Half Width (degree)", self.kernel_var1,
                            bold=False)
        add_label_and_entry(main_frame1, next(line), "Euler Angles (degree)", self.euler_var1)

        # create buttons
        button_frame1 = ttk.Frame(main_frame1)
        button_frame1.grid(row=next(line), column=0, columnspan=3, pady=10, sticky='ew')

        button_import_ebsd = ttk.Button(button_frame1, text="Import EBSD", style='TButton',
                                        command=self.import_ebsd)
        button_import_ebsd.grid(row=0, column=0, padx=(10, 5), pady=5, sticky='ew')
        button_import_stats = ttk.Button(button_frame1, text="Import Statistics", style='TButton',
                                         command=self.import_stats)
        button_import_stats.grid(row=0, column=1, padx=(10, 5), pady=5, sticky='ew')
        write_stats_button = ttk.Button(button_frame1, text="Export Statistics", style='TButton',
                                        command=self.write_stat_param)
        write_stats_button.grid(row=0, column=2, padx=(10, 5), pady=5, sticky='ew')
        button_statistics = ttk.Button(button_frame1, text="Plot Statistics", style='TButton',
                                       command=self.create_and_plot_stats)
        button_statistics.grid(row=1, column=0, padx=(10, 5), pady=5, sticky='ew')
        button_create_rve = ttk.Button(button_frame1, text="Create RVE", style='TButton',
                                       command=self.create_and_plot_rve)
        button_create_rve.grid(row=1, column=1, padx=(10, 5), pady=5, sticky='ew')
        button_create_ori = ttk.Button(button_frame1, text="Create Orientations", style='TButton',
                                       command=self.create_orientation)
        button_create_ori.grid(row=1, column=2, padx=(10, 5), pady=5, sticky='ew')
        write_files_button = ttk.Button(button_frame1, text="Write Abaqus Input", style='TButton',
                                        command=self.export_abq)
        write_files_button.grid(row=2, column=0, padx=(10, 5), pady=5, sticky='ew')
        button_exit1 = ttk.Button(button_frame1, text="Exit", style='TButton', command=self.close)
        button_exit1.grid(row=2, column=2, padx=(10, 5), pady=5, sticky='ew')

    def close(self):
        """
        Quit and destroy the particle_rve GUI main window

        Notes
        -----
        This method quits the Tkinter application and destroys the main window
        """
        self.app.quit()
        self.app.destroy()

    def update_kernel_var(self, *args):
        """
        Update the kernel variable based on the current texture selection

        Parameters
        ----------
        *args : tuple
            Optional arguments passed by the tkinter trace callback, not used directly

        Notes
        -----
        If `self.texture_var1` is 'unimodal', `self.kernel_var1` is set to "7.5"
        Otherwise, `self.kernel_var1` is set to "-"
        """
        self.kernel_var1.set("7.5" if self.texture_var1.get() == 'unimodal' else "-")

    def update_euler_var(self, *args):
        """
        Update the Euler angles variable based on the current texture selection

        Parameters
        ----------
        *args : tuple
            Optional arguments passed by the tkinter trace callback, not used directly

        Notes
        -----
        If `self.texture_var1` is 'unimodal', `self.euler_var1` is set to "0.0, 45.0, 0.0"
        Otherwise, `self.euler_var1` is set to "-"
        """
        self.euler_var1.set("0.0, 45.0, 0.0" if self.texture_var1.get() == 'unimodal' else "-")

    def display_plot(self, fig, plot_type):
        """
        Show a statistics graph or RVE plot on the GUI canvas

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to be displayed
        plot_type : str
            Type of plot to display, either 'stats' for statistics graph or 'rve' for RVE plot

        Notes
        -----
        Existing canvas for the specified plot type will be destroyed before displaying the new figure
        The GUI window size is updated to fit the new plot
        """
        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")
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

        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")

    def import_ebsd(self):
        """
        Import an EBSD map and extract microstructure parameters

        Notes
        -----
        Opens a file dialog to select an EBSD map with extensions '.ctf' or '.ang'
        Displays messages indicating progress and success or failure
        Calls `extract_microstructure_params` and `reset_act` after successful import
        """
        file_path = filedialog.askopenfilename(title="Select EBSD map", filetypes=[("EBSD maps", ".ctf .ang")])
        if file_path:
            self_closing_message("Reading EBSD map, please wait..")
            try:
                self.ebsd = EBSDmap(file_path, show_plot=True)
                self.extract_microstructure_params()
                self.reset_act()
                self_closing_message("Data from EBSD map imported successfully.")
            except:
                self_closing_message("ERROR: Could not read EBSD map!")

    def write_stat_param(self):
        """
        Save the current microstructure statistics to a file

        Notes
        -----
        If `self.ms_stats` is None, a message is shown indicating no stats are available
        Opens a file dialog to select the save location
        Calls `write_stats` to write the statistics to the selected file
        """
        if self.ms_stats is None:
            self_closing_message("No stats created yet.")
        else:
            file_path = filedialog.asksaveasfilename()
            if file_path:
                self_closing_message(f"Saving stats file as {file_path}.")
                write_stats(self.ms_stats, file_path)

    def reset_act(self):
        """
        Reset all activity-related variables to zero

        Notes
        -----
        Sets `vf_act`, `eqd_act_sig`, `eqd_act_scale`, `ar_act_sig`, and `ar_act_scale` to 0.0
        """
        self.vf_act.set(0.0)
        self.eqd_act_sig.set(0.0)
        self.eqd_act_scale.set(0.0)
        self.ar_act_sig.set(0.0)
        self.ar_act_scale.set(0.0)

    def readout_stats(self):
        """
        Collect and return the current microstructure and RVE statistics

        Returns
        -------
        dict
            A dictionary containing:
            - 'Grain type': Type of grain
            - 'Equivalent diameter': Distribution parameters (sig, scale, loc, cutoff_min, cutoff_max)
            - 'Aspect ratio': Distribution parameters (sig, scale, loc, cutoff_min, cutoff_max)
            - 'Tilt angle': Orientation parameters (kappa, loc, cutoff_min, cutoff_max)
            - 'RVE': RVE size and voxel counts
            - 'Simulation': Simulation settings including periodicity and output units
            - 'Phase': Material phase information including name, number, and volume fraction
        """
        matname = self.matname_var1.get()
        nvox = int(self.nvox_var1.get())
        size = int(self.size_var1.get())
        periodic = self.periodic_var1.get()
        sd = {
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
            'Simulation': {'periodicity': periodic, 'output_units': 'mm'},
            'Phase': {'Name': matname, 'Number': 0, 'Volume fraction': self.volume_fraction.get()}
        }
        return sd

    def import_stats(self):
        """
        Import statistical data from a JSON file and update GUI variables

        Notes
        -----
        Opens a file dialog to select a statistics file with '.json' extension
        Updates GUI variables for material name, RVE size, voxel counts, periodicity, volume fraction,
        equivalent diameter, aspect ratio, and tilt angle based on imported data
        Calls `reset_act` and clears `ebsd` after successful import
        Displays a message indicating success or failure
        """
        file_path = filedialog.askopenfilename(title="Select statistics file", filetypes=[("Stats files", ".json")])
        if file_path:
            try:
                ms_stats = import_stats(file_path)
                self.matname_var1.set(ms_stats['Phase']['Name'])
                self.nvox_var1.set(ms_stats['RVE']['Nx'])
                self.size_var1.set(ms_stats['RVE']['sideX'])
                self.periodic_var1.set(ms_stats['Simulation']['periodicity'])
                self.volume_fraction.set(ms_stats['Phase']['Volume fraction'])
    
                self.eq_diameter_sig.set(ms_stats['Equivalent diameter']['sig'])
                self.eq_diameter_scale.set(ms_stats['Equivalent diameter']['scale'])
                if 'loc' in ms_stats['Equivalent diameter'].keys():
                    self.eq_diameter_loc.set(ms_stats['Equivalent diameter']['loc'])
                else:
                    self.eq_diameter_loc.set(0.0)
                self.eq_diameter_min.set(ms_stats['Equivalent diameter']['cutoff_min'])
                self.eq_diameter_max.set(ms_stats['Equivalent diameter']['cutoff_max'])
    
                self.aspect_ratio_sig.set(ms_stats['Aspect ratio']['sig'])
                self.aspect_ratio_scale.set(ms_stats['Aspect ratio']['scale'])
                if 'loc' in ms_stats['Equivalent diameter'].keys():
                    self.aspect_ratio_loc.set(ms_stats['Aspect ratio']['loc'])
                else:
                    self.aspect_ratio_loc.set(0.0)
                self.aspect_ratio_min.set(ms_stats['Aspect ratio']['cutoff_min'])
                self.aspect_ratio_max.set(ms_stats['Aspect ratio']['cutoff_max'])
    
                self.tilt_angle_kappa.set(ms_stats['Tilt angle']['kappa'])
                self.tilt_angle_loc.set(ms_stats['Tilt angle']['loc'])
                self.tilt_angle_min.set(ms_stats['Tilt angle']['cutoff_min'])
                self.tilt_angle_max.set(ms_stats['Tilt angle']['cutoff_max'])
                self.ms_stats = ms_stats
                self.reset_act()
                self.ebsd = None
                self_closing_message("Statistical data imported successfully.")
            except Exception as e:
                self_closing_message(f"ERROR: Statistical parameters could not be imported! {e}")

    def extract_microstructure_params(self):
        """
        Extract microstructure parameters from the EBSD data and update GUI variables

        Notes
        -----
        If `self.ebsd` is None, the method does nothing
        Extracts grain size, aspect ratio, and orientation parameters from the first EBSD dataset
        Updates GUI variables for material name, volume fraction, equivalent diameter, aspect ratio,
        tilt angle, and texture type
        Computed min and max cutoff values are based on statistical scaling of the extracted parameters
        """
        if self.ebsd is None:
            return

        ms_data = self.ebsd.ms_data[0]
        gs_param = ms_data['gs_param']
        ar_param = ms_data['ar_param']
        om_param = ms_data['om_param']
        self.matname_var1.set(ms_data['name'])
        self.volume_fraction.set(1.0)
        self.eq_diameter_sig.set(gs_param[0].round(3))
        self.eq_diameter_scale.set(gs_param[2].round(3))
        self.eq_diameter_loc.set(gs_param[1].round(3))
        co_min = gs_param[2] * 0.5 + gs_param[1]
        co_max = gs_param[2] * 1.1 + gs_param[1]
        self.eq_diameter_min.set(co_min.round(3))
        self.eq_diameter_max.set(co_max.round(3))

        self.aspect_ratio_sig.set(ar_param[0].round(3))
        self.aspect_ratio_scale.set(ar_param[2].round(3))
        self.aspect_ratio_loc.set(ar_param[1].round(3))
        ar_max = ar_param[2] * 1.4 + ar_param[1]
        self.aspect_ratio_min.set(0.95)
        self.aspect_ratio_max.set(ar_max.round(3))

        self.tilt_angle_kappa.set(om_param[0].round(3))
        self.tilt_angle_loc.set(om_param[1].round(3))
        self.tilt_angle_min.set(0.0)
        self.tilt_angle_max.set(3.1416)

        self.texture_var1.set('EBSD-ODF')

    def create_and_plot_stats(self):
        """
        Plot statistics of the current microstructure descriptors and initialize a new Microstructure object

        Notes
        -----
        Will erase the global microstructure object `self.ms` if it exists
        Reads current statistics using `readout_stats`
        Initializes `self.ms` with the selected material and texture
        Plots statistics using `plot_stats_init` and displays each figure on the GUI
        Resets activity-related variables after plotting
        """
        self.ms_stats = self.readout_stats()
        texture = self.texture_var1.get()
        matname = self.matname_var1.get()
        self.ms = Microstructure(descriptor=self.ms_stats, name=f"{matname}_{texture}_texture")
        self.ms.init_RVE()
        if self.ebsd is None:
            gs_data = None
            ar_data = None
        else:
            gs_data = self.ebsd.ms_data[0]['gs_data']
            ar_data = self.ebsd.ms_data[0]['ar_data']
        flist, descs = self.ms.plot_stats_init(gs_data=gs_data, ar_data=ar_data, silent=True)
        for fig in flist:
            self.display_plot(fig, plot_type="stats")
        self.reset_act()

    def create_and_plot_rve(self):
        """
        Create the RVE from current statistics and plot it along with associated statistics

        Notes
        -----
        Will overwrite existing `ms_stats` and `ms` objects
        Displays a progress message and measures processing time
        Initializes `self.ms` with the selected material and voxelizes the RVE
        Plots the voxelized RVE using `plot_voxels` and displays it on the GUI
        Updates activity-related variables based on voxelized statistics
        Also plots statistics figures returned from `plot_stats_init`
        """

        self_closing_message("The process has been started, please wait...")
        start_time = time.time()
        matname = self.matname_var1.get()
        self.ms_stats = self.readout_stats()
        self.ms = Microstructure(descriptor=self.ms_stats, name=f"{matname}")
        self.ms.init_RVE()
        self.ms.pack()
        self.ms.voxelize()
        fig = self.ms.plot_voxels(silent=True, sliced=False)
        self.display_plot(fig, plot_type="rve")
        flist, descs = self.ms.plot_stats_init(silent=True, get_res=True, return_descriptors=True)
        self.vf_act.set(self.ms.vf_vox[0])
        self.eqd_act_sig.set(descs[0]['eqd']['std'])
        self.eqd_act_scale.set(descs[0]['eqd']['mean'])
        self.ar_act_sig.set(descs[0]['ar']['std'])
        self.ar_act_scale.set(descs[0]['ar']['mean'])
        end_time = time.time()
        duration = end_time - start_time
        self_closing_message(f"Process completed in {duration:.2f} seconds")
        for fig in flist:
            self.display_plot(fig, plot_type="stats")

    def create_orientation(self):
        """
        Generate grain orientations based on the selected texture and assign them to the RVE

        Notes
        -----
        Supports 'unimodal' textures with user-specified kernel and Euler angles, or 'EBSD-ODF' textures
        Automatically generates the RVE if `ms_stats` or `ms` objects are not initialized
        Calls `generate_orientations` to assign orientations to the microstructure
        Writes voxel data to a JSON file and plots the voxelized RVE with orientations
        Displays messages indicating progress and processing time
        """
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
        if texture == 'EBSD-ODF':
            texture = self.ebsd
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
        """
        Export the RVE mesh to an Abaqus input file

        Notes
        -----
        If `ms_stats`, `ms`, or `ms.mesh` is not initialized, the RVE is first generated without orientations
        Checks if the material is dual-phase based on volume fraction
        Calls `write_abq` to export the mesh in millimeter units
        Displays a progress message if RVE generation is required
        """
        if self.ms_stats is None or self.ms is None or self.ms.mesh is None:
            self_closing_message("Generating and exporting RVE without orientations.")
            self.create_and_plot_rve()
        dp = self.volume_fraction.get() < 1.0
        self.ms.write_abq('v', dual_phase=dp, units='mm')


class cuboid_rve(object):
    """
    Class for managing RVEs with cuboid grains and associated GUI controls

    Parameters
    ----------
    app: tk.Tk or tk.Frame
      Reference to the main Tkinter application instance
    notebook: ttk.Notebook
      Reference to the parent notebook widget for GUI tab placement

    Attributes
    ----------
    app: tk.Tk or tk.Frame
      Reference to the main Tkinter application instance
    canvas: tk.Canvas
      Canvas for displaying the cuboid RVE
    ms: object
      Placeholder for the microstructure object
    texture_var2: tk.StringVar
      Selected texture type (default "random")
    matname_var2: tk.StringVar
      Material name (default "Simulanium")
    ialloy: tk.IntVar
      Number of alloys (default 2)
    ngr_var: tk.StringVar
      Number of grains in each direction (default "5, 5, 5")
    nv_gr_var: tk.StringVar
      Number of voxels per grain in each direction (default "3, 3, 3")
    size_var2: tk.StringVar
      RVE size in each direction (default "45, 45, 45")
    kernel_var2: tk.StringVar
      Kernel parameter for unimodal textures (default "-" or "7.5")
    euler_var2: tk.StringVar
      Euler angles for unimodal textures (default "-" or "0.0, 45.0, 0.0")

    Notes
    -----
    The class handles creation, visualization, and parameter management for cuboid RVEs.
    GUI variable traces are used to automatically update dependent parameters such as
    kernel and Euler angle values based on the selected texture type.
    """

    def __init__(self, app, notebook):
        # define standard parameters
        self.app = app
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
        add_label_and_entry(main_frame2, next(line), "Material Name", self.matname_var2)
        add_label_and_entry(main_frame2, next(line), "Material Number", self.ialloy)
        add_label_and_entry(main_frame2, next(line), "Number of Voxels", self.nv_gr_var)
        add_label_and_entry(main_frame2, next(line), "Number of Grains", self.ngr_var)
        add_label_and_entry(main_frame2, next(line), "Size of RVE (in micron)", self.size_var2)

        ttk.Label(main_frame2, text="Orientation Parameters", font=("Helvetica", 12, "bold")) \
            .grid(row=next(line), column=0, columnspan=2, pady=(10, 0), sticky='w')
        add_label_and_entry(main_frame2, next(line), "Texture", self.texture_var2, entry_type="combobox",
                            options=["random", "unimodal"])
        add_label_and_entry(main_frame2, next(line), "Kernel Half Width (degree)", self.kernel_var2,
                            bold=False)
        add_label_and_entry(main_frame2, next(line), "Euler Angles (degree)", self.euler_var2)

        # add buttons
        button_frame2 = ttk.Frame(main_frame2)
        button_frame2.grid(row=next(line), column=0, columnspan=2, pady=10, sticky='ew')
        run_simulation_button = ttk.Button(button_frame2, text="Create RVE", style='TButton',
                                           command=self.create_cubes_and_plot)
        run_simulation_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        create_orientation_button = ttk.Button(button_frame2, text="Create Orientations", style='TButton',
                                               command=self.create_orientation)
        create_orientation_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        write_files_button = ttk.Button(button_frame2, text="Write Abaqus Input", style='TButton',
                                        command=self.export_abq)
        write_files_button.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        button_exit2 = ttk.Button(button_frame2, text="Exit", style='TButton', command=self.close)
        button_exit2.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

    def close(self):
        """
        Quit and destroy the cuboid_rve GUI main window

        Notes
        -----
        This method terminates the Tkinter application and closes the main window
        """
        self.app.quit()
        self.app.destroy()

    def update_kernel_var(self, *args):
        """
        Update the kernel parameter variable based on the current texture selection

        Parameters
        ----------
        *args : tuple
          Optional arguments passed by the tkinter trace callback, not used directly

        Notes
        -----
        Sets `kernel_var2` to "7.5" if `texture_var2` is 'unimodal', otherwise sets it to "-"
        """
        self.kernel_var2.set("7.5" if self.texture_var2.get() == 'unimodal' else "-")

    def update_euler_var(self, *args):
        """
        Update the Euler angles variable based on the current texture selection

        Parameters
        ----------
        *args : tuple
          Optional arguments passed by the tkinter trace callback, not used directly

        Notes
        -----
        Sets `euler_var2` to "0.0, 45.0, 0.0" if `texture_var2` is 'unimodal', otherwise sets it to "-"
        """
        self.euler_var2.set("0.0, 45.0, 0.0" if self.texture_var2.get() == 'unimodal' else "-")

    def display_cuboid(self, fig):
        """
        Display the cuboid RVE figure on the GUI canvas

        Parameters
        ----------
        fig : matplotlib.figure.Figure
          The figure object representing the cuboid RVE to display

        Notes
        -----
        Destroys any existing canvas before displaying the new figure
        Updates the Tkinter window geometry to fit the figure
        """
        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.rve_plot_frame2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.app.update_idletasks()
        width, height = self.app.winfo_reqwidth(), self.app.winfo_reqheight()
        self.app.geometry(f"{width}x{height}")

    def create_cubes_and_plot(self):
        """
        Create a microstructure object with cuboid grains and plot the RVE

        Notes
        -----
        Parses GUI parameters for number of grains, voxels per grain, and RVE size
        Initializes a Microstructure object and sets up the voxel mesh
        Assigns each cuboid grain an index and populates grain dictionaries
        Calls `plot_voxels` to generate the RVE figure and displays it on the GUI
        Returns nothing; updates the GUI canvas with the plotted RVE
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
        self.ms = Microstructure('from_voxels')
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
        """
        Create grain orientations for the cuboid RVE based on texture descriptors

        Notes
        -----
        Supports 'unimodal' textures with user-specified kernel and Euler angles
        Automatically generates the cuboid RVE if `ms` or `ms.mesh` is not initialized
        Calls `generate_orientations` to assign orientations to the microstructure
        Writes voxel data to a JSON file and plots the voxelized RVE with orientations
        Displays messages indicating progress and processing time
        """
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
            self_closing_message("Generating RVE with cuboid grains to assign orientation to.")
            self.create_cubes_and_plot()
        self.ms.generate_orientations(texture, ang=ang, omega=omega)
        self.ms.write_voxels(file=f'{matname}_voxels.json', script_name=__file__, mesh=False, system=False)
        fig = self.ms.plot_voxels(silent=True, sliced=False, ori=True)
        self.display_cuboid(fig)
        end_time = time.time()
        duration = end_time - start_time
        self_closing_message(f"Process completed in {duration:.2f} seconds, the Voxel file has been saved.")

    def export_abq(self):
        """
        Export the cuboid RVE mesh to an Abaqus input file

        Notes
        -----
        Automatically generates the cuboid RVE without orientations if `ms` is not initialized
        Calls `write_abq` to export the mesh in millimeter units
        Displays a progress message if RVE generation is required
        """
        if self.ms is None:
            self_closing_message("Generating and exporting RVE with cuboid grains w/o orientations.")
            self.create_cubes_and_plot()
        self.ms.write_abq('v', units='mm')
