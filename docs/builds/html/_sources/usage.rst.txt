.. highlight:: shell

======
Usage
======

Kanapy's CLI
------------

Kanapy provides some basic command line instructions to manage the installation. 
The  

``$ kanapy --help`` command 
details this as shown:

.. note:: Make sure that you are within the virtual environment created during the kanapy installation, as 
          this environment contains the kanapy instalation and its required dependencies.
          
.. code-block:: console

    $ conda activate knpy
    (knpy) $ kanapy --help
            Usage: kanapy [OPTIONS] COMMAND [ARGS]...

            Options:
              --help  Show this message and exit.

            Commands:
              copyExamples  Creates local copy of Kanapy's example directory.
              genDocs       Generates a HTML-based reference documentation.
              gui           Start Kanapy's GUI
              runTests      Runs unittests built-in to Kanapy.
              setupTexture  Stores the user provided MATLAB & MTEX paths for texture.
  

Kanapy's API (incomplete)
-------------------------

Under construction ...

The central element of Kanapy are statistical microstructure descriptors, which are used 
to generate the microstructure geometry and its texture in the different Kanapy modules. 
For multiphase microstructures, each phase has its own statistical descriptors and the 
respective dictionaries and combined to a list.


""""""""""""""""""""""""""
Microstructure descriptors
""""""""""""""""""""""""""
An exemplary structure of the descriptor dictionary is shown below:

.. code-block::

    {
      "Grain type": 
        "Elongated",
      "Equivalent diameter": 
        {
          "sig": 0.39,
          "scale": 2.418,
          "loc": 0.0,
          "cutoff_min": 8.0,
          "cutoff_max": 25.0
        },
      "Aspect ratio": 
        {
          "sig": 0.3,
          "scale": 0.918,
          "loc": 0.0
          "cutoff_min": 1.0,
          "cutoff_max": 4.0        
        },           
      "Tilt angle":
        {
          "kappa": 15.8774,
          "loc": 87.4178, 
          "cutoff_min": 75.0,
          "cutoff_max": 105.0            
        },            
      "RVE": 
        {
          "sideX": 85.9,
          "sideY": 112.33076,
          "sideZ": 85.9,
          "Nx": 65,
          "Ny": 85,
          "Nz": 65
        },
      "Simulation":
        {
          "periodicity": "True",                                         
          "output_units": "mm"         
        },
      "Phase": 
        {
          "Name": "Simulanium",
          "Number": 1,
          "Volume fraction": 1.0
        }
    }
    
Sample dictionaries are provided in the examples shown below. Note that descriptors can 
be stored and read from JSON files. In the following the essentail keywords: 
``Grain type, Equivalent diameter, Aspect ratio, Tilt angle, RVE, Simulation, Phase`` are described 

  - ``Grain type`` (mandatory): Either "Elongated" or "Equiaxed"
  - ``Equivalent diameter`` (mandatory): takes in four arguments to generate a 
    log-normal distribution for the particle's equivalent diameter; they are the 
    `Normal distribution's`_ standard deviation and mean, and the minimum 
    and maximum cut-off values for the diameter. The values should correspond to :math:`\mu m` scale.
  - ``Aspect ratio`` takes the mean and the standard deviation value value as input. If the resultant 
    microstructure contains equiaxed grains then this field is not necessary.
  - ``Tilt angle`` keyword represents the tilt angle of particles with 
    respect to the positive x-axis. Hence, to generate a distribution, it takes in 
    two arguments: the normal distribution's mean and the standard deviation. If the resultant 
    microstructure contains equiaxed grains then this field is also not necessary. 
  - ``RVE`` takes two types of input: the side lengths of the final RVE 
    required and the number of voxels per RVE side length. 
  - ``Simulation``  takes in two inputs: A boolean value for periodicity (True/False) 
    and the required unit scale ('mm' or 'um' = :math:`\mu m`) for the output 
    ABAQUS .inp file.
  - ``Phase`` (optional) Information about phase name, number and volume fraction.

.. note:: 1. The user may choose not to use the built-in voxelization (meshing) routine 
             for meshing the final RVE. Nevertheless, a value for the number of voxels 
             per side has to be provided.
          2. A good estimation for `Nx, Ny & Nz` value can be made by keeping the 
             following point in mind: The smallest dimension of the smallest ellipsoid/sphere 
             should contain at least 3 voxels.
          3. The size of voxels should be the same along X, Y & Z directions 
             (voxel_sizeX = voxel_sizeY = voxel_sizeZ). 
             It is determined using: voxel size = RVE side length/Voxel per side. 
          4. Particles grow during the simulation. At the start of the simulation, all particles 
             are initialized with null volume. At each time step, they grow in size by the 
             value: diameter/1000. Since the maximum packing density of ellipsoides is 
             about 65%, the full space filling structure is achieved during voxelization.
          5. The input unit scale should be in :math:`\mu m` and the user can choose between 
             'mm' or 'um' (= :math:`\mu m`) as the unit scale in which output to the 
             ABAQUS .inp file will be written. 

.. _Normal distribution's: https://en.wikipedia.org/wiki/Normal_distribution   

"""""""""
Tutorials
"""""""""     
Several examples in form of Python scripts and Jupyter notebooks come bundled along with 
the kanapy package. Each example describes a particular workflow and demonstrates the use
of the geometry module and the texture module. Note, that the latter requires a complete 
Kanapy installation including MTEX, that can be setup with the 'kanapy setupTexture' 
command.

^^^^^^^^^
Notebooks
^^^^^^^^^

-  | `generate_rve.ipynb <_static/generate_rve.html>`__
   | The basic steps of using Kanapy to generate synthetic
     representative volume elements (RVE) are demonstrated. The RVE can
     be exported in standard FEA formats.

-  | `ebsd2rve.ipynb <_static/ebsd2rve.html>`__
   | An EBSD map aof a austenitic steel produced by additive
     manufacturing is analyzed with respect to statistical
     microstructure descriptors. The descriptors are the basis to build
     an RVE.

-  | `dispersed_phase.ipynb <_static/dispersed_phase.html>`__
   | A polycrystal with a second phase dispersed along the grain
     boundaries is generated.


Outdated CLI contents
---------------------

The following sections refer mainly to CLI commands from previous Kanapy versions.
To be updated ...


Hence the examples are sub-categorized here under 
two sub-sections: Geometry examples and Texture examples. A detailed description of these examples is presented here. 
The two examples ``sphere_packing`` and ``ellipsoid_packing`` depict the different workflows 
that have to be setup for generating synthetic microstructures with equiaxed and elongated 
grains, respectively. And the two examples ``ODF_reconstruction`` and ``ODF_reconstruction_with_orientation_assignment``
depict the workflows for reconstructing ODF from EBSD data and assigning orientations to RVE grains, respectively. 
For a detailed understanding of the general framework of the packing simulations or the ODF reconstruction, please 
refer to: :ref:`Modeling`.

.. note:: New examples must be created in a separate directory. It allows the kanapy modules 
          an easy access to the json, dump and other files created during the simulation.


"""""""""""""""""""""""""""""
Workflows for sphere packing 
"""""""""""""""""""""""""""""
This example demonstrates the workflow for generating synthetic microstructures with
equiaxed grains. The principle involved in generating such microstructures are described
in the sub-section :ref:`Microstructure with equiaxed grains`. With respect to the final RVE mesh, 
the user has the flexibility to choose between the in-built voxelization routine and external meshing softwares.

If external meshing is required, the positions and weights of the particles (spheres) after packing 
can be written out to be post-processed. The positions and weights can be read by the Voronoi tessellation 
and meshing software Neper_ for generating tessellations and FEM mesh. For more details refer to Neper's 
documentation_.

If the in-built voxelization routine is prefered, then kanapy will generate
hexahedral element (C3D8) mesh that can be read by the commercial FEM software Abaqus_. The Abaqus .inp 
file will be written out in either :math:`mm` or :math:`\mu m` scale.

.. _Neper: http://neper.sourceforge.net/
.. _documentaion: http://neper.sourceforge.net/docs/neper.pdf
.. _Abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/


.. code-block:: python

    > ms = knpy.Microstructure(descriptor=ms_elong, name=matname + '_' + texture + '_texture')
    > ms.init_RVE()  # initialize RVE including particle distribution and structured mesh
    > ms.plot_stats_init()  # plot initial statistics of equivalent grain diameter and aspect ratio
    > ms.pack()  # perform particle simulation to distribute grain nuclei in RVE volume
    > ms.plot_ellipsoids()  # plot final configuration of particles
    > ms.voxelize()  # assign voxels to grains according to particle configuration
    > ms.plot_voxels(sliced=True)  # plot voxels colored according to grain number
    > ms.generate_grains()  # generate a polyhedral hull around each voxelized grain
    > ms.plot_grains()  # plot polyhedral grains
    > ms.plot_stats()  # compared final grain statistics with initial parameters

After navigating to the directory where the input file ``stat_input.json`` is located, kanapy's CLI 
command ``genStats`` is executed along with its argument (name of the input file). It creates an exemplary
'Input_distribution.png' file depicting the Log-normal distribution corresponding to the input statistics defined in 
``stat_input.json``. Modifications can be made to the input statistics based on this plot. Next the ``genRVE`` 
command is executed to generate the necessary particle, RVE, and the simulation attributes, and it writes it 
to json files. Next the ``pack`` command is called to run the particle packing simulation. This command looks 
for the json files generated by ``genRVE`` and reads the files for extracting the information required for the 
packing simulation. At each time step of the packing simulation, kanapy will write out a dump file containing 
information of particle positions and other attributes. Finally, the ``neperOutput`` command (Optional) can be 
called to write out the position and weights files required for further post-processing. This function takes 
the specified timestep value as an input parameter and reads the corresponding, previously generated dump file. 
By extracting the particle's position and dimensions, it creates the ``sphere_positions.txt`` & ``sphere_weights.txt`` files.  

.. note:: 1. The ``neperOutput`` command requires the simulation timestep as input. The choice of the timestep is very important. 
             It is suggested to choose the time step at which the spheres are tightly packed and at which there is the least 
             amount of overlap. The remaining empty spaces will get assigned to the closest sphere when it is sent to the 
             tessellation and meshing routine. Please refer to :ref:`Microstructure with equiaxed grains` for more details.   
          2. The values of position and weights for Neper will be written in :math:`\mu m` scale only.

          
.. note:: For comparing the input and output statistics:          
            
            1. The json file ``particle_data.json`` in the directory ``../json_files/`` can be used to read the 
               particle's equivalent diameter as input statistics.
            2. After tessellation, Neper can be used to generate the equivalent diameter for output statistics.


If the built-in voxelization is prefered, then the ``voxelize`` command can be called to generate the hexahedral mesh. 
It populates the simulation box with voxels and assigns the voxels to the respective particles (Spheres). The 
``abaqusOutput`` command can be called to write out the Abaqus (.inp) input file. The workflow for this looks like:

.. code-block:: console

    $ conda activate knpy
    (knpy) $ cd kanapy-master/examples/sphere_packing/
    (knpy) $ kanapy genStats -f stat_input.json
    (knpy) $ kanapy genRVE -f stat_input.json
    (knpy) $ kanapy pack
    (knpy) $ kanapy voxelize
    (knpy) $ kanapy abaqusOutput
    (knpy) $ kanapy outputStats    
    (knpy) $ kanapy plotStats      
    
.. note:: 1. The Abaqus (.inp) file will be written out in either :math:`mm` or :math:`\mu m` scale, depending 
             on the user requirement specified in the input file.          
          2. For comparing the input and the output equivalent diameter statistics the ``outputStats`` command can be 
             called. This command writes the diameter values in either :math:`mm` or :math:`\mu m` scale, depending 
             on the user requirement specified in the input file.            
          3. The ``outputStats`` command also writes out the L1-error between the input and output diameter distributions.
          4. The ``plotStats`` command outputs a figure comparing the input and output diameter distributions.           
                  
Storing information in json & dump files is effective in making the workflow stages independent of one another. 
But the sequence of the workflow is important, for example: Running the packing routine before the statistics generation 
is not advised as the packing routine would not have any input to work on. Both the json and the dump files are human readable, 
and hence they help the user debug the code in case of simulation problems. The dump files can be read by the visualization 
software OVITO_; this provides the user a visual aid to understand the physics behind packing. For more information regarding 
visualization, refer to :ref:`Visualize the packing simulation`.

.. _OVITO: https://ovito.org/                           
                                   

""""""""""""""""""""""""""""""""
Workflows for ellipsoid packing 
""""""""""""""""""""""""""""""""
This example demonstrates the workflow for generating synthetic microstructures with
elongated grains. The principle involved in generating such microstructures is described
in the sub-section :ref:`Microstructure with elongated grains`. With respect to the final RVE mesh, 
the built-in voxelization routine has to be used due to the inavailability of anisotropic tessellation techniques.
The :ref:`Module voxelization` will generate a hexahedral element (C3D8) mesh that can be read by the commercial FEM software Abaqus_.

.. _Abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/

.. code-block:: console

    $ conda activate knpy
    (knpy) $ cd kanapy-master/examples/ellipsoid_packing/
    (knpy) $ kanapy genStats -f stat_input.json
    (knpy) $ kanapy genRVE -f stat_input.json
    (knpy) $ kanapy pack
    (knpy) $ kanapy voxelize
    (knpy) $ kanapy abaqusOutput
    (knpy) $ kanapy outputStats    
    (knpy) $ kanapy plotStats      

The workflow is similar to the one described earlier for sphere packing. The only difference being, that the ``neperOutput``
command is not applicable here. The ``outputStats`` command not only writes out the equivalent diameters, but also the 
major and minor diameters of the ellipsoidal particles and grains.
    
.. note:: 1. A good estimation for the RVE side length values can be made by keeping the following point in mind: The 
             biggest dimension of the biggest ellipsoid/sphere should fit within the corresponding RVE side length.
          2. For comparing the input and output equivalent, major and minor diameter statistics, the command 
             ``outputStats`` can be called. Kanapy writes the diameter values in either :math:`mm` or :math:`\mu m` scale, 
             depending on the user requirement specified in the input file.            
          3. The ``outputStats`` command also writes out the L1-error between the input and output diameter distributions  
          4. The ``plotStats`` command outputs figures comparing the input and output diameter & aspect ratio distributions.          
          
^^^^^^^^^^^^^^^^^
Texture examples
^^^^^^^^^^^^^^^^^
Both examples ``ODF_reconstruction`` and ``ODF_reconstruction_with_orientation_assignment`` require MATLAB & MTEX to be
installed in your system. If your kanapy is not configured for texture analysis, please run the following command:

.. code-block:: console

    $ conda activate knpy
    (knpy) $ kanapy setupTexture

.. note:: 1. Your MATLAB version must be 2015 and above.
          2. The required input files must be placed in the working directory from where the kanapy commands are run.

""""""""""""""""""""""""""""""""
Workflow for ODF reconstruction 
""""""""""""""""""""""""""""""""
This example demonstrates the workflow for reconstructing ODF from experimental EBSD data. The principle involved 
in generating the reduced ODF is described in the sub-section :ref:`ODF reconstruction`. Kanapy requires the EBSD data 
saved as (.mat) file format. In this regard, an exemplary EBSD file (`ebsd_316L.mat`) is provided in the ``../kanapy-master/examples/ODF_reconstruction/`` folder.

.. code-block:: console

    $ conda activate knpy
    (knpy) $ cd kanapy-master/examples/ODF_reconstruction/
    (knpy) $ kanapy reduceODF -ebsd ebsd_316L.mat
    
After navigating to the directory where ``ebsd_316L.mat`` is located, kanapy's CLI 
command ``reduceODF`` is executed along with its argument (name of the EBSD (.mat) file). If kanapy's 
geometry module is executed already, then the number of reduced orientations are read directly. Else kanapy requests 
the user to provide the number of reduced orientations required before calling the MATLAB ODF reconstruction algorithm. 

.. note:: 1. The EBSD (.mat) file is a mandatory requirement for the ODF reconstruction algorithm.
          2. Note here the value of the kernel shape parameter (:math:`\kappa`) is set to a default value of 0.0873 rad.          

Alternatly, an initial kernel shape parameter (:math:`\kappa`) can be specified as an user input (OR) the grains 
estimated using MTEX can be provided as an input in the (.mat) file format. The value of :math:`\kappa` must be in radians, 
if user specified. Else if the grains (.mat) file is provided, then the optimum :math:`\kappa` is estimated by kanapy using 
the mean orientation of the grains. In this regard, an exemplary grains file (``grains_316L.mat``) is
provided in the ``../kanapy-master/examples/ODF_reconstruction/`` folder. The workflow for this looks like: 

.. code-block:: console

    $ conda activate knpy
    (knpy) $ cd kanapy-master/examples/ODF_reconstruction/
    (knpy) $ kanapy reduceODF -ebsd ebsd_316L.mat -kernel 0.096
                                     (OR)
    (knpy) $ kanapy reduceODF -ebsd ebsd_316L.mat -grains grains_316L.mat

.. note:: 1. The output files are saved to the ``/mat_files`` folder under the current working directory. 
          2. The output (.txt) file contains the following information: :math:`L_1` error of ODF reconstruction, 
             the initial (:math:`\kappa`) and the optimized (:math:`\kappa^\prime`) values, and a list of discrete orientations.
          3. Additionaly kanapy saves the reduced ODF and the reduced orientations (.mat) files in this folder.
          4. Kanapy writes a log file (``kanapyTexture.log``) in the current working directory for possible errors and warnings debugging.
              
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Workflow for ODF reconstruction with orientation assignment 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This example demonstrates the workflow for reconstructing ODF from experimental EBSD data and then determining the optimal 
assignment of orientations to RVE grains. The principle involved in optimal orientation assignment is described in the 
sub-section :ref:`ODF reconstruction with orientation assignment`. In addition to the EBSD data, Kanapy requires 
grain (.mat) file, and the grain boundary shared surface area information as input. In this regard, an exemplary 
EBSD file (``ebsd_316L.mat``), and a grains file (``grains_316L.mat``) are provided in the 
``../kanapy-master/examples/ODF_reconstruction_with_orientation_assignment/`` folder. It is important to note that 
the grain boundary shared surface area file is created whilst generating an RVE by kanapy's geometry module.

.. code-block:: console

    $ conda activate knpy
    (knpy) $ cd kanapy-master/examples/ODF_reconstruction_orientation_assignment/
    (knpy) $ kanapy genStats -f stat_input.json
    (knpy) $ kanapy genRVE -f stat_input.json
    (knpy) $ kanapy pack
    (knpy) $ kanapy voxelize    
    (knpy) $ kanapy outputStats
    (knpy) $ kanapy reduceODF -ebsd ebsd_316L.mat -grains grains_316L.mat -fit_mad yes

After navigating to the directory where the input file ``ebsd_316L.mat`` is located, generate an RVE by calling kanapy's 
geometry CLI commands: ``genStats``, ``genRVE``, ``pack`` & ``voxelize``. To generate the shared surface area file, 
run ``outputStats`` command. Kanapy will write a ``shared_surfaceArea.csv`` 
file to the ``/json_files/`` folder. This file contains the grain boundary shared surface area 
information between neighbouring grains. Now, kanapy's texture CLI command ``reduceODF`` can be called along with 
its arguments (name of the EBSD, grains (.mat) files). The key ``-fit_mad`` must be used with this command to tell 
kanapy that orientation assignment to grains is required. Since kanapy's geometry module is executed already, kanapy recognizes 
the number of reduced orientations required (=number of grains in the RVE). Else kanapy requests the user to provide 
the number of reduced orientations required before calling the MATLAB functions. 

.. note:: 1. The EBSD, grains (.mat) files and the grain boundary shared surface file are mandatory requirements for the 
             orientation assignment algorithm.          
          2. The ``shared_surfaceArea.csv`` file is generated by running ``kanapy outputStats``.
          
Additionally an optional input that can be provided is the grain volume information, which is used for weighting the 
orientations after assignment and for estimating the ODF represented by the RVE. Kanapy also writes the grains volume file 
(``grainsVolumes.csv``) to the ``/json_files/`` folder, when the ``outputStats`` command is executed after RVE generation. 

.. note:: 1. The ``grainsVolumes.csv`` file lists the volume of each grain in the ascending order of the grain ID.
          2. Kanapy automatically detects the presence of the ``shared_surfaceArea.csv`` & ``grainsVolumes.csv`` files, 
             if they are present in the ``/json_files/`` folder.
          3. The output files are saved to the ``/mat_files`` folder under the current working directory. 
          4. The output (.txt) file contains the following information: :math:`L_1` error of ODF reconstruction, 
             :math:`L_1` error between disorientation angle distributions from the EBSD data and the RVE, the initial 
             (:math:`\kappa`) and the optimized (:math:`\kappa^\prime`) values, and a list of discrete orientations each 
             with a specific grain number that it should be assigned to.
          5. Additionaly kanapy saves the reduced ODF and the reduced orientations (.mat) files in this folder.
          6. Kanapy writes a log file (``kanapyTexture.log``) in the current working directory for possible errors and warnings debugging.           


"""""""""""""""""""""""""""""""""
Visualize the packing simulation
""""""""""""""""""""""""""""""""" 

You can view the data generated by the simulation (after the simulation
is complete or during the simulation) by launching OVITO and reading in 
the dump files generated by kanapy from the ``../sphere_packing/dump_files/`` directory. 
The dump file is generated at each timestep of the particle packing simulation. It contains 
the timestep, the number of particles, the simulation box dimensions and the particle's attributes 
such as its ID, position (x, y, z), axes lengths (a, b, c) and tilt angle (Quaternion format - X, Y, Z, W).
The OVITO user interface when loaded, should look similar to this:

.. image:: /figs/UI.png
    :width: 750px    

By default, OVITO loads the particles as spheres, this option can be changed to visualize ellipsoids. 
The asphericalshapex, asphericalshapey, and asphericalshapez columns need to be mapped to 
Aspherical Shape.X, Aspherical Shape.Y, and Aspherical Shape.Z properties of OVITO when 
importing the dump file. Similarily, the orientationx, orientationy, orientationz, and 
orientationw particle properties need to be mapped to the Orientation.X, Orientation.Y, 
Orientation.Z, and Orientation.W. OVITO cannot set up this mapping automatically, you have 
to do it manually by using the ``Edit column mapping`` button (at the bottom-right corner 
of the GUI) in the file import panel after loading the dump files. The required assignment 
and components are shown here:

.. image:: /figs/UI_options.png
    :width: 750px    

For further viewing customizations refer to OVITO's documentation_.

.. _documentation: https://ovito.org/manual/           
