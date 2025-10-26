=======
History
=======

0.1 (2019-07-01)
----------------

* First release on GitHub.

1.0 (2019-11-05)
----------------

* Completed JOSS review for Kanapy.

1.1 (2020-01-28)
----------------

* Texture reduction module included to Kanapy with coupling to MATLAB & MTEX.
* Matlab unittests written for texture reduction and misorientation algorithms.
* New CLI commands for texture reduction.

1.2 (2020-02-01)
----------------

* Updated the 'Modeling' section of the documentation with Texture.
* Updated the 'Applications' section of the documentation with Texture module. 
* Updated the 'Usage' section of the documentation with the two Textures examples.

2.0 (2020-02-11)
----------------

* Second release on GitHub.

2.1 (2021-03-09)
----------------

* Grain boundary (GB) smoothing code implemented in Kanapy
* New CLI commands for GB smoothing

3.0 (2021-11-29)
----------------

* Application Programming Interface (API) introduced
* Improved plotting options

3.1 (2022-04-01)
----------------

* Support for dual-phase materials
* Construction of polyhedral hull for grains

4.0 (2022-09-08)
----------------

* Updated name conventions for functions and files
* Significant code refactoring
* Import and export of voxels

5.0 (2023-09-23)
----------------

* Pure Python version for easier installation (previous versions used C++ code for collision detection)
* Improvements in installation procedure of MTEX module
* Improved handling of dual-phase microstructures (EBSD import, descriptors, packing, voxelization)

5.0.4 (2023-12-28)
------------------

* Last version with support of Command Line Interface (CLI)

6.0 (2024-01-04)
----------------

 * Completely new internal data structure for improved support of dual-phase and porous structures
 * Full support of voxel files in JSON format for input and output
 * CLI tools are deactivated
 
6.1 (2024-02-03)
----------------

 * Full support of dual-phase and porous microstructures in analysis of microstructure descriptors
 * Modified keywords of microstructure descriptors for compatibility with scipy fit functions
 * Unified notation of input/output methods
 * New methods and attributes for generation and handling of orientations
 
6.2 (2024-02-25)
----------------

 * Improved installation process, paths stored in local copy of package
 * Possibility to include inner polyhedral structures into ellipsoids
 
6.3 (2024-03-03)
----------------

 * Trajectories integrated by Verlet algorithm during packing
 * Extract statistical microstructure data from voxels, particles and polyhedra 
 
6.4 (2025-06-07)
----------------

 * Definition of 2D EBSD microstructures as graphs based on networkx
 * Support of the modular materials data schema for import and export of microstructures
 
6.5 (2025-08-29)
----------------

 * Pure Python version for reading and analyzing EBSD maps based on Orix. The MTEX backend is still available with Kanapy-mtex.
 