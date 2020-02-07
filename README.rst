.. highlight:: shell

=========
Overview
=========

.. image:: https://joss.theoj.org/papers/10.21105/joss.01732/status.svg
   :target: https://doi.org/10.21105/joss.01732

.. image:: https://img.shields.io/badge/Platform-Linux%2C%20MacOS-critical
   
.. image:: https://img.shields.io/travis/mrgprasad/kanapy.svg
    :target: https://travis-ci.org/mrgprasad/kanapy

.. image:: https://codecov.io/gh/mrgprasad/kanapy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mrgprasad/kanapy
    
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/v/release/mrgprasad/kanapy?color=lightgray

Kanapy is a python package for generating complex synthetic polycrystalline microstructures. The general implementation is done in Python_ with the performance critical part for the geometry module implemented in C++. The Python bindings for the code written in C++ are generated using the lightweight header-only library pybind11_. The C++ part of the implementation utilizes the Eigen_ library for efficient linear algebra calculations. The texture module of Kanapy is implemented as MATLAB_ functions. It also utilizes several algorithms implemented in MTEX_ for texture analysis. 

.. _Python: http://www.python.org
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/

Motivation
----------
An accurate representation of the material microstructure is fundamental in understanding the relationship between microstructure and its corresponding mechanical behavior. In this regard, Kanapy is developed to be a robust and an efficient tool to generate synthetic microstructures within the micro mechanical framework for Finite Element Method (FEM) simulations. It is designed to model simple and complex grain morphologies and to systematically incorporate texture data directly from experiments concurrently maintaining the numerical efficiency of the micromechanical model. Kanapy is developed to overcome the limitations of spatial tessellation methods and to provide an alternative to the existing Random Sequential Addition technique of microstructure geometry generation. 

Features
--------

* User interface to kanapy through CLI.   
* Efficient collision handling of particles through a two-layer collision detection method employing the Octree spatial data structure and the bounding sphere hierarchy. 
* Efficient ODF reconstruction directly using orientations from experimantal data.
* Optimal orientaion assignment based on Measured misorientation distribution.
* Independent execution of individual modules through easy data storage and handling.
* In-built hexahedral mesh generator for complex polycrystalline microstructures.        
* Flexibility in the choice of the particle packing time step to be sent for voxelization (meshing).
* Option to generate spherical particle position- and radius files that can be read by the Voronoi tessellation software Neper_.
* Option to generate input files for the commercial finite-element software Abaqus_.    
* High-performance for the critical part of the geometry code using Python-C++ bindings.  

.. _Neper: http://neper.sourceforge.net/
.. _Abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/

.. role:: bash(code)
   :language: bash
   
Installation
------------
CMake is used for building extensions; if it is not installed on your machine, follow this 
`CMake documentation`_ to install it. The preferred method to install kanapy is through 
Anaconda or Miniconda Python distributions. If you do not have any, we suggest installing miniconda_. 

.. _CMake documentation: https://cgold.readthedocs.io/en/latest/first-step/installation.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

Once done, create a virtual environment for Kanapy installation and clone the repository to 
a desired location.

.. code-block:: console

    $ conda create -n myenv python=3.6 pip git
    $ conda activate myenv    
    (myenv) $ git clone https://github.com/mrgprasad/kanapy.git <location to clone>/kanapy-master
    (myenv) $ cd kanapy-master/
    (myenv) $ kanapy install

Kanapy is now installed along with all its dependencies. To use Kanapy's texture module (optional), 
you need to link Kanapy with MATLAB_ and MTEX_ installations. Run: :bash:`kanapy setuptexture` 
and follow the instructions.

.. note:: 1. ``myenv`` can be replaced with any name for your environment.
          2. For older versions of anaconda/miniconda use: ``source activate myenv``
                    
.. tip:: To learn more about managing environments see Anaconda documentation_.

.. _documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html    
.. _Github repo: https://github.com/mrgprasad/kanapy
.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/
            
Running tests
--------------
Kanapy uses pytest to perform all its unit testing. Run: :bash:`(myenv) $ kanapy unittests`
      
Documentation build
-------------------
Documentation for kanapy is generated using Sphinx. Run: :bash:`(myenv) $ kanapy builddocs`.
The HTML documentation can be found at *../kanapy-master/docs/index.html*.

Dependencies
-------------

Kanapy requires a working C/C++ compiler on your machine. On Linux/Mac OS,
the gcc toolchain will work well. The lightweight header-only library pybind11 
is used to create Python bindings for the code written in C++.
The C++ function will be complied by linking the Eigen library 
(present in the directory *../kanapy-master/libs/*). CMake builds this extension.

Kanapy's texture module requires MATLAB_ and MTEX_ to be installed on your machine.         
Make sure to use MATLAB v2015a and above.

.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/

^^^^^^^^^^^^^^^^^^
Core dependencies
^^^^^^^^^^^^^^^^^^

Below are the listed dependencies for running kanapy:

  - NumPy_ for array manipulation.
  - Scipy_ for functionalities like Convexhull.
  - pybind11_ for creating python bindings for C++ code.
  - Eigen_ for C++ linear algebra operations.
  - pytest_ for running kanapy unit tests.
  - sphinx_ for generating documentation.
  - MATLAB_ for texture modules.
  - MTEX_ for texture modules.
  
.. _NumPy: http://numpy.scipy.org
.. _Scipy: https://www.scipy.org/
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _pytest: https://www.pytest.org
.. _sphinx: http://www.sphinx-doc.org/en/master/
.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/

^^^^^^^^^^^^^^^^^^^^^^
Optional dependencies
^^^^^^^^^^^^^^^^^^^^^^

  - Matplotlib_ for plotting and visualizing.
  - OVITO_ for visualizing simulation data. 

.. _Matplotlib: https://matplotlib.org/
.. _OVITO: https://ovito.org/


Citation
---------
The preferred way to cite Kanapy is: 

Prasad et al., (2019). Kanapy: A Python package for generating complex synthetic polycrystalline microstructures. Journal of Open Source Software, 4(43), 1732, https://doi.org/10.21105/joss.01732

License
--------
Kanapy is made available under the MIT license_.

.. _license: https://opensource.org/licenses/MIT


About
-------
The name kanapy is derived from the sanskrit word káṇa_ meaning particle. Kanapy is primarily developed at the `Interdisciplinary Center for Advanced Materials Simulation (ICAMS), Ruhr-University Bochum - Germany <http://www.icams.de/content/>`__. Our goal is to build a complete synthetic microstructure generation tool for research and industry use. 

.. _káṇa: https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3
