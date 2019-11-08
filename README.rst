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

Kanapy is a python package for generating complex synthetic polycrystalline microstructures. The general implementation is done in Python_ with the performance critical part implemented in C++. The Python bindings for the code written in C++ are generated using the lightweight header-only library pybind11_. The C++ part of the implementation utilizes the Eigen_ library for efficient linear algebra calculations.

.. _Python: http://www.python.org
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page

Motivation
----------
An accurate representation of the material microstructure is fundamental in understanding the relationship between microstructure and its corresponding mechanical behavior. In this regard, Kanapy is developed to be a robust and an efficient tool to generate synthetic microstructures within the micro mechanical framework for Finite Element Method (FEM) simulations. It is designed not only to provide an alternative to the existing Random Sequential Addition technique of microstructure generation, but also to model simple and complex grain morphologies, thus overcoming the limitations of spatial tessellation methods. 

Features
--------

* User interfaces to kanapy through scripts are written in pure Python.
* Grains are approximated by ellipsoids (particles) and are packed into a pre-defined domain representing RVE.   
* Efficient collision handling of particles through a two-layer collision detection method employing the Octree spatial data structure and the bounding sphere hierarchy. 
* In-built hexahedral mesh generator for complex polycrystalline microstructures.    
* Independent execution of individual modules through easy data storage and handling.    
* Flexibility in the choice of the particle packing time step to be sent for voxelization (meshing).
* Option to generate spherical particle position- and radius files that can be read by the Voronoi tessellation software Neper_.
* Option to generate input files for the commercial finite-element software Abaqus_.    
* High-performance for the critical part of the code using Python-C++ bindings.  

.. _Neper: http://neper.sourceforge.net/
.. _Abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/


Installation
------------
CMake is used for building extensions; if it is not installed on your machine, follow this 
`CMake documentation`_ to install it.

.. _CMake documentation: https://cgold.readthedocs.io/en/latest/first-step/installation.html

The preferred method to install kanapy is through Anaconda or Miniconda Python distributions. 
If you do not have any, we suggest installing miniconda_. 

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html


Once done, create a virtual environment for installing Python-specific packages required for kanapy and 
activate it.

.. code-block:: console

    $ conda create -n myenv python=3.6 pip git
    $ conda activate myenv    

.. note:: 1. ``myenv`` can be replaced with any name for your environment.
          2. For older versions of anaconda/miniconda use: ``source activate myenv``
                    
.. tip:: To learn more about managing environments see Anaconda documentation_.

.. _documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html           

You can either clone the kanapy public repository by using git or 
download the kanapy source code from the `Github repo`_ to a desired location. 

.. code-block:: console

    (myenv) $ git clone https://github.com/mrgprasad/kanapy.git <location to clone>/kanapy-master
    (myenv) $ cd kanapy-master/
    (myenv) $ conda install -y -c conda-forge --file requirements.txt
    (myenv) $ pip install -e .

Kanapy is now installed along with all its dependencies.

.. _Github repo: https://github.com/mrgprasad/kanapy
          
Running tests
--------------

Kanapy uses ``pytest`` to perform all its unit testing. From the kanapy main directory (``kanapy-master``), run the tests:

.. code-block:: console
    
    (myenv) $ pytest tests/ -v
   
   
Documentation build
-------------------
Documentation for kanapy is generated using ``Sphinx``. The following command generates a HTML-based reference documentation; 
for other formats, please refer to the Sphinx manual. From the kanapy main directory (``kanapy-master``), do:

.. code-block:: console

    (myenv) $ cd docs/
    (myenv) $ make clean && make html

The HTML documentation can be found at ``/kanapy-master/docs/index.html``.


Dependencies
-------------

Kanapy requires a working C/C++ compiler on your machine. On Linux/Mac OS,
the gcc toolchain will work well. The lightweight header-only library pybind11 
is used to create Python bindings for the code written in C++.
The C++ function will be complied by linking the Eigen library 
(present in the directory ``/kanapy-master/libs/``). CMake builds this extension.
         
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

.. _NumPy: http://numpy.scipy.org
.. _Scipy: https://www.scipy.org/
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _pytest: https://www.pytest.org
.. _sphinx: http://www.sphinx-doc.org/en/master/

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
