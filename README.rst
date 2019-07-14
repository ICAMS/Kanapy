.. highlight:: shell

=========
Overview
=========


.. image:: https://img.shields.io/travis/mrgprasad/kanapy.svg
    :target: https://travis-ci.org/mrgprasad/kanapy
    :width: 20 %

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://lbesson.mit-license.org/
   :width: 17 %

kanapy is a python package for generating complex synthetic polycrystalline microstructures. The general implementation is done in Python_ with performance critical part implemented in C++. Python bindings for the code written in C++ is generated using the lightweight header-only library pybind11_. The C++ part of the implementation utilizes the Eigen_ library for efficient linear algebra calculations.

.. _Python: http://www.python.org
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page


Features
--------

* User interface to kanapy through scripts are written in pure Python.  
* Efficient collision handling through a two-layer collision detection method  employing the Octree spatial data structure and the bounding sphere hierarchy. 
* Efficient in-built hexahedral mesh generator for complex polycrystalline microstructures.    
* Independent execution of individual modules through easy data storage and handling.    
* Flexibility in the choice of particle packing time step to be sent for voxelization (meshing).
* Option to generate spherical particle position and radius files that can be read by voronoi tessellation software Neper_.
* Option to generate input files for commercial finite-element software Abaqus_.    
* High-performance for the critical part of the code using Python-C++ bindings.  

.. _Neper: http://neper.sourceforge.net/
.. _Abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/

Documentation
-------------

Online documentation can be found `here <https://mrgprasad.github.io/kanapy/index.html>`_.  

Installation
------------
CMake is used for building extensions, if it is not installed on your machine follow this 
`CMake documentation`_ to install it.

.. _CMake documentation: https://cgold.readthedocs.io/en/latest/first-step/installation.html

The preferred method to install kanapy is through Anaconda or Miniconda python distributions. 
If you do not have any we suggest installing miniconda_. 

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html


Once done, create a virtual environment for installing python specific packages required for kanapy and 
activate it.

.. code-block:: console

    $ conda create -n myenv python=3.6 pip git
    $ conda activate myenv    

.. note:: 1. ``myenv`` can be replaced with any name for your environment.
          2. For older versions of anaconda/miniconda use: ``source activate myenv``
                    
.. tip:: To learn more on managing environments see Anaconda documentation_.

.. _documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html           

You can either clone the kanapy public repository using git or 
download the kanapy source code from the `Github repo`_ to a desired location. 

.. code-block:: console

    (myenv) $ git clone https://github.com/mrgprasad/kanapy.git <location to clone>/kanapy-master
    (myenv) $ cd kanapy-master/
    (myenv) $ conda install --file requirements.txt
    (myenv) $ pip install -e .

kanapy is now installed along with all its dependencies.

.. _Github repo: https://github.com/mrgprasad/kanapy
          
Running tests
--------------

kanapy uses ``pytest`` to perform all its unit testing. From the kanapy main directory (``kanapy-master``) run the tests using ``pytest``:

.. code-block:: console
    
    (myenv) $ pytest tests/ -v
   
   
Documentation build
-------------------
Documentation for kanapy is generated using ``Sphinx``. The following command generates HTML-based reference documentation; 
for other formats please refer to the Sphinx manual. From the kanapy main directory (``kanapy-master``):

.. code-block:: console

    (myenv) $ cd docs/
    (myenv) $ make html

.. note:: The HTML documentation can be found at ``/kanapy-master/docs/builds/html/index.html``


Dependencies
-------------

kanapy requires a working C/C++ compiler on your machine. On Linux/Mac OS
the gcc toolchain will work well. The lightweight header-only library pybind11 
is used to create Python bindings for the code written in C++.
The C++ function will be complied by linking the Eigen library 
(present in the directory ``/kanapy-master/libs/``). CMake builds this extension.
         
^^^^^^^^^^^^^^^^^^
Core dependencies
^^^^^^^^^^^^^^^^^^

Below are the listed dependencies for running kanapy:

  - NumPy_ for array manipulation.
  - Scipy_ for functionalities like Convexhull and KDTree.
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


License
--------
kanapy is made available under the MIT license


About
-------
The name kanapy is derived from the sanskrit word káṇa_ meaning particle. It is primarily developed at the `Interdisciplinary Center for Advanced Materials Simulation (ICAMS), Ruhr-University Bochum - Germany <http://www.icams.de/content/>`__. Our goal is to build a complete synthetic microstructure generation tool for research and industry use. 

.. _káṇa: https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3
