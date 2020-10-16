.. highlight:: shell

.. image:: https://joss.theoj.org/papers/10.21105/joss.01732/status.svg
   :target: https://doi.org/10.21105/joss.01732

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3662366.svg
   :target: https://doi.org/10.5281/zenodo.3662366
   
.. image:: https://img.shields.io/badge/Platform-Linux%2C%20MacOS%2C%20Windows-critical
   
.. image:: https://img.shields.io/travis/mrgprasad/kanapy.svg
    :target: https://travis-ci.org/mrgprasad/kanapy

.. image:: https://codecov.io/gh/mrgprasad/kanapy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mrgprasad/kanapy
    
.. image:: https://img.shields.io/badge/License-GNU%20AGPLv3-blue
   :target: https://www.gnu.org/licenses/agpl-3.0.html

.. image:: https://img.shields.io/github/v/release/mrgprasad/kanapy

Kanapy is a python package for generating complex synthetic polycrystalline microstructures. The general implementation is done in Python_ with the performance critical part for the geometry module implemented in C++. The Python bindings for the code written in C++ are generated using the lightweight header-only library pybind11_. The C++ part of the implementation utilizes the Eigen_ library for efficient linear algebra calculations. The texture module of Kanapy is implemented as MATLAB_ functions. It also utilizes several algorithms implemented in MTEX_ for texture analysis. 

.. _Python: http://www.python.org
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/

.. figure:: /docs/figs/Kanapy_graphical_abstract.png
    :align: center
    
    
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
The preferred method to install kanapy is through 
Anaconda or Miniconda Python distributions. If you do not have any, we suggest installing miniconda_. 

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

Once done, create a virtual environment for Kanapy installation and clone the repository to 
a desired location and install.

.. code-block:: console

    $ conda create -y -n knpy python=3.6 pip git
    $ conda activate knpy    
    (knpy) $ git clone https://github.com/mrgprasad/kanapy.git <location to clone>/kanapy-master
    (knpy) $ cd kanapy-master/
    (knpy) $ conda install -y -c conda-forge --file requirements.txt    
    (knpy) $ pip install -e .
    
    
Kanapy is now installed along with all its dependencies. If you intend to use Kanapy's 
texture module, link Kanapy with MATLAB_ and MTEX_ installations by 
running: :bash:`kanapy setupTexture` and follow the instructions.

.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/
            
Running tests
--------------
Kanapy uses pytest to perform all its unit testing.        
 
.. code-block:: console  
     
    (knpy) $ kanapy runTests          
    
            
Documentation build
-------------------
Documentation for kanapy is generated using Sphinx. The HTML documentation can be 
found at *../kanapy-master/docs/builds/html/index.html*

.. code-block:: console  
    
    (knpy) $ kanapy genDocs                    
        

Citation
---------
The preferred way to cite Kanapy is: 

.. code-block:: bibtex

  @article{Prasad2019,
    doi = {10.21105/joss.01732},
    url = {https://doi.org/10.21105/joss.01732},
    year = {2019},
    publisher = {The Open Journal},
    volume = {4},
    number = {43},
    pages = {1732},
    author = {Mahesh R.G. Prasad and Napat Vajragupta and Alexander Hartmaier},
    title = {Kanapy: A Python package for generating complex synthetic polycrystalline microstructures},
    journal = {Journal of Open Source Software}
  }


Related works and applications
------------------------------
* Prasad et al., (2019). Kanapy: A Python package for generating complex synthetic polycrystalline microstructures. Journal of Open Source Software, 4(43), 1732. https://doi.org/10.21105/joss.01732

* Biswas, Abhishek, R.G. Prasad, Mahesh, Vajragupta, Napat, & Hartmaier, Alexander. (2020, February 11). Kanapy: Synthetic polycrystalline microstructure generator with geometry and texture (Version v2.0.0). Zenodo. http://doi.org/10.5281/zenodo.3662366

* Biswas, A., Prasad, M.R.G., Vajragupta, N., ul Hassan, H., Brenne, F., Niendorf, T. and Hartmaier, A. (2019), Influence of Microstructural Features on the Strain Hardening Behavior of Additively Manufactured Metallic Components. Adv. Eng. Mater., 21: 1900275. http://doi.org/10.1002/adem.201900275

* Biswas, A., Vajragupta, N., Hielscher, R. & Hartmaier, A. (2020). J. Appl. Cryst. 53, 178-187. https://doi.org/10.1107/S1600576719017138

* Biswas, A., Prasad, M.R.G., Vajragupta, N., Kostka, A., Niendorf, T. and Hartmaier, A. (2020), Effect of Grain Statistics on Micromechanical Modeling: The Example of Additively Manufactured Materials Examined by Electron Backscatter Diffraction. Adv. Eng. Mater., 22: 1901416. http://doi.org/10.1002/adem.201901416


License
--------
Kanapy is made available under the MIT license_.

.. _license: https://opensource.org/licenses/MIT


About
-------
The name kanapy is derived from the sanskrit word káṇa_ meaning particle. Kanapy is primarily developed at the `Interdisciplinary Center for Advanced Materials Simulation (ICAMS), Ruhr-University Bochum - Germany <http://www.icams.de/content/>`__. Our goal is to build a complete synthetic microstructure generation tool for research and industry use. 

.. _káṇa: https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3
