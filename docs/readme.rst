.. highlight:: shell

=========
Overview
=========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3662366.svg
   :target: https://doi.org/10.5281/zenodo.3662366
   
.. image:: https://joss.theoj.org/papers/10.21105/joss.01732/status.svg
   :target: https://doi.org/10.21105/joss.01732

.. image:: https://img.shields.io/badge/Platform-Linux%2C%20MacOS%2C%20Windows-critical
    
.. image:: https://img.shields.io/badge/License-GNU%20AGPLv3-blue
   :target: https://www.gnu.org/licenses/agpl-3.0.html

Kanapy is a python package for generating complex three-dimensional (3D) synthetic
polycrystalline microstructures. The microstructures are built based on statistical 
information about grain geometry, given as grain size distribution and aspect ratio of 
grains, and crystallographic texture, in form of orientation distribution functions 
(ODF) and misorientation distribution functions (MDF). Kanapy offers tools to analyze 
EBSD maps with respect to the geometry and texture of microstructures. Based on this 
experimental data, it generates 3D synthetic microstructures mimicking real ones in a 
statistical sense. The implementation is done in
`Python <http://www.python.org>`__.

The texture module of Kanapy is implemented as
`MATLAB <https://www.mathworks.com/products/matlab.html>`__ functions
using several algorithms implemented in
`MTEX <https://mtex-toolbox.github.io/>`__ for texture analysis.

Motivation
----------
An accurate representation of a material's microstructure is fundamental in
understanding the relationship between microstructure and its corresponding
mechanical behavior. In this regard, Kanapy is developed to be a robust and
an efficient tool to generate synthetic microstructures as the basis for
micromechanical simulations with crystal plasticity methods. The generated
microstructures can be used for mechanical analysis with Finite Element (FE)
of spectral solvers. Kanapy is designed to model simple and complex grain
morphologies and to systematically incorporate texture data directly from
experiments concurrently maintaining the numerical efficiency of the
micromechanical model. Kanapy is developed to overcome the limitations of
spatial tessellation methods and to provide an alternative to the existing
Random Sequential Addition technique of microstructure geometry generation.

.. figure:: /figs/kanapy_graphical_abstract.png
    :align: center
    
    **Figure**: Kanapy workflow.
    
Features
--------
-  Kanapy offers a Python Application Programming Interface (API).
-  Possibility to analyze experimental microstructures based on
   `MTEX <https://mtex-toolbox.github.io/>`__ functions.
-  Generation of microstructure geometry based on statistical features
   as grain size distribution and grain aspect ratio distribution.
-  Crystallographic texture reconstruction using orientations from
   experimental data in form of Orientation Distribution Function (ODF).
-  Optimal orientation assignment based on measured Misorientation
   Distribution Function (MDF) that maintains correct statistical
   description of high-angle or low-angle grain boundary
   characteristics.
-  Independent execution of individual modules through easy data storage
   and handling.
-  In-built hexahedral mesh generator for complex polycrystalline
   microstructures.
-  Efficient generation of space filling structures by particle dynamics
   method.
-  Collision handling of particles through a two-layer collision
   detection method employing the Octree spatial data structure and the
   bounding sphere hierarchy.
-  Option to generate spherical particle position- and radius files that
   can be read by the Voronoi tessellation software
   `Neper <http://neper.sourceforge.net/>`__.
-  Option to generate input files for the commercial finite-element
   software
   `Abaqus <https://www.3ds.com/products-services/simulia/products/abaqus/>`__.
-  Analysis and RVE generation for dual-phase microstructures.
-  Import and export of voxelated structures for data transfer between different tools.
   
Installation
------------
The preferred method to install kanapy is through 
Anaconda or Miniconda Python distributions. If you do not have any, we suggest installing miniconda_. 

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

Once done, create a virtual environment for Kanapy installation and clone the repository to 
a desired location and install.

.. code-block:: console

    $ git clone https://github.com/ICAMS/Kanapy.git <location to clone>/kanapy-master
    $ cd kanapy
    $ conda env create -f environment.yml
    $ conda activate knpy
    (knpy) $ python -m pip install .
    
    
Kanapy is now installed along with all its dependencies. If you intend to use Kanapy's 
texture module, a MATLAB_ installation
is required because the texture module is based on MTEX_ functions. Kanapy uses a local 
version of MTEX stored in libs/mtex, if you want to use another version, please set the 
paths accordingly.  If MATLAB is available on your system, the texture module is 
initialized by the command

.. code-block:: console

   (knpy) $ kanapy setupTexture

.. note:: 1. ``knpy`` can be replaced with any name for your environment.  
        2. The absolute paths to {user\_dependent\_path}/kanapy/src/kanapy and 
        {user\_dependent\_path}/kanapy/libs/mtex should to be added to the MATLABPATH 
        environment variable, see `Mathworks documentation`_.  
        
        3. The installation scripts have been tested for Matlab R2023a with Python 3.9 
        and 3.10. If you are using other Matlab versions, the script
        "setupTexture" might fail. In that case, you can setup the Matlab
        Engine API for Python manually. To do so, please follow the instructions
        given on the Mathworks_ website, to (i) Verify your configuration, (ii) 
        Install Engine API, and (iii) Start MATLAB Engine.
        The Python version of the *knpy*-environment can be changed according to the 
        requirements of the Matlab Engine API by editing the "environment.yml" file 
        and re-creating the conda environment *knpy*.
                    
.. tip:: To learn more about managing environments see Anaconda documentation_.

.. _documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html    
.. _Github repo: https://github.com/mrgprasad/kanapy
.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/
.. _Mathworks: https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
.. _Mathworks documentation: https://de.mathworks.com/help/matlab/matlab_env/add-folders-to-matlab-search-path-at-startup.html#

Installation as system administrator
------------------------------------

If system administrators with write access to the root directory of the knpy-environment want to install Kanapy for all users, they need to run

.. code-block:: console

   (knpy) $ python admin_setup.py


after activating the *knpy*-environment. This will automatically execute the Kanapy installation and texture setup in administrator mode. After that step, users can directly access Kanapy in the conda environment.

Running tests
--------------
Kanapy uses pytest to perform all its unit testing.        
 
.. code-block:: console  
     
    (knpy) $ kanapy runTests          
    
Updates
-------

Kanapy is constantly under development and there will be frequent updates with bugfixes and new features. To update Kanapy, follow these steps:

.. code-block:: console

    $ cd kanapy
    $ git pull
    $ conda activate knpy
    (knpy) $ python -m pip install .


This will make the new version available in your *knpy*-environment. If your Kanapy installation has been setup for textures (MTEX module), this feature will not be affected by such updates. This update routine is also valid for global installations as system administrator.

Examples
--------

Kanapy comes with several examples in form of Python scripts and Juypter notebooks. If you want 
to create a local copy of the kanapy/examples directory within the current working directory (cwd),
please run the command

.. code-block:: console

    (knpy) $ kanapy copyExamples          

      
Documentation build
-------------------
The complete documentation for kanapy is available online on GitHub
Pages: https://icams.github.io/Kanapy/

Documentation for kanapy is generated using Sphinx. You can create or
update your local documentation with the command 

.. code-block:: console  
    
    (knpy) $ kanapy genDocs                    
     
The HTML documentation is then found at *kanapy/docs/builds/html/index.html*

Dependencies
------------

Kanapy’s texture module requires MATLAB_ to be
installed on your machine. Make sure to use MATLAB v2015a and above. The
module uses a local version of MTEX_ contained in *kanapy/libs*
and does not interfere with other installations of MTEX.

.. _MATLAB: https://www.mathworks.com/products/matlab.html
.. _MTEX: https://mtex-toolbox.github.io/

^^^^^^^^^^^^^^^^^^
Core dependencies
^^^^^^^^^^^^^^^^^^

Below are the listed dependencies for running kanapy:

  - NumPy_ for array manipulation.
  - Scipy_ for functionalities like Convexhull.
  - pytest_ for running kanapy unit tests.
  - sphinx_ for generating documentation.
  - MATLAB_ for texture modules.
  - MTEX_ for texture modules.
  
.. _NumPy: http://numpy.scipy.org
.. _Scipy: https://www.scipy.org/
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
--------
The preferred way to cite Kanapy is: 

.. code-block:: bibtex

  @article{Biswas2020,
    doi = {10.5281/zenodo.3662366},
    url = {https://doi.org/10.5281/zenodo.3662366},
    author = {Abhishek Biswas and Mahesh R.G. Prasad and Napat Vajragupta and Alexander Hartmaier},
    title = {Kanapy: Synthetic polycrystalline microstructure generator with geometry and texture},
    journal = {Zenodo},
    year = {2020}
  }



Related works and applications
------------------------------
* Prasad et al., (2019). Kanapy: A Python package for generating complex synthetic polycrystalline microstructures. Journal of Open Source Software, 4(43), 1732. https://doi.org/10.21105/joss.01732

* Biswas, Abhishek, R.G. Prasad, Mahesh, Vajragupta, Napat, & Hartmaier, Alexander. (2020, February 11). Kanapy: Synthetic polycrystalline microstructure generator with geometry and texture (Version v2.0.0). Zenodo. http://doi.org/10.5281/zenodo.3662366

* Biswas, A., Prasad, M.R.G., Vajragupta, N., ul Hassan, H., Brenne, F., Niendorf, T. and Hartmaier, A. (2019), Influence of Microstructural Features on the Strain Hardening Behavior of Additively Manufactured Metallic Components. Adv. Eng. Mater., 21: 1900275. http://doi.org/10.1002/adem.201900275

* Biswas, A., Vajragupta, N., Hielscher, R. & Hartmaier, A. (2020). J. Appl. Cryst. 53, 178-187. https://doi.org/10.1107/S1600576719017138

* Biswas, A., Prasad, M.R.G., Vajragupta, N., Kostka, A., Niendorf, T. and Hartmaier, A. (2020), Effect of Grain Statistics on Micromechanical Modeling: The Example of Additively Manufactured Materials Examined by Electron Backscatter Diffraction. Adv. Eng. Mater., 22: 1901416. http://doi.org/10.1002/adem.201901416

* R.G. Prasad, M., Biswas, A., Geenen, K., Amin, W., Gao, S., Lian, J., Röttger, A., Vajragupta, N. and Hartmaier, A. (2020), Influence of Pore Characteristics on Anisotropic Mechanical Behavior of Laser Powder Bed Fusion–Manufactured Metal by Micromechanical Modeling. Adv. Eng. Mater., https://doi.org/10.1002/adem.202000641


License
-------
Kanapy is made available under the GNU AGPLv3 `license <https://www.gnu.org/licenses/agpl-3.0.html>`__.

The additional materials under examples and in the documentation are published under the Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0) `license <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__.



About
-----
The name kanapy is derived from the sanskrit word káṇa_ meaning particle. Kanapy is primarily developed at the `Interdisciplinary Center for Advanced Materials Simulation (ICAMS), Ruhr University Bochum - Germany <http://www.icams.de/content/>`__. Our goal is to build a complete synthetic microstructure generation tool for research and industry use. 

.. _káṇa: https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3

Disclaimer
----------

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS
IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
