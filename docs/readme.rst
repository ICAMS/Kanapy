.. highlight:: shell

=========
Overview
=========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3662366.svg
   :target: https://doi.org/10.5281/zenodo.3662366
   
.. image:: https://joss.theoj.org/papers/10.21105/joss.01732/status.svg
   :target: https://doi.org/10.21105/joss.01732

.. image:: https://anaconda.org/conda-forge/kanapy/badges/platforms.svg
   :target: https://anaconda.org/conda-forge/kanapy
   
.. image:: https://anaconda.org/conda-forge/kanapy/badges/license.svg
   :target: https://anaconda.org/conda-forge/kanapy
   
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ICAMS/Kanapy.git/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb
   
.. image:: https://anaconda.org/conda-forge/kanapy/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/kanapy


Kanapy is a `python <http://www.python.org>`__ package for generating complex 
three-dimensional (3D) synthetic polycrystalline microstructures. The microstructures 
are built based on statistical information about phase and grain morphologies, given 
as size distributions and aspect ratio distrubitions of grains and phase regions. 
Furthermore, crystallographic texture is 
considered in form of orientation distribution functions (ODF) and misorientation 
distribution functions (MDF). Kanapy offers tools to analyze EBSD maps with respect to 
the morphology and texture of microstructures. Based on this experimental data, it 
generates 3D synthetic microstructures mimicking real ones in a statistical sense.  

The basic implementation of Kanapy is done in form of a Python Appplication Programming 
Interface (API). There is also a command line interface (CLI) for administrative 
functions and and Graphical User Interface (GUI), which is still under development.


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

.. figure:: figs/kanapy_graphical_abstract.png
    :align: center
    
    **Figure**: Kanapy workflow.
    
Features
--------
-  Kanapy offers a Python Application Programming Interface (API).
-  Kanapy offers a Python Application Programming Interface (API).
-  Possibility to analyze experimental microstructures based on 
   `orix <https://orix.readthedocs.io/en/stable/#>`__ functions.
-  Support of multiphase microstructures.
-  Generation of 3D microstructure morphology based on statistical features as size 
   distributions and aspect ratio distributions of grains and phase regions.
-  Crystallographic texture reconstruction using orientations from
   experimental data in form of Orientation Distribution Function (ODF).
-  Optimal orientation assignment based on measured Misorientation Distribution Function 
   (MDF) that maintains correct statistical description of high-angle or low-angle grain 
   boundary characteristics.
-  Independent execution of individual modules through easy data storage and handling.
-  In-built hexahedral mesh generator for representation of complex polycrystalline 
   microstructures in form of voxels.
-  Efficient generation of space filling structures by particle dynamics method.
-  Collision handling of particles through a two-layer collision detection method 
   employing the Octree spatial data structure and the bounding sphere hierarchy.
-  Option to generate spherical particle position and radius files that can be read by 
   the Voronoi tessellation software `Neper <http://neper.sourceforge.net/>`__.
-  Option to generate input files for finite-element packages.
-  Import and export of voxel structures according to following the modular materials 
   data schema published on `GitHub <https://github.com/Ronakshoghi/MetadataSchema.git>`__
   for data transfer between different tools.

   
Installation
------------
The preferred method to use Kanapy is within `Anaconda <https://www.anaconda.com>`__ 
or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, into which it can be 
easily installed from `conda-forge <https://conda-forge.org>`__ by

.. code-block:: console

   $ conda install kanapy -c conda-forge

Generally, it can be installed within any 
Python environment supporting the package installer for python 
(`pip <https://pypi.org/project/pip/>`__) from its latest 
`PyPi <https://pypi.org/project/kanapy/>`__ release via the shell command

.. code-block:: console

   $ pip install kanapy


Alternatively, the most recent version of the complete repository, including the source 
code, documentation and examples, can be cloned and installed locally from
`GitHub <https://github.com/ICAMS/kanapy>`__. It is recommended 
to create a conda environment before installation. This can be done by the following the 
command line instructions

.. code-block:: console

   $ git clone https://github.com/ICAMS/Kanapy.git ./kanapy
   $ cd kanapy
   $ conda env create -f environment.yml
   $ conda activate knpy
   (knpy) $ python -m pip install .

Kanapy is now installed along with all its dependencies. The correct installation with 
this method can be tested with

.. code-block:: console

   (knpy) $ kanapy runTests


.. note:: ``knpy`` can be replaced with any name for your environment.  
                    
.. tip:: To learn more about managing environments see Anaconda documentation_.

.. _documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Using Kanapy in your Python scripts
"""""""""""""""""""""""""""""""""""

After installation by any of those methods, the package can be used as API within python,
e.g. by importing the entire package with

.. code-block:: python

   >>> import kanapy as knpy


Command line tools
""""""""""""""""""

Kanapy supports some command line tools, a list of supported tools can be displayed with

.. code-block:: console

   (knpy) $ kanapy --help

Graphical User Interface (GUI)
""""""""""""""""""""""""""""""

The alpha-version of the GUI can be started with the shell command

.. code-block:: console

   (knpy) $ kanapy gui



Examples
--------

Kanapy comes with several examples in form of Python scripts and Juypter notebooks. 
If you want to create a local copy of the kanapy/examples directory within the current 
working directory (cwd), please run the command

.. code-block:: console

   (knpy) $ kanapy copyExamples          


Kanapy notebooks can also be used on 
`Binder <https://mybinder.org/v2/gh/ICAMS/Kanapy.git/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb>`__.      


Dependencies
------------

Below are the listed dependencies for running Kanapy:

-  `NumPy <https://numpy.org`__ for array manipulation.
-  `SciPy <https://www.scipy.org/>`__ for functionalities like Convexhull.
-  `Matplotlib <https://matplotlib.org/>`__ for plotting and visualizing.
-  `orix <https://orix.readthedocs.io/en/stable/#>`__ for reading and analyzing EBSD maps 
   and for generation of crystal orientations.
-  `NetworkX <https://networkx.org>`__ generating graph networks of microstructures.
-  `scikit-image <https://scikit-image.org>`__ processing of microstructure images.


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
