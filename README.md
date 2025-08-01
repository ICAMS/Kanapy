[![image](https://joss.theoj.org/papers/10.21105/joss.01732/status.svg)](https://doi.org/10.21105/joss.01732)
[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.3662366.svg)](https://doi.org/10.5281/zenodo.3662366)
![image](https://img.shields.io/badge/Platform-Linux%2C%20MacOS%2C%20Windows-critical)
[![image](https://img.shields.io/badge/License-GNU%20AGPLv3-blue)](https://www.gnu.org/licenses/agpl-3.0.html)


Kanapy is a python package for generating complex three-dimensional (3D) synthetic
polycrystalline microstructures. The microstructures are built based on statistical information about grain geometry, given as grain size distribution and aspect ratio of grains, and crystallographic texture, in form of orientation distribution functions (ODF) and misorientation distribution functions (MDF). Kanapy offers tools to analyze EBSD maps with respect to the geometry and texture of microstructures. Based on this experimental data, it generates 3D synthetic microstructures mimicking real ones in a statistical sense. The
implementation is done in [Python](http://www.python.org).  

The texture module of Kanapy is implemented as
[MATLAB](https://www.mathworks.com/products/matlab.html) functions using several algorithms implemented in
[MTEX](https://mtex-toolbox.github.io/) for texture analysis.

![](docs/figs/kanapy_graphical_abstract.png)

# Features

-   Kanapy offers a Python Application Programming Interface (API).
-   Possibility to analyze experimental microstructures based on [MTEX](https://mtex-toolbox.github.io/) functions.
-   Generation of microstructure geometry based on statistical features as grain size distribution and grain aspect ratio distribution.
-   Crystallographic texture reconstruction using orientations from
    experimental data in form of Orientation Distribution Function (ODF).
-   Optimal orientation assignment based on measured Misorientation Distribution Function (MDF) that maintains correct statistical description of high-angle or low-angle grain boundary characteristics.
-   Independent execution of individual modules through easy data
    storage and handling.
-   In-built hexahedral mesh generator for complex polycrystalline
    microstructures.
-   Efficient generation of space filling structures by particle dynamics method.
-   Collision handling of particles through a two-layer
    collision detection method employing the Octree spatial data
    structure and the bounding sphere hierarchy.
-   Option to generate spherical particle position and radius files
    that can be read by the Voronoi tessellation software
    [Neper](http://neper.sourceforge.net/).
-   Option to generate input files for the commercial finite-element
    software
    [Abaqus](https://www.3ds.com/products-services/simulia/products/abaqus/)
-   Analysis and RVE generation for dual-phase microstructures.
-   Import and export of voxelated structures for data transfer between different tools.

# Installation

The preferred method to install Kanapy is through Anaconda or Miniconda
Python distributions. If you do not have any, we suggest installing
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

It can be installed from [conda-forge](https://conda-forge.org) by

```
$ conda install kanapy
```

or from [PyPi](https://pypi.org/project/kanapy/) via pip

```
$ pip install kanapy
```

Alternatively, the most recent version of the complete repository, including the source code, documentation and examples, can be cloned and installed locally. It is recommended to create a conda environment before installation. This can be done by the following the command line instructions

```
$ git clone https://github.com/ICAMS/Kanapy.git ./kanapy
$ cd kanapy
$ conda env create -f environment.yml
$ conda activate knpy
(knpy) $ python -m pip install .
```

Kanapy is now installed along with all its dependencies. The correct installation with this method can be tested with

```
(knpy) $ pytest tests -v
```

After this, the package can be used as API within python, e.g. by importing the entire package with

```python
import kanapy as knpy
```

### Kanapy texture module

If you intend
to use Kanapy's texture module, a
[MATLAB](https://www.mathworks.com/products/matlab.html) installation is required because the texture module is based on [MTEX](https://mtex-toolbox.github.io/) functions. Kanapy uses a local version of MTEX stored in `src/kanapy/libs/mtex`, if you want to use another MTEX version, please set the paths accordingly. If MATLAB is available on your system, the texture module is initialized by the command

``` 
(knpy) $ kanapy setupTexture
```
**Note:** The absolute paths to {user\_dependent\_path}/site-packages/kanapy and {user\_dependent\_path}/site-packages/kanapy/libs/mtex should to be added to the MATLABPATH environment variable, see [Mathworks&reg; documentation](https://de.mathworks.com/help/matlab/matlab_env/add-folders-to-matlab-search-path-at-startup.html#). If the texture module is installed as described above, this is done automatically within Kanapy.


**Note:** The installation scripts have been tested for Matlab R2024a and R2025a with Python 3.9 and 3.10 on current Linux, MacOS and Windows systems. If you are using other Matlab versions or if you do not have write permission in the Matlab-extern/engines/python directory, the script "setupTexture" might fail. In that case, you or a system administrator can setup the Matlab Engine API for Python manually. To do so, please follow the instructions given on [Mathworks&reg;](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), to (i) verify your configuration, (ii) install Engine API, and (iii) start MATLAB Engine. The Python version of the *knpy*-environment can be changed according to the requirements of the Matlab Engine API by editing the "environment.yml" file and re-creating the conda environment *knpy*.


# Examples
Kanapy comes with several examples in form of Python scripts and Juypter notebooks. If you want to create a local copy of the kanapy/examples directory within the current working directory (cwd), please run the command

```
(knpy) $ kanapy copyExamples          
```

# Documentation

The Kanapy documentation is available online on GitHub Pages: [https://icams.github.io/Kanapy/](https://icams.github.io/Kanapy/)

The documentation for Kanapy is generated using [Sphinx](http://www.sphinx-doc.org/en/master/). 

# Dependencies
### Third-party components


Kanapy's texture module requires
[MATLAB](https://www.mathworks.com/products/matlab.html) to be installed on your machine. Make sure to use MATLAB v2024a or above. This package includes code from the project [MTEX](https://mtex-toolbox.github.io/) contained in `src/kanapy/libs/mtex`, which is licensed under the GNU GPL v2 license. See the LICENSE file in that directory for details.

### Core dependencies

Below are the listed dependencies for running Kanapy:

> -   [NumPy](http://numpy.scipy.org) for array manipulation.
> -   [Scipy](https://www.scipy.org/) for functionalities like
>     Convexhull.
> -   [Matplotlib](https://matplotlib.org/) for plotting and
>     visualizing.

### Optional dependencies
> -   [MATLAB](https://www.mathworks.com/products/matlab.html) for
>     texture module (commercial software).
> -   [MTEX](https://mtex-toolbox.github.io/) for texture module (Kanapy package contains MTEX version 5.2.2).
> -   [OVITO](https://ovito.org/) for visualizing simulation data.


# Citation

The preferred way to cite Kanapy is:

``` bibtex
@article{Biswas2020,
  doi = {10.5281/zenodo.3662366},
  url = {https://doi.org/10.5281/zenodo.3662366},
  author = {Abhishek Biswas and Mahesh R.G. Prasad and Napat Vajragupta and Alexander Hartmaier},
  title = {Kanapy: Synthetic polycrystalline microstructure generator with geometry and texture},
  journal = {Zenodo},
  year = {2020}
}
```

# Related works and applications

-   Prasad et al., (2019). Kanapy: A Python package for generating
    complex synthetic polycrystalline microstructures. Journal of Open
    Source Software, 4(43), 1732. <https://doi.org/10.21105/joss.01732>
-   Biswas, Abhishek, R.G. Prasad, Mahesh, Vajragupta, Napat, &
    Hartmaier, Alexander. (2020, February 11). Kanapy: Synthetic
    polycrystalline microstructure generator with geometry and texture
    (Version v2.0.0). Zenodo. <http://doi.org/10.5281/zenodo.3662366>
-   Biswas, A., Prasad, M.R.G., Vajragupta, N., ul Hassan, H., Brenne,
    F., Niendorf, T. and Hartmaier, A. (2019), Influence of
    Microstructural Features on the Strain Hardening Behavior of
    Additively Manufactured Metallic Components. Adv. Eng. Mater.,
    21: 1900275. <http://doi.org/10.1002/adem.201900275>
-   Biswas, A., Vajragupta, N., Hielscher, R. & Hartmaier, A. (2020). J.
    Appl. Cryst. 53, 178-187.
    <https://doi.org/10.1107/S1600576719017138>
-   Biswas, A., Prasad, M.R.G., Vajragupta, N., Kostka, A., Niendorf, T.
    and Hartmaier, A. (2020), Effect of Grain Statistics on
    Micromechanical Modeling: The Example of Additively Manufactured
    Materials Examined by Electron Backscatter Diffraction. Adv. Eng.
    Mater., 22: 1901416. <http://doi.org/10.1002/adem.201901416>
-   R.G. Prasad, M., Biswas, A., Geenen, K., Amin, W., Gao, S., Lian,
    J., Röttger, A., Vajragupta, N. and Hartmaier, A. (2020), Influence
    of Pore Characteristics on Anisotropic Mechanical Behavior of Laser
    Powder Bed Fusion--Manufactured Metal by Micromechanical Modeling.
    Adv. Eng. Mater., <https://doi.org/10.1002/adem.202000641>

## Version history

 - v3: Introduction of API
 - v4: Import and export of microstructures in form of voxels
 - v5: Pure Python version, support of CLI suspended
 - v6: Major revision of internal data structure and statistical microstructure parameters
 - v6.1: Full support of dual-phase and porous microstructures
 - v6.2: Possibility of other geometries than ellipsoids as basic microstructure shapes
 - v6.3: Implementation of velocity-Verlet algorithm to integrate particle trajectories during packing
 - v6.4: Implmenentation of universal data structure for import and export of microstructures 

# Licenses

<a rel="license" href="https://www.gnu.org/licenses/agpl-3.0.html"><img alt="AGPLv3" style="border-width:0;max-height:30px;height:50%;" src="https://www.gnu.org/graphics/agplv3-155x51.png" /></a>
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
   <img alt="Creative Commons License" style="border-width:0;max-height:30px;height:100%;" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

Kanapy is made available under the GNU Affero General Public License (AGPL) v3
[license](https://www.gnu.org/licenses/agpl-3.0.html).   
MTEX is licensed under the GNU GPL v2 [license](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).  
The additional materials under examples and in the documentation are published under the Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0) [license](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

# About

The name Kanapy is derived from the sanskrit word
[káṇa](https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3) meaning
particle. Kanapy is primarily developed at the [Interdisciplinary Center
for Advanced Materials Simulation (ICAMS), Ruhr University Bochum -
Germany](http://www.icams.de/content/). Our goal is to build a complete
synthetic microstructure generation tool for research and industry use.

# Authors

Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier  
[ICAMS](http://www.icams.de/content/) / [Ruhr-Universität Bochum](https://www.ruhr-uni-bochum.de/en), Germany 


# Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
