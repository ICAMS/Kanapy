[![image](https://joss.theoj.org/papers/10.21105/joss.01732/status.svg)](https://doi.org/10.21105/joss.01732)
[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.3662366.svg)](https://doi.org/10.5281/zenodo.3662366)
![image](https://img.shields.io/badge/Platform-Linux%2C%20MacOS%2C%20Windows-critical)
[![image](https://img.shields.io/badge/License-GNU%20AGPLv3-blue)](https://www.gnu.org/licenses/agpl-3.0.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICAMS/Kanapy.git/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb)

# Kanapy

### Python tool for microstructure analysis and generation of 3D microstructure models

  - Authors: Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier  
  - Organization: [ICAMS](http://www.icams.de/content/) / [Ruhr-Universität Bochum](https://www.ruhr-uni-bochum.de/en), Germany 
  - Contact: <alexander.hartmaier@rub.de>

Kanapy is a [python](http://www.python.org) package for generating complex three-dimensional (3D) synthetic
polycrystalline microstructures. The microstructures are built based on statistical information about phase and grain morphologies, given as size distributions and aspect ratio distrubitions of grains and phase regions. Furthermore, crystallographic texture is considered in form of orientation distribution functions (ODF) and misorientation distribution functions (MDF). Kanapy offers tools to analyze EBSD maps with respect to the morphology and texture of microstructures. Based on this experimental data, it generates 3D synthetic microstructures mimicking real ones in a statistical sense.  

The
basic implementation of Kanapy is done in form of a Python Appplication Programming Interface (API). There is also a command line interface (CLI) for administrative functions and and Graphical User Interface (GUI), which is still under development.

![](docs/figs/kanapy_graphical_abstract.png)

## Features

-   Kanapy offers a Python Application Programming Interface (API).
-   Possibility to analyze experimental microstructures based on [orix](https://orix.readthedocs.io/en/stable/#) functions.
-   Support of multiphase microstructures.
-   Generation of 3D microstructure morphology based on statistical features as size distributions and aspect ratio distributions of grains and phase regions.
-   Crystallographic texture reconstruction using orientations from
    experimental data in form of Orientation Distribution Function (ODF).
-   Optimal orientation assignment based on measured Misorientation Distribution Function (MDF) that maintains correct statistical description of high-angle or low-angle grain boundary characteristics.
-   Independent execution of individual modules through easy data
    storage and handling.
-   In-built hexahedral mesh generator for representation of complex polycrystalline microstructures in form of voxels.
-   Efficient generation of space filling structures by particle dynamics method.
-   Collision handling of particles through a two-layer
    collision detection method employing the Octree spatial data
    structure and the bounding sphere hierarchy.
-   Option to generate spherical particle position and radius files
    that can be read by the Voronoi tessellation software
    [Neper](http://neper.sourceforge.net/).
-   Option to generate input files for finite-element packages.
-   Import and export of voxel structures according to following the modular materials data schema published on [GitHub](https://github.com/Ronakshoghi/MetadataSchema.git) for data transfer between different tools.

## Installation

The preferred method to use Kanapy is within [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), into which it can be easily installed from [conda-forge](https://conda-forge.org) by

```
$ conda install kanapy -c conda-forge
```

Generally, it can be installed within any 
Python environment supporting the package installer for python ([pip](https://pypi.org/project/pip/)) from its latest [PyPi](https://pypi.org/project/kanapy/) image via the shell command

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
(knpy) $ kanapy runTests
```

### Using Kanapy in your Python scripts
After installation by any of those methods, the package can be used as API within python, e.g. by importing the entire package with

```python
import kanapy as knpy
```

### Command line tools
Kanapy supports some command line tools, a list of supported tools can be displayed with

```
(knpy) $ kanapy --help          
```

### Graphical User Interface (GUI)
The alpha-version of the GUI can be started with the shell command

```
(knpy) $ kanapy gui
```


## Examples [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICAMS/Kanapy.git/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb)
Kanapy comes with several examples in form of Python scripts and Juypter notebooks. If you want to create a local copy of the kanapy/examples directory within the current working directory (cwd), please run the command

```
(knpy) $ kanapy copyExamples          
```

Kanapy notebooks can also be used on [Binder](https://mybinder.org/v2/gh/ICAMS/Kanapy.git/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb).


## Documentation

The Kanapy documentation is available online on GitHub Pages: [https://icams.github.io/Kanapy/](https://icams.github.io/Kanapy/) and can directly be displayed with

```
(knpy) $ kanapy readDocs           
```

The documentation for Kanapy is generated using [Sphinx](http://www.sphinx-doc.org/en/master/). 

## Dependencies

Below are the listed dependencies for running Kanapy:

-   [NumPy](https://numpy.org) for array manipulation.
-   [SciPy](https://www.scipy.org/) for functionalities like Convexhull.
-   [Matplotlib](https://matplotlib.org/) for plotting and visualizing.
-   [orix](https://orix.readthedocs.io/en/stable/#) for reading and analyzing EBSD maps and for generation of crystal orientations
-   [NetworkX](https://networkx.org) generating graph networks of microstructures
-   [scikit-image](https://scikit-image.org) processing of microstructure images




## Citation

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

## Related works and applications

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

 - v3: Introduction of Python API
 - v4: Import and export of microstructures in form of voxels
 - v5: Pure Python version, support of CLI functions suspended
 - v6: Major revision of internal data structure and statistical microstructure parameters
 - v6.1: Full support of dual-phase and porous microstructures
 - v6.2: Possibility of other geometries than ellipsoids as basic microstructure shapes
 - v6.3: Implementation of velocity-Verlet algorithm to integrate particle trajectories during packing
 - v6.4: Support of the [modular materials data schema](https://github.com/Ronakshoghi/MetadataSchema.git) for import and export of microstructures 
 - v6.5: Switched to orix library for EBSD import and analysis and generation of textures to have a pure Python code. The MTEX backend is still available with [Kanapy-mtex](https://github.com/ICAMS/kanapy-mtex.git).

## Licenses

<a rel="license" href="https://www.gnu.org/licenses/agpl-3.0.html"><img alt="AGPLv3" style="border-width:0;max-height:30px;height:50%;" src="https://www.gnu.org/graphics/agplv3-155x51.png" /></a>
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
   <img alt="Creative Commons License" style="border-width:0;max-height:30px;height:100%;" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

Kanapy is made available under the GNU Affero General Public License (AGPL) v3
[license](https://www.gnu.org/licenses/agpl-3.0.html).    
The additional materials under examples and in the documentation are published under the Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0) [license](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

&copy; 2025 by Authors, ICAMS/Ruhr University Bochum, Germany

## About

The name Kanapy is derived from the sanskrit word
[káṇa](https://en.wiktionary.org/wiki/%E0%A4%95%E0%A4%A3) meaning
particle. Kanapy is primarily developed at the [Interdisciplinary Center
for Advanced Materials Simulation (ICAMS), Ruhr University Bochum -
Germany](http://www.icams.de/content/). Our goal is to build a complete
synthetic microstructure generation tool for research and industry use.


## Disclaimer

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
