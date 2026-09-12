"""
Top-level package for kanapy with default ORIX backend

use pip install kanapy-mtex for version based on MTEX library, depending on Matlab

 Copyright (C) 2025, 2026  by {__author__} ICAMS / Ruhr University Bochum, Germany

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
from importlib.metadata import version as distribution_version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib


def _get_version() -> str:
    """Read the project version from pyproject.toml when running from source."""
    pyproject = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    if pyproject.is_file():
        with pyproject.open('rb') as file:
            return tomllib.load(file)['project']['version']
    return distribution_version('kanapy')

# Re-export shared core and texture modules for convenience
from .core import Microstructure, set_stats, pickle2microstructure, import_voxels,\
     import_stats, write_stats, start, triple_surf, plot_voxels_3D, plot_polygons_3D,\
     plot_mean_ellipsoids_from_stats

from .texture import EBSDmap, ODF, createOriset, createOrisetRandom, \
    get_ipf_colors, plot_pole_figure, plot_pole_figure_proj
from .graph_workflow import (
    EBSDGraphConfig,
    EBSDGraphOutputOptions,
    EBSDGraphResult,
    build_ebsd_graph,
    load_ebsd_graph,
    write_graph_result_outputs,
)

logger = logging.getLogger(__name__)
poly_scale = 1.6
__author__ = ('Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, '
              'Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier')
__email__ = 'alexander.hartmaier@rub.de'
__version__ = _get_version()
__backend__ = "orix"
__all__ = ["Microstructure", "set_stats", "pickle2microstructure", "import_voxels",
           "import_stats", "write_stats", "start", "EBSDmap", "ODF",  "createOriset",
           "createOrisetRandom", "get_ipf_colors", "plot_pole_figure",
           "plot_pole_figure_proj", "plot_voxels_3D", "plot_polygons_3D", "plot_mean_ellipsoids_from_stats",
           "triple_surf", "EBSDGraphConfig", "EBSDGraphOutputOptions",
           "EBSDGraphResult", "build_ebsd_graph", "load_ebsd_graph",
           "write_graph_result_outputs"]

if triple_surf:
    from .core import create_ref_ell
    __all__.append("create_ref_ell")

MTEX_AVAIL = False  # legacy flag for downwards compatibility
