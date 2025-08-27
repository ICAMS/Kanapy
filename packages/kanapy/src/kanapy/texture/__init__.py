"""
Top-level package for kanapy with default ORIX backend

 Copyright (C) 2025  by {__author__} ICAMS / Ruhr University Bochum, Germany

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

import os
import logging
from importlib.metadata import version
from importlib.resources import files

from .textures_orix import EBSDmap, ODF, createOriset, createOrisetRandom, \
    get_ipf_colors, plot_inverse_pole_figure, plot_inverse_pole_figure_density

# Re-export shared core modules for convenience
try:
    from kanapy_core import Microstructure, set_stats, pickle2microstructure, import_voxels,\
         import_stats, write_stats, poly_scale, log_level, triple_surf
except ImportError:
    raise ModuleNotFoundError('Failed to import kanapy-core routines. Please install kanapy-core.')

logging.basicConfig(level=log_level)  # set log level
__author__ = ('Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, '
              'Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier')
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('kanapy-orix')
__all__ = ["Microstructure", "EBSDmap", "ODF", "set_stats", "pickle2microstructure", "import_voxels",
           "import_stats", "write_stats", "createOriset", "createOrisetRandom",
           "get_ipf_colors", "plot_inverse_pole_figure", "plot_inverse_pole_figure_density"]
__backend__ = 'orix'
MTEX_AVAIL = False  # legacy flag