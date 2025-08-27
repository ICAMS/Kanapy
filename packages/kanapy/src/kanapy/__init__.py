"""
Top-level package for kanapy with default ORIX backend

use pip install kanapy-mtex for version based on MTEX library, depending on Matlab

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
import sys
import logging
from importlib.metadata import version
from importlib.resources import files

# Re-export shared core and texture modules for convenience
#if os.path.abspath('.core') not in sys.path:
#    sys.path.insert(0, os.path.abspath('.core'))
#if os.path.abspath('./texture') not in sys.path:
#    sys.path.insert(0, os.path.abspath('./texture'))
from .core import Microstructure, set_stats, pickle2microstructure, import_voxels,\
     import_stats, write_stats, start, triple_surf

from .texture import EBSDmap, ODF, createOriset, createOrisetRandom, \
    get_ipf_colors, plot_inverse_pole_figure, plot_inverse_pole_figure_density

log_level = 20  # Levels for logging: 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR
logging.basicConfig(level=log_level)  # set log level
poly_scale = 1.6
__author__ = ('Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, '
              'Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier')
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('kanapy')
__backend__ = "orix"
__all__ = ["Microstructure", "EBSDmap", "ODF", "set_stats", "pickle2microstructure", "import_voxels",
           "import_stats", "write_stats", "createOriset", "createOrisetRandom",
           "get_ipf_colors", "plot_inverse_pole_figure", "plot_inverse_pole_figure_density",
           "start"]
MTEX_AVAIL = False  # legacy flag
