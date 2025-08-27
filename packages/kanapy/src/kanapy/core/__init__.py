# -*- coding: utf-8 -*-

"""
Top-level package for Kanapy (Core utilities).
defines API, CLI and GUI, 
imported as core into kanapy-orix or kanapy-mtex

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
from .api import Microstructure
from .initializations import set_stats
from .plotting import plot_voxels_3D, plot_polygons_3D
from .input_output import export2abaqus,writeAbaqusMat, pickle2microstructure, import_voxels, \
    import_stats, write_stats
from .rve_stats import find_rot_axis, bbox

log_level = 20  # Levels for logging: 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR
logging.basicConfig(level=log_level)  # set log level
poly_scale = 1.6

try:
    from .triple_surface import create_ref_ell
    triple_surf = True
except:
    triple_surf = False

__author__ = ('Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, '
              'Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier')
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('kanapy-core')
__all__ = ["Microstructure", "set_stats", "plot_voxels_3D", "plot_polygons_3D", 
           "export2abaqus", "writeAbaqusMat", "pickle2microstructure", "import_voxels",
           "import_stats", "write_stats", "find_rot_axis", "bbox"]
if triple_surf:
    __all__.append("create_ref_ell")
