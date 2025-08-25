# -*- coding: utf-8 -*-

"""Top-level package for Kanapy."""
#
# Copyright (C) 2025  by {__author__} ICAMS / Ruhr University Bochum, Germany
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import logging
from importlib.metadata import version
from importlib.resources import files
from .api import Microstructure
from .initializations import set_stats
from .plotting import plot_voxels_3D, plot_polygons_3D
from .input_output import export2abaqus,writeAbaqusMat, pickle2microstructure, import_voxels, \
    import_stats, write_stats
from .rve_stats import find_rot_axis, bbox

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # root directory of package, where source code is located
MTEX_DIR = os.path.normpath(os.path.join(files("kanapy"), 'libs', 'mtex'))  # directory where MTEX routines are located

log_level = 20  # Levels for logging: 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR
logging.basicConfig(level=log_level)  # set log level

poly_scale = 1.6

try:
    raise ValueError('MTEX deactivated.')
    from .textures import EBSDmap, createOriset, createOrisetRandom, get_ipf_colors
    MTEX_AVAIL = True
except:
    from .textures_pp import EBSDmap, ODF, createOriset, createOrisetRandom, \
        get_ipf_colors, plot_inverse_pole_figure, plot_inverse_pole_figure_density
    MTEX_AVAIL = False

try:
    from .triple_surface import create_ref_ell
    triple_surf = True
except:
    triple_surf = False

__author__ = ('Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Ronak Shoghi, '
              'Napat Vajragupta, Yousef Rezek, Hrushikesh Uday Bhimavarapu, Alexander Hartmaier')
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('kanapy')
