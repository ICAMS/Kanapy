# -*- coding: utf-8 -*-

"""
Top-level package for kanapy.core module with core utilities.
defines all user interfaces: API, CLI and GUI

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

from .api import Microstructure
from .initializations import set_stats
from .plotting import plot_voxels_3D, plot_polygons_3D
from .input_output import export2abaqus, writeAbaqusMat, pickle2microstructure, import_voxels, \
    import_stats, write_stats
from .rve_stats import find_rot_axis, bbox, get_grain_geom
from .cli import start

try:
    from .triple_surface import create_ref_ell
    triple_surf = True
except:
    triple_surf = False


__all__ = ["Microstructure", "set_stats", "plot_voxels_3D", "plot_polygons_3D", 
           "export2abaqus", "writeAbaqusMat", "pickle2microstructure", "import_voxels",
           "import_stats", "write_stats", "find_rot_axis", "bbox", "particle_rve", "cuboid_rve",
           "collision_routine", "get_grain_geom", "start"]
           
if triple_surf:
    __all__.append("create_ref_ell")
