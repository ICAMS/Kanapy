# -*- coding: utf-8 -*-

"""Top-level package for kanapy."""
import logging
from importlib.metadata import version
from kanapy.api import Microstructure
from kanapy.initializations import set_stats
from kanapy.plotting import plot_voxels_3D, plot_polygons_3D
from kanapy.input_output import export2abaqus,writeAbaqusMat, pickle2microstructure, import_voxels, \
    import_stats, write_stats
from kanapy.rve_stats import find_rot_axis, bbox
from kanapy.util import ROOT_DIR, MAIN_DIR, MTEX_DIR, poly_scale

log_level = 20  # Levels for logging: 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR
logging.basicConfig(level=log_level)  # set log level
try:
    #raise ValueError('MTEX deactivated.')
    from kanapy.textures import EBSDmap, createOriset, createOrisetRandom, get_ipf_colors
    MTEX_AVAIL = True
except:
    from kanapy.textures_pp import EBSDmap
    MTEX_AVAIL = False
if MTEX_DIR is None and MTEX_AVAIL:
    logging.error('Inconsistent installation of Kanapy package. MTEX tools will not be available.')
    MTEX_AVAIL = False

try:
    from kanapy.triple_surface import create_ref_ell
    triple_surf = True
except:
    triple_surf = False

__author__ = 'Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Alexander Hartmaier'
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('kanapy')
