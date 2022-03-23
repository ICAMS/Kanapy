# -*- coding: utf-8 -*-

"""Top-level package for kanapy."""

from pkg_resources import get_distribution
from kanapy.api import Microstructure
from kanapy.plotting import plot_voxels_3D, plot_polygons_3D
from kanapy.input_output import writeAbaqusMat, writeAbaqusPhase
from kanapy.util import ROOT_DIR, MAIN_DIR, MTEX_DIR
try:
    from kanapy.analyse_micrograph import EBSDmap, set_stats, createOriset, createOrisetRandom
    MTEX_AVAIL = True
except:
    MTEX_AVAIL = False

__author__ = 'Mahesh R.G Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi,\
Alexander Hartmaier'
__email__ = 'alexander.hartmaier@rub.de'
__version__ = get_distribution('kanapy').version
