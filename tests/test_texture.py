"""
Test for reading of EBSD map and reconstruction of texture.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
December 2023
"""

import os
import numpy as np
import pytest
from kanapy import MAIN_DIR, MTEX_AVAIL, MTEX_DIR

@pytest.mark.skipif(not MTEX_AVAIL, reason="Kanapy is not configured for texture analysis yet!")
def test_mex():
    path_mex = os.path.join(MTEX_DIR, 'mex')
    dir_list = os.listdir(path_mex)
    exist = False
    for fn in dir_list:
        if fn[:16] == 'SO3Grid_find.mex':
            exist = True
            break
    assert exist

@pytest.mark.skipif(not MTEX_AVAIL, reason="Kanapy is not configured for texture analysis yet!")
def test_readEBSD():
    from kanapy.textures import EBSDmap
    fname = MAIN_DIR + '/tests/ebsd_316L_500x500.ang'  # name of ang file to be imported
    # read EBSD map and evaluate statistics of microstructural features
    ebsd = EBSDmap(os.path.normpath(fname), plot=False)
    assert len(ebsd.ms_data) == 1
    gs_param = ebsd.ms_data[0]['gs_param']
    assert np.abs(gs_param[0] - 0.7177939893510182) < 1.e-5
    # get list of orientations for grains in RVE matching the ODF of the EBSD map
    ori_rve = ebsd.calcORI(20)
    assert np.abs(ori_rve[0, 1] - 0.5817764173314431) < 1.e-5

@pytest.mark.skipif(not MTEX_AVAIL, reason="Kanapy is not configured for texture analysis yet!")
def test_createORI():
    from kanapy.textures import createOriset
    Ngr = 10
    ang = [0., 45., 0.]    # Euler angles for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = createOriset(Ngr, ang, omega)
    assert (np.abs(ori_rve[4, 0] - 0.1121997376282069) < 1.e-5)
