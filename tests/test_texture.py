"""
Test for reading of EBSD map and reconstruction of texture.

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
December 2023
"""

import os
import numpy as np
import pytest
from kanapy import MAIN_DIR, MTEX_AVAIL, EBSDmap, createOriset

@pytest.mark.skipif(MTEX_AVAIL == False, reason="Kanapy is not configured for texture analysis yet!")
def test_readEBSD():
    fname = MAIN_DIR + '/tests/ebsd_316L_1000x1000.ang'  # name of ang file to be imported
    # read EBSD map and evaluate statistics of microstructural features
    ebsd = EBSDmap(os.path.normpath(fname), plot=False)
    assert(len(ebsd.ms_data) == 1)
    gs_param = ebsd.ms_data[0]['gs_param']
    assert(np.abs(gs_param[0] - 0.99765477) < 1.e-5)
    # get list of orientations for grains in RVE matching the ODF of the EBSD map
    ori_rve = ebsd.calcORI(20)
    assert(np.abs(ori_rve[0, 1] - 0.26179939) < 1.e-5)
    # write Euler angles of grains into Abaqus input file
    #knpy.writeAbaqusMat(0, ori_rve)

def test_createORI():
    Ngr = 10
    ang = [0., 45., 0.]    # Euler angles for Goss texture
    omega = 7.5         # kernel half-width
    ori_rve = createOriset(Ngr, ang, omega)
    assert (np.abs(ori_rve[4, 0] - 0.1121997376282069) < 1.e-5)

if __name__ == "__main__":
    pytest.main([__file__])