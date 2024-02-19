#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 08:51:22 2021

@author: alexander
"""

from kanapy.util import MTEX_DIR, ROOT_DIR, MAIN_DIR
import os
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(MTEX_DIR, nargout=0)
eng.addpath(ROOT_DIR, nargout=0)
mat_path = os.path.join(MAIN_DIR, 'src', 'kanapy')
eng.addpath(mat_path, nargout=0)
eng.startup(nargout=0)
