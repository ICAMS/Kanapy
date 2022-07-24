#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 08:51:22 2021

@author: alexander
"""

import matlab.engine
eng = matlab.engine.start_matlab()
eng.startup(nargout=0)

