#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copy the PATH.json file from MAIN_DIR to ROOT_DIR,
necessary for installation of type "--admin" as ROOT_DIR is not known during installation.
"""
import shutil
import os
from kanapy.util import MAIN_DIR, ROOT_DIR

with open('admin_flag', 'w') as file:
    file.write('1')
os.system('python -m pip install .')
src = os.path.join(MAIN_DIR, 'PATHS.json')
dst = os.path.join(ROOT_DIR, 'PATHS.json')
shutil.copy(src, dst)
os.system('kanapy setupTexture -admin=True')
