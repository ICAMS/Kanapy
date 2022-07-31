# -*- coding: utf-8 -*-
import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # root directory of package
WORK_DIR = os.path.expanduser('~') + '/.kanapy'  # working directory for temporary files
if not os.path.exists(WORK_DIR):
    raise FileNotFoundError('Package not properly installed, working directory is missing.')
with open(WORK_DIR + '/PATHS.json') as json_file:
    paths = json.load(json_file)
MAIN_DIR = paths['MAIN_DIR']  # directory in which repository is cloned
MTEX_DIR = paths['MTEXpath']  # path to MTEX
