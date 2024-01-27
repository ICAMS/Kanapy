# -*- coding: utf-8 -*-
import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # root directory of package
WORK_DIR = os.path.expanduser('~') + '/.kanapy'  # working directory for temporary files
path_json = os.path.join(WORK_DIR, 'PATHS.json')
if not os.path.exists(path_json):
    # no user-specific path file in working directory, try if admin installation with paths in ROOT_DIR
    path_json = os.path.join(ROOT_DIR, 'PATHS.json')
    if not os.path.exists(path_json):
        raise FileNotFoundError('Package not properly installed, no PATH.json in working or root directory.')
with open(path_json) as json_file:
    paths = json.load(json_file)
MAIN_DIR = paths['MAIN_DIR']  # directory in which repository is cloned
MTEX_DIR = paths['MTEXpath']  # path to MTEX
log_level = 20  # Levels for logging: 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR
