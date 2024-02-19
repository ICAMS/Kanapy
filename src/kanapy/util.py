# -*- coding: utf-8 -*-
import os
import json
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # root directory of package, where source code is located
try:
    path_path = os.environ['CONDA_PREFIX']  # if path to conda env is set, use this one
except:
    path_path = os.path.expanduser('~')  # otherwise fall back to user home
    path_path = os.path.join(path_path, '.kanapy')

path_json = os.path.join(path_path, 'PATHS.json')
if not os.path.exists(path_json):
    logging.error(f'Package not properly installed for MTEX, no file {path_json}.')
    MAIN_DIR = None
    MTEX_DIR = None
else:
    with open(path_json) as json_file:
        paths = json.load(json_file)
    MAIN_DIR = paths['MAIN_DIR']  # directory in which repository is cloned
    MTEX_DIR = paths['MTEXpath']  # path to MTEX
    ENV_DIR = paths["ENV_DIR"]  # path to knpy environment
