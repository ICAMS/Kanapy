# -*- coding: utf-8 -*-
import os
import json
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # root directory of package, where source code is located
paths = None
try:
    # if path to conda env is set, use this one
    path_json = os.path.join(os.environ['CONDA_PREFIX'], 'PATHS.json')
    with open(path_json) as json_file:
        paths = json.load(json_file)
except Exception as e:
    # otherwise fall back to user home
    path_json = os.path.join(os.path.expanduser('~'), '.kanapy', 'PATHS.json')
    if os.path.isfile(path_json):
        with open(path_json, 'r') as f:
            paths = json.load(json_file)
    else:
        logging.error(f'No file {path_json} with MTEX paths in conda env or user home: {e}')
        logging.error(f'Package not properly installed for MTEX, will continue without.')
        MAIN_DIR = None
        MTEX_DIR = None
if paths is not None:
    MAIN_DIR = paths['MAIN_DIR']  # directory in which repository is cloned
    MTEX_DIR = paths['MTEXpath']  # path to MTEX
    ENV_DIR = paths["ENV_DIR"]  # path to knpy environment
