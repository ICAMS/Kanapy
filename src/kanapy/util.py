# -*- coding: utf-8 -*-
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Kanapy's root
path = ''
pel = ROOT_DIR.split('/')
for hs in pel[0:pel.index('kanapy')+1]:
    path += hs+'/'
MAIN_DIR = path[0:-1]
MTEX_DIR = path + '/libs/mtex'

HOME_DIR = os.path.expanduser('~')  # User home directory
KNPY_DIR = HOME_DIR + '/.kanapy'
if not os.path.exists(KNPY_DIR):
    os.makedirs(KNPY_DIR)
    os.makedirs(KNPY_DIR + '/tests')
    os.system(f'cp {MAIN_DIR}/tests/unitTest_ODF_MDF_reconstruction.m {KNPY_DIR}/tests')