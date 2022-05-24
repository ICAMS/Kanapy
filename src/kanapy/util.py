# -*- coding: utf-8 -*-
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))       # Kanapy's root
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
HOME_DIR = os.path.expanduser('~')

path = ''
pel = ROOT_DIR.split('/')
for hs in pel[0:pel.index('kanapy')+1]:
    path += hs+'/'
    
MAIN_DIR = path
MTEX_DIR = path + 'libs/mtex'
KNPY_DIR = HOME_DIR + '/.kanapy'
if not os.path.exists(KNPY_DIR):
    os.makedirs(KNPY_DIR)
    os.makedirs(KNPY_DIR + '/tests')
    os.system(f'cp {MAIN_DIR}/tests/unitTest_ODF_MDF_reconstruction.m {KNPY_DIR}/tests')