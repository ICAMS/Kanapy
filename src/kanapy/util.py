# -*- coding: utf-8 -*-
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))       # Kanapy's root
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

path = ''
pel = ROOT_DIR.split('/')
for hs in pel[0:pel.index('kanapy')+1]:
    path += hs+'/'
    
MAIN_DIR = path
MTEX_DIR = path + 'libs/mtex'
