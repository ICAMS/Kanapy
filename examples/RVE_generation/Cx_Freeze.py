# -*- coding: utf-8 -*-
"""
Author: Ronak Shoghi
Date: 16.05.24
Time: 21:19

"""
import datetime
from cx_Freeze import setup, Executable

def generate_version():
    """
    Generate a version number based on the current timestamp.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d.%H%M%S")

setup(
    name="GUI",
    version=generate_version(),
    description="Kanapy",
    executables=[Executable("RVE_generation_GUI.py")]
)


