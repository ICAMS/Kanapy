#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools
from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
    
setup(
    author="Mahesh R.G Prasad",
    author_email='mahesh.prasad@rub.de',
    classifiers=[        
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],    
    description="A python package for generating complex synthetic polycrystalline microstructures.",
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='kanapy',
    name='kanapy',
    test_suite='tests',    
    url='https://github.com/mrgprasad/kanapy',
    version='0.0.1',
    zip_safe=False,    
)        
