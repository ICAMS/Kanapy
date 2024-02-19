#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This software is distributed under the  GNU AFFERO GENERAL PUBLIC LICENSE, Version 3, 19 November 2007
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
from setuptools import setup, find_packages

# create file structure for MTEX support
# not required otherwise
MAIN_DIR = os.getcwd()  # directory in which repository is cloned
env_path = os.environ['CONDA_PREFIX']  # get path to environment

#create PATHS dictionary
path_path = os.path.join(env_path, 'PATHS.json')
path_dict = {'MAIN_DIR': MAIN_DIR,
             'ENV_DIR': env_path,
             'MTEXpath': os.path.join(MAIN_DIR, 'libs', 'mtex')}

# safe paths in installation directory and in working directory (latter will have priority for reading in kanapy.util)
with open(path_path, 'w') as outfile:
    json.dump(path_dict, outfile, indent=2)

# execute setup procedure
setup(
    name='kanapy',
    version='6.1.6',
    author='Mahesh R.G. Prasad, Abhishek Biswas, Golsa Tolooei Eshlaghi, Napat Vajragupta, Alexander Hartmaier',
    author_email='alexander.hartmaier@rub.de',
    classifiers=[        
        'License :: GNU AFFERO GENERAL PUBLIC LICENSE v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],     
    description="Python package for generating complex synthetic polycrystalline microstructures",
    install_requires=['numpy', 'matplotlib', 'scipy', 'seaborn', 'click', 'tqdm',
                      'pytest-cov', 'pytest-mock', 'pytest'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license="GNU AGPL v3 license",    
    url='https://github.com/ICAMS/Kanapy.git',
    entry_points={'console_scripts': ['kanapy = kanapy.cli:start']},
    zip_safe=False,
)
