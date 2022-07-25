#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 23:26:59 2022

"""

import os
import shutil
import json
import numpy as np
import pytest
from kanapy import WORK_DIR, MAIN_DIR, MTEX_AVAIL
            
@pytest.mark.skipif(MTEX_AVAIL == False, reason="Kanapy is not configured for texture analysis yet!")
def test_analyzeTexture():
           
    pathFile = WORK_DIR + '/PATHS.json'
    testDir = WORK_DIR + '/tests'
    utFile = testDir + '/unitTest_ODF_MDF_reconstruction.m'
    logFile = testDir + '/matlabUnitTest.log'
    resultFile = testDir + '/matlabResults.txt'
    
    # Read the MATLAB & MTEX paths                 
    with open(pathFile) as json_file:  
        path_dict = json.load(json_file)
        
    if type(path_dict['MATLABpath']) != str:
        raise ModuleNotFoundError('Matlab not installed properly.')
    if not MTEX_AVAIL:
        raise ModuleNotFoundError('MTEX not installed properly.')

    # Read the MATLAB unittest file and replace the MTEX path in it
    with open (utFile,'r') as f:
        data = f.readlines()
    data[3] = '        r = {\'' + path_dict['MTEXpath'] + '\'};\n' 
    with open (utFile,'w') as f:
        data = f.writelines(data)
                 
    # Copy file temporarily into the 'tests/' folder
    filelist = ['ODF_reduction_algo.m', 'splitMean.m', 'odfEst.m',
                'mdf_Anglefitting_algo_MC.m']
    for i in filelist:
        shutil.copy2(MAIN_DIR+'/src/kanapy/'+i, testDir)

    # Create a temporary matlab script file that runs Texture reduction algorithm      
    TRfile = testDir + '/runUnitTest.m'  # Temporary '.m' file

    with open (TRfile, 'w') as f:
        f.write("result = runtests('{0}');\n".format(utFile))
        f.write("T=table(result)\n")  
        f.write("writetable(T,'{}','Delimiter',' ');\n".format(resultFile))    
        f.write("exit;")

    cmd1 = "{0} -nosplash -nodesktop -nodisplay -r ".format(path_dict['MATLABpath']) 
    cmd2 = '"run(' + "'{}')".format(TRfile) + '; exit;"'
    cmd = cmd1+cmd2
    os.system(cmd + '> {}'.format(logFile))               
        
    # Remove the files once done! 
    os.remove(TRfile)
    for i in filelist:
        os.remove(testDir+'/'+i)
    
    # Read the tabulated result written by MATLAB
    tabRes = np.loadtxt(resultFile, delimiter=' ', skiprows=1, usecols=(1, 2, 3))
    
    # Remove the MATLAB tabulated results file
    os.remove(resultFile)
    
    # Report back to pytest 
    passed = int(sum(tabRes[:, 0]))
    # failed = int(sum(tabRes[:, 1]))
    # incomplete = int(sum(tabRes[:, 2])
    assert passed == np.shape(tabRes)[0]
               

if __name__ == "__main__":   
    pytest.main([__file__])
