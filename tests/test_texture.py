#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil, json
from pathlib import Path
from subprocess import PIPE, run 

import numpy as np
import pytest

import kanapy
from kanapy.util import KNPY_DIR, ROOT_DIR

# Check if kanapy has been configured with MATLAB & MTEX
pathFile = KNPY_DIR + '/PATHS.json'
if not os.path.exists(pathFile):
    skipVal = True
else:
    skipVal = False
            
@pytest.mark.skipif(skipVal == True, reason="Your Kanapy is not configured for texture analysis yet! Run: kanapy setuptexture to set it up.")
def test_analyzeTexture():
           
    pathFile = KNPY_DIR + '/PATHS.json'
    testDir = KNPY_DIR + '/tests'
    utFile = testDir + '/unitTest_ODF_MDF_reconstruction.m'
    logFile = testDir + '/matlabUnitTest.log'
    resultFile = 'matlabResults.txt'
    
    # Read the MATLAB & MTEX paths                 
    with open(pathFile) as json_file:  
        path_dict = json.load(json_file) 

    # Read the MATLAB unittest file and replace the MTEX path in it
    with open (utFile,'r') as f:
        data = f.readlines()
    data[3] = '        r = {\'' + path_dict['MTEXpath'] + '\'};\n' 
    with open (utFile,'w') as f:
        data = f.writelines(data)
                 
    # Copy file temporarily into the 'tests/' folder
    filelist = ['ODF_reduction_algo.m','splitMean.m','odfEst.m','mdf_Anglefitting_algo_MC.m']
    for i in filelist:
        shutil.copy2(ROOT_DIR+'/'+i, testDir)

    # Create a temporary matlab script file that runs Texture reduction algorithm      
    TRfile = testDir+'/runUnitTest.m'           # Temporary '.m' file

    with open (TRfile,'w') as f:
        f.write("MTEXpath='{}';\n".format(path_dict['MTEXpath']))
        f.write("run([MTEXpath 'install_mtex.m'])\n")
        f.write('\n')
        f.write("result = runtests('{0}');\n".format(utFile))
        f.write("T=table(result)\n")  
        f.write("writetable(T,'{}','Delimiter',' ');\n".format(resultFile))    
        f.write("exit;")

    # Run from the terminal                                    
    #command = ["{}".format(path_dict['MATLABpath']), " -nosplash -nodesktop -nodisplay -r ", '"run(', "'{}')".format(TRfile), '; exit;"']
    #with open(logFile, "w") as outfile:
    #    result = run(command, stdout=outfile, stderr=PIPE, universal_newlines=True)             

    cmd1 = "{0} -nosplash -nodesktop -nodisplay -r ".format(path_dict['MATLABpath']) 
    cmd2 = '"run(' + "'{}')".format(TRfile) + '; exit;"'
    cmd = cmd1+cmd2
    os.system(cmd + '> {}'.format(logFile))               
        
    # Remove the files once done! 
    os.remove(TRfile)        
    for i in filelist:
        os.remove(testDir+'/'+i)
    
    # Read the tabulated result written by MATLAB
    tabRes = np.loadtxt(testDir + '/' + resultFile, delimiter=' ', skiprows=1, usecols = (1,2,3))
    
    # Remove the MATLAB tabulated results file
    os.remove(testDir + '/' + resultFile)
    
    # Report back to pytest 
    passed, failed, incomplete = int(sum(tabRes[:,0])), int(sum(tabRes[:,1])), int(sum(tabRes[:,2]))        
    assert passed==np.shape(tabRes)[0]    
               

if __name__ == "__main__":   
    pytest.main([__file__])
