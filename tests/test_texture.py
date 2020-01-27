#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil, json
from pathlib import Path
from subprocess import PIPE, run 

import pytest

import kanapy
from kanapy.util import ROOT_DIR, MAIN_DIR

# Check if kanapy has been configured with MATLAB & MTEX
pathFile = ROOT_DIR + '/PATHS.json'
if not os.path.exists(pathFile):
    skipVal = True
else:
    skipVal = False
            
@pytest.mark.skipif(skipVal == True, reason="Your Kanapy is not configured for texture analysis yet! Run: kanapy setuptexture to set it up.")
def test_analyzeTexture():
           
    pathFile = ROOT_DIR + '/PATHS.json'
    testDir = MAIN_DIR + '/tests'
    utFile = testDir + '/unitTest_ODF_MDF_reconstruction.m'
    logFile = testDir + '/matlabUnitTest.log'

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
        f.write("run([MTEXpath '/install_mtex.m'])\n")
        f.write('\n')
        f.write("mtexdata('titanium')\n")
        f.write("mtexdata('alu')\n")
        f.write("mtexdata('epidote')\n")
        f.write('\n')
        f.write("result = runtests('{0}');\n".format(utFile))
        f.write("table(result)\n")      
        f.write("exit;")
        
    # Run from the terminal                                    
    command = ["{}".format(path_dict['MATLABpath']), " -nosplash -nodesktop -nodisplay -r ", '"run(', "'{}')".format(TRfile), '; exit;"']
    with open(logFile, "w") as outfile:
        result = run(command, stdout=outfile, stderr=PIPE, universal_newlines=True)             

    # Remove the files once done! 
    os.remove(TRfile)        
    for i in filelist:
        os.remove(testDir+'/'+i)
                
    # Read the LOG file 
    with open(logFile) as f:
        lines = f.read().splitlines()        
    lines = lines[-24:-5]                
    
    # Print the MATLAB unittest results
    print("MATLAB unitests result: \n")
    for l in lines:
        print(l)
    print("Created 'matlabUnitTest.log' file, check for error messages (if any) here!")

    # Report back to pytest 
    report = [string.split()[1] for string in lines[5:]]      
    status = all(elem == report[0] for elem in report)
    assert status == True
               

if __name__ == "__main__":    
    pytest.main([__file__])
