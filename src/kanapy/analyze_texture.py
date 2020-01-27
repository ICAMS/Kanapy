# -*- coding: utf-8 -*-
import os, sys
import json

from kanapy.util import ROOT_DIR    
    
def checkConfiguration():
    ''' Evaluates if Kanapy has been configured for texture analysis.'''
    
    path_file = ROOT_DIR + '/PATHS.json'
    print('')
    
    if not os.path.exists(path_file):        
        print("    Your Kanapy is not configured for texture analysis yet! Please run 'kanapy setuptexture' to do so.\n")
        sys.exit(0)
    else:            
        with open(path_file) as json_file:  
            paths = json.load(json_file)   
        print("    Your Kanapy has been configured for texture analysis!")
        print("        MATLAB executable: {}".format(paths['MATLABpath']))
        print("        MTEX path:         {}".format(paths['MTEXpath']))
        print("        (To update these paths, run: kanapy setuptexture)")
    return paths


def getGrainNumber(wd):    
    ''' Get the grain number'''
    grain_info = wd + '/particle_data.json'         # Kanapy generated particle file
    print('')
    
    # If kanapy's geometry module is executed before: Use it to get the grain number!
    if os.path.exists(grain_info):    
        with open(grain_info) as json_file:  
            particle_data = json.load(json_file)            
        grain_num = particle_data['Number']     
           
        decision = input('    Will generate {0} reduced orientations, continue(yes/no): '.format(grain_num))        
        if decision == 'yes' or decision == 'y' or decision == 'Y' or decision == 'YES':
            oriNum = grain_num
        elif decision == 'no' or decision == 'n' or decision == 'N' or decision == 'NO':
            oriNum = input('    Please provide the number of reduced orientations required (integer): ')
        else:
            print('    Invalid entry!, run: kanapy reducetexture again\n')
            sys.exit(0)                
    # Else ask the user for input.
    elif not os.path.exists(grain_info):        
        oriNum = input('    Please provide the number of reduced orientations required (integer): ')  
        if oriNum=='':
            print('    Invalid entry!, run: kanapy reducetexture again\n')
            sys.exit(0) 
    return oriNum


def getSharedSurfaceArea(wd):    
    ''' Get the shared surface area'''
    ssa_info = wd + '/shared_surfaceArea.csv'       # Kanapy generated file

    # If kanapy's geometry module is executed before: Use it to get the shared area info!
    if os.path.exists(ssa_info):                   
        decision = input("    Found a shared surface area file in the current directory under: '/json_files', continue(yes/no): ")        
        if decision == 'yes' or decision == 'y' or decision == 'Y' or decision == 'YES':
            ssafile = ssa_info
        elif decision == 'no' or decision == 'n' or decision == 'N' or decision == 'NO':
            ssafileName = input("    Please provide the shared surface area file name located in the 'current/working/directory/json_files' directory! (.csv format): ")
            ssafile = wd + '/json_files/' + ssafileName

            if not os.path.exists(ssafile):
                print("    Mentioned file: '{}' does not exist in the current working directory!\n".format(ssafileName))
                sys.exit(0)
            
        else:
            print('    Invalid entry!, run: kanapy reducetexture again\n')
            sys.exit(0)              
              
    # Else ask the user for input.
    elif not os.path.exists(ssa_info):        
        ssafileName = input("    Please provide the shared surface area file name located in the 'current/working/directory/json_files' directory! (.csv format): ")       
        ssafile = wd + '/json_files/' + ssafileName
        
        if not os.path.exists(ssafile):
            print("    Mentioned file: '{}' does not exist in the current working directory!\n".format(ssafileName))
            sys.exit(0)

    return ssafile
                       

def getGrainVolume(wd):    
    ''' Get the grain volume'''
    
    status = input("    During MAD fitting, grain orientations can be weighted based on their volumes. This option required (yes/no): ")           
    if status == 'yes' or status == 'y' or status == 'Y' or status == 'YES':
        volfile = True
    elif status == 'no' or status == 'n' or status == 'N' or status == 'NO':
        volfile = False            
    else:
        print('    Invalid entry!, run: kanapy reducetexture again\n')
        sys.exit(0) 
            
    # If the user requests the grain volumes to be used
    if volfile == True:
        vol_info = wd + '/grainVolumes.csv'       # Kanapy generated file
        
        # If kanapy's geometry module is executed before: Use it to get the grain volume info!
        if os.path.exists(vol_info):                   
            decision = input("    Found a grain volume file in the current directory under: '/json_files', continue(yes/no): ".format(wd))        
            if decision == 'yes' or decision == 'y' or decision == 'Y' or decision == 'YES':
                volfile = vol_info
            elif decision == 'no' or decision == 'n' or decision == 'N' or decision == 'NO':
                volfileName = input("    Please provide the grain volumes file name located in the 'current/working/directory/json_files' directory! (.csv format): ")
                volfile = wd + '/json_files/' + volfileName 

                if not os.path.exists(volfile):
                    print("    Mentioned file: '{}' does not exist in the current working directory!\n".format(volfileName))
                    sys.exit(0)
                                       
            else:
                print('    Invalid entry!, run: kanapy reducetexture again\n')
                sys.exit(0)              
              
        # Else ask the user for input.
        elif not os.path.exists(vol_info):        
            volfileName = input("    Please provide the grain volumes file name located in the 'current/working/directory/json_files' directory! (.csv format): ")       
            volfile = wd + '/json_files/' + volfileName

        if not os.path.exists(volfile):
            print("    Mentioned file: '{}' does not exist in the current working directory!\n".format(volfileName))
            sys.exit(0)
                
        return volfile
    
    # Else: if user requests not to use it
    elif volfile == False:
        return None
    
    
def textureReduction(kdict):
    ''' Calls the MATLAB texture reduction algorithm'''
    
    print('')
    print('Starting texture reduction') 

    path_dict = checkConfiguration()            # Get the MATLAB & MTEX paths    
    
    # Get the number of grains info
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'    
    ori_num = getGrainNumber(json_dir)
    
    # If MDF fitting is requested get the shared surface area and bin info!
    if 'MisAngDist' in kdict.keys():
        nbins = input('    Please provide the number of bins required for MDF fitting (integer, Default=13): ')
        if nbins=='':
            nbins = 13        
        ssa_file = getSharedSurfaceArea(json_dir)        
        
    # If MDF fitting is requested get the grain volume info!
    if 'MisAngDist' in kdict.keys():
        vol_file = getGrainVolume(json_dir)                
        
    # If MDF fitting is requested get the grains info!
    if 'MisAngDist' in kdict.keys():
        if 'grainsMatFile' not in kdict.keys():
            print('')
            user_grainsFileName = input('    Misorientation fitting requires input misorientation info! Please provide the required file in the current directory (.mat): ')
            user_grainsFile = cwd + '/' + user_grainsFileName
            
            if not os.path.exists(user_grainsFile):
                print("    Mentioned file: '{}' does not exist in the current working directory!\n".format(user_grainsFileName))
                sys.exit(0)
            
    # Create a temporary matlab script file that runs Texture reduction algorithm    
    TRfile = ROOT_DIR+'/textureReduction.m'      # Temporary '.m' file
    logFile = cwd + '/kanapyTexture.log'        # Log file.
    
    with open (TRfile,'w') as f:
        f.write("MTEXpath='{}';\n".format(path_dict['MTEXpath']))
        f.write("ori_num={};\n".format(ori_num))
        
        if "ebsdMatFile" in kdict.keys():
            f.write("ebsdFile='{}';\n".format(kdict['ebsdMatFile']))

        if "grainsMatFile" in kdict.keys():
            f.write("grainsFile='{}';\n".format(kdict['grainsMatFile']))        

        if "MisAngDist" in kdict.keys():
            f.write("sharedAreaFile='{}';\n".format(ssa_file))  
            
            if vol_file != None:
                f.write("grainsVolumeFile='{}';\n".format(vol_file))  
                
            if 'grainsMatFile' not in kdict.keys():
                f.write("grainsFile='{}';\n".format(user_grainsFile))  
        

        # Path for writing texture outputs
        f.write("outputPath='{}';\n".format(cwd))
        
        f.write("\n")
        f.write('addpath {0}\n'.format(ROOT_DIR))                        
        
        command = "ODF_reduction_algo(MTEXpath,ori_num,"
        if "ebsdMatFile" in kdict.keys():
            command += "'ebsdMatFile',ebsdFile"
        if "grainsMatFile" in kdict.keys():
            command += ",'grainsMatFile',grainsFile"
        if "kernelShape" in kdict.keys():
            command += ",'kernelShape',{0}".format(kdict['kernelShape'])
        if "MisAngDist" in kdict.keys():
            command += ",'MisAngDist','sharedArea',sharedAreaFile,'nbins',{0}".format(nbins)
            if 'grainsMatFile' not in kdict.keys():
                command += ",'grainsMatFile',grainsFile"
            if vol_file != None:
                command += ",'GrainVolume',grainsVolumeFile"
        
        command += ",'path',outputPath"     # For texture output
        command += ');'
        
        f.write('{0}\n'.format(command))
        f.write('exit;\n')

    # Run from the terminal 
    #os.system('cat {0} | {1} -nosplash -nodesktop -nodisplay -nojvm;'.format(TRfile, path_dict['MATLABpath']))         
    print('')
    print('    Calling Kanapy-MATLAB scripts ...')
    
    cmd1 = "{0} -nosplash -nodesktop -nodisplay -r ".format(path_dict['MATLABpath']) 
    cmd2 = '"run(' + "'{}')".format(TRfile) + '; exit;"'
    cmd = cmd1+cmd2
    
    print('    Coupling with MTEX and reducing.')
    os.system(cmd + '> {}'.format(logFile))   

    print("    Generated output files & placed it in the '/mat_files' folder under the current directory.")
    print("    Wrote Log file ( 'kanapyTexture.log') in the current directory. Check for possible errors and warnings.")
    
    os.remove(TRfile)        # Remove the file once done!
    print('')
    print("Completed texture reduction!\n")
    return


