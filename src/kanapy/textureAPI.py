#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 18:02:45 2021

@author: Biswas, A 
"""

import matlab.engine
import numpy as np
#from kanapy.util import ROOT_DIR    

eng = matlab.engine.start_matlab()
path = '/home/users/biswaa5w/mtex-5.5.2'
#path = ROOT_DIR+'/libs/mtex'
eng.addpath(path,nargout=0)
eng.startup(nargout=0)

class Texture:
    
    def __init__(self):
        self.ori    = None
        self.grain  = None   # grains mat file
        self.ebsd   = None   # ebsd matfile 
        self.kernel = None   # kernel halfwidth in radian float      
                              
    def textureReconstruction(self,ng):        
        
        # check for multiple input
        
        if self.ori is not None and self.ebsd is not None:
            raise ValueError('Multiple inputs for orientations detected give either orientation or ebsd mat file name')
        
        if self.kernel is not None and self.grain is not None:
            raise ValueError('Multiple inputs for kernel detected give either kernel shape value or grain mat file name')
            
                
        if self.ori is None:
            if self.grain is None:                          
                if self.kernel is None:
                    [orired,odfred,ero] = eng.textureReconstruction(ng,'ebsdMatFile',self.ebsd,nargout=3)
                else:
                    [orired,odfred,ero] = eng.textureReconstruction(ng,'ebsdMatFile',self.ebsd,'kernelShape',self.kernel,nargout=3)                
            else:
                [orired,odfred,ero] = eng.textureReconstruction(ng,'ebsdMatFile',self.ebsd,'grainsMatFile',self.grain,nargout=3)                 
        else:
            if self.kernel is None:
                [orired,odfred,ero] = eng.textureReconstruction(ng,'orientation',self.ori,nargout=3)            
            else:          
                [orired,odfred,ero] = eng.textureReconstruction(ng,'orientation',self.ori,'kernelShape',self.kernel,nargout=3)                         
 
        return orired,odfred,ero
    
    def gb_texture(self,ori,nbins,gbArea):
        
        if self.grain is None:
            raise ValueError('Please provide the grains mat file name')
        else:
            
            g = eng.importdata(self.grain)    
            [orilist,ein,eout,mbin] = eng.gb_textureReconstruction(g,ori,matlab.double(gbArea.tolist()),nbins,nargout=4)
            
        return orilist,ein,eout,mbin    
        
               
d = Texture()
d.ebsd  = 'ebsd_316L.mat'
d.grain = 'grains_316L.mat'
#d.grain = 10*np.pi/180.0
a,b,c = d.textureReconstruction(96)

fl = np.genfromtxt('shared_surfaceArea.csv',delimiter=',',skip_header=1)

e,f,g,h = d.gb_texture(a,12,fl)

eng.quit()






       
