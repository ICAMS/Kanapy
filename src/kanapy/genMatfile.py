#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: biswaa5w
"""

'''
angles : Euler angles with number of rows= number of grains and three columns
         phi1, Phi, phi2
ialloy : alloy number in the umat, mod_alloys.f 
nsdv : number of state dependant variables default value is 200         
'''

def writeAbaqusMat(ialloy,angles,nsdv=200):
    with open('Material.inp','w') as f:
        f.write('** MATERIALS'+'\n')
        f.write('**'+'\n')
        for i in range(len(angles)):
            f.write('*Material, name=GRAIN'+str(i+1)+'_mat'+'\n')
            f.write('*Depvar'+'\n')
            f.write('    '+str(nsdv)+'\n')
            f.write('*User Material, constants=4'+'\n')
            f.write(str(ialloy)+'.'+','+str(angles[i,0])+','+str(angles[i,1])+','+str(angles[i,2])+'\n') 
    return None




