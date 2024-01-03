#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 07:52:11 2021

@author: Golsa Tolooei Eshlaghi, Alexander Hartmaier
@insititution: ICAMS, Ruhr-University Bochum, Germany
"""

import numpy as np
import kanapy as knpy

# define statistcal information about microstructure
Grain_type = 'Equiaxed' 
#Equivalent Diameter:
std = 0.39 
mean = 2.418 
cutoff_min = 8.0 
cutoff_max = 25.0 

#RVE:
sideX = 67.3965
sideY = 67.3965
sideZ = 67.3965
Nx = 51
Ny = 51
Nz = 51

#Simulation:
periodicity = 'True'
output_units = 'mm'

#Phase 1 Volume Fraction: 
VF = 0.5

#store statistical information in Python dictionary
stat_input = {
"Grain type": Grain_type,
"Equivalent diameter": 
{
    "std": std,
    "mean": mean,
    "cutoff_min": cutoff_min,
    "cutoff_max": cutoff_max
},
"RVE": 
{
    "sideX": sideX,
    "sideY": sideY,
    "sideZ": sideZ,
    "Nx": Nx,
    "Ny": Ny,
    "Nz": Nz
},
"Simulation": 
{
    "periodicity": periodicity,
    "output_units": output_units
}
}

# create kanapy.Microstructure instance with statistical information
ms = knpy.Microstructure(stat_input, name='equiaxed')
ms.create_stats()

# follow kanapy workflow to obtain an RVE
ms.create_RVE()
ms.pack()
ms.voxelize()

#assign phases to voxelized microstructure
'''nodes = np.array(list(ms.nodeDict.values()))
elmts = np.array(list(ms.elmtDict.values()))
elmtSetKeys = np.array(list(ms.elmtSetDict.keys()))
elmtSets = np.array(list(ms.elmtSetDict.values()))
nGrains =  ms.particle_data['Number']
nelmt = Nx * Ny * Nz
nnode = nelmt + Nx * Ny + Ny * Nz + Nx * Nz + Nx + Ny + Nz + 1
Voxel_resolutionX = sideX/Nx
Voxel_resolutionY = sideY/Ny
Voxel_resolutionZ = sideZ/Nz
Voxelvolume = Voxel_resolutionX * Voxel_resolutionY * Voxel_resolutionZ
Volume =  nelmt*Voxelvolume
nElmtspergrain = np.zeros(nGrains, dtype = int)
    
for i in range(0,nGrains):
    nElmtspergrain[i] = len(elmtSets[i])
GrainV = nElmtspergrain * Voxelvolume
    
grainPhase = np.zeros((1,nGrains))
VF1 = 0
nphase1 = 0
ii = 0
while (VF1 < VF):
    if abs(VF-VF1) < 0.01:
        break
    rand = np.random.randint(0,nGrains-1)
    ii = ii + 1
    if grainPhase[0,rand] == 0:
        grainPhase[0,rand] = 1
        VF1 = VF1 + GrainV[rand]/Volume
        nphase1 = nphase1 + 1;
        if VF1 - VF > 0.01:
            grainPhase[0,rand] = 0;
            VF1 = VF1 - GrainV[rand]/Volume;
            nphase1 = nphase1 - 1;
                
    
Data = np.zeros((Nx,Ny,Nz))
xyz = np.zeros((3,8))
iii = 0
jjj = 0
kkk = 0
SUM = 0
for gg in range(0,nGrains):
    for ee in range(0,nElmtspergrain[gg]):
        Element = elmtSets[gg][ee]
        ElementNodes = elmts[Element-1,0:8]
        for nn in range(0,8):
            NNN = ElementNodes[nn]
            xyz[:,nn] = nodes[NNN-1,:]
         
        iii = int(min(xyz[0,:])/Voxel_resolutionX)
        jjj = int(min(xyz[1,:])/Voxel_resolutionY)
        kkk = int(min(xyz[2,:])/Voxel_resolutionZ)
        Data[iii,jjj,kkk] = grainPhase[0,gg]  
        SUM = SUM + Data[iii,jjj,kkk]
        
Datafilename = 'two_phase_microstructure_data.txt'
f = open(Datafilename,'w')

for iii in range(0,Nx):
    for jjj in range(0,Ny):
        for kkk in range(0,Nz):
            f.write('%d\t%d\t%d\t%d\n' % (iii,jjj,kkk,Data[iii,jjj,kkk]))'''

pass