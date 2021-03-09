# -*- coding: utf-8 -*-
import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np


class Node(object):    

    def __init__(self,iden,px,py,pz):
        self.id = iden
        self.px = px
        self.py = py
        self.pz = pz        
                
        # Store the original position of the node 
        self.oripx = px
        self.oripy = py
        self.oripz = pz   
        
        # Velocity of the node
        self.vx = 0.
        self.vy = 0.
        self.vz = 0.

        # Acceleration of the node
        self.ax = 0.
        self.ay = 0.
        self.az = 0.        
        
        # List anchors connected to the node
        self.anchors = []    

    def get_pos(self):        
        return np.array([self.px, self.py, self.pz])

    def get_Oripos(self):        
        return np.array([self.oripx, self.oripy, self.oripz])                 
    
    def get_vel(self):        
        return np.array([self.vx, self.vy, self.vz])
        
    def update_pos(self,dt):               
        self.px += self.vx * dt
        self.py += self.vy * dt
        self.pz += self.vz * dt   
        
    def update_vel(self,dt):        
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt   

    def compute_acc(self,fx,fy,fz,mass):        
        self.ax = fx/mass
        self.ay = fy/mass
        self.az = fz/mass       
        
        
        
def readGrainFaces(nodeDict,elmtDict,elmtSetDict,RVE_xmin,RVE_xmax, RVE_ymin,RVE_ymax,RVE_zmin, RVE_zmax):

    grain_facesDict = dict()      # {Grain: faces} 
    for gid, elset in elmtSetDict.items():               

        outer_faces = set()          # Stores only outer face IDs
        face_nodeDict = dict()       # {Faces: nodal connectivity} 
        
        nodeConn = [elmtDict[el] for el in elset]    # Nodal connectivity of a voxel

        # For each voxel, re-create its 6 faces
        for nc in nodeConn:
            faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                     [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                     [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]
                        
            # Sort in ascending order
            sorted_faces = [sorted(fc) for fc in faces]  
            
            # create face ids by joining node id's                        
            face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]        

            # Update {Faces: nodal connectivity} dictionary
            for enum, fid in enumerate(face_ids):
                if fid not in face_nodeDict:       
                    face_nodeDict[fid] = faces[enum]                
            
            # Update the set
            for fid in face_ids:        
                if fid not in outer_faces:
                    outer_faces.add(fid)
                else:
                    outer_faces.remove(fid)        
        
        # Update {Grain: faces} dictionary
        grain_facesDict[gid] = dict()    
        for of in outer_faces:
        
            # Don't add faces belonging to RVE surface
            conn = face_nodeDict[of]
            n1 = nodeDict[conn[0]]
            n2 = nodeDict[conn[1]]
            n3 = nodeDict[conn[2]]
            n4 = nodeDict[conn[3]]
            
            if (n1[0] == n2[0] == n3[0] == n4[0] == RVE_xmin):
                continue
            if (n1[0] == n2[0] == n3[0] == n4[0] == RVE_xmax):
                continue
            
            if (n1[1] == n2[1] == n3[1] == n4[1] == RVE_ymin):
                continue
            if (n1[1] == n2[1] == n3[1] == n4[1] == RVE_ymax):
                continue
                
            if (n1[2] == n2[2] == n3[2] == n4[2] == RVE_zmin):
                continue
            if (n1[2] == n2[2] == n3[2] == n4[2] == RVE_zmax):
                continue
                
            grain_facesDict[gid][of] = face_nodeDict[of]    
        
    return grain_facesDict      
        


def initalizeSystem(nodeDict,grain_facesDict):
    
    # Initialize all nodes as masses
    allNodes = [Node(nid,coords[0],coords[1],coords[2]) for nid,coords in nodeDict.items()]  
    
    # Create anchors at each face center    
    print('    Creating anchors for the spring-mass system')
    pbar = tqdm(total = len(grain_facesDict))   # progress bar tqdm
    
    anchDict = {}
    nodeAnch = defaultdict(list)
    fcList = []   
    for gid,ginfo in grain_facesDict.items():
        for fc,conn in ginfo.items():                   
            if fc not in fcList:
                fcList.append(fc)
                n1 = nodeDict[conn[0]]      # A corner node
                n2 = nodeDict[conn[2]]      # it's opposite corner
                ancrpos = ((n1[0]+n2[0])/2, (n1[1]+n2[1])/2, (n1[2]+n2[2])/2)
                anchDict[fc] = ancrpos
                
                nodeAnch[conn[0]].append(fc)
                nodeAnch[conn[1]].append(fc)
                nodeAnch[conn[2]].append(fc)
                nodeAnch[conn[3]].append(fc)
        
        pbar.reset()
        pbar.update(gid) 
        pbar.refresh()
        
    pbar.close()                               # Close progress bar
    # Add the anchor ID to the node class
    for node in allNodes:
        node.anchors.extend(nodeAnch[node.id])

    return allNodes,anchDict


def relaxSystem(allNodes,anchDict,dt,N,k,c,RVE_xmin,RVE_xmax,
                         RVE_ymin,RVE_ymax,RVE_zmin,RVE_zmax):

    print('    Relaxing the system for equilibrium')                 
    nodeMass = 30
    for i in tqdm(range(0, N)):        
        for node in allNodes:
        
            # Force calculations
            fX, fY, fZ = 0.,0.,0.
            for anch in node.anchors:
                anX,anY,anZ = anchDict[anch]
                sfX = -k*(node.px - anX)
                sfY = -k*(node.py - anY)
                sfZ = -k*(node.pz - anZ)
                
                dfX = c*node.vx
                dfY = c*node.vy
                dfZ = c*node.vz
                     
                fX += sfX - dfX
                fY += sfY - dfY 
                fZ += sfZ - dfZ
            
            # Set force components to zeros for RVE boundary nodes
            if (node.oripx == RVE_xmin) or (node.oripx == RVE_xmax):
                fX = 0.0
            if (node.oripy == RVE_ymin) or (node.oripy == RVE_ymax):
                fY = 0.0
            if (node.oripz == RVE_zmin) or (node.oripz == RVE_zmax):
                fZ = 0.0  
            
            # Update node positions
            node.compute_acc(fX,fY,fZ,nodeMass)    
            node.update_vel(dt)
            node.update_pos(dt)
            
    return allNodes
            

def writeOutput(typeGB,fileName,allNodes,nodeDict,grain_facesDict):
    
    with open(fileName, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('** Nodal coordinates scale in mm\n')
        f.write('*HEADING\n')
        f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=PART-1\n')
        f.write('*Node\n')

        if typeGB == 'voxelated':        
            for node in allNodes:
                coords = nodeDict[node.id]        
                f.write('%d, %f, %f, %f\n' % (node.id, coords[0], coords[1], coords[2]))
                
        elif typeGB == 'smoothed':   
            for node in allNodes:        
                f.write('%d, %f, %f, %f\n' % (node.id, node.px, node.py, node.pz))
                
        f.write('*ELEMENT, TYPE=SFM3D4\n')
        fcList = {}
        fcNum = 0
        gr_fcs = defaultdict(list)
        for gid,ginfo in grain_facesDict.items():                
            for fc,conn in ginfo.items():
                if fc not in fcList.keys():
                    fcNum += 1
                    fcList[fc] = fcNum
                    f.write('%d,%d,%d,%d,%d\n'%(fcNum,conn[0],conn[1],conn[2], conn[3]))            
                    gr_fcs[gid].append(fcNum)  
                elif fc in fcList.keys():
                    f.write('%d,%d,%d,%d,%d\n'%(fcList[fc],conn[0],conn[1],conn[2], conn[3]))       
                    gr_fcs[gid].append(fcList[fc])

        for gid,fcs in gr_fcs.items():             
            f.write('*ELSET, ELSET=GRAIN%d_SET\n'% (gid))    
            for enum, el in enumerate(fcs, 1):
                if enum % 16 != 0:
                    if enum == len(fcs):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d, ' % el)
                else:
                    if enum == len(fcs):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d\n' % el)    

        for gid,fcs in gr_fcs.items():    
            f.write('*SURFACE SECTION, ELSET=GRAIN%d_SET\n'% (gid))
                                                                
        f.write('*End Part\n')
        f.write('**\n')
        f.write('**\n')
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=PART-1-1, part=PART-1\n')
        f.write('*End Instance\n')
        f.write('*End Assembly\n')                    
        
    return
    
    
def smoothingRoutine():

    try:
        print('')
        print('Starting Grain boundary smoothing')
            
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'
        
        try:                
            with open(json_dir + '/nodeDict.json') as json_file: 
                nodeDict = {int(k):v for k,v in json.load(json_file).items()}
                    
            with open(json_dir + '/elmtDict.json') as json_file:
                elmtDict = {int(k):v for k,v in json.load(json_file).items()}

            with open(json_dir + '/elmtSetDict.json') as json_file:    
                elmtSetDict = {int(k):v for k,v in json.load(json_file).items()}
            
        except FileNotFoundError:
            print('Json files not found, make sure "nodeDict.json", "elmtDict.json" and "elmtSetDict.json" files exist!')
            raise FileNotFoundError
        
        
        total_grains = len(elmtSetDict)

        coords = np.array(list(nodeDict.values()))
        xvals, yvals, zvals = coords[:,0], coords[:,1], coords[:,2]
        RVE_xmin, RVE_xmax = min(xvals), max(xvals)
        RVE_ymin, RVE_ymax = min(yvals), max(yvals)
        RVE_zmin, RVE_zmax = min(zvals), max(zvals)
        
        # Find each grain's outer face ids and its nodal connectivities
        grain_facesDict = readGrainFaces(nodeDict,elmtDict,elmtSetDict,
                            RVE_xmin, RVE_xmax, RVE_ymin, RVE_ymax,
                            RVE_zmin, RVE_zmax)
        
        # Initialize the spring-mass-anchor system
        allNodes,anchDict = initalizeSystem(nodeDict,grain_facesDict)
        
        # Run the simulation
        dt = 0.2   # time-step
        N = 100    # total steps
        k = 10     # spring constant
        c = 10     # damping constant
        allNodes = relaxSystem(allNodes,anchDict,dt,N,k,c,RVE_xmin,
                               RVE_xmax,RVE_ymin, RVE_ymax,RVE_zmin,RVE_zmax)
                
        # Write out voxelated & smoothed ABAQUS (.inp) files
        print('    Writing voxelated and smoothed GB ABAQUS (.inp) files')    
        
        typeGB = 'voxelated'
        voxName = cwd + '/kanapy_{0}grainsVoxelatedGB.inp'.format(total_grains)
        writeOutput(typeGB,voxName,allNodes,nodeDict,grain_facesDict)
        
        typeGB = 'smoothed'
        smoothName = cwd + '/kanapy_{0}grainsSmoothedGB.inp'.format(total_grains)
        writeOutput(typeGB,smoothName,allNodes,nodeDict,grain_facesDict)    
                
        print('Completed Grain boundary smoothing')
        print('')
        return 
    
    except KeyboardInterrupt:
        sys.exit(0)
        return  

 
