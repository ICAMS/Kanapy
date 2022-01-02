# -*- coding: utf-8 -*-
import os
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
        
        
""" The function readGrainFaces has some redundancies with
kanapy/input_output/extract_volume_sharedGBarea which is only used in 
kanapy CLI and 
kanapy/api/calcPolygons which offers more functionality, e.g. polygonization
"""
def readGrainFaces(nodes_v,elmtDict,elmtSetDict):
    RVE_min = np.amin(nodes_v, axis=0)
    RVE_max = np.amax(nodes_v, axis=0)
    grain_facesDict = dict()      # {Grain: faces} 
    for gid, elset in elmtSetDict.items():               

        outer_faces = set()          # Stores only outer face IDs
        face_nodes = dict()       # {Faces: nodal connectivity} 
        
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
                if fid not in face_nodes:       
                    face_nodes[fid] = faces[enum]                
            
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
            conn = face_nodes[of]
            n1 = nodes_v[conn[0]-1,:]
            n2 = nodes_v[conn[1]-1,:]
            n3 = nodes_v[conn[2]-1,:]
            n4 = nodes_v[conn[3]-1,:]
            
            if (n1[0] == n2[0] == n3[0] == n4[0] == RVE_min[0]):
                continue
            if (n1[0] == n2[0] == n3[0] == n4[0] == RVE_max[0]):
                continue
            
            if (n1[1] == n2[1] == n3[1] == n4[1] == RVE_min[1]):
                continue
            if (n1[1] == n2[1] == n3[1] == n4[1] == RVE_max[1]):
                continue
                
            if (n1[2] == n2[2] == n3[2] == n4[2] == RVE_min[2]):
                continue
            if (n1[2] == n2[2] == n3[2] == n4[2] == RVE_max[2]):
                continue
                
            grain_facesDict[gid][of] = face_nodes[of]  
        
    return grain_facesDict
        


def initalizeSystem(nodes_v,grain_facesDict):
    
    # Initialize all nodes as masses
    allNodes = [Node(nid+1,coords[0],coords[1],coords[2]) for nid,coords in enumerate(nodes_v)]  
    
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
                n1 = nodes_v[conn[0]-1,:]      # A corner node
                n2 = nodes_v[conn[2]-1,:]      # its opposite corner
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
    
    
def smoothingRoutine(nodes_v, elmtDict, elmtSetDict, save_files=False):

    #coords = np.array(list(nodes_v.values()))
    xvals, yvals, zvals = nodes_v[:,0], nodes_v[:,1], nodes_v[:,2]
    RVE_xmin, RVE_xmax = min(xvals), max(xvals)
    RVE_ymin, RVE_ymax = min(yvals), max(yvals)
    RVE_zmin, RVE_zmax = min(zvals), max(zvals)
    
    # Find each grain's outer face ids and its nodal connectivities
    grain_facesDict = readGrainFaces(nodes_v,elmtDict,elmtSetDict)
    
    # Initialize the spring-mass-anchor system
    allNodes,anchDict = initalizeSystem(nodes_v,grain_facesDict)
    
    # Run the simulation
    dt = 0.2   # time-step
    N = 100    # total steps
    k = 10     # spring constant
    c = 10     # damping constant
    allNodes = relaxSystem(allNodes,anchDict,dt,N,k,c,RVE_xmin,
                           RVE_xmax,RVE_ymin, RVE_ymax,RVE_zmin,RVE_zmax)
            
    nodes_s = np.array([(n.px, n.py, n.pz) for n in allNodes])
    
    if save_files:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        
        with open(json_dir + '/nodes_s.csv', 'w') as f:
            for v in nodes_s:
                f.write('{0}, {1}, {2}\n'.format(v[0], v[1], v[2]))
            
        with open(json_dir + '/grain_facesDict.json', 'w') as outfile:
            json.dump(grain_facesDict, outfile, indent=2)
               
    return nodes_s, grain_facesDict
