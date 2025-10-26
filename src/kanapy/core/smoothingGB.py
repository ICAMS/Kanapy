# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np


class Node(object):
    """
    Class representing a single node in 3D space with position, velocity, acceleration, and anchors

    Parameters
    ----------
    iden : int
        Identifier to assign to the node
    px : float
        Initial x-coordinate of the node
    py : float
        Initial y-coordinate of the node
    pz : float
        Initial z-coordinate of the node

    Attributes
    ----------
    id : int
        Unique identifier of the node
    px, py, pz : float
        Current x, y, z coordinates of the node
    oripx, oripy, oripz : float
        Original x, y, z coordinates of the node
    vx, vy, vz : float
        Velocity components along x, y, z
    ax, ay, az : float
        Acceleration components along x, y, z
    anchors : list
        List of anchors connected to this node

    Notes
    -----
    This class represents a single node in 3D space, storing both its kinematic state
    (position, velocity, acceleration) and connections to anchors.
    """


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
        """
        Return the current position of the node as a NumPy array

        Returns
        -------
        pos : ndarray
            Array containing the x, y, z coordinates of the node
        """
        return np.array([self.px, self.py, self.pz])

    def get_Oripos(self):
        """
        Return the original position of the node as a NumPy array

        Returns
        -------
        oripos : ndarray
            Array containing the original x, y, z coordinates of the node
        """
        return np.array([self.oripx, self.oripy, self.oripz])                 
    
    def get_vel(self):
        """
        Return the current velocity of the node as a NumPy array

        Returns
        -------
        vel : ndarray
            Array containing the velocity components vx, vy, vz
        """
        return np.array([self.vx, self.vy, self.vz])
        
    def update_pos(self,dt):
        """
        Update the position of the node based on its velocity and time step

        Parameters
        ----------
        dt : float
            Time step for the update
        """
        self.px += self.vx * dt
        self.py += self.vy * dt
        self.pz += self.vz * dt   
        
    def update_vel(self,dt):
        """
        Update the velocity of the node based on its acceleration and time step

        Parameters
        ----------
        dt : float
            Time step for the update
        """
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt   

    def compute_acc(self,fx,fy,fz,mass):
        """
        Compute the acceleration of the node given applied forces and mass

        Parameters
        ----------
        fx : float
            Force component along x
        fy : float
            Force component along y
        fz : float
            Force component along z
        mass : float
            Mass of the node
        """
        self.ax = fx/mass
        self.ay = fy/mass
        self.az = fz/mass       
        
        
""" The function readGrainFaces has some redundancies with
kanapy/input_output/extract_volume_sharedGBarea which is only used in 
kanapy CLI and 
kanapy/api/calcPolygons which offers more functionality, e.g. polygonization
"""
def readGrainFaces(nodes_v,elmtDict,elmtSetDict):
    """
    Extract outer faces of polyhedral grains from voxel connectivity

    Constructs a dictionary of outer faces for each grain by analyzing voxel elements,
    excluding faces that lie on the boundary of the RVE.

    Parameters
    ----------
    nodes_v : ndarray
        Array of nodal coordinates with shape (Nnodes, 3)
    elmtDict : dict
        Dictionary mapping element ID to its 8 node indices
    elmtSetDict : dict
        Dictionary mapping grain ID to a list of element IDs belonging to that grain

    Returns
    -------
    grain_facesDict : dict
        Dictionary mapping grain ID to a dictionary of outer faces.
        Each face is stored as {face_id: [node1, node2, node3, node4]}.
    """
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
    """
    Initialize a spring-mass system from nodes and grain faces

    Creates Node objects for all nodes, generates anchors at the center of each face,
    and links nodes to their associated anchors for the spring-mass system.

    Parameters
    ----------
    nodes_v : ndarray
        Array of nodal coordinates with shape (Nnodes, 3)
    grain_facesDict : dict
        Dictionary mapping grain ID to a dictionary of outer faces, as returned by readGrainFaces

    Returns
    -------
    allNodes : list of Node
        List of Node objects with anchors assigned
    anchDict : dict
        Dictionary mapping face ID to anchor position (x, y, z)
    """
    
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
    """
    Relax the spring-mass system to reach equilibrium

    Integrates the motion of nodes over N time steps with time step dt,
    considering spring forces to anchors and damping. Boundary nodes at RVE surfaces
    are fixed in the corresponding directions.

    Parameters
    ----------
    allNodes : list of Node
        List of Node objects with positions, velocities, and anchors
    anchDict : dict
        Dictionary mapping face ID to anchor position (x, y, z)
    dt : float
        Time step for integration
    N : int
        Number of integration steps
    k : float
        Spring stiffness
    c : float
        Damping coefficient
    RVE_xmin, RVE_xmax : float
        Minimum and maximum x-coordinates of RVE boundary
    RVE_ymin, RVE_ymax : float
        Minimum and maximum y-coordinates of RVE boundary
    RVE_zmin, RVE_zmax : float
        Minimum and maximum z-coordinates of RVE boundary

    Returns
    -------
    allNodes : list of Node
        Updated list of Node objects after relaxation
    """

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
    
    
def smoothingRoutine(nodes_v, elmtDict, elmtSetDict):
    """
    Smooth a voxel-based microstructure by relaxing a spring-mass-anchor system

    Constructs outer faces of grains, initializes nodes with anchors at face centers,
    and iteratively relaxes the system to reduce irregularities in node positions.

    Parameters
    ----------
    nodes_v : ndarray, shape (N_nodes, 3)
        Coordinates of all nodes in the voxel mesh
    elmtDict : dict
        Dictionary mapping element ID to its node indices
    elmtSetDict : dict
        Dictionary mapping grain ID to the set of element IDs it contains

    Returns
    -------
    nodes_s : ndarray, shape (N_nodes, 3)
        Smoothed coordinates of all nodes
    grain_facesDict : dict
        Dictionary mapping grain ID to its outer faces and corresponding node connectivity
    """

    #coords = np.array(list(nodes_v.values()))
    xvals, yvals, zvals = nodes_v[:,0], nodes_v[:,1], nodes_v[:,2]
    RVE_xmin, RVE_xmax = min(xvals), max(xvals)
    RVE_ymin, RVE_ymax = min(yvals), max(yvals)
    RVE_zmin, RVE_zmax = min(zvals), max(zvals)
    
    # Find each grain's outer face ids and it's nodal connectivities
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
            
    nodes_s = np.array([[n.px, n.py, n.pz] for n in allNodes])
               
    return nodes_s, grain_facesDict
