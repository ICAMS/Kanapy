# -*- coding: utf-8 -*-
import os
import sys
import json
import itertools
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

def points_in_convexHull(Points, hull):
    """
    Determines whether the given array of points lie inside the convex hull or outside.        

    :param Points: Array of points to be tested whether they lie inside the hull or not. 
    :type Points: numpy array
    :param hull: Ellipsoid represented by a convex hull created from its outer surface points.  
    :type hull: Scipy's :obj:`ConvexHull` object

    :returns: Boolean values representing the status. If inside: **True**, else **False**
    :rtype: numpy array

    .. seealso:: https://stackoverflow.com/questions/21698630/how-can-i-find-if-a-point-lies-inside-or-outside-of-convexhull
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    return np.all((A @ np.transpose(Points)) <= np.tile(-b, (1, len(Points))), axis=0)


def create_voxels(sim_box, voxNums):
    """
    Generates voxels inside the defined RVE (Simulation box)      

    :param sim_box: Simulation box representing RVE dimensions 
    :type sim_box: :obj:`entities.Cuboid`
    :param voxNums: Number of voxels along the RVE sides X, Y & Z  
    :type voxNums: tuple of int

    :returns: * Node list containing coordinates.
              * Element dictionary containing element IDs and nodal connectivities. 
              * Voxel dictionary containing voxel ID and center coordinates.
    :rtype: Tuple of Python dictionaries.
    """
    print('    Generating voxels inside RVE')
    # generate nodes of all voxels from RVE side dimensions        
    lim_minX, lim_maxX = sim_box.left, sim_box.right
    lim_minY, lim_maxY = sim_box.top, sim_box.bottom
    lim_minZ, lim_maxZ = sim_box.front, sim_box.back   # define the cuboidal RVE limits        
    
    # generate points within these limits
    voxNumX,voxNumY,voxNumZ = voxNums[:]    
    pointsX = np.linspace(lim_minX, lim_maxX, num=voxNumX + 1, endpoint=True)
    pointsY = np.linspace(lim_minY, lim_maxY, num=voxNumY + 1, endpoint=True)
    pointsZ = np.linspace(lim_minZ, lim_maxZ, num=voxNumZ + 1, endpoint=True)
    
    # duplicate neighbouring points
    pointsX_dup = [(first, second) for first, second in zip(pointsX, pointsX[1:])]
    pointsY_dup = [(first, second) for first, second in zip(pointsY, pointsY[1:])]
    pointsZ_dup = [(first, second) for first, second in zip(pointsZ, pointsZ[1:])]

    verticesDict = {}                       # dictionary to store vertices    
    elmtDict = defaultdict(list)            # dictionary to store elements and its node ids    
    vox_centerDict = {}                     # dictionary to store center of each element/voxel
    node_count, elmt_count = 0, 0
    # loop over the duplicate pairs
    for (mk, nk), (mj, nj), (mi, ni) in itertools.product(pointsX_dup, pointsY_dup, pointsZ_dup):

        # Find the center of each voxel and update the center dictionary
        elmt_count += 1
        vox_centerDict[elmt_count] = (0.5*(mi + ni), 0.5*(mj + nj), 0.5*(mk + nk))

        # group the 8 nodes of an element and update node & element dictonary accordingly
        # C3D8 element connectivity is maintained by this list (DON'T change this order)
        vertices = [(ni, mj, nk), (ni, mj, mk), (mi, mj, mk), (mi, mj, nk),
                    (ni, nj, nk), (ni, nj, mk), (mi, nj, mk), (mi, nj, nk)]

        for coo in vertices:
            if coo not in verticesDict.keys():
                node_count += 1
                verticesDict[coo] = node_count
                elmtDict[elmt_count].append(node_count)
            else:
                elmtDict[elmt_count].append(verticesDict[coo])

    # node list
    nodes_v = np.zeros((node_count,3))
    print('### create voxels', node_count, nodes_v.shape)
    for pos, i in verticesDict.items():
        #print('***   ', i, pos)
        nodes_v[i-1,:] = pos
    return nodes_v, elmtDict, vox_centerDict


def assign_voxels_to_ellipsoid(cooDict, Ellipsoids, elmtDict):
    """
    Determines voxels belonging to each ellipsoid    

    :param cooDict: Voxel dictionary containing voxel IDs and center coordinates. 
    :type cooDict: Python dictionary
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list
    :param elmtDict: Element dictionary containing element IDs and nodal connectivities. 
    :type elmtDict: Python dictionary       
    """
    print('    Assigning voxels to grains')

    # ALl the voxel centers as numpy 2D array and voxel ids
    test_ids = np.array(list(cooDict.keys()))
    test_points = np.array(list(cooDict.values()))

    # array defining ellipsoid growth for each stage of while loop
    growth = iter(list(np.arange(1.0, 10.0, 0.1)))    
    remaining_voxels = set(list(cooDict.keys()))
    assigned_voxels = set()
    
    # Initialize a tqdm progress bar to the Number of voxels in the domain
    pbar = tqdm(total = len(remaining_voxels))
    for ellipsoid in Ellipsoids:
        ellipsoid.inside_voxels = []
    while len(remaining_voxels) > 0:
        
        # call the growth value for the ellipsoids
        scale = next(growth)

        for ellipsoid in Ellipsoids:

            # scale ellipsoid dimensions by the growth factor and generate surface points
            ellipsoid.a, ellipsoid.b, ellipsoid.c = scale*ellipsoid.a, scale*ellipsoid.b, scale*ellipsoid.c
            ellipsoid.surfacePointsGen()
            
            # Find the new surface points of the ellipsoid at their final position
            new_surfPts = ellipsoid.surface_points + ellipsoid.get_pos()
            
            # Find the bounding box extremums along x, y, and z
            bbox_xmin, bbox_xmax = np.amin(new_surfPts[:, 0]), np.amax(new_surfPts[:, 0])
            bbox_ymin, bbox_ymax = np.amin(new_surfPts[:, 1]), np.amax(new_surfPts[:, 1])
            bbox_zmin, bbox_zmax = np.amin(new_surfPts[:, 2]), np.amax(new_surfPts[:, 2])
            
            # Find the numpy indices of all voxels within the bounding box                                                    
            in_bbox_idx = np.where((test_points[:, 0] >= bbox_xmin) & (test_points[:, 0] <= bbox_xmax) 
                             & (test_points[:, 1] >= bbox_ymin) & (test_points[:, 1] <= bbox_ymax)
                             & (test_points[:, 2] >= bbox_zmin) & (test_points[:, 2] <= bbox_zmax))[0] 
            
            # extract the actual voxel ids and coordinates from the reduced numpy indices
            bbox_testids = test_ids[in_bbox_idx]        # voxels ids 
            bbox_testPts = test_points[in_bbox_idx]     # coordinates            
            
            # create a convex hull from the surface points
            hull = ConvexHull(new_surfPts, incremental=False)
            
            # check if the extracted points are within the hull
            results = points_in_convexHull(bbox_testPts, hull)
            
            # Extract the voxel ids inside the hull
            inside_ids = list(bbox_testids[results])
            
            # Check if the new found voxels share atlest 4 nodes with ellipsoid nodes
            if scale != 1.0:               
                # Extract single instance of all nodes currently belonging to the ellipsoid
                all_nodes = [elmtDict[i] for i in ellipsoid.inside_voxels]
                merged_nodes = list(itertools.chain(*all_nodes))
                ell_nodes = set(merged_nodes)
                
                # Extract single instance of all nodes currently to be tested
                nids = [elmtDict[vid] for vid in inside_ids]
                m_nids = list(itertools.chain(*nids))
                e_nids = set(m_nids)
                
                while True:
                    # Find the common nodes
                    common_nodes = ell_nodes.intersection(e_nids)
                    
                    # If there are no nodes in the ellipsoid
                    if len(common_nodes) == 0 and len(ell_nodes) == 0:
                        ellipsoid.inside_voxels.extend(inside_ids)
                        assigned_voxels.update(set(inside_ids))
                        break

                    # If there are more than 4 common nodes
                    elif len(common_nodes) >= 4:
                        
                        # Find the voxels that have these common nodes
                        int_assigned = set()
                        for vid in inside_ids:
                            nds = elmtDict[vid]
                           
                            if len(ell_nodes.intersection(nds)) >= 4:                            
                                int_assigned.add(vid)
                            else:
                                continue
                        
                        if len(int_assigned) != 0:
                            # update the ellipsoid instance and assigned set
                            ellipsoid.inside_voxels.extend(list(int_assigned))
                            assigned_voxels.update(int_assigned) 
                            
                            # Remove them and test again
                            for i in int_assigned:
                                inside_ids.remove(i)           # Remove the assigned voxel from the list                                                                        
                                
                                nds = set(elmtDict[i])
                                ell_nodes.update(nds)          # Update the actual ellipsoid node list                        
                                e_nids -= nds                  # update the current node list (testing)
                        else:
                            break
                            
                    # If there are no common nodes
                    else:
                        break
                
                # scale ellipsoid dimensions back to original by the growth factor
                ellipsoid.a, ellipsoid.b, ellipsoid.c = ellipsoid.a/scale, ellipsoid.b/scale, ellipsoid.c/scale   
    
                continue   
                                                
            # If scale == 1.0         
            else:    
                # Each voxel should share atleast 4 nodes with the remaining voxels
                for vid in inside_ids:
                    nds = elmtDict[vid]
        
                    rem_ids = [i for i in inside_ids if i != vid]
                    all_nodes = [elmtDict[i] for i in rem_ids]
                    merged_nodes = list(itertools.chain(*all_nodes))
                    rem_nodes = set(merged_nodes)                    
                    
                    common_nodes = rem_nodes.intersection(nds)
                    if len(common_nodes) >= 4:
                        # update the ellipsoid instance and assigned set
                        ellipsoid.inside_voxels.append(vid)
                        assigned_voxels.add(vid)                                                                        
                
                # scale ellipsoid dimensions back to original by the growth factor
                ellipsoid.a, ellipsoid.b, ellipsoid.c = ellipsoid.a/scale, ellipsoid.b/scale, ellipsoid.c/scale
     
        
        # find the remaining voxels
        remaining_voxels = set(list(cooDict.keys())) - assigned_voxels
        
        # Reassign at the end of each growth cycle
        reassign_shared_voxels(cooDict, Ellipsoids, elmtDict)

        # Update the test_points and ids to remaining voxels
        test_ids = np.array(list(remaining_voxels))        
        test_points = np.array([cooDict[pt_id] for pt_id in test_ids])
        
        # Reset the progress bar to '0' and update it and then refresh the view again
        pbar.reset()
        pbar.update(len(assigned_voxels)) 
        pbar.refresh()
            
    pbar.close()    # Close the progress bar
    return


def reassign_shared_voxels(cooDict, Ellipsoids, elmtDict):
    """
    Assigns shared voxels between ellipsoids to the ellispoid with the closest center.

    :param cooDict: Voxel dictionary containing voxel IDs and center coordinates. 
    :type cooDict: Python dictionary            
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list               
    """
    # Find all combination of ellipsoids to check for shared voxels
    combis = list(itertools.combinations(Ellipsoids, 2))
    
    # Create a dictionary for linking voxels and their containing ellipsoids
    vox_ellDict = defaultdict(set)
    for cb in combis:
        shared_voxels = set(cb[0].inside_voxels).intersection(cb[1].inside_voxels)
        
        if shared_voxels:        
            for vox in shared_voxels:
                vox_ellDict[vox].update(cb)

    assigned_voxel = []
    if len(list(vox_ellDict.keys())) != 0:       
        for vox, ells in vox_ellDict.items():
            # Remove the shared voxel for all the ellipsoids containing it
            for el in ells:
                el.inside_voxels.remove(vox)                    
                
    shared_voxels = set(vox_ellDict.keys())
    while len(shared_voxels) > 0: 
        for vox, ells in vox_ellDict.items():  
            if vox in shared_voxels:
                ells = list(ells) 
                nids = set(elmtDict[vox])                                                
                common_nodes = dict()
                len_common_nodes = list()
                
                for ellipsoid in ells:
                    all_nodes = [elmtDict[i] for i in ellipsoid.inside_voxels]
                    merged_nodes = list(itertools.chain(*all_nodes))
                    ell_nodes = set(merged_nodes)
                    common_nodes[ellipsoid.id] = ell_nodes.intersection(nids)
                    len_common_nodes.append(len(common_nodes[ellipsoid.id])) 
                
                loc_common_nodes_max = [i for i, x in enumerate(len_common_nodes) if x == max(len_common_nodes)]
                
                if np.any(len_common_nodes) and max(len_common_nodes) >= 4:
                    if len(loc_common_nodes_max) == 1:
                        assigned_voxel.append(vox) 
                        shared_voxels.remove(vox)
                        ells[loc_common_nodes_max[0]].inside_voxels.append(vox)
                    else:
                        ells = [ells[i] for i in loc_common_nodes_max]
                        vox_coord = cooDict[vox]                              # Get the voxel position        
                        ells_pos = np.array([el.get_pos() for el in ells])    # Get the ellipsoids positions
                                
                        # Distance b/w points along three axes
                        XDiff = vox_coord[0] - ells_pos[:, 0]  # 'x'-axis
                        YDiff = vox_coord[1] - ells_pos[:, 1]  # 'y'-axis
                        ZDiff = vox_coord[2] - ells_pos[:, 2]  # 'z'-axis
                
                        # Find the distance from the 1st ellipsoid
                        dist = np.sqrt((XDiff**2)+(YDiff**2)+(ZDiff**2))
                        
                        clo_loc = np.where(dist == dist.min())[0]             # closest ellipsoid index        
                        clo_ells = [ells[loc] for loc in clo_loc]             # closest ellipsoids
                        
                        # If '1' closest ellipsoid: assign voxel to it        
                        if len(clo_ells) == 1:
                            clo_ells[0].inside_voxels.append(vox)
                        # Else: Determine the smallest and assign to it
                        else:
                            clo_vol = np.array([ce.get_volume() for ce in clo_ells])    # Determine the volumes
                            small_loc = np.where(clo_vol == clo_vol.min())[0]     # Smallest ellipsoid index
                            #small_ells = [ells[loc] for loc in clo_loc]           # Smallest ellipsoids
                            
                            # assign to the smallest one regardless how many are of the same volume
                            #small_ells[0].inside_voxels.append(vox)
                            clo_ells[small_loc].inside_voxels.append(vox)
                        assigned_voxel.append(vox) 
                        shared_voxels.remove(vox)     
                                       
            # ells = list(ells)                                     # convert to list
            # vox_coord = cooDict[vox]                              # Get the voxel position        
            # ells_pos = np.array([el.get_pos() for el in ells])    # Get the ellipsoids positions
                    
            # # Distance b/w points along three axes
            # XDiff = vox_coord[0] - ells_pos[:, 0]  # 'x'-axis
            # YDiff = vox_coord[1] - ells_pos[:, 1]  # 'y'-axis
            # ZDiff = vox_coord[2] - ells_pos[:, 2]  # 'z'-axis
    
            # # Find the distance from the 1st ellipsoid
            # dist = np.sqrt((XDiff**2)+(YDiff**2)+(ZDiff**2))
       
            # clo_loc = np.where(dist == dist.min())[0]             # closest ellipsoid index        
            # clo_ells = [ells[loc] for loc in clo_loc]             # closest ellipsoids
        
            # # If '1' closest ellipsoid: assign voxel to it        
            # if len(clo_ells) == 1:
            #     clo_ells[0].inside_voxels.append(vox)
            # # Else: Determine the smallest and assign to it
            # else:
            #     #clo_vol = np.array([ce.get_volume() for ce in clo_ells])    # Determine the volumes
            #     #small_loc = np.where(clo_vol == clo_vol.min())[0]     # Smallest ellipsoid index
            #     small_ells = [ells[loc] for loc in clo_loc]           # Smallest ellipsoids
            
            #     # assign to the smallest one regardless how many are of the same volume
            #     small_ells[0].inside_voxels.append(vox)
        
            # assigned_voxel.append(vox)                                                          
    return
    

def voxelizationRoutine(particle_data, RVE_data, Ellipsoids, sim_box, save_files=False, dual_phase=False):
    """
    The main function that controls the voxelization routine using: :meth:`kanapy.input_output.read_dump`, :meth:`create_voxels`
    , :meth:`assign_voxels_to_ellipsoid`, :meth:`reassign_shared_voxels`
    
    .. note:: 1. The RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution 
                 is read by loading the JSON file that is generated by :meth:`kanapy.input_output.read_dump`.
              2. The following dictionaries are written as json files into a folder in the current working directory.

                * Node list containing coordinates.
                * Element dictionary containing element ID and nodal connectivities.
                * Element set dictionary containing element set ID and group of 
                  elements each representing a grain of the RVE.                                 
    """
    print('')
    print('Starting RVE voxelization')
    
    #RVE_sizeX, RVE_sizeY, RVE_sizeZ = RVE_data['RVE_sizeX'], RVE_data['RVE_sizeY'], RVE_data['RVE_sizeZ']
    voxX, voxY, voxZ = RVE_data['Voxel_numberX'], RVE_data['Voxel_numberY'], RVE_data['Voxel_numberZ']        
    #voxel_resX, voxel_resY, voxel_resZ = RVE_data['Voxel_resolutionX'], RVE_data['Voxel_resolutionY'], RVE_data['Voxel_resolutionZ']           

    # create voxels inside the RVE
    nodes_v, elmtDict, vox_centerDict = create_voxels(sim_box, (voxX,voxY,voxZ))              

    # Find the voxels belonging to each grain by growing ellipsoid each time
    assign_voxels_to_ellipsoid(vox_centerDict, Ellipsoids, elmtDict)

    # Create element sets
    elmtSetDict = {}
    for ellipsoid in Ellipsoids:
        if ellipsoid.inside_voxels:
            # If the ellipsoid is a duplicate add the voxels to the original ellipsoid
            if ellipsoid.duplicate is not None:
                iel = int(ellipsoid.duplicate)
                if iel not in elmtSetDict:
                    elmtSetDict[iel] =\
                        [int(iv) for iv in ellipsoid.inside_voxels]
                else:
                    elmtSetDict[iel].extend(
                        [int(iv) for iv in ellipsoid.inside_voxels])
            # Else it is the original ellipsoid
            else:
                elmtSetDict[int(ellipsoid.id)] =\
                    [int(iv) for iv in ellipsoid.inside_voxels]
        else:
            # continue
            # If ellipsoid does'nt contain any voxel inside
            print('        Grain {0} is not voxelized, as particle {0} overlap condition is inadmissable'
                  .format(int(ellipsoid.id)))
            sys.exit(0)
    
    # generate array of voxelized structure
    hh = np.zeros(voxX*voxY*voxZ, dtype=int)
    for ih, il in elmtSetDict.items():
        il = np.array(il) - 1
        hh[il] = ih
    voxels = np.reshape(hh, (voxX,voxY,voxZ), order='F')
    #voxels = np.array(voxels, dtype=int)
    
    hh = np.zeros(voxX*voxY*voxZ, dtype=int)
    voxels_phase = np.reshape(hh, (voxX,voxY,voxZ), order='F')
   
    if dual_phase == True:
        for ih, il in elmtSetDict.items():
            il = np.array(il) - 1
            hh[il] = Ellipsoids[ih-1].phasenum
        voxels_phase = np.reshape(hh, (voxX,voxY,voxZ), order='F')

    print('Completed RVE voxelization')
    print('')
    
    if save_files:
        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Dump the Dictionaries as json files
        with open(json_dir + '/nodes_v.csv', 'w') as f:
            for v in nodes_v:
                f.write('{0}, {1}, {2}\n'.format(v[0], v[1], v[2]))

        with open(json_dir + '/elmtDict.json', 'w') as outfile:
            json.dump(elmtDict, outfile, indent=2)

        with open(json_dir + '/elmtSetDict.json', 'w') as outfile:
            json.dump(elmtSetDict, outfile, indent=2)  
                                                                                   
    return nodes_v, elmtDict, elmtSetDict, vox_centerDict, voxels, voxels_phase
        
