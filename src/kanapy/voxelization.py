# -*- coding: utf-8 -*-
import os
import sys
import json
import itertools
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

from kanapy.input_output import read_dump, printProgressBar


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


def create_voxels(sim_box, voxel_num):
    """
    Generates voxels inside the defined RVE (Simulation box)      

    :param sim_box: Simulation box representing RVE dimensions 
    :type sim_box: :obj:`entities.Cuboid`
    :param voxel_num: Number of voxels along the RVE side length  
    :type voxel_num: int

    :returns: * Node dictionary containing node ID and coordinates.
              * Element dictionary containing element IDs and nodal connectivities. 
              * Voxel dictionary containing voxel ID and center coordinates.
    :rtype: Tuple of Python dictionaries.
    """
    print('    Generating voxels inside RVE')
    # generate nodes of all voxels from RVE side dimensions    
    lim_min, lim_max = sim_box.front, sim_box.back   # define the cubic RVE limits
    
    # generate points within these limits
    points = np.linspace(lim_min, lim_max, num=voxel_num + 1, endpoint=True)
    points_dup = [(first, second) for first, second in zip(points, points[1:])]     # duplicate neighbouring points
    
    verticesDict = {}                       # dictionary to store vertices    
    elmtDict = defaultdict(list)            # dictionary to store elements and its node ids    
    vox_centerDict = {}                     # dictionary to store center of each element/voxel
    node_count, elmt_count = 0, 0
    # loop over the duplicate pairs
    for (mk, nk), (mj, nj), (mi, ni) in itertools.product(points_dup, points_dup, points_dup):

        # Find the center of each voxel and update the center dictionary
        elmt_count += 1
        vox_centerDict[elmt_count] = ((mi + ni) / 2., (mj + nj) / 2., (mk + nk) / 2.)

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

    # node dictionary
    nodeDict = {v: k for k, v in verticesDict.items()}

    return nodeDict, elmtDict, vox_centerDict


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
    growth = iter(list(np.arange(0.6, 10.0, 0.5)))    
    remaining_voxels = set(list(cooDict.keys()))
    assigned_voxels = set()
    while True:

        # Initial call to print 0% progress
        printProgressBar(0, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)

        # call the growth value for the ellipsoids
        scale = next(growth)
        for enum, ellipsoid in enumerate(Ellipsoids):

            # scale ellipsoid dimensions by the growth factor and generate surface points
            ellipsoid.a, ellipsoid.b, ellipsoid.c = scale*ellipsoid.a, scale*ellipsoid.b, scale*ellipsoid.c
            ellipsoid.surfacePointsGen()

            # Find the new surface points of the ellipsoid at their final position
            new_surfPts = ellipsoid.surface_points + ellipsoid.get_pos()

            # create a convex hull from the surface points
            hull = ConvexHull(new_surfPts, incremental=False)

            # check if points are within the hull
            results = points_in_convexHull(test_points, hull)
            
            # Extract the ids of voxels inside the hull
            inside_ids = list(test_ids[results])                        
            
            # Check if the new found voxels share atlest 4 nodes with ellipsoid nodes
            if scale != 0.6:
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
                                inside_ids.remove(i)                   # Remove the assigned voxel from the list                                                                        
                                
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
    
                # Update Progress Bar
                printProgressBar(enum + 1, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)
    
                continue   
                                                
            # If scale == 0.6          
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

            # Update Progress Bar
            printProgressBar(enum + 1, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)        
        
        # find the remaining voxels
        remaining_voxels = set(list(cooDict.keys())) - assigned_voxels
        
        # Reassign at the end of each growth cycle
        reassign_shared_voxels(cooDict, Ellipsoids)

        # Update the test_points and ids to remaining voxels
        test_ids = np.array(list(remaining_voxels))        
        test_points = np.array([cooDict[pt_id] for pt_id in test_ids])
        
        print('    Number of unassigned voxels: ', len(remaining_voxels))

        if len(remaining_voxels) == 0:
            break

    return


def reassign_shared_voxels(cooDict, Ellipsoids):
    """
    Assigns shared voxels between ellipsoids to the ellispoid with the closest center.

    :param cooDict: Voxel dictionary containing voxel IDs and center coordinates. 
    :type cooDict: Python dictionary            
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list               
    """
    print('    Assigning shared voxels between grains to the closest grain')

    # Find all combination of ellipsoids to check for shared voxels
    combis = list(itertools.combinations(Ellipsoids, 2))
    
    # Create a dictionary for linking voxels and their containing ellipsoids
    vox_ellDict = defaultdict(set)
    for cb in combis:
        shared_voxels = set(cb[0].inside_voxels).intersection(cb[1].inside_voxels)
        
        if shared_voxels:        
            for vox in shared_voxels:
                vox_ellDict[vox].update(cb)

    if len(list(vox_ellDict.keys())) != 0:
       
        # Initial call to print 0% progress
        printProgressBar(0, len(vox_ellDict), prefix='    Progress:', suffix='', length=50)

        assigned_voxel = []
        for vox, ells in vox_ellDict.items():
        
            # Remove the shared voxel for all the ellipsoids containing it
            for el in ells:
                el.inside_voxels.remove(vox)                    
                
            ells = list(ells)                                     # convert to list
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
                small_ells = [ells[loc] for loc in clo_loc]           # Smallest ellipsoids
            
                # assign to the smallest one regardless how many are of the same volume
                small_ells[0].inside_voxels.append(vox)
        
            assigned_voxel.append(vox)                                             
        
            # Update Progress Bar
            printProgressBar(len(assigned_voxel), len(vox_ellDict), prefix='    Progress:', suffix='', length=50)                

    return


def voxelizationRoutine(file_num):
    """
    The main function that controls the voxelization routine using: :meth:`src.kanapy.input_output.read_dump`, :meth:`create_voxels`
    , :meth:`assign_voxels_to_ellipsoid`, :meth:`reassign_shared_voxels`

    :param file_num: Simulation time step to be voxelized. 
    :type file_num: int
    
    .. note:: 1. The RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution 
                 is read by loading the JSON file that is generated by :meth:`src.kanapy.input_output.read_dump`.
              2. The following dictionaries are written as json files into a folder in the current working directory.

                * Node dictionary containing node IDs and coordinates.
                * Element dictionary containing element ID and nodal connectivities.
                * Element set dictionary containing element set ID and group of 
                  elements each representing a grain of the RVE.                                 
    """
    try:
        print('\n')
        print('Starting RVE voxelization')

        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files

        try:
            with open(json_dir + '/RVE_data.txt') as json_file:
                RVE_data = json.load(json_file)

        except FileNotFoundError:
            print('Json file not found, make sure "RVE_data.txt" file exists!')
            raise FileNotFoundError
            
        filename = cwd + '/dump_files/particle.{0}.dump'.format(file_num)

        # Read the required dump file
        sim_box, Ellipsoids = read_dump(filename)

        voxel_per_side = RVE_data['Voxel_number_per_side']
        RVE_size = RVE_data['RVE_size']
        voxel_res = RVE_data['Voxel_resolution']

        # create voxels inside the RVE
        nodeDict, elmtDict, vox_centerDict = create_voxels(sim_box, voxel_per_side)

        # Find the voxels belonging to each grain by growing ellipsoid each time
        assign_voxels_to_ellipsoid(vox_centerDict, Ellipsoids, elmtDict)

        # Create element sets
        elmtSetDict = {}
        for ellipsoid in Ellipsoids:
            if ellipsoid.inside_voxels:
                # If the ellipsoid is a duplicate add the voxels to the original ellipsoid
                if ellipsoid.duplicate:
                    if int(ellipsoid.duplicate) not in elmtSetDict:
                        elmtSetDict[int(ellipsoid.duplicate)] = [int(iv) for iv in ellipsoid.inside_voxels]
                    else:
                        elmtSetDict[int(ellipsoid.duplicate)].extend(
                            [int(iv) for iv in ellipsoid.inside_voxels])
                # Else it is the original ellipsoid
                else:
                    elmtSetDict[int(ellipsoid.id)] = [int(iv) for iv in ellipsoid.inside_voxels]
            else:
                continue
                # If ellipsoid does'nt contain any voxel inside
                print('        Grain {0} is not voxelized, as particle {0} overlap condition is inadmissable'.format(
                    enum + 1))
                sys.exit(0)

        print('Completed RVE voxelization')

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Dump the Dictionaries as json files
        with open(json_dir + '/nodeDict.txt', 'w') as outfile:
            json.dump(nodeDict, outfile, indent=2)

        with open(json_dir + '/elmtDict.txt', 'w') as outfile:
            json.dump(elmtDict, outfile, indent=2)

        with open(json_dir + '/elmtSetDict.txt', 'w') as outfile:
            json.dump(elmtSetDict, outfile, indent=2)        
            
        return

    except KeyboardInterrupt:
        sys.exit(0)
        return
