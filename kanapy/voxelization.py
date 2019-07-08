# -*- coding: utf-8 -*-
import os
import itertools
import sys
import json
from collections import defaultdict
from operator import itemgetter

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from kanapy.input_output import read_dump, printProgressBar


def points_in_convexHull(Points, hull):
    """
    Determines if the given array of points lie inside the convex hull or outside.        

    :param Points: Array of points to be tested if they lie inside the hull or not. 
    :type Points: numpy array
    :param hull: Ellipsoid represented by a convex hull created from its outer surface points.  
    :type hull: Scipy's :obj:`cKDTree` object

    :returns: Boolean values representing the status. If inside: **True**, else **False**
    :rtype: numpy array

    .. seealso:: https://stackoverflow.com/questions/21698630/how-can-i-find-if-a-point-lies-inside-or-outside-of-convexhull
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    return np.all((A @ np.transpose(Points)) <= np.tile(-b, (1, len(Points))), axis=0)


def distance_away(shared, cooDict, ell1, ell2):
    """
    Determines the closest ellipsoid to a given voxel based on ellipsoid's center        

    :param shared: voxel IDs  
    :type shared: numpy array
    :param cooDict: Contains information on the position (x, y, z) of the voxels  
    :type cooDict: python dictionary
    :param ell1: Ellipsoid :math:`i` 
    :type ell1: :obj:`entities.Ellipsoid`
    :param ell2: Ellipsoid :math:`j`
    :type ell2: :obj:`entities.Ellipsoid`

    :returns: Array of 0 and 1, where 0 & 1 represents ellipsoid :math:`i` & :math:`j` respectively
    :rtype: numpy array    

    .. note:: Closest ellipsoid is evaluated based on the distance from the voxel to the ellipsoid centers and not 
               with respect to the ellipsoid surface points. 
    """
    vox_coo = np.array(itemgetter(*(shared))(cooDict),
                       ndmin=2)                 # Get the voxel coordinates as an array

    # Distance b/w points along 
    XDiff = ell1.x - vox_coo[:, 0]  # 'x'-axis
    YDiff = ell1.y - vox_coo[:, 1]  # 'y'-axis
    ZDiff = ell1.z - vox_coo[:, 2]  # 'z'-axis
    
    # Find the distance from the 1st ellipsoid
    dist1 = np.sqrt((XDiff**2)+(YDiff**2)+(ZDiff**2))

    # Distance b/w points along 
    XDiff2 = ell2.x - vox_coo[:, 0]  # 'x'-axis
    YDiff2 = ell2.y - vox_coo[:, 1]  # 'y'-axis
    ZDiff2 = ell2.z - vox_coo[:, 2]  # 'z'-axis
    
    # Find the distance from the 2nd ellipsoid
    dist2 = np.sqrt((XDiff2**2)+(YDiff2**2)+(ZDiff2**2))

    
    ell_dist = np.stack((dist1, dist2), axis=-1)   # stack the 2 arrays    
    ell_dist_min = np.argmin(ell_dist, axis=1)     # closest ellipsoid for each voxel

    # return closest ellipsoid for each voxel
    return ell_dist_min


def create_voxels(sim_box, voxel_num):
    """
    Generates voxels inside the defined RVE (Simulation box)      

    :param sim_box: Simulation box representing RVE dimensions 
    :type sim_box: :obj:`entities.Cuboid`
    :param voxel_num: Number of voxels along the RVE side length  
    :type voxel_num: int

    :returns: * Node dictionary containing node ID and coordinates.
              * Element dictionary containing element Id's and nodal connectivities. 
              * Voxel dictionary containing voxel ID and center coordinates.
              * Scipy's cKDTree object representing voxel centers.
    :rtype: Tuple of python objects (dictionaries, :obj:`cKDTree`)
    """
    print('    Generating voxels inside RVE')
    # generate nodes of all voxels from RVE side dimensions    
    lim_min, lim_max = sim_box.front, sim_box.back   # define the cubic RVE limits
    
    # generate points within these limits
    points = np.linspace(lim_min, lim_max, num=voxel_num + 1, endpoint=True)
    points_dup = [(first, second) for first, second in zip(points, points[1:])]     # duplicate neighbouring points
    
    verticesDict = {}                       # dictionary to store vertices    
    elmtDict = defaultdict(list)            # dictionary to store elements and its node id's    
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

    # Create a cKDTree for fast lookups of voxel center's
    vox_centers = [vox_center for vox_id, vox_center in vox_centerDict.items()]
    vox_centertree = cKDTree(vox_centers)

    # node dictionary
    nodeDict = {v: k for k, v in verticesDict.items()}

    return nodeDict, elmtDict, vox_centerDict, vox_centertree


def assign_voxels_to_ellipsoid(cooTree, cooDict, Ellipsoids, sim_box):
    """
    Determines voxels belonging to each ellipsoid    

    :param cooTree: Scipy's cKDTree object representing voxel centers. 
    :type cooTree: :obj:`cKDTree`
    :param cooDict: Voxel dictionary containing voxel ID's and center coordinates. 
    :type cooDict: Python dictionary
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list
    :param sim_box: Simulation box representing RVE dimensions 
    :type sim_box: :obj:`entities.Cuboid`
    """
    print('    Assigning voxels to grains')

    # array defining ellipsoid growth for each stage of while loop
    growth = iter(list(np.arange(1.0, 5.05, 0.1)))
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

            # coordinates near the ellipsoid center & within the radius 'a'
            neighsIdx = np.asarray(cooTree.query_ball_point(ellipsoid.get_pos(), r=ellipsoid.a))
            
            if len(neighsIdx) == 0:
                continue

            # Find only those voxels which don't belong to the ellipsoid
            actual_neighsIdx = np.asarray(list(set(list(neighsIdx + 1)) - set(ellipsoid.inside_voxels)))

            # If the difference is '0' (No change btween previous and current number of voxels)
            if len(actual_neighsIdx) == 0:
                continue
                
            # create a convex hull from the surface points
            hull = ConvexHull(new_surfPts, incremental=False)

            # Find the indices within the hull
            # points coordinates as array extracted from dictionary
            test_points = np.array(itemgetter(*(actual_neighsIdx))(cooDict), ndmin=2)
            
            # check if points are within the hull
            results = points_in_convexHull(test_points, hull)
            
            # if the results is True, then index is inside
            insideCoos = actual_neighsIdx[results]

            # Assign the voxel only if it is in remaining voxel list (ELSE: its already assigned)
            assign_it = list(set(insideCoos) & remaining_voxels)

            # update the ellipsoid instance
            ellipsoid.inside_voxels.extend(assign_it)
            
            # update the set
            assigned_voxels.update(assign_it)

            # scale ellipsoid dimensions back to original by the growth factor
            ellipsoid.a, ellipsoid.b, ellipsoid.c = ellipsoid.a/scale, ellipsoid.b/scale, ellipsoid.c/scale

            # Update Progress Bar
            printProgressBar(enum + 1, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)

        # find the unassigned voxels
        remaining_voxels = set(list(cooDict.keys())) - assigned_voxels

        if len(remaining_voxels) == 0:
            break

        print('    Number of unassigned voxels : ', len(remaining_voxels))
    return


def reassign_shared_voxels(cooDict, centerTree, centerDict, Ellipsoids, sim_box):
    """
    Assigns shared voxels between ellipsoids to the ellispoid with the closest center using :meth:`distance_away`

    :param cooDict: Voxel dictionary containing voxel ID's and center coordinates. 
    :type cooDict: Python dictionary
    :param centerTree: Scipy's cKDTree object representing ellipsoid's centers. 
    :type centerTree: :obj:`cKDTree`
    :param centerDict: Voxel dictionary containing ellipsoid ID's and center coordinates. 
    :type centerDict: Python dictionary    
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list   
    :param sim_box: Simulation box representing RVE dimensions 
    :type sim_box: :obj:`entities.Cuboid`        
    """
    print('    Assigning shared voxels between grains to the closest grain')

    # Initial call to print 0% progress
    printProgressBar(0, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)

    for enum, ellipsoid in enumerate(Ellipsoids):
        ell_pos = [ellipsoid.x, ellipsoid.y, ellipsoid.z]

        search_radius = (sim_box.right - sim_box.left)
        # Find all the ellipsoid centers close to the current ellipsoid
        neighsIdx = centerTree.query_ball_point(ell_pos, r=search_radius)
        # neighbour ellipsoid objects
        neighbour_ellipsoid = [centerDict[idx + 1] for idx in neighsIdx]

        if ellipsoid in neighbour_ellipsoid:
            neighbour_ellipsoid.remove(ellipsoid)      # to avoid self checking

        # If there are shared voxels, find the ellipsoids sharing them with current ellipsoid
        for n_el in neighbour_ellipsoid:
            shared_voxels = list(
                set(ellipsoid.inside_voxels).intersection(n_el.inside_voxels))

            if shared_voxels:
                # remove the shared voxels from both ellipsoids, so that they can be assigned to the correct one later
                ellipsoid.inside_voxels = [x for x in ellipsoid.inside_voxels if x not in shared_voxels]
                n_el.inside_voxels = [x for x in n_el.inside_voxels if x not in shared_voxels]

                # Find the closest ellipsoid for all voxels
                shared_voxels = np.asarray(shared_voxels)
                closest_ell = distance_away(shared_voxels, cooDict, ellipsoid, n_el)

                # Assign voxel to the closest ellipsoid
                for idx, vox in zip(closest_ell, shared_voxels):
                    if idx == 0:
                        ellipsoid.inside_voxels.append(vox)
                    elif idx == 1:
                        n_el.inside_voxels.append(vox)

        # Update Progress Bar
        printProgressBar(enum + 1, len(Ellipsoids), prefix='    Progress:', suffix='', length=50)

    return


def voxelizationRoutine(file_num):
    """
    The main function that controls the voxelization routine using: :meth:`kanapy.input_output.read_dump`, :meth:`create_voxels`
    , :meth:`assign_voxels_to_ellipsoid`, :meth:`reassign_shared_voxels`

    :param file_num: Simulation time step to be voxelized. 
    :type file_num: int
    
    .. note:: 1. The RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution 
                 is read by loading the JSON file that is generated by :meth:`kanapy.input_output.read_dump`.
              2. The following dictionaries are written as json files into a folder in the current working directory.

                * Node dictionary containing node IDs and coordinates.
                * Element dictionary containing element ID and nodal connectivities.
                * Element set dictionary containing element set ID and group of 
                  elements each representing a grain of the RVE.                                 
    """
    try:
        print('Starting RVE voxelization')

        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files

        try:
            with open(json_dir + '/RVE_data.txt') as json_file:
                RVE_data = json.load(json_file)

        except FileNotFoundError:
            print('Json file not found, make sure "RVE_data.txt" file exists!')

        filename = cwd + '/dump_files/particle.{0}.dump'.format(file_num)

        # Read the required dump file
        sim_box, Ellipsoids, ell_centerDict, ell_centerTree = read_dump(filename)

        voxel_per_side = RVE_data['Voxel_number_per_side']
        RVE_size = RVE_data['RVE_size']
        voxel_res = RVE_data['Voxel_resolution']

        # create voxels inside the RVE
        nodeDict, elmtDict, vox_centerDict, vox_centerTree = create_voxels(sim_box, voxel_per_side)

        # Find the voxels belonging to each grain by growing ellipsoid by 10% each time
        assign_voxels_to_ellipsoid(vox_centerTree, vox_centerDict, Ellipsoids, sim_box)

        # reassign shared voxels between ellipsoids
        reassign_shared_voxels(vox_centerDict, ell_centerTree, ell_centerDict, Ellipsoids, sim_box)

        # Create element sets
        elmtSetDict = {}
        for ellipsoid in Ellipsoids:
            if ellipsoid.inside_voxels:
                # If the ellipsoid is a duplicate add the voxels to the original ellipsoid
                if ellipsoid.duplicate:
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
            json.dump(nodeDict, outfile)

        with open(json_dir + '/elmtDict.txt', 'w') as outfile:
            json.dump(elmtDict, outfile)

        with open(json_dir + '/elmtSetDict.txt', 'w') as outfile:
            json.dump(elmtSetDict, outfile)        
            
        return

    except KeyboardInterrupt:
        sys.exit(0)
        return
