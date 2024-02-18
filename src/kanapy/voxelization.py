# -*- coding: utf-8 -*-
import numpy as np
import itertools
import logging
from tqdm import tqdm
from collections import defaultdict
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

    .. seealso::
       https://stackoverflow.com/questions/21698630/how-can-i-find-if-a-point-lies-inside-or-outside-of-convexhull
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    return np.all((A @ np.transpose(Points)) <= np.tile(-b, (1, len(Points))), axis=0)


def assign_voxels_to_ellipsoid(cooDict, Ellipsoids, voxel_dict, vf_target=None):
    """
    Determines voxels belonging to each ellipsoid    

    :param cooDict: Voxel dictionary containing voxel IDs and center coordinates. 
    :type cooDict: Python dictionary
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list
    :param voxel_dict: Element dictionary containing element IDs and nodal connectivities.
    :type voxel_dict: Python dictionary
    :param vf_target: target value for volume fraction of particles (optional, default: None)
    :type: float
    """
    print('    Assigning voxels to grains')
    if vf_target is None or vf_target > 1.0:
        vf_target = 1.0

    # All the voxel centers as numpy 2D array and voxel ids
    test_ids = np.array(list(cooDict.keys()))
    Nvox = len(test_ids)
    test_points = np.array(list(cooDict.values()))

    # array defining ellipsoid growth for each stage of while loop
    growth = iter(list(np.arange(1.0, 15.0, 0.1)))
    remaining_voxels = set(list(cooDict.keys()))
    assigned_voxels = set()
    vf_cur = 0.

    # Initialize a tqdm progress bar to the Number of voxels in the domain
    pbar = tqdm(total=len(remaining_voxels))
    for ellipsoid in Ellipsoids:
        ellipsoid.inside_voxels = []
    while vf_cur < vf_target:
        # call the growth value for the ellipsoids
        scale = next(growth)

        for ellipsoid in Ellipsoids:
            # scale ellipsoid dimensions by the growth factor and generate surface points
            ellipsoid.a, ellipsoid.b, ellipsoid.c = scale * ellipsoid.a, scale * ellipsoid.b, scale * ellipsoid.c
            ellipsoid.surfacePointsGen()

            # Find the new surface points of the ellipsoid at their final position
            new_surfPts = ellipsoid.surface_points + ellipsoid.get_pos()

            # Find the bounding box extrema along x, y, and z
            bbox_xmin, bbox_xmax = np.amin(new_surfPts[:, 0]), np.amax(new_surfPts[:, 0])
            bbox_ymin, bbox_ymax = np.amin(new_surfPts[:, 1]), np.amax(new_surfPts[:, 1])
            bbox_zmin, bbox_zmax = np.amin(new_surfPts[:, 2]), np.amax(new_surfPts[:, 2])

            # Find the numpy indices of all voxels within the bounding box                                                    
            in_bbox_idx = np.where((test_points[:, 0] >= bbox_xmin) & (test_points[:, 0] <= bbox_xmax)
                                   & (test_points[:, 1] >= bbox_ymin) & (test_points[:, 1] <= bbox_ymax)
                                   & (test_points[:, 2] >= bbox_zmin) & (test_points[:, 2] <= bbox_zmax))[0]

            # extract the actual voxel ids and coordinates from the reduced numpy indices
            bbox_testids = test_ids[in_bbox_idx]  # voxels ids
            bbox_testPts = test_points[in_bbox_idx]  # coordinates

            # create a convex hull from the surface points
            hull = ConvexHull(new_surfPts, incremental=False)

            # check if the extracted points are within the hull
            results = points_in_convexHull(bbox_testPts, hull)

            # Extract the voxel ids inside the hull
            inside_ids = list(bbox_testids[results])

            # Check if the newly found voxels share at least 4 nodes with
            # ellipsoid nodes
            if not np.isclose(scale, 1.0):
                # Extract single instance of all nodes currently belonging
                # to the ellipsoid
                all_nodes = [voxel_dict[i] for i in ellipsoid.inside_voxels]
                merged_nodes = list(itertools.chain(*all_nodes))
                ell_nodes = set(merged_nodes)

                # Extract single instance of all nodes currently to be tested
                nids = [voxel_dict[vid] for vid in inside_ids]
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
                            nds = voxel_dict[vid]

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
                                inside_ids.remove(
                                    i)  # Remove the assigned voxel from the list

                                nds = set(voxel_dict[i])
                                ell_nodes.update(nds)  # Update the actual ellipsoid node list
                                e_nids -= nds  # update the current node list (testing)
                        else:
                            break

                    # If there are no common nodes
                    else:
                        break

                # scale ellipsoid dimensions back to original by the growth factor
                ellipsoid.a, ellipsoid.b, ellipsoid.c = \
                    ellipsoid.a / scale, ellipsoid.b / scale, ellipsoid.c / scale

                continue

                # If scale == 1.0
            else:
                # Each voxel should share at least 4 nodes with the remaining voxels
                for vid in inside_ids:
                    nds = voxel_dict[vid]

                    rem_ids = [i for i in inside_ids if i != vid]
                    all_nodes = [voxel_dict[i] for i in rem_ids]
                    merged_nodes = list(itertools.chain(*all_nodes))
                    rem_nodes = set(merged_nodes)

                    common_nodes = rem_nodes.intersection(nds)
                    if len(common_nodes) >= 4:
                        # update the ellipsoid instance and assigned set
                        ellipsoid.inside_voxels.append(vid)
                        assigned_voxels.add(vid)

        # find the remaining voxels
        remaining_voxels = set(list(cooDict.keys())) - assigned_voxels

        # Reassign at the end of each growth cycle
        reassign_shared_voxels(cooDict, Ellipsoids, voxel_dict)

        # Update the test_points and ids to remaining voxels
        test_ids = np.array(list(remaining_voxels))
        test_points = np.array([cooDict[pt_id] for pt_id in test_ids])

        # Reset the progress bar to '0' and update it and then refresh the view again
        pbar.reset()
        pbar.update(len(assigned_voxels))
        pbar.refresh()

        # Calculate volume fraction of assigned voxels
        vf_cur = len(assigned_voxels) / Nvox

    pbar.close()  # Close the progress bar
    return


def reassign_shared_voxels(cooDict, Ellipsoids, voxel_dict):
    """
    Assigns shared voxels between ellipsoids to the ellipsoid with the closest center.

    :param cooDict: Voxel dictionary containing voxel IDs and center coordinates. 
    :type cooDict: Python dictionary            
    :param Ellipsoids: Ellipsoids from the packing routine.
    :type Ellipsoids: list
    :param voxel_dict: Dictionary of element definitions
    :type voxel_dict: dict
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
    ncyc = 0
    while len(shared_voxels) > 0 and ncyc < 5:
        for vox, ells in vox_ellDict.items():
            if vox in shared_voxels:
                ells = list(ells)
                nids = set(voxel_dict[vox])
                common_nodes = dict()
                len_common_nodes = list()

                for ellipsoid in ells:
                    all_nodes = [voxel_dict[i] for i in ellipsoid.inside_voxels]
                    merged_nodes = list(itertools.chain(*all_nodes))
                    ell_nodes = set(merged_nodes)
                    common_nodes[ellipsoid.id] = ell_nodes.intersection(nids)
                    len_common_nodes.append(len(common_nodes[ellipsoid.id]))

                loc_common_nodes_max = \
                    [i for i, x in enumerate(len_common_nodes)
                     if x == max(len_common_nodes)]

                if np.any(len_common_nodes) and max(len_common_nodes) >= 4:
                    if len(loc_common_nodes_max) == 1:
                        assigned_voxel.append(vox)
                        shared_voxels.remove(vox)
                        ells[loc_common_nodes_max[0]].inside_voxels.append(vox)
                    else:
                        ells = [ells[i] for i in loc_common_nodes_max]
                        vox_coord = cooDict[vox]  # Get the voxel position
                        ells_pos = np.array([el.get_pos() for el in ells])  # Get the ellipsoids positions

                        # Distance b/w points along three axes
                        XDiff = vox_coord[0] - ells_pos[:, 0]  # 'x'-axis
                        YDiff = vox_coord[1] - ells_pos[:, 1]  # 'y'-axis
                        ZDiff = vox_coord[2] - ells_pos[:, 2]  # 'z'-axis

                        # Find the distance from the 1st ellipsoid
                        dist = np.sqrt((XDiff ** 2) + (YDiff ** 2) + (ZDiff ** 2))

                        clo_loc = np.where(dist == dist.min())[0]  # closest ellipsoid index
                        clo_ells = [ells[loc] for loc in clo_loc]  # closest ellipsoids

                        # If '1' closest ellipsoid: assign voxel to it        
                        if len(clo_ells) == 1:
                            clo_ells[0].inside_voxels.append(vox)
                        # Else: Determine the smallest and assign to it
                        else:
                            clo_vol = np.array([ce.get_volume() for ce in clo_ells])  # Determine the volumes
                            small_loc = np.where(clo_vol == clo_vol.min())[0]  # Smallest ellipsoid index
                            # small_ells = [ells[loc] for loc in clo_loc]           # Smallest ellipsoids

                            # assign to the smallest one regardless how many are of the same volume
                            # small_ells[0].inside_voxels.append(vox)
                            clo_ells[small_loc].inside_voxels.append(vox)
                        assigned_voxel.append(vox)
                        shared_voxels.remove(vox)

            """ells = list(ells)                                     # convert to list
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
                #clo_vol = np.array([ce.get_volume() for ce in clo_ells])    # Determine the volumes
                #small_loc = np.where(clo_vol == clo_vol.min())[0]     # Smallest ellipsoid index
                small_ells = [ells[loc] for loc in clo_loc]           # Smallest ellipsoids
            
                # assign to the smallest one regardless how many are of the same volume
                small_ells[0].inside_voxels.append(vox)
        
            assigned_voxel.append(vox)"""
        ncyc += 1
    return


def voxelizationRoutine(Ellipsoids, mesh, nphases, prec_vf=None):
    """
    The main function that controls the voxelization routine using: :meth:`kanapy.input_output.read_dump`,
    :meth:`create_voxels`, :meth:`assign_voxels_to_ellipsoid`, :meth:`reassign_shared_voxels`
    
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

    # Find the voxels belonging to each grain by growing ellipsoid each time
    if prec_vf is None:
        prec_vf = 1.

    assign_voxels_to_ellipsoid(mesh.vox_center_dict, Ellipsoids, mesh.voxel_dict, vf_target=prec_vf)

    # Create element sets
    for ellipsoid in Ellipsoids:
        if ellipsoid.inside_voxels:
            # If the ellipsoid is a duplicate add the voxels to the original ellipsoid
            if ellipsoid.duplicate is not None:
                iel = int(ellipsoid.duplicate)
                if iel not in mesh.grain_dict:
                    mesh.grain_dict[iel] = \
                        [int(iv) for iv in ellipsoid.inside_voxels]
                else:
                    mesh.grain_dict[iel].extend(
                        [int(iv) for iv in ellipsoid.inside_voxels])
            # Else it is the original ellipsoid
            else:
                mesh.grain_dict[int(ellipsoid.id)] = \
                    [int(iv) for iv in ellipsoid.inside_voxels]
        else:
            # If ellipsoid doesn't contain any voxel inside
            # grain should be removed from list!!!
            logging.debug('        Grain {0} is not voxelized, as particle {0} overlap condition is inadmissible'
                          .format(ellipsoid.id))
            # sys.exit(0)

    # generate array of voxelized structure with grain IDs
    # if vf < 1.0, empty voxels will have grain ID 0
    gr_arr = np.zeros(mesh.nvox, dtype=int)
    for igr, vlist in mesh.grain_dict.items():
        vlist = np.array(vlist) - 1
        gr_arr[vlist] = igr
    mesh.grains = np.reshape(gr_arr, mesh.dim, order='F')

    # generate array of voxelized structure with phase numbers
    # and dict of phase numbers for each grain
    # empty voxels will get phase number 1 and be assigned to grain with key 0
    ph_arr = -np.ones(mesh.nvox, dtype=int)
    mesh.grain_phase_dict = dict()
    mesh.ngrains_phase = np.zeros(nphases, dtype=int)
    for igr, vlist in mesh.grain_dict.items():
        vlist = np.array(vlist) - 1
        ip = Ellipsoids[igr - 1].phasenum
        ph_arr[vlist] = ip
        mesh.grain_phase_dict[igr] = ip
        mesh.ngrains_phase[ip] += 1
    ind = np.nonzero(ph_arr < 0.0)[0]
    ph_arr[ind] = 1  # assign phase 1 to empty voxels
    mesh.phases = np.reshape(ph_arr, mesh.dim, order='F')
    vf_cur = 1.0 - len(ind) / mesh.nvox

    print('Completed RVE voxelization')
    if prec_vf is not None and prec_vf < 1.0:
        print('Dispersed phase (precipitates/porosity):')
        print(f'Volume fraction in voxelized grains: {vf_cur}')
        print(f'Target volume fraction = {prec_vf}')
        mesh.prec_vf_voxels = vf_cur
        if 0 in mesh.grain_dict.keys():
            raise ValueError('Grain with key "0" already exists. Should be reserved for matrix phase in structures ' +
            'with precipitates or porosity. Cannot continue with precipitate simulation.')
        mesh.grain_dict[0] = ind
        mesh.grain_phase_dict[0] = 1
        mesh.ngrains_phase[1] += 1
    elif vf_cur < 1.0:
        logging.warning(f'WARNING: {len(ind)} voxels have not been assigned to grains.')
        """Try to assign empty voxels to neighbor grain"""
    print('')

    return mesh
