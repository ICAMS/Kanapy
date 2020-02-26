# -*- coding: utf-8 -*-
import os
import sys
import json
import itertools
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

from kanapy.input_output import read_dump, write_output_stat


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
    growth = iter(list(np.arange(0.5, 10.0, 0.1)))    
    remaining_voxels = set(list(cooDict.keys()))
    assigned_voxels = set()
    
    # Initialize a tqdm progress bar to the Number of voxels in the domain
    pbar = tqdm(total = len(remaining_voxels))
    
    while True:
        
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
            if scale != 0.5:               
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
                                                
            # If scale == 0.5          
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
        reassign_shared_voxels(cooDict, Ellipsoids)

        # Update the test_points and ids to remaining voxels
        test_ids = np.array(list(remaining_voxels))        
        test_points = np.array([cooDict[pt_id] for pt_id in test_ids])
        
        # Reset the progress bar to '0' and update it and then refresh the view again
        pbar.reset()
        pbar.update(len(assigned_voxels)) 
        pbar.refresh()

        if len(remaining_voxels) == 0:            
            break
            
    pbar.close()    # Close the progress bar
    return


def reassign_shared_voxels(cooDict, Ellipsoids):
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

    if len(list(vox_ellDict.keys())) != 0:       

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

    return


def voxelizationIteration(ts, vox_centerDict, elmtDict, nodeDict):
    ''' Iterates through the timestep defined by voxelization routine'''
    
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files
    
    # Read the required dump file
    filename = cwd + '/dump_files/particle.{0}.dump'.format(ts)            
    sim_box, Ellipsoids = read_dump(filename)

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
                int(ellipsoid.id)))
            sys.exit(0)

    print('Completed RVE voxelization')
    print('')
    
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # Dump the Dictionaries as json files
    with open(json_dir + '/nodeDict.json', 'w') as outfile:
        json.dump(nodeDict, outfile, indent=2)

    with open(json_dir + '/elmtDict.json', 'w') as outfile:
        json.dump(elmtDict, outfile, indent=2)

    with open(json_dir + '/elmtSetDict.json', 'w') as outfile:
        json.dump(elmtSetDict, outfile, indent=2)  
                
    return
    

def voxelizationRoutine():
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
        print('')
        print('Starting RVE voxelization')

        cwd = os.getcwd()
        json_dir = cwd + '/json_files'          # Folder to store the json files

        try:
            with open(json_dir + '/RVE_data.json') as json_file:
                RVE_data = json.load(json_file)

        except FileNotFoundError:
            print('Json file not found, make sure "RVE_data.json" file exists!')
            raise FileNotFoundError
                
        voxel_per_side = RVE_data['Voxel_number_per_side']
        RVE_size = RVE_data['RVE_size']
        voxel_res = RVE_data['Voxel_resolution']          

        # Read one dump file to get simulation box dimensions
        sim_box, notEll = read_dump(cwd + '/dump_files/particle.{0}.dump'.format(0))
        
        # create voxels inside the RVE
        nodeDict, elmtDict, vox_centerDict = create_voxels(sim_box, voxel_per_side)
                          
        # Loop over timesteps 650-800 and find the ideal timestep for which L1-error is the least
        ts_error = {}
        time, err = [], []
        import matplotlib.pyplot as plt
        for ts in np.arange(500,820+10,10):            
            
            # Find the voxels belonging to ellipsoids and dump them to json files
            voxelizationIteration(ts, vox_centerDict, elmtDict, nodeDict)                  
            
            # Write output statistics to a file: 'output_statistics.json'
            write_output_stat()
            
            # Read the particle and grain diameters and compute the L1-error
            try:
                with open(json_dir + '/output_statistics.json') as json_file:
                    output_data = json.load(json_file)    
                  
            except FileNotFoundError:
                print('Json file not found!')
                raise FileNotFoundError
            
            if 'Particle_Major_diameter' in output_data.keys():
                par_majDia = np.asarray(output_data['Particle_Major_diameter']) 
                par_minDia = np.asarray(output_data['Particle_Minor_diameter'])                  
                grain_majDia = np.asarray(output_data['Grain_Major_diameter'])
                grain_minDia = np.asarray(output_data['Grain_Minor_diameter'])
                
                # calculate the optimal histogram bin number
                bins_pmaj = len(knuth_bin_width(par_majDia, return_bins=True, quiet=True)[1])
                bins_pmin = len(knuth_bin_width(par_minDia, return_bins=True, quiet=True)[1])                
                bins_gmaj = len(knuth_bin_width(grain_majDia, return_bins=True, quiet=True)[1])
                bins_gmin = len(knuth_bin_width(grain_minDia, return_bins=True, quiet=True)[1])
   
                # Scale the array between '0 & 1'
                par_majDia = par_majDia/np.amax(par_majDia)
                par_minDia = par_minDia/np.amax(par_minDia)                
                grain_majDia = grain_majDia/np.amax(grain_majDia)
                grain_minDia = grain_minDia/np.amax(grain_minDia)
                
                # Concatenate into Multi-D array
                particles = np.c_[par_majDia, par_minDia]   
                grains = np.c_[grain_majDia, grain_minDia]                  

                # Calculate the multidimensional histogram
                hist_par, edge_par = np.histogramdd(particles, bins=(bins_gmaj, bins_gmin), range=((0,1),(0,1)))
                hist_gr, edge_gr = np.histogramdd(grains, bins=(bins_gmaj, bins_gmin), range=((0,1),(0,1)))
                
                hist_par = hist_par/np.sum(hist_par)
                hist_gr = hist_gr/np.sum(hist_gr)
                
                # Compute the L1-error and update the dictionary
                l1 = np.sum(np.abs(hist_par-hist_gr))
                ts_error[ts] = l1
                print(ts, l1)
                
                time.append(ts)
                err.append(l1)
                plt.plot(ts,l1,'-b')
            else:
                continue           

        plt.show()
        print(ts_error)
        quit()            
        return

    except KeyboardInterrupt:
        sys.exit(0)
        return
        
class _KnuthF:

    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

        # import here rather than globally: scipy is an optional dependency.
        # Note that scipy is imported in the function which calls this,
        # so there shouldn't be any issue importing here.
        from scipy import special

        # create a reference to gammaln to use in self.eval()
        self.gammaln = special.gammaln

    def bins(self, M):
        """Return the bin edges given a width dx"""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function

        Parameters
        ----------
        dx : float
            Width of bins

        Returns
        -------
        F : float
            evaluation of the negative Knuth likelihood function:
            smaller values indicate a better fit.
        """
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M) +
                 self.gammaln(0.5 * M) -
                 M * self.gammaln(0.5) -
                 self.gammaln(self.n + 0.5 * M) +
                 np.sum(self.gammaln(nk + 0.5)))
        

def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using the Freedman-Diaconis rule
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    """
    
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    v25, v75 = np.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) / (n ** (1 / 3))

    if return_bins:
        dmin, dmax = data.min(), data.max()
        Nbins = max(1, np.ceil((dmax - dmin) / dx))
        try:
            bins = dmin + dx * np.arange(Nbins + 1)
        except ValueError as e:
            if 'Maximum allowed size exceeded' in str(e):
                raise ValueError(
                    'The inter-quartile range of the data is too small: '
                    'failed to construct histogram with {} bins. '
                    'Please use another bin method, such as '
                    'bins="scott"'.format(Nbins + 1))
            else:  # Something else  # pragma: no cover
                raise
        return dx, bins
    else:
        return dx

        
def knuth_bin_width(data, return_bins=False, quiet=True):
    r"""Return the optimal histogram bin width using Knuth's rule.

    Knuth's rule is a fixed-width, Bayesian approach to determining
    the optimal bin width of a histogram.

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges
    quiet : bool, optional
        if True (default) then suppress stdout output from scipy.optimize

    Returns
    -------
    dx : float
        optimal bin width. Bins are measured starting at the first data point.
    bins : ndarray
        bin edges: returned if ``return_bins`` is True

    """
    # import here because of optional scipy dependency
    from scipy import optimize

    knuthF = _KnuthF(data)
    dx0, bins0 = freedman_bin_width(data, True)
    M = optimize.fmin(knuthF, len(bins0), disp=not quiet)[0]
    bins = knuthF.bins(M)
    dx = bins[1] - bins[0]

    if return_bins:
        return dx, bins
    else:
        return dx        
