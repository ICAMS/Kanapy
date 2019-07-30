# -*- coding: utf-8 -*-
import os
import re
import math
import json

import numpy as np
from scipy.special import erfc
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull 
from scipy.spatial.distance import euclidean

from kanapy.entities import Ellipsoid, Cuboid


def particleStatGenerator(inputFile):
    """
    Generates ellipsoid size distribution (Log-normal) based on user defined statistics

    :param inputFile: User defined statistics file for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Standard deviation for ellipsoid equivalent diameter (Log-normal distribution)
                * Mean value of ellipsoid equivalent diameter (Log-normal distribution)
                * Minimum and Maximum cut offs for ellipsoid equivalent diameters 
                * Mean value for aspect ratio
                * Mean value for ellipsoid tilt angles (Normal distribution)
                * Standard deviation for ellipsoid tilt angles (Normal distribution)
                * Side dimension of the RVE 
                * Discretization along the RVE side          

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current 
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attribute such as total number of timesteps and periodicity.

    """
    print('Generating particle distribution based on user defined statistics')

    # Open the user input statistics file and read the data
    try:
        with open(inputFile, 'r+') as fd:
            lookup = "@ Equivalent diameter"
            lookup2 = "@ Aspect ratio"
            lookup3 = "@ Orientation"
            lookup4 = "@ RVE"
            lookup5 = "@ Simulation"

            for num, lines in enumerate(fd, 1):
                if lookup in lines:
                    content = next(fd).split()
                    sd_lognormal = float(content[2])

                    content = next(fd).split()
                    mean_lognormal = float(content[2])

                    content = next(fd).split()
                    dia_cutoff_min = float(content[2])

                    content = next(fd).split()
                    dia_cutoff_max = float(content[2])

                if lookup2 in lines:
                    content = next(fd).split()
                    mean_AR = float(content[2])

                if lookup3 in lines:
                    content = next(fd).split()
                    sigma_Ori = float(content[2])

                    content = next(fd).split()
                    mean_Ori = float(content[2])

                if lookup4 in lines:
                    content = next(fd).split()
                    RVEsize = float(content[2])

                    content = next(fd).split()
                    voxel_per_side = int(content[2])

                if lookup5 in lines:
                    content = next(fd).split()
                    nsteps = float(content[2])

                    content = next(fd).split()
                    periodicity = str(content[2])

        if type(voxel_per_side) is not int:
            raise ValueError('Number of voxel per RVE side can only take integer values!')

    except FileNotFoundError:
        print('Input file not found, make sure "stat_input.txt" file is present in the working directory!')
        raise FileNotFoundError
        
    # Generate the x-gaussian
    exp_array = np.arange(-10, +10, 0.01)
    x_lognormal = np.exp(exp_array)
    x_lognormal_mean = np.vstack([x_lognormal[1:], x_lognormal[:-1]]).mean(axis=0)
    
    # Mean, variance for normal distribution (For back verification)
    m = np.exp(mean_lognormal + (sd_lognormal**2)/2.0)                            
    v = np.exp((sd_lognormal**2) - 1) * np.exp(2*mean_lognormal*(sd_lognormal**2))

    # From wikipedia page for Log-normal distribution
    # Calculate the CDF using the error function    
    erfInput = -(np.log(x_lognormal) - mean_lognormal)/(math.sqrt(2.0)*sd_lognormal)
    y_CDF = 0.5*erfc(erfInput)

    # Calculate the number fraction
    number_fraction = np.ediff1d(y_CDF)

    # Based on the cutoff specified, get the restricted distribution
    index_array = np.where((x_lognormal_mean >= dia_cutoff_min) & (x_lognormal_mean <= dia_cutoff_max))    
    eq_Dia = x_lognormal_mean[index_array]          # Selected diameters within the cutoff
    
    # corresponding number fractions
    numFra_Dia = number_fraction[index_array]

    # Volume of each ellipsoid
    volume_array = (4/3)*np.pi*(eq_Dia**3)*(1/8)

    # Volume fraction for each ellipsoid
    individualK = np.multiply(numFra_Dia, volume_array)
    K = individualK/np.sum(individualK)

    # Total number of ellipsoids
    num = np.divide(K*(RVEsize**3), volume_array)    
    num = np.rint(num).astype(int)                  # Round to the nearest integer    
    totalEllipsoids = np.sum(num)                   # Total number of ellipsoids

    # Duplicate the diameter values
    eq_Dia = np.repeat(eq_Dia, num)
    
    # Raise value error in case the RVE side length is too small to fit grains inside.
    if len(eq_Dia) == 0:
         raise ValueError('RVE volume too less to fit grains inside, please increase the RVE side length (or) decrease the mean size for diameters!')
    
    # Ellipsoid tilt angles
    ori_array = np.random.normal(mean_Ori, sigma_Ori, totalEllipsoids)

    # Calculate the major, minor axes lengths for pores using: (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b    
    minDia = eq_Dia / (mean_AR)**(1/3)                          # Minor axis length
    majDia = minDia * mean_AR                                   # Major axis length    
    minDia2 = minDia.copy()                                     # Minor2 axis length (assuming spheroid)

    # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain atleast 3 voxels
    voxel_size = RVEsize / voxel_per_side

    # raise value error in case the grains are not voxelated well
    if voxel_size >= np.amin(minDia) / 3.:
        raise ValueError('Grains will not be voxelated well, please increase the number of voxels per RVE side (or) decrease the RVE side length!')

    print('    Total number of particles = ', totalEllipsoids)
    print('    RVE side length = ', RVEsize)
    print('    Voxel resolution = ', voxel_size)
    print('    Total number of hexahedral elements (C3D8) = ', (voxel_per_side)**3)

    # Create dictionaries to store the data generated
    particle_data = {'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia), 'Major_diameter': list(majDia),
                     'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Orientation': list(ori_array)}

    RVE_data = {'RVE_size': RVEsize, 'Voxel_number_per_side': voxel_per_side,
                'Voxel_resolution': voxel_size}

    simulation_data = {'Time steps': nsteps, 'Periodicity': periodicity}

    # Dump the Dictionaries as json files
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/particle_data.txt', 'w') as outfile:
        json.dump(particle_data, outfile, indent=2)

    with open(json_dir + '/RVE_data.txt', 'w') as outfile:
        json.dump(RVE_data, outfile, indent=2)

    with open(json_dir + '/simulation_data.txt', 'w') as outfile:
        json.dump(simulation_data, outfile, indent=2)

    return


def write_dump(Ellipsoids, sim_box, num_particles):
    """
    Writes the (.dump) file which can be read by visualization software OVITO.  

    :param Ellipsoids: Contains information of ellipsoids such as its position, axes lengths and orientation 
    :type Ellipsoids: list    
    :param sim_box: Contains information of the dimensions of the simulation box
    :type sim_box: :obj:`Cuboid`    
    :param num_particles: Total number of ellipsoids in the simulation box 
    :type num_particles: int    

    .. note:: This function writes (.dump) files containing simulation domain and ellipsoid attribute information. 
    """
    cwd = os.getcwd()
    output_dir = cwd + '/dump_files'    # output directory
    dump_outfile = output_dir + '/particle'	    # output dump file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(dump_outfile + ".{0}.dump".format(sim_box.sim_ts), 'w') as f:
        f.write('ITEM: TIMESTEP\n')
        f.write('{0}\n'.format(sim_box.sim_ts))
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write('{0}\n'.format(num_particles))
        f.write('ITEM: BOX BOUNDS ff ff ff\n')
        f.write('{0} {1}\n'.format(sim_box.left, sim_box.right))
        f.write('{0} {1}\n'.format(sim_box.bottom, sim_box.top))
        f.write('{0} {1}\n'.format(sim_box.back, sim_box.front))
        f.write('ITEM: ATOMS id x y z AsphericalShapeX AsphericalShapeY AsphericalShapeZ OrientationX OrientationY OrientationZ OrientationW\n')
        for ell in Ellipsoids:
            qw, qx, qy, qz = ell.quat
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(
                ell.id, ell.x, ell.y, ell.z, ell.a, ell.b, ell.c, qx, qy, qz, qw))


def read_dump(dump_file):
    """
    Reads the (.dump) file to extract information for voxelization (meshing) routine    

    :param dump_file: Contains information of ellipsoids generated in the packing routine.
    :type dump_file: document

    :returns: * Cuboid object that represents the RVE.
              * Ellipsoid object that represents the grains.
              * Coordinates of ellipsoid centers. 
              * Scipy's cKDTree object representing ellipsoid centers.
    :rtype: Tuple of python objects (:obj:`Cuboid`, :obj:`Ellipsoid`, dictionary, :obj:`cKDTree`)
    """
    print('    Reading the .dump file for particle information\n')

    try:
        # Read the Simulation box dimensions
        with open(dump_file, 'r+') as fd:
            lookup = "ITEM: NUMBER OF ATOMS"
            lookup2 = "ITEM: BOX BOUNDS ff ff ff"
            for num, lines in enumerate(fd, 1):
                if lookup in lines:
                    number_particles = int(next(fd))
                    par_line_num = num + 7

                if lookup2 in lines:
                    values = re.findall(r'\S+', next(fd))
                    RVE_min, RVE_max = list(map(float, values))

    except FileNotFoundError:
        print('    .dump file not found, make sure "packingRoutine()" function is executed first!')
        raise FileNotFoundError
        
    # Create an instance of simulation box
    sim_box = Cuboid(RVE_min, RVE_min, RVE_max, RVE_max, RVE_min, RVE_max)

    # Read the particle shape & position information
    # Create instances for ellipsoids & assign values from dump files
    Ellipsoids = []
    ell_centers = []
    # create a dictionary for ellipsoid centers
    ell_centerDict = {}
    with open(dump_file, "r") as f:
        count = 0
        for num, lines in enumerate(f, 1):
            if num >= par_line_num:

                count += 1
                values = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                iden = count                                        # ellipsoid 'id'                
                a, b, c = values[4], values[5], values[6]           # Semi-major length, Semi-minor length 1 & 2
                x, y, z = values[1], values[2], values[3]
                qx, qy, qz, qw = values[7], values[8], values[9], values[10]
                quat = np.array([qw, qx, qy, qz])                
                ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat) # instance of Ellipsoid class

                # Find the original particle if the current is duplicate
                for c in values[0]:
                    if c == '_':
                        split_str = values[0].split("_")
                        original_id = int(split_str[0])
                        ellipsoid.duplicate = original_id
                        break
                    else:
                        continue

                Ellipsoids.append(ellipsoid)

                ell_centers.append((x, y, z))
                ell_centerDict[iden] = ellipsoid
                
        # Create a ckDTree for ellipsoid centers
        ell_centerTree = cKDTree(ell_centers)

    return sim_box, Ellipsoids, ell_centerDict, ell_centerTree


def write_position_weights(file_num):
    """
    Reads the (.dump) file to extract information and ouputs the position and weight files for tessellation.

    :param file_num: Simulation time step for which position and weights output. 
    :type file_num: int
    
    .. note:: 1. Applicable only to spherical particles.         
              2. The generated 'sphere_positions.txt' and 'sphere_weights.txt' files can be inputted 
                 into NEPER for tessellation and meshing.
    """
    print('Writing position and weights files')
    cwd = os.getcwd()
    dump_file = cwd + '/dump_files/particle.{0}.dump'.format(file_num)

    try:
        with open(dump_file, 'r+') as fd:
            lookup = "ITEM: NUMBER OF ATOMS"
            lookup2 = "ITEM: BOX BOUNDS ff ff ff"
            for num, lines in enumerate(fd, 1):
                if lookup in lines:
                    number_particles = int(next(fd))
                    par_line_num = num + 7

                if lookup2 in lines:
                    values = re.findall(r'\S+', next(fd))
                    RVE_min, RVE_max = list(map(float, values))

    except FileNotFoundError:
        print('    .dump file not found, make sure "packingRoutine()" function is executed first!')
        raise FileNotFoundError
        
    par_dict = dict()
    with open(dump_file, "r") as f:
        count = 0
        for num, lines in enumerate(f, 1):
            if num >= par_line_num:

                values = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                if '_' in values[0]:
                    # Duplicates exists (ignore them when writing position and weight files)
                    continue
                else:
                    count += 1
                    iden = count
                    a, b, c = values[4], values[5], values[6]
                    x, y, z = values[1], values[2], values[3]

                    par_dict[iden] = [x, y, z, a]

    with open('sphere_positions.txt', 'w') as fd:
        for key, value in par_dict.items():
            fd.write('{0} {1} {2}\n'.format(value[0], value[1], value[2]))

    with open('sphere_weights.txt', 'w') as fd:
        for key, value in par_dict.items():
            fd.write('{0}\n'.format(value[3]))

    return


def write_abaqus_inp():
    """
    Creates an ABAQUS input file with microstructure morphology information
    in the form of nodes, elements and element sets.

    .. note:: 1. JSON files generated by :meth:`src.kanapy.voxelization.voxelizationRoutine` are read to generate the ABAQUS (.inp) file.
                 The json files contain:

                 * Node ID and its corresponding coordinates
                 * Element ID with its nodal connectivities
                 * Element sets representing grains (Assembly of elements) 
                 
              2. The nodal coordinates are written out in 'mm' scale.
    """
    print('Writing ABAQUS (.inp) file')

    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    try:
        with open(json_dir + '/nodeDict.txt') as json_file:
            nodeDict = json.load(json_file)

        with open(json_dir + '/elmtDict.txt') as json_file:
            elmtDict = json.load(json_file)

        with open(json_dir + '/elmtSetDict.txt') as json_file:
            elmtSetDict = json.load(json_file)

    except FileNotFoundError:
        print('Json file not found, make sure "voxelizationRoutine()" function is executed first!')
        raise FileNotFoundError
        
    abaqus_file = cwd + '/kanapy_{0}grains.inp'.format(len(elmtSetDict))
    if os.path.exists(abaqus_file):
        os.remove(abaqus_file)                  # remove old file if it exists

    with open(abaqus_file, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('*HEADING\n')
        f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=PART-1\n')
        f.write('*Node\n')

        # Create nodes
        for k, v in nodeDict.items():
            # Write out coordinates in 'mm'
            f.write('{0}, {1}, {2}, {3}\n'.format(k, v[0]/1000, v[1]/1000, v[2]/1000))

        # Create Elements
        f.write('*ELEMENT, TYPE=C3D8\n')
        for k, v in elmtDict.items():
            f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))

        # Create element sets
        for k, v in elmtSetDict.items():
            f.write('*ELSET, ELSET=Grain{0}_set\n'.format(k))
            for enum, el in enumerate(v, 1):
                if enum % 16 != 0:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d, ' % el)
                else:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d\n' % el)

        # Create sections
        for k, v in elmtSetDict.items():
            f.write(
                '*Solid Section, elset=Grain{0}_set, material=Grain{1}_mat\n'.format(k, k))
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

        
def write_output_stat():
    """
    Evaluates particle and output RVE grain statistics with respect to Major, Minor & Equivalent diameters for comparison.

    .. note:: Particle information is read from (.json) file generated by :meth:`src.kanapy.input_output.particleStatGenerator`.
              And RVE grain information is read from the (.json) files generated by :meth:`src.kanapy.voxelization.voxelizationRoutine`.
    """  
    print('Comparing input & output statistics')
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    try:
        with open(json_dir + '/nodeDict.txt') as json_file:
            nodeDict = json.load(json_file)

        with open(json_dir + '/elmtDict.txt') as json_file:
            elmtDict = json.load(json_file)

        with open(json_dir + '/elmtSetDict.txt') as json_file:
            elmtSetDict = json.load(json_file)

        with open(json_dir + '/particle_data.txt') as json_file:  
            particle_data = json.load(json_file)
        
        with open(json_dir + '/RVE_data.txt') as json_file:  
            RVE_data = json.load(json_file)

        with open(json_dir + '/simulation_data.txt') as json_file:  
            simulation_data = json.load(json_file)    
          
    except FileNotFoundError:
        print('Json file not found, make sure "particleStatGenerator(), packingRoutine(), voxelizationRoutine()" function is executed first!')
        raise FileNotFoundError
    
    # Extract from dictionaries
    par_eqDia = particle_data['Equivalent_diameter']
    par_majDia = particle_data['Major_diameter']
    par_minDia = particle_data['Minor_diameter1']
    voxel_size = RVE_data['Voxel_resolution']
    RVEsize = RVE_data['RVE_size']
    
    if simulation_data['Periodicity'] == 'True':
        periodic_status = True
    elif simulation_data['Periodicity'] == 'False':
        periodic_status = False
            
    # Check if Equiaxed or elongated particles
    if np.array_equal(par_majDia, par_minDia):          # Equiaxed grains (spherical particles)    
        
        # Find each grain's equivalent diameter
        grain_eqDia = []    
        for k, v in elmtSetDict.items():
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)
        
        print('Writing particle & grain equivalent diameter files')
            
        # write out the particle and grain equivalent diameters to files
        np.savetxt('particle_equiDiameters.txt', par_eqDia)
        np.savetxt('grain_equiDiameters.txt', grain_eqDia)        
    
    else:                                               # Elongated grains (ellipsoidal particles)

        grain_eqDia, grain_majDia, grain_minDia = [], [], []                                                              
        # Find all the nodal coordinates belonging to the grain
        grain_node = {}    
        for k, v in elmtSetDict.items():                   
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)  
            
            # All nodes belonging to grain                                     
            nodeset = set()
            for el in v:
                nodes = elmtDict[str(el)]
                for n in nodes:
                    if n not in nodeset:
                        nodeset.add(n)
            
            # Get the coordinates as an array 
            points = [nodeDict[str(n)] for n in nodeset]
            points = np.asarray(points)                  
            grain_node[k] = points
        
        if periodic_status:                       
            # If periodic, find the grains whose perodic halves have to be shifted
            shiftRight, shiftTop, shiftBack = [], [], [] 
            for key, value in grain_node.items():                             
                
                # Find all nodes on left, Right, Top, Bottom, Front & Back faces
                nodeLS, nodeRS = set(), set()
                nodeTS, nodeBS = set(), set()
                nodeFS, nodeBaS = set(), set()        
                for enum, coord in enumerate(value):        
                        
                    if abs(0.0000 - coord[0]) <= 0.00000001:       # nodes on Left face
                        nodeLS.add(enum)
                    elif abs(RVEsize - coord[0]) <= 0.00000001:    # nodes on Right face
                        nodeRS.add(enum)
                    
                    if abs(0.0000 - coord[1]) <= 0.00000001:       # nodes on Bottom face
                        nodeBS.add(enum)
                    elif abs(RVEsize - coord[1]) <= 0.00000001:    # nodes on Top face
                        nodeTS.add(enum)

                    if abs(0.0000 - coord[2]) <= 0.00000001:       # nodes on Front face
                        nodeFS.add(enum)
                    elif abs(RVEsize - coord[2]) <= 0.00000001:    # nodes on Back face
                        nodeBaS.add(enum)                
                                                                                                                                                                
                if len(nodeLS) != 0 and len(nodeRS) != 0:   # grain is periodic, has faces on both Left & Right sides
                    shiftRight.append(key)                  # left set has to be moved to right side 
                if len(nodeBS) != 0 and len(nodeTS) != 0:   # grain is periodic, has faces on both Top & Bottom sides
                    shiftTop.append(key)                    # bottom set has to be moved to Top side 
                if len(nodeFS) != 0 and len(nodeBaS) != 0:  # grain is periodic, has faces on both Front & Back sides
                    shiftBack.append(key)                   # front set has to be moved to Back side                         
                                  
            # For each grain that has to be shifted, pad along x, y, z respectively
            for grain in shiftRight:
                pts = grain_node[grain]                     
                # Pad the nodes on the left side by RVE x-dimension
                for enum, val in enumerate(pts[:, 0]):
                    if val>=0.0 and val<=RVEsize/2.:
                        pts[enum, 0] += RVEsize

            for grain in shiftBack:
                pts = grain_node[grain]        
                # Pad the nodes on the front side by RVE z-dimension
                for enum, val in enumerate(pts[:, 2]):
                    if val>=0.0 and val<=RVEsize/2.:
                        pts[enum, 2] += RVEsize

            for grain in shiftTop:
                pts = grain_node[grain]        
                # Pad the nodes on the bottom side by RVE y-dimension
                for enum, val in enumerate(pts[:, 1]):
                    if val>=0.0 and val<=RVEsize/2.:
                        pts[enum, 1] += RVEsize                                
                    
        # For periodic & Non-periodic: create the convex hull and find the major & minor diameters
        for grain, points in grain_node.items():   
            
            hull = ConvexHull(points)             
            hull_pts = points[hull.vertices]
            
            # Find the approximate center of the grain using extreme surface points
            xmin, xmax = np.amin(points[:, 0]), np.amax(points[:, 0])
            ymin, ymax = np.amin(points[:, 1]), np.amax(points[:, 1])
            zmin, zmax = np.amin(points[:, 2]), np.amax(points[:, 2])             
            center = np.array([xmin + (xmax-xmin)/2.0, ymin + (ymax-ymin)/2.0, zmin + (zmax-zmin)/2.0])
            
            # Find the euclidean distance to all surface points from the center
            dists = [euclidean(center, pt) for pt in hull_pts]
            a2 = 2.0*np.amax(dists)
            b2 = 2.0*np.amin(dists)

            grain_majDia.append(a2)                 # update the major diameter list
            grain_minDia.append(b2)                 # update the minor diameter list
        
        print('Writing particle & grain equivalent, major & minor diameter files')
        
        # write out the particle and grain equivalent diameters to files
        np.savetxt('particle_equiDiameters.txt', par_eqDia)
        np.savetxt('grain_equiDiameters.txt', grain_eqDia)
        
        # write out the particle and grain equivalent diameters to files
        np.savetxt('particle_majorDiameters.txt', par_majDia)
        np.savetxt('grain_majorDiameters.txt', grain_majDia)
        # write out the particle and grain equivalent diameters to files
        np.savetxt('particle_minorDiameters.txt', par_minDia)
        np.savetxt('grain_minorDiameters.txt', grain_minDia)                        
                         
    return
    
    
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Prints the progress bar in the terminal for a given loop

    :param iteration: current iteration
    :type iteration: int
    :param total: total iterations
    :type total: int
    :param prefix: prefix string
    :type prefix: str
    :param suffix: suffix string
    :type suffix: str
    :param decimals: positive number of decimals in percent complete
    :type decimals: int
    :param length: character length of bar
    :type length: int
    :param fill: bar fill character
    :type fill: str
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
    return
