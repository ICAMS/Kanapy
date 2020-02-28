# -*- coding: utf-8 -*-
import os, sys
import re, json
import csv, itertools

import numpy as np
from scipy.special import erfc
from scipy.spatial import ConvexHull 
from scipy.spatial.distance import euclidean

from kanapy.entities import Ellipsoid, Cuboid


def particleStatGenerator(inputFile):
    r"""
    Generates ellipsoid size distribution (Log-normal) based on user-defined statistics

    :param inputFile: User-defined statistics file for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Standard deviation for ellipsoid equivalent diameter (Log-normal distribution)
                * Mean value of ellipsoid equivalent diameter (Log-normal distribution)
                * Minimum and Maximum cut-offs for ellipsoid equivalent diameters 
                * Mean value for aspect ratio
                * Mean value for ellipsoid tilt angles (Normal distribution)
                * Standard deviation for ellipsoid tilt angles (Normal distribution)
                * Side dimension of the RVE 
                * Discretization along the RVE side          

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current 
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as periodicity and output unit scale (:math:`mm` 
                  or :math:`\mu m`) for ABAQUS .inp file.

    """
    print('')
    print('------------------------------------------------------------------------')    
    print('Welcome to KANAPY - A synthetic polycrystalline microstructure generator')
    print('------------------------------------------------------------------------')
    
    print('Generating particle distribution based on user defined statistics')
    
    # Open the user input statistics file and read the data
    try:
                
        with open(inputFile) as json_file:  
            stats_dict = json.load(json_file)    
        
        # Extract grain diameter statistics info from input file 
        sd_lognormal = stats_dict["Equivalent diameter"]["std"]
        mean_lognormal = stats_dict["Equivalent diameter"]["mean"]
        dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
        dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]
        
        # Extract mean grain aspect ratio value info from input file 
        mean_AR = stats_dict["Aspect ratio"]["mean"]

        # Extract grain tilt angle statistics info from input file 
        sigma_Ori = stats_dict["Tilt angle"]["sigma"]
        mean_Ori = stats_dict["Tilt angle"]["mean"]        
        
        # Extract RVE side lengths and voxel numbers info from input file 
        RVEsizeX = stats_dict["RVE"]["sideX"]
        RVEsizeY = stats_dict["RVE"]["sideY"]
        RVEsizeZ = stats_dict["RVE"]["sideZ"]        
        Nx = int(stats_dict["RVE"]["Nx"]) 
        Ny = int(stats_dict["RVE"]["Ny"]) 
        Nz = int(stats_dict["RVE"]["Nz"]) 
        
        # Extract other simulation attrributes from input file 
        nsteps = 1000
        periodicity = str(stats_dict["Simulation"]["periodicity"])       
        output_units = str(stats_dict["Simulation"]["output_units"])                                                                                     
        
        # Raise ValueError if units are not specified as 'mm' or 'um'
        if output_units != 'mm':
            if output_units != 'um':
                raise ValueError('Output units can only be "mm" or "um"!')
            
    except FileNotFoundError:
        print('Input file not found, make sure "stat_input.json" file is present in the working directory!')
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
    erfInput = -(np.log(x_lognormal) - mean_lognormal)/(np.sqrt(2.0)*sd_lognormal)
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
    num = np.divide(K*(RVEsizeX*RVEsizeY*RVEsizeZ), volume_array)    
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
    voxel_sizeX = RVEsizeX / Nx
    voxel_sizeY = RVEsizeY / Ny
    voxel_sizeZ = RVEsizeZ / Nz
    
    # raise value error if voxel sizes along the 3 directions are not equal
    if (voxel_sizeX != voxel_sizeY != voxel_sizeZ):
        print(" ")
        print("    The voxel resolution along (X,Y,Z): ({0:.2f},{1:.2f},{2:.2f}) are not equal!".format(voxel_sizeX,voxel_sizeY, voxel_sizeZ))
        print("    Change the RVE side lengths (OR) the voxel numbers\n")
        sys.exit(0) 
    
    # raise value error in case the grains are not voxelated well
    if voxel_sizeX >= np.amin(minDia) / 3.:
        print(" ")
        print("    Grains will not be voxelated well!")
        print("    Please increase the voxel numbers (OR) decrease the RVE side lengths\n")
        sys.exit(0)                     
   
    print('    Total number of grains        = {}'.format(totalEllipsoids))
    print('    RVE side lengths (X, Y, Z)    = {0}, {1}, {2}'.format(RVEsizeX, RVEsizeY, RVEsizeZ))
    print('    Number of voxels (X, Y, Z)    = {0}, {1}, {2}'.format(Nx, Ny, Nz))
    print('    Voxel resolution (X, Y, Z)    = {0:.2f}, {1:.2f}, {2:.2f}'.format(voxel_sizeX, voxel_sizeY, voxel_sizeZ))
    print('    Total number of voxels (C3D8) = {}\n'.format(Nx*Ny*Nz))
    
    # Create dictionaries to store the data generated
    particle_data = {'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia), 'Major_diameter': list(majDia),
                     'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Tilt angle': list(ori_array)}

    RVE_data = {'RVE_sizeX': RVEsizeX, 'RVE_sizeY': RVEsizeY, 'RVE_sizeZ': RVEsizeZ, 
                'Voxel_numberX': Nx, 'Voxel_numberY': Ny, 'Voxel_numberZ': Nz,
                'Voxel_resolutionX': voxel_sizeX,'Voxel_resolutionY': voxel_sizeY, 'Voxel_resolutionZ': voxel_sizeZ}

    simulation_data = {'Time steps': nsteps, 'Periodicity': periodicity, 'Output units': output_units}

    # Dump the Dictionaries as json files
    cwd = os.getcwd()     
    json_dir = cwd + '/json_files'          # Folder to store the json files

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/particle_data.json', 'w') as outfile:
        json.dump(particle_data, outfile, indent=2)

    with open(json_dir + '/RVE_data.json', 'w') as outfile:
        json.dump(RVE_data, outfile, indent=2)

    with open(json_dir + '/simulation_data.json', 'w') as outfile:
        json.dump(simulation_data, outfile, indent=2)

    return


def particleCreator(inputFile, periodic='True', units="mm"):
    r"""
    Generates ellipsoid particles based on user-defined inputs.

    :param inputFile: User-defined grain informationfile for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Grain major diameter (:math:`\mu m`)
                * Grain minor diameter (:math:`\mu m`)               
                * Grain's major axis tilt angle (degrees) with respect to the +ve X-axis (horizontal axis)                
              
              2. Other user defined inputs: Periodicity & output units format (:math:`mm` or :math:`\mu m`).
                 Default values: periodicity=True & units= :math:`\mu m`.
              
              3. Particle, RVE and simulation data are written as JSON files in a folder in the current 
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as total number of timesteps, periodicity and Output unit scale (:math:`mm` 
                  or :math:`\mu m`) for ABAQUS .inp file.

    """
    print('')
    print('------------------------------------------------------------------------')    
    print('Welcome to KANAPY - A synthetic polycrystalline microstructure generator')
    print('------------------------------------------------------------------------')
    
    print('Generating particles based on user defined grains')
    
    # Open the user input grain file and read the data
    try:
        input_data = np.loadtxt(inputFile, delimiter=',')
    except FileNotFoundError:
        print('Input file not found, make sure {0} file is present in the working directory!'.format(inputFile))
        raise FileNotFoundError
    
    # User defined major, minor axes lengths using: (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b    
    minDia = input_data[:,1]                          # Minor axis length
    majDia = input_data[:,0]                          # Major axis length    
    minDia2 = minDia.copy()                           # Minor2 axis length (assuming spheroid)
    ori_array = input_data[:,2]
    
    # Volume of each ellipsoid       
    volume_array = (4/3)*np.pi*(majDia*minDia*minDia2)*(1/8)
    
    # RVE size: RVE volume = sum(ellipsoidal volume)
    RVEsize = (np.sum(volume_array))**(1/3)
    
    # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain atleast 3 voxels
    voxel_size = 1.1*(np.amin(minDia) / 3.)
    voxel_per_side = int(round(RVEsize / voxel_size))  # Number of voxel/RVE side                
    voxel_size = RVEsize / voxel_per_side              # Re-calculate
                        
    totalEllipsoids = len(majDia)
    print('    Total number of particles = {}'.format(totalEllipsoids))
    print('    RVE side length = {}'.format(RVEsize))
    print('    Voxel resolution = {}'.format(voxel_size))
    print('    Total number of voxels (C3D8) = {}\n'.format(voxel_per_side**3))
        
    # Create dictionaries to store the data generated
    particle_data = {'Number': int(totalEllipsoids), 'Major_diameter': list(majDia),
                     'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Tilt angle': list(ori_array)}

    RVE_data = {'RVE_size': RVEsize, 'Voxel_number_per_side': voxel_per_side,
                'Voxel_resolution': voxel_size}

    simulation_data = {'Time steps': 1000, 'Periodicity': "{}".format(periodic), 'Output units': units}

    # Dump the Dictionaries as json files
    cwd = os.getcwd()     
    json_dir = cwd + '/json_files'          # Folder to store the json files

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(json_dir + '/particle_data.json', 'w') as outfile:
        json.dump(particle_data, outfile, indent=2)

    with open(json_dir + '/RVE_data.json', 'w') as outfile:
        json.dump(RVE_data, outfile, indent=2)

    with open(json_dir + '/simulation_data.json', 'w') as outfile:
        json.dump(simulation_data, outfile, indent=2)

    return
    

def write_dump(Ellipsoids, sim_box, num_particles):
    """
    Writes the (.dump) file, which can be read by visualization software OVITO.  

    :param Ellipsoids: Contains information of ellipsoids such as its position, axes lengths and tilt angles 
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
              * List of ellipsoid objects that represent the grains.
    :rtype: Tuple of python objects (:obj:`Cuboid`, :obj:`Ellipsoid`)
    """
    print('    Reading the .dump file for particle information')

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

    return sim_box, Ellipsoids


def write_position_weights(file_num):
    r"""
    Reads the (.dump) file to extract information and ouputs the position and weight files for tessellation.

    :param file_num: Simulation time step for which position and weights output. 
    :type file_num: int
    
    .. note:: 1. Applicable only to spherical particles.         
              2. The generated 'sphere_positions.txt' and 'sphere_weights.txt' files can be inputted 
                 into NEPER for tessellation and meshing.
              3. The values of positions and weights are written in :math:`\mu m` scale only.
    """
    print('')
    print('Writing position and weights files for NEPER', end="")
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
    print('---->DONE!\n') 
    return


def write_abaqus_inp():
    r"""
    Creates an ABAQUS input file with microstructure morphology information
    in the form of nodes, elements and element sets.

    .. note:: 1. JSON files generated by :meth:`kanapy.voxelization.voxelizationRoutine` are read to generate the ABAQUS (.inp) file.
                 The json files contain:

                 * Node ID and its corresponding coordinates
                 * Element ID with its nodal connectivities
                 * Element sets representing grains (Assembly of elements) 
                 
              2. The nodal coordinates are written out in :math:`mm` or :math:`\mu m` scale, as requested by the user in the input file.
    """
    print('')
    print('Writing ABAQUS (.inp) file', end="")

    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files   
        
    try:
        with open(json_dir + '/simulation_data.json') as json_file:  
            simulation_data = json.load(json_file)     
    
        with open(json_dir + '/nodeDict.json') as json_file:
            nodeDict = json.load(json_file)

        with open(json_dir + '/elmtDict.json') as json_file:
            elmtDict = json.load(json_file)

        with open(json_dir + '/elmtSetDict.json') as json_file:
            elmtSetDict = json.load(json_file)

    except FileNotFoundError:
        print('Json file not found, make sure "voxelizationRoutine()" function is executed first!')
        raise FileNotFoundError

    # Factor used to generate nodal cordinates in 'mm' or 'um' scale
    if simulation_data['Output units'] == 'mm':
        scale = 'mm'
        divideBy = 1000
    elif simulation_data['Output units'] == 'um':
        scale = 'um'
        divideBy = 1
                
    abaqus_file = cwd + '/kanapy_{0}grains.inp'.format(len(elmtSetDict))
    if os.path.exists(abaqus_file):
        os.remove(abaqus_file)                  # remove old file if it exists

    with open(abaqus_file, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('** Nodal coordinates scale in {0}\n'.format(scale))
        f.write('*HEADING\n')
        f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=PART-1\n')
        f.write('*Node\n')

        # Create nodes
        for k, v in nodeDict.items():
            # Write out coordinates in 'mm' or 'um'
            f.write('{0}, {1}, {2}, {3}\n'.format(k, v[0]/divideBy, v[1]/divideBy, v[2]/divideBy))

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
    print('---->DONE!\n') 
    return

        
def write_output_stat():
    r"""
    Evaluates particle- and output RVE grain statistics with respect to Major, Minor & Equivalent diameters for comparison
    and writes them to 'output_statistics.json' file. 

    .. note:: 1. Particle information is read from (.json) file generated by :meth:`kanapy.input_output.particleStatGenerator`.
                 And RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.
                 
              2. The particle and grain diameter values are written in either :math:`mm` or :math:`\mu m` scale, 
                 as requested by the user in the input file.                       
    """ 
    print('') 
    print('Comparing input & output statistics')
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    try:
        with open(json_dir + '/nodeDict.json') as json_file:
            nodeDict = json.load(json_file)

        with open(json_dir + '/elmtDict.json') as json_file:
            elmtDict = json.load(json_file)

        with open(json_dir + '/elmtSetDict.json') as json_file:
            elmtSetDict = json.load(json_file)

        with open(json_dir + '/particle_data.json') as json_file:  
            particle_data = json.load(json_file)
        
        with open(json_dir + '/RVE_data.json') as json_file:  
            RVE_data = json.load(json_file)

        with open(json_dir + '/simulation_data.json') as json_file:  
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

    # Factor used to generate particle and grains diameters in 'mm' or 'um' scale
    if simulation_data['Output units'] == 'mm':
        scale = 'mm'
        divideBy = 1000
    elif simulation_data['Output units'] == 'um':
        scale = 'um'
        divideBy = 1
                                           
    # Check if Equiaxed or elongated particles
    if np.array_equal(par_majDia, par_minDia):          # Equiaxed grains (spherical particles)    
        
        # Find each grain's equivalent diameter
        grain_eqDia = []    
        for k, v in elmtSetDict.items():
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)
        
        print('Writing particle & grain equivalent diameter to files', end="")
            
        # write out the particle and grain equivalent diameters to files            
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        # Compute the L1-error
        kwargs = {'Particles': par_eqDia, 'Grains': grain_eqDia}
        error = l1_error_est(**kwargs)
                
        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)), 
                       'Unit_scale': scale,
                       'L1-error':error,                       
                       'Particle_Equivalent_diameter': par_eqDia, 
                       'Grain_Equivalent_diameter': grain_eqDia}
        
        with open(json_dir + '/output_statistics.json', 'w') as outfile:
            json.dump(output_data, outfile, indent=2)             
    
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
        
        print('Writing particle & grain equivalent, major & minor diameter to files', end="")

        # write out the particle and grain equivalent, major, minor diameters to file            
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        par_majDia = list(np.array(par_majDia)/divideBy)
        grain_majDia = list(np.array(grain_majDia)/divideBy)

        par_minDia = list(np.array(par_minDia)/divideBy)
        grain_minDia = list(np.array(grain_minDia)/divideBy)                
        
        # Compute the L1-error
        kwargs = {'Particles': par_eqDia, 'Grains': grain_eqDia}
        error = l1_error_est(**kwargs)
        
        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)), 
                       'Unit_scale': scale,
                       'L1-error':error,
                       'Particle_Equivalent_diameter': par_eqDia, 
                       'Particle_Major_diameter': par_majDia,
                       'Particle_Minor_diameter': par_minDia,
                       'Grain_Equivalent_diameter': grain_eqDia,
                       'Grain_Major_diameter': grain_majDia,
                       'Grain_Minor_diameter': grain_minDia}
        
        with open(json_dir + '/output_statistics.json', 'w') as outfile:
            json.dump(output_data, outfile, indent=2)                                                                                           
    
    print('---->DONE!')             
    return
    
    
def l1_error_est(**kwargs):
    r"""
    Evaluates the L1-error between the particle- and output RVE grain statistics with respect to Major, Minor & 
    Equivalent diameters. 

    .. note:: 1. Particle information is read from (.json) file generated by :meth:`kanapy.input_output.particleStatGenerator`.
                 And RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.
                 
              2. The L1-error value is written to the 'output_statistics.json' file.
    """ 
        
    print('') 
    print('Computing the L1-error between input and output diameter distributions', end="")
    
    par_eqDia = np.asarray(kwargs['Particles'])
    grain_eqDia = np.asarray(kwargs['Grains'])
    
    # Scale the array between '0 & 1'                
    par_eqDia = par_eqDia/np.amax(par_eqDia)                
    grain_eqDia = grain_eqDia/np.amax(grain_eqDia)
    
    # Calculate the multidimensional histogram
    hist_par, edge_par = np.histogramdd(par_eqDia, bins=10, range=((0,1),))
    hist_gr, edge_gr = np.histogramdd(grain_eqDia, bins=10, range=((0,1),))
    
    # normalize the histogram
    hist_par = hist_par/np.sum(hist_par)
    hist_gr = hist_gr/np.sum(hist_gr)
    
    # Compute the L1-error
    l1_error = np.sum(np.abs(hist_par - hist_gr))    
    return l1_error               
    
    
def extract_volume_sharedGBarea():
    r"""
    Evaluates the grain volume and the grain boundary shared surface area between neighbouring grains 
    and writes them to 'grainVolumes.csv' & 'shared_surfaceArea.csv' files.

    .. note:: 1. RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.
                 
              2. The grain volumes written to the 'grainVolumes.csv' file are sorted in ascending order of grain IDs. And the values are written 
                 in either :math:`mm` or :math:`\mu m` scale, as requested by the user in the input file.
                 
              3. The shared surface area written to the 'shared_surfaceArea.csv' file are in either :math:`mm` or :math:`\mu m` scale, 
                 as requested by the user in the input file.
    """ 
    print('') 
    print('Evaluating grain volumes.')
    print('Evaluating shared Grain Boundary surface area between grains.')
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files
    
    try:
        with open(json_dir + '/nodeDict.json') as json_file:
            nodeDict = json.load(json_file)

        with open(json_dir + '/elmtDict.json') as json_file:
            elmtDict = json.load(json_file)

        with open(json_dir + '/elmtSetDict.json') as json_file:
            elmtSetDict = json.load(json_file)

        with open(json_dir + '/RVE_data.json') as json_file:
            RVE_data = json.load(json_file)
                      
    except FileNotFoundError:
        print('Json file not found, make sure "particleStatGenerator(), packingRoutine(), voxelizationRoutine()" function is executed first!')
        raise FileNotFoundError
                    
    voxel_size = RVE_data['Voxel_resolution']

    grain_vol = {}
    # For each grain find its volume and output it
    for gid, elset in elmtSetDict.items():
        # Convert to  
        gvol = len(elset) * (voxel_size**3)
        grain_vol[gid] = gvol 
        
    # Sort the grain volumes in ascending order of grain IDs
    gv_sorted_keys = sorted(grain_vol, key=grain_vol.get)
    gv_sorted_values = [[grain_vol[gk]] for gk in gv_sorted_keys]            

    print('Writing grain volumes info. to file', end="")
        
    # Write out grain volumes to a file
    with open(json_dir + '/grainVolumes.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(gv_sorted_values)
                
    # For each grain find its outer face ids
    grain_facesDict = dict()
    for gid, elset in elmtSetDict.items():               
        outer_faces = set()    
        nodeConn = [elmtDict[str(el)] for el in elset]        # For each voxel/element get node connectivity
        
        # create the 6 faces of the voxel
        for nc in nodeConn:
            faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                     [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                     [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]
            
            # Sort each list in ascending order
            sorted_faces = [sorted(fc) for fc in faces]     
            
            # create face ids by joining node id's
            face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]
            
            # Update the set to include only the outer face id's
            for fid in face_ids:        
                if fid not in outer_faces:
                    outer_faces.add(fid)
                else:
                    outer_faces.remove(fid)
        
        grain_facesDict[gid] = outer_faces
    
    # Find all combination of grains to check for common area
    combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))

    # Find the shared area
    shared_area = []
    for cb in combis:
        finter = grain_facesDict[cb[0]].intersection(grain_facesDict[cb[1]])    
        if finter:
            sh_area = len(finter) * (voxel_size**2)
            shared_area.append([cb[0], cb[1], sh_area])
        else:
            continue

    print('Writing shared GB surface area info. to file', end="")
    
    # Write out shared grain boundary area to a file
    with open(json_dir + '/shared_surfaceArea.csv', "w", newline="") as f:
        f.write('GrainA, GrainB, SharedArea\n')
        writer = csv.writer(f)
        writer.writerows(shared_area)
    
    print('---->DONE!\n')       
    return    

