# -*- coding: utf-8 -*-
import os, sys
import re, json
import csv, itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from scipy.spatial import ConvexHull 
from scipy.spatial.distance import euclidean
    
from kanapy.entities import Ellipsoid, Cuboid


def particleStatGenerator(stats_dict, save_files=False):
    r"""
    Generates ellipsoid size distribution (Log-normal) based on user-defined statistics

    :param inputFile: User-defined statistics file for ellipsoid generation.
    :type inputFile: document

    .. note:: 1. Input parameters provided by the user in the input file are:

                * Standard deviation for ellipsoid equivalent diameter (Normal distribution)
                * Mean value of ellipsoid equivalent diameter (Normal distribution)
                * Minimum and Maximum cut-offs for ellipsoid equivalent diameters 
                * Mean value for aspect ratio
                * Mean value for ellipsoid tilt angles (Normal distribution)
                * Standard deviation for ellipsoid tilt angles (Normal distribution)
                * Side dimension of the RVE 
                * Discretization along the RVE sides          

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current 
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as periodicity and output unit scale (:math:`mm` 
                  or :math:`\mu m`) for ABAQUS .inp file.

    """    
    print('Generating particle distribution based on user defined statistics')   

    # Extract grain diameter statistics info from input file 
    sd = stats_dict["Equivalent diameter"]["std"]
    mean = stats_dict["Equivalent diameter"]["mean"]
    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]
    
    #Equivalent diameter statistics
    # NOTE: SCIPY's lognorm takes in sigma & mu of Normal distribution
    # https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma/13837335#13837335
       
    # Compute the Log-normal PDF & CDF.              
    frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
    xaxis = np.linspace(0.1,200,1000)
    ypdf, ycdf = frozen_lognorm.pdf(xaxis), frozen_lognorm.cdf(xaxis)
        
    # Find the location at which CDF > 0.99    
    #cdf_idx = np.where(ycdf > 0.99)[0][0]
    #x_lim = xaxis[cdf_idx]
    x_lim = dia_cutoff_max + 20       

    if stats_dict["Grain type"] == "Equiaxed":         
        sns.set(color_codes=True)                                 
        plt.figure(figsize=(12, 9))      
        ax = plt.subplot(111)  
        plt.ion()
        
        # Plot grain size distribution                                  
        plt.plot(xaxis, ypdf, linestyle='-', linewidth=3.0)              
        ax.fill_between(xaxis, 0, ypdf, alpha=0.3) 
        ax.set_xlim(left=0.0, right=x_lim)    
        ax.set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(labelsize=14)
        ax.axvline(dia_cutoff_min, linestyle='--', linewidth=3.0, label='Minimum cut-off: {}'.format(dia_cutoff_min))
        ax.axvline(dia_cutoff_max, linestyle='-', linewidth=3.0, label='Maximum cut-off: {}'.format(dia_cutoff_max))    
        plt.title("Grain equivalent diameter distribution", fontsize=20)     
        plt.legend(fontsize=16)
        plt.show()
    elif stats_dict["Grain type"] == "Elongated":       
        # Extract mean grain aspect ratio value info from input file 
        sd_AR = stats_dict["Aspect ratio"]["std"]
        mean_AR = stats_dict["Aspect ratio"]["mean"]        
        ar_cutoff_min = stats_dict["Aspect ratio"]["cutoff_min"]
        ar_cutoff_max = stats_dict["Aspect ratio"]["cutoff_max"]  
    
        sns.set(color_codes=True)                                    
        fig, ax = plt.subplots(2, 1, figsize=(9, 9))  
        plt.ion()
        
        # Plot grain size distribution                                  
        ax[0].plot(xaxis, ypdf, linestyle='-', linewidth=3.0)              
        ax[0].fill_between(xaxis, 0, ypdf, alpha=0.3) 
        ax[0].set_xlim(left=0.0, right=x_lim)    
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[0].set_ylabel('Density', fontsize=18)
        ax[0].tick_params(labelsize=14)
        ax[0].axvline(dia_cutoff_min, linestyle='--', linewidth=3.0, label='Minimum cut-off: {}'.format(dia_cutoff_min))
        ax[0].axvline(dia_cutoff_max, linestyle='-', linewidth=3.0, label='Maximum cut-off: {}'.format(dia_cutoff_max))    
        ax[0].legend(fontsize=14) 
        
        # Plot aspect ratio statistics   
        # Compute the Log-normal PDF & CDF.              
        frozen_lognorm = lognorm(s=sd_AR, scale=np.exp(mean_AR))
        xaxis = np.linspace(0.1,20,1000)
        ypdf, ycdf = frozen_lognorm.pdf(xaxis), frozen_lognorm.cdf(xaxis)                                          
        ax[1].plot(xaxis, ypdf, linestyle='-', linewidth=3.0)              
        ax[1].fill_between(xaxis, 0, ypdf, alpha=0.3)    
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)            
        ax[1].axvline(ar_cutoff_min, linestyle='--', linewidth=3.0, label='Minimum cut-off: {}'.format(ar_cutoff_min))
        ax[1].axvline(ar_cutoff_max, linestyle='-', linewidth=3.0, label='Maximum cut-off: {}'.format(ar_cutoff_max))    
        ax[1].legend(fontsize=14)
        plt.show()
    else:
        raise ValueError('Value for "Grain type" must be either "Equiaxed" or "Elongated".')
                   
    if save_files:
        plt.savefig("Input_distribution.png", bbox_inches="tight")  
        plt.draw()
        plt.pause(0.001)
        print(' ') 
        input("    Press [enter] to continue")     
        print("    'Input_distribution.png' is placed in the current working directory\n")
                                         
    return


def RVEcreator(stats_dict, save_files=False):    
    r"""
    Creates an RVE based on user-defined statistics

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
                * Discretization along the RVE sides          

              2. Particle, RVE and simulation data are written as JSON files in a folder in the current 
                 working directory for later access.

                * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
                * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
                * Simulation attributes such as periodicity and output unit scale (:math:`mm` 
                  or :math:`\mu m`) for ABAQUS .inp file.

    """
    print('Creating an RVE based on user defined statistics')
    # Extract grain diameter statistics info from input file 
    sd = stats_dict["Equivalent diameter"]["std"]
    mean = stats_dict["Equivalent diameter"]["mean"]
    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]    
            
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
    if output_units != 'mm' and output_units != 'um':
        raise ValueError('Output units can only be "mm" or "um"!')  
                
    # Compute the Log-normal PDF & CDF.              
    frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
    xaxis = np.linspace(0.1,200,1000)
    ypdf, ycdf = frozen_lognorm.pdf(xaxis), frozen_lognorm.cdf(xaxis)
            
    # Get the mean value for each pair of neighboring points                
    xaxis = np.vstack([xaxis[1:], xaxis[:-1]]).mean(axis=0)
    
    # Based on the cutoff specified, get the restricted distribution
    index_array = np.where((xaxis >= dia_cutoff_min) & (xaxis <= dia_cutoff_max))    
    eq_Dia = xaxis[index_array]          # Selected diameters within the cutoff            
    
    # Compute the number fractions and extract them based on the cut-off
    number_fraction = np.ediff1d(ycdf)
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
         raise ValueError('RVE volume too small to fit grains inside, please increase the RVE side length (or) decrease the mean size for diameters!')     
    
    # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain atleast 3 voxels
    voxel_sizeX = round(RVEsizeX / Nx, 4)
    voxel_sizeY = round(RVEsizeY / Ny, 4)
    voxel_sizeZ = round(RVEsizeZ / Nz, 4)
        
    # raise value error if voxel sizes along the 3 directions are not equal
    dif1 = np.abs(voxel_sizeX-voxel_sizeY)
    dif2 = np.abs(voxel_sizeY-voxel_sizeZ)
    dif3 = np.abs(voxel_sizeZ-voxel_sizeX)

    if (dif1 > 1e-5) or (dif2 > 1e-5) or (dif3 > 1e-5):
        print(" ")
        print("    The voxel resolution along (X,Y,Z): ({0:.4f},{1:.4f},{2:.4f}) are not equal!".format(voxel_sizeX,voxel_sizeY, voxel_sizeZ))
        print("    Change the RVE side lengths (OR) the voxel numbers\n")
        sys.exit(0) 
        
    # raise value error in case the grains are not voxelated well
    if voxel_sizeX >= np.amin(eq_Dia) / 3.:
        print(" ")
        print("    Grains will not be voxelated well!")
        print("    Please increase the voxel numbers (OR) decrease the RVE side lengths\n")
        sys.exit(0)                     
    
    print('    Total number of grains        = {}'.format(int(totalEllipsoids)))
    print('    RVE side lengths (X, Y, Z)    = {0}, {1}, {2}'.format(RVEsizeX, RVEsizeY, RVEsizeZ))
    print('    Number of voxels (X, Y, Z)    = {0}, {1}, {2}'.format(Nx, Ny, Nz))
    print('    Voxel resolution (X, Y, Z)    = {0:.4f}, {1:.4f}, {2:.4f}'.format(voxel_sizeX, voxel_sizeY, voxel_sizeZ))
    print('    Total number of voxels (C3D8) = {}\n'.format(Nx*Ny*Nz))                

    if stats_dict["Grain type"] == "Equiaxed":

        # Create dictionaries to store the data generated
        particle_data = {'Type': stats_dict["Grain type"], 'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia)}
        
    elif stats_dict["Grain type"] == "Elongated":     
        # Extract mean grain aspect ratio value info from dict
        sd_AR = stats_dict["Aspect ratio"]["std"]
        mean_AR = stats_dict["Aspect ratio"]["mean"]        
        ar_cutoff_min = stats_dict["Aspect ratio"]["cutoff_min"]
        ar_cutoff_max = stats_dict["Aspect ratio"]["cutoff_max"]   
        
        # Extract grain tilt angle statistics info from dict
        std_Ori = stats_dict["Tilt angle"]["std"]
        mean_Ori = stats_dict["Tilt angle"]["mean"]  
        ori_cutoff_min = stats_dict["Tilt angle"]["cutoff_min"]
        ori_cutoff_max = stats_dict["Tilt angle"]["cutoff_max"] 
        
        # Tilt angle statistics      
        # Sample from Normal distribution: It takes mean and std of normal distribution                                                     
        tilt_angle = []
        num = int(totalEllipsoids)
        while True:
            tilt = np.random.normal(mean_Ori, std_Ori, num)
            index_array = np.where((tilt >= ori_cutoff_min) & (tilt <= ori_cutoff_max))
            TA = tilt[index_array].tolist()            
            tilt_angle.extend(TA)                                                
            num = int(totalEllipsoids) - len(tilt_angle)            
            if len(tilt_angle) >= int(totalEllipsoids):
                break  
                
        # Aspect ratio statistics    
        # Sample from lognormal distribution: It takes mean and std of the underlying normal distribution                                                     
        finalAR = []
        num = int(totalEllipsoids)
        while True:
            ar = np.random.lognormal(mean_AR, sd_AR, num)  
            index_array = np.where((ar >= ar_cutoff_min) & (ar <= ar_cutoff_max))
            AR = ar[index_array].tolist()            
            finalAR.extend(AR)                                                
            num = int(totalEllipsoids) - len(finalAR)            
            if len(finalAR) >= int(totalEllipsoids):
                break                
        finalAR = np.array(finalAR)

        # Calculate the major, minor axes lengths for pores using: (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b    
        minDia = eq_Dia / (finalAR)**(1/3)                          # Minor axis length
        majDia = minDia * finalAR                                   # Major axis length    
        minDia2 = minDia.copy()                                     # Minor2 axis length (assuming spheroid)                                                       

        # Create dictionaries to store the data generated
        particle_data = {'Type': stats_dict["Grain type"], 'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia), 'Major_diameter': list(majDia),
                         'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Tilt angle': list(tilt_angle)}
    else:
        raise ValueError('The value for "Grain type" must be either "Equiaxed" or "Elongated".')
    
    RVE_data = {'RVE_sizeX': RVEsizeX, 'RVE_sizeY': RVEsizeY, 'RVE_sizeZ': RVEsizeZ, 
                'Voxel_numberX': Nx, 'Voxel_numberY': Ny, 'Voxel_numberZ': Nz,
                'Voxel_resolutionX': voxel_sizeX,'Voxel_resolutionY': voxel_sizeY, 'Voxel_resolutionZ': voxel_sizeZ}

    simulation_data = {'Time steps': nsteps, 'Periodicity': periodicity, 'Output units': output_units}     
    
    if save_files:
        cwd = os.getcwd() 
        json_dir = cwd + '/json_files'          # Folder to store the json files

        # Dump the Dictionaries as json files    
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open(json_dir + '/particle_data.json', 'w') as outfile:
            json.dump(particle_data, outfile, indent=2) 
            
        with open(json_dir + '/RVE_data.json', 'w') as outfile:
            json.dump(RVE_data, outfile, indent=2)

        with open(json_dir + '/simulation_data.json', 'w') as outfile:
            json.dump(simulation_data, outfile, indent=2)


    return particle_data, RVE_data, simulation_data


def particleCreator(inputFile, periodic='True', units="mm", output=False):
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
    majDia = input_data[:,0]                          # Major axis length  
    minDia = input_data[:,1]                          # Minor axis length  
    minDia2 = minDia.copy()                           # Minor2 axis length (assuming spheroid)
    tilt_angle = input_data[:,2]                      # Tilt angle 
    
    # Volume of each ellipsoid       
    volume_array = (4/3)*np.pi*(majDia*minDia*minDia2)*(1/8)
    
    # Equivalent diameter of each ellipsoid
    eq_Dia = (majDia*minDia*minDia2)**(1/3)
    
    # RVE size: RVE volume = sum(ellipsoidal volume)
    RVEvol = (np.sum(volume_array))

    # Determine the RVE side lengths
    dia_max = np.amax(majDia)    
    RVEsizeY = 1.1*dia_max                 # The Y-side length should accomodate the Biggest dimension of the biggest ellipsoid
    RVEsizeX = round((RVEvol/ RVEsizeY)**0.5, 4)
    RVEsizeZ = RVEsizeX 
    
    # Voxel resolution : Smallest dimension of the smallest ellipsoid should contain atleast 3 voxels
    voxel_size = 1.1*(np.amin(minDia) / 3.)
    Nx = int(round(RVEsizeX / voxel_size))         # Number of voxel/RVE side
    Ny = int(round(RVEsizeY / voxel_size))
    Nz = int(round(RVEsizeZ / voxel_size))    
    
    # Re-calculate the voxel resolution
    voxel_sizeX = RVEsizeX / Nx            
    voxel_sizeY = RVEsizeY / Ny
    voxel_sizeZ = RVEsizeZ / Nz
    
    totalEllipsoids = len(majDia)
    print('    Total number of grains        = {}'.format(totalEllipsoids))
    print('    RVE side lengths (X, Y, Z)    = {0}, {1}, {2}'.format(RVEsizeX, RVEsizeY, RVEsizeZ))
    print('    Number of voxels (X, Y, Z)    = {0}, {1}, {2}'.format(Nx, Ny, Nz))
    print('    Voxel resolution (X, Y, Z)    = {0:.4f}, {1:.4f}, {2:.4f}'.format(voxel_sizeX, voxel_sizeY, voxel_sizeZ))
    print('    Total number of voxels (C3D8) = {}\n'.format(Nx*Ny*Nz))           
                
    # Create dictionaries to store the data generated
    particle_data = {'Type': 'Elongated', 'Number': int(totalEllipsoids), 'Equivalent_diameter': list(eq_Dia), 'Major_diameter': list(majDia),
                     'Minor_diameter1': list(minDia), 'Minor_diameter2': list(minDia2), 'Tilt angle': list(tilt_angle)}

    RVE_data = {'RVE_sizeX': RVEsizeX, 'RVE_sizeY': RVEsizeY, 'RVE_sizeZ': RVEsizeZ, 
                'Voxel_numberX': Nx, 'Voxel_numberY': Ny, 'Voxel_numberZ': Nz,
                'Voxel_resolutionX': voxel_sizeX,'Voxel_resolutionY': voxel_sizeY, 'Voxel_resolutionZ': voxel_sizeZ}                    

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
    

def write_dump(Ellipsoids, sim_box):
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
    num_particles = len(Ellipsoids)
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
                    valuesX = re.findall(r'\S+', next(fd))
                    RVE_minX, RVE_maxX = list(map(float, valuesX))
                    
                    valuesY = re.findall(r'\S+', next(fd))
                    RVE_minY, RVE_maxY = list(map(float, valuesY))
                    
                    valuesZ = re.findall(r'\S+', next(fd))
                    RVE_minZ, RVE_maxZ = list(map(float, valuesZ))

    except FileNotFoundError:
        print('    .dump file not found, make sure "Packing" command is executed first!')
        raise FileNotFoundError
        
    # Create an instance of simulation box
    sim_box = Cuboid(RVE_minX, RVE_maxY, RVE_maxX, RVE_minY, RVE_maxZ, RVE_minZ)

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
                    valuesX = re.findall(r'\S+', next(fd))
                    RVE_minX, RVE_maxX = list(map(float, valuesX))
                    
                    valuesY = re.findall(r'\S+', next(fd))
                    RVE_minY, RVE_maxY = list(map(float, valuesY))
                    
                    valuesZ = re.findall(r'\S+', next(fd))
                    RVE_minZ, RVE_maxZ = list(map(float, valuesZ))
                                        

    except FileNotFoundError:
        print('    .dump file not found, make sure "Packing" command is executed first!')
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
        print('Json file not found, make sure "Voxelization" command is executed first!')
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
        print('Json file not found, make sure "Input statistics, Packing, & Voxelization" commands are executed first!')
        raise FileNotFoundError
    
    # Extract from dictionaries
    par_eqDia = particle_data['Equivalent_diameter']    
    voxel_size = RVE_data['Voxel_resolutionX']
    RVE_sizeX, RVE_sizeY, RVE_sizeZ = RVE_data['RVE_sizeX'], RVE_data['RVE_sizeY'], RVE_data['RVE_sizeZ']
    
    if particle_data['Type'] == 'Elongated':
        par_majDia = particle_data['Major_diameter']
        par_minDia = particle_data['Minor_diameter1']   
    
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
    if particle_data['Type'] == 'Equiaxed':          # Equiaxed grains (spherical particles)    
        
        # Find each grain's equivalent diameter
        grain_eqDia = []    
        for k, v in elmtSetDict.items():
            num_voxels = len(v)
            grain_vol = num_voxels * (voxel_size)**3
            grain_dia = 2 * (grain_vol * (3/(4*np.pi)))**(1/3)
            grain_eqDia.append(grain_dia)
        
        print("Writing particle & grain equivalent, major & minor diameter to file ('output_statistics.json')", end="")
            
        # write out the particle and grain equivalent diameters to files            
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        # Compute the L1-error
        kwargs = {'Spheres': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia}}}                
        error = l1_error_est(**kwargs)
                
        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)), 
                       'Grain type': particle_data['Type'],
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
                    elif abs(RVE_sizeX - coord[0]) <= 0.00000001:    # nodes on Right face
                        nodeRS.add(enum)
                    
                    if abs(0.0000 - coord[1]) <= 0.00000001:       # nodes on Bottom face
                        nodeBS.add(enum)
                    elif abs(RVE_sizeY - coord[1]) <= 0.00000001:    # nodes on Top face
                        nodeTS.add(enum)

                    if abs(0.0000 - coord[2]) <= 0.00000001:       # nodes on Front face
                        nodeFS.add(enum)
                    elif abs(RVE_sizeZ - coord[2]) <= 0.00000001:    # nodes on Back face
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
                    if val>=0.0 and val<=RVE_sizeX/2.:
                        pts[enum, 0] += RVE_sizeX

            for grain in shiftBack:
                pts = grain_node[grain]        
                # Pad the nodes on the front side by RVE z-dimension
                for enum, val in enumerate(pts[:, 2]):
                    if val>=0.0 and val<=RVE_sizeZ/2.:
                        pts[enum, 2] += RVE_sizeZ

            for grain in shiftTop:
                pts = grain_node[grain]        
                # Pad the nodes on the bottom side by RVE y-dimension
                for enum, val in enumerate(pts[:, 1]):
                    if val>=0.0 and val<=RVE_sizeY/2.:
                        pts[enum, 1] += RVE_sizeY                                
                    
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
            
            # Calculate ellipsoid dimensions using eigen values
            #ellPoints = points.T            
            #eigvals, eigvecs = np.linalg.eig(np.cov(ellPoints))
            #eigvals = np.sort(eigvals)
            #a2, b2 = eigvals[-1], eigvals[-2]            
            
            grain_majDia.append(a2)                 # update the major diameter list
            grain_minDia.append(b2)                 # update the minor diameter list
        
        print("Writing particle & grain equivalent, major & minor diameter to file ('output_statistics.json')", end="")

        # write out the particle and grain equivalent, major, minor diameters to file            
        par_eqDia = list(np.array(par_eqDia)/divideBy)
        grain_eqDia = list(np.array(grain_eqDia)/divideBy)

        par_majDia = list(np.array(par_majDia)/divideBy)
        grain_majDia = list(np.array(grain_majDia)/divideBy)

        par_minDia = list(np.array(par_minDia)/divideBy)
        grain_minDia = list(np.array(grain_minDia)/divideBy)                
        
        # Compute the L1-error
        kwargs = {'Ellipsoids': {'Equivalent': {'Particles': par_eqDia, 'Grains': grain_eqDia},
                                'Major diameter': {'Particles': par_majDia, 'Grains': grain_majDia},
                                'Minor diameter': {'Particles': par_minDia, 'Grains': grain_minDia}}}  
        error = l1_error_est(**kwargs)
        
        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)), 
                       'Grain type': particle_data['Type'],
                       'Unit_scale': scale,
                       'L1-error': error,
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
   
    if 'Spheres' in kwargs.keys():
        # Extract the values
        par_eqDia = np.asarray(kwargs['Spheres']['Equivalent']['Particles'])
        grain_eqDia = np.asarray(kwargs['Spheres']['Equivalent']['Grains'])
        
        # Concatenate both arrays to compute shared bins
        # NOTE: 'doane' produces better estimates for non-normal datasets
        total_eqDia = np.concatenate([par_eqDia, grain_eqDia]) 
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')
        
        # Compute the histogram for particles and grains
        hist_par, _ = np.histogram(par_eqDia, bins=shared_bins)
        hist_gr, _ = np.histogram(grain_eqDia, bins=shared_bins)
        
        # Normalize the values
        hist_par = hist_par/np.sum(hist_par)
        hist_gr = hist_gr/np.sum(hist_gr)
        
        # Compute the L1-error between particles and grains    
        l1_value = np.sum(np.abs(hist_par - hist_gr))        
        return l1_value

    elif 'Ellipsoids' in kwargs.keys():
        # Extract the values
        par_eqDia = np.asarray(kwargs['Ellipsoids']['Equivalent']['Particles'])
        grain_eqDia = np.asarray(kwargs['Ellipsoids']['Equivalent']['Grains'])
        
        # Concatenate both arrays to compute shared bins
        # NOTE: 'doane' produces better estimates for non-normal datasets
        total_eqDia = np.concatenate([par_eqDia, grain_eqDia]) 
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')
        
        # Compute the histogram for particles and grains
        hist_par, _ = np.histogram(par_eqDia, bins=shared_bins)
        hist_gr, _ = np.histogram(grain_eqDia, bins=shared_bins)
        
        # Normalize the values
        hist_par = hist_par/np.sum(hist_par)
        hist_gr = hist_gr/np.sum(hist_gr)
        
        # Compute the L1-error between particles and grains    
        l1_value = np.sum(np.abs(hist_par - hist_gr))        
        return l1_value
    '''
    elif 'Ellipsoids' in kwargs.keys():
        # Extract the values       
        par_majDia = np.asarray(kwargs['Ellipsoids']['Major diameter']['Particles'])
        grain_majDia = np.asarray(kwargs['Ellipsoids']['Major diameter']['Grains'])
        
        par_minDia = np.asarray(kwargs['Ellipsoids']['Minor diameter']['Particles'])
        grain_minDia = np.asarray(kwargs['Ellipsoids']['Minor diameter']['Grains'])
        
        # Concatenate corresponding arrays to compute shared bins
        total_majDia = np.concatenate([par_majDia, grain_majDia]) 
        total_minDia = np.concatenate([par_minDia, grain_minDia])
                                
        # Find the corresponding shared bin edges 
        # NOTE: 'doane' produces better estimates for non-normal datasets
        shared_majDia = np.histogram_bin_edges(total_majDia, bins='doane')
        shared_minDia = np.histogram_bin_edges(total_minDia, bins='doane') 
        
        # Concatenate arrays along columns for creating multidimensional histogram
        par_dd = np.c_[par_majDia, par_minDia]
        grain_dd = np.c_[grain_majDia, grain_minDia]
        
        # Compute the multidimensional histogram for both: particles and grains
        hist_par, _ = np.histogramdd(par_dd, bins=(shared_majDia, shared_minDia))
        hist_gr, _ = np.histogramdd(grain_dd, bins=(shared_majDia, shared_minDia))        
    
        # Normalize the values
        hist_par = hist_par/np.sum(hist_par)
        hist_gr = hist_gr/np.sum(hist_gr)
        
        # Compute the L1-error between particles and grains    
        l1_value = np.sum(np.abs(hist_par - hist_gr))                
        return l1_value                     
    '''

def plot_output_stats():
    r"""
    Evaluates particle- and output RVE grain statistics with respect to Major, Minor & Equivalent diameters and plots the distributions

    .. note:: 1. Particle information is read from (.json) file generated by :meth:`kanapy.input_output.particleStatGenerator`.
                 And RVE grain information is read from the (.json) files generated by :meth:`kanapy.voxelization.voxelizationRoutine`.                                     
    """ 
    print('') 
    print('Plotting input & output statistics')
    cwd = os.getcwd()
    json_dir = cwd + '/json_files'          # Folder to store the json files

    try:
        with open(json_dir + '/output_statistics.json') as json_file:
            dataDict = json.load(json_file) 
          
    except FileNotFoundError:
        print('Json file not found, make sure "Input statistics, Packing, Voxelization & Output Statistics" commands are executed first!')
        raise FileNotFoundError
    
    # read the data from the file
    if dataDict['Grain type'] == 'Equiaxed':
        
        par_eqDia = np.sort(np.asarray(dataDict['Particle_Equivalent_diameter']))
        grain_eqDia = np.sort(np.asarray(dataDict['Grain_Equivalent_diameter']))

        # Convert to micro meter for plotting   
        if dataDict['Unit_scale'] == 'mm':            
            par_eqDia, grain_eqDia = par_eqDia*1000, grain_eqDia*1000
        
        # Concatenate both arrays to compute shared bins
        # NOTE: 'doane' produces better estimates for non-normal datasets
        total_eqDia = np.concatenate([par_eqDia, grain_eqDia]) 
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')

        # Get the mean & std of the underlying normal distribution
        par_data, grain_data = np.log(par_eqDia), np.log(grain_eqDia)
        mu_par, std_par = np.mean(par_data), np.std(par_data)
        mu_gr, std_gr = np.mean(grain_data), np.std(grain_data)

        # Lognormal mean & variance & std  
        #log_mean = np.exp(mean + (sd**2)/2.0)
        #log_var = np.exp((sd**2)-1.0)*np.exp(2.0*mean + sd**2)        
        #print(log_mean, (log_var)**0.5, log_var)
        
        # NOTE: lognorm takes mean & std of normal distribution
        par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par)) 
        grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr)) 
                        
        # Plot the histogram & PDF
        sns.set(color_codes=True)        
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))

        # Plot histogram
        ax[0].hist([par_eqDia, grain_eqDia], density=False, bins=len(shared_bins), label=['Input', 'Output'])                 
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)
        
        # Plot PDF                             
        ypdf1 = par_lognorm.pdf(par_eqDia)
        ypdf2 = grain_lognorm.pdf(grain_eqDia)
        ax[1].plot(par_eqDia, ypdf1, linestyle='-', linewidth=3.0, label='Input')              
        ax[1].fill_between(par_eqDia, 0, ypdf1, alpha=0.3)
        ax[1].plot(grain_eqDia, ypdf2, linestyle='-', linewidth=3.0, label='Output')              
        ax[1].fill_between(grain_eqDia, 0, ypdf2, alpha=0.3) 
                    
        #sns.distplot(ypdf1, hist = False, kde = True, 
        #             kde_kws = {'shade': True, 'linewidth': 3}, 
        #             label = 'Input', ax=ax[1])
        #sns.distplot(ypdf2, hist = False, kde = True, 
        #             kde_kws = {'shade': True, 'linewidth': 3}, 
        #            label = 'Output', ax=ax[1])        
        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('Equivalent diameter (μm)', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)        
        plt.savefig(cwd + "/Equivalent_diameter.png", bbox_inches="tight")                      
                
        
    elif dataDict['Grain type'] == 'Elongated':
        par_eqDia = np.sort(np.asarray(dataDict['Particle_Equivalent_diameter']))        
        grain_eqDia = np.sort(np.asarray(dataDict['Grain_Equivalent_diameter']))
            
        par_majDia = np.sort(np.asarray(dataDict['Particle_Major_diameter']))
        par_minDia = np.sort(np.asarray(dataDict['Particle_Minor_diameter']))
    
        grain_majDia = np.sort(np.asarray(dataDict['Grain_Major_diameter']))
        grain_minDia = np.sort(np.asarray(dataDict['Grain_Minor_diameter']))
        
        # Convert to micro meter for plotting   
        if dataDict['Unit_scale'] == 'mm':            
            par_eqDia, grain_eqDia = par_eqDia*1000, grain_eqDia*1000
            par_majDia, grain_majDia = par_majDia*1000, grain_majDia*1000
            par_minDia, grain_minDia = par_minDia*1000, grain_minDia*1000        
                
        # Concatenate corresponding arrays to compute shared bins
        total_eqDia = np.concatenate([par_eqDia, grain_eqDia])                 
        total_majDia = np.concatenate([par_majDia, grain_majDia]) 
        total_minDia = np.concatenate([par_minDia, grain_minDia])
                                
        # Find the corresponding shared bin edges 
        # NOTE: 'doane' produces better estimates for non-normal datasets
        shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')
        shared_majDia = np.histogram_bin_edges(total_majDia, bins='doane')
        shared_minDia = np.histogram_bin_edges(total_minDia, bins='doane')                                          
        
#        loop_list = [(par_eqDia, grain_eqDia, len(shared_bins), 'Equivalent'), 
#                     (par_majDia, grain_majDia, len(shared_majDia), 'Major'), 
#                     (par_minDia, grain_minDia, len(shared_minDia), 'Minor')]

        loop_list = [(par_eqDia, grain_eqDia, len(shared_bins), 'Equivalent')]
                               
        for (particles, grains, binNum, name) in loop_list:
        
            # Get the mean & std of the underlying normal distribution
            par_data, grain_data = np.log(particles), np.log(grains)
            mu_par, std_par = np.mean(par_data), np.std(par_data)
            mu_gr, std_gr = np.mean(grain_data), np.std(grain_data)
            
            # Lognormal mean & variance & std  
            #log_mean = np.exp(mean + (sd**2)/2.0)
            #log_var = np.exp((sd**2)-1.0)*np.exp(2.0*mean + sd**2)        
            #print(log_mean, (log_var)**0.5, log_var)
            
            # NOTE: lognorm takes mean & std of normal distribution
            par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par)) 
            grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr))         

            # Plot the histogram & PDF        
            sns.set(color_codes=True)                                      
            fig, ax = plt.subplots(1, 2, figsize=(15, 9))
            
            # Plot histogram
            ax[0].hist([particles, grains], density=False, bins=binNum, label=['Input', 'Output'])                 
            ax[0].legend(loc="upper right", fontsize=16)
            ax[0].set_xlabel('{0} diameter (μm)'.format(name), fontsize=18)
            ax[0].set_ylabel('Frequency', fontsize=18)
            ax[0].tick_params(labelsize=14)
            
            # Plot PDF                             
            ypdf1 = par_lognorm.pdf(particles)
            ypdf2 = grain_lognorm.pdf(grains)
            ax[1].plot(particles, ypdf1, linestyle='-', linewidth=3.0, label='Input')              
            ax[1].fill_between(particles, 0, ypdf1, alpha=0.3)
            ax[1].plot(grains, ypdf2, linestyle='-', linewidth=3.0, label='Output')              
            ax[1].fill_between(grains, 0, ypdf2, alpha=0.3) 
                        
            #sns.distplot(ypdf1, hist = False, kde = True, 
            #             kde_kws = {'shade': True, 'linewidth': 3}, 
            #             label = 'Input', ax=ax[1])
            #sns.distplot(ypdf2, hist = False, kde = True, 
            #             kde_kws = {'shade': True, 'linewidth': 3}, 
            #            label = 'Output', ax=ax[1])        
            ax[1].legend(loc="upper right", fontsize=16)
            ax[1].set_xlabel('{0} diameter (μm)'.format(name), fontsize=18)
            ax[1].set_ylabel('Density', fontsize=18)
            ax[1].tick_params(labelsize=14)        
            plt.savefig(cwd + "/{0}_diameter.png".format(name), bbox_inches="tight") 
            print("    '{0}_diameter.png' is placed in the current working directory\n".format(name))
        
        
        ''' Plot the aspect ratio comparison '''
        par_AR = np.sort(np.asarray(dataDict['Particle_Major_diameter']) / np.asarray(dataDict['Particle_Minor_diameter']))
        grain_AR = np.sort(np.asarray(dataDict['Grain_Major_diameter']) / np.asarray(dataDict['Grain_Minor_diameter']))

        # Concatenate corresponding arrays to compute shared bins
        total_AR = np.concatenate([par_AR, grain_AR])                         
                                
        # Find the corresponding shared bin edges
        shared_AR = np.histogram_bin_edges(total_AR, bins='doane')
        
        # Get the mean & std of the underlying normal distribution
        par_data, grain_data = np.log(par_AR), np.log(grain_AR)
        mu_par, std_par = np.mean(par_data), np.std(par_data)
        mu_gr, std_gr = np.mean(grain_data), np.std(grain_data)        
        
        par_lognorm = lognorm(s=std_par, scale=np.exp(mu_par)) 
        grain_lognorm = lognorm(s=std_gr, scale=np.exp(mu_gr)) 
        
        # Plot the histogram & PDF        
        sns.set(color_codes=True)                                      
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        
        # Plot histogram
        ax[0].hist([par_AR, grain_AR], density=False, bins=len(shared_AR), label=['Input', 'Output'])                 
        ax[0].legend(loc="upper right", fontsize=16)
        ax[0].set_xlabel('Aspect ratio', fontsize=18)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].tick_params(labelsize=14)
        
        # Plot PDF                             
        ypdf1 = par_lognorm.pdf(par_AR)
        ypdf2 = grain_lognorm.pdf(grain_AR)
        ax[1].plot(par_AR, ypdf1, linestyle='-', linewidth=3.0, label='Input')              
        ax[1].fill_between(par_AR, 0, ypdf1, alpha=0.3)
        ax[1].plot(grain_AR, ypdf2, linestyle='-', linewidth=3.0, label='Output')              
        ax[1].fill_between(grain_AR, 0, ypdf2, alpha=0.3) 
                    
        #sns.distplot(ypdf1, hist = False, kde = True, 
        #             kde_kws = {'shade': True, 'linewidth': 3}, 
        #             label = 'Input', ax=ax[1])
        #sns.distplot(ypdf2, hist = False, kde = True, 
        #             kde_kws = {'shade': True, 'linewidth': 3}, 
        #            label = 'Output', ax=ax[1])        
        ax[1].legend(loc="upper right", fontsize=16)
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)        
        plt.savefig(cwd + "/Aspect_ratio.png", bbox_inches="tight") 
        print("    'Aspect_ratio.png' is placed in the current working directory\n".format(name))                                            
        
    print('---->DONE!\n')                          
    return
    
        
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
        print('Json file not found, make sure "Input statistics, Packing, Voxelization & Output Statistics" commands are executed first!')
        raise FileNotFoundError
                    
    voxel_size = RVE_data['Voxel_resolutionX']

    grain_vol = {}
    # For each grain find its volume and output it
    for gid, elset in elmtSetDict.items():
        # Convert to  
        gvol = len(elset) * (voxel_size**3)
        grain_vol[gid] = gvol 

    # Sort the grain volumes in ascending order of grain IDs
    gv_sorted_keys = sorted(grain_vol, key=grain_vol.get)
    gv_sorted_values = [[grain_vol[gk]] for gk in gv_sorted_keys]            
#    gv_sorted_values = [[gk,gv] for gk,gv in grain_vol.items()]
    
    print("Writing grain volumes info. to file ('grainVolumes.csv')\n", end="")
        
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

    print("Writing shared GB surface area info. to file ('shared_surfaceArea.csv')", end="")
    
    # Write out shared grain boundary area to a file
    with open(json_dir + '/shared_surfaceArea.csv', "w", newline="") as f:
        f.write('GrainA, GrainB, SharedArea\n')
        writer = csv.writer(f)
        writer.writerows(shared_area)
    
    print('---->DONE!\n')       
    return    

