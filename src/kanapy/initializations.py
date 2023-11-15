import json
import os
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import lognorm, norm


def particleStatGenerator(stats_dict, gs_data=None, ar_data=None, save_files=False):
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
    if "offs" in stats_dict["Equivalent diameter"]:
        offs = stats_dict["Equivalent diameter"]["offs"]
    else:
        offs = None
    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]

    #Equivalent diameter statistics
    # NOTE: SCIPY's lognorm takes in sigma & mu of Normal distribution
    # https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma/13837335#13837335

    # Compute the Log-normal PDF & CDF.
    if offs is None:
        frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
    else:
        frozen_lognorm = lognorm(s=sd, loc=offs, scale=mean)

    xaxis = np.linspace(0.1,200,1000)
    ypdf = frozen_lognorm.pdf(xaxis)

    # Find the location at which CDF > 0.99
    #cdf_idx = np.where(ycdf > 0.99)[0][0]
    #x_lim = xaxis[cdf_idx]
    x_lim = dia_cutoff_max*1.5

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
        ax.axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                   label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax.axvline(dia_cutoff_max, linestyle='-', linewidth=3.0,
                   label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        if gs_data is not None:
            ind = np.nonzero(gs_data<x_lim)[0]
            ax.hist(gs_data[ind], bins=80, density=True, label='experimental data')
        plt.title("Grain equivalent diameter distribution", fontsize=20)
        plt.legend(fontsize=16)
        plt.show()
    elif stats_dict["Grain type"] == "Elongated":
        # Extract mean grain aspect ratio value info from input file
        sd_AR = stats_dict["Aspect ratio"]["std"]
        mean_AR = stats_dict["Aspect ratio"]["mean"]
        if "offs" in stats_dict["Aspect ratio"]:
            offs_AR = stats_dict["Aspect ratio"]["offs"]
        else:
            offs_AR = None
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
        ax[0].axvline(dia_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(dia_cutoff_min))
        ax[0].axvline(dia_cutoff_max, linestyle='-', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(dia_cutoff_max))
        if gs_data is not None:
            ax[0].hist(gs_data, bins=80, density=True, label='experimental data')
        ax[0].legend(fontsize=14)

        # Plot aspect ratio statistics
        # Compute the Log-normal PDF & CDF.
        if offs_AR is None:
            frozen_lognorm = lognorm(s=sd_AR, scale=np.exp(mean_AR))
        else:
            frozen_lognorm = lognorm(sd_AR, loc=offs_AR, scale=mean_AR)
        xaxis = np.linspace(0.5, 2*ar_cutoff_max, 500)
        ypdf = frozen_lognorm.pdf(xaxis)
        ax[1].plot(xaxis, ypdf, linestyle='-', linewidth=3.0)
        ax[1].fill_between(xaxis, 0, ypdf, alpha=0.3)
        ax[1].set_xlabel('Aspect ratio', fontsize=18)
        ax[1].set_ylabel('Density', fontsize=18)
        ax[1].tick_params(labelsize=14)
        ax[1].axvline(ar_cutoff_min, linestyle='--', linewidth=3.0,
                      label='Minimum cut-off: {:.2f}'.format(ar_cutoff_min))
        ax[1].axvline(ar_cutoff_max, linestyle='-', linewidth=3.0,
                      label='Maximum cut-off: {:.2f}'.format(ar_cutoff_max))
        if ar_data is not None:
            ax[1].hist(ar_data, bins=15, density=True, label='experimental data')
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


def RVEcreator(stats_dict, nsteps=1000, save_files=False):
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
    if "offs" in stats_dict["Equivalent diameter"]:
        offs = stats_dict["Equivalent diameter"]["offs"]
    else:
        offs = None
    dia_cutoff_min = stats_dict["Equivalent diameter"]["cutoff_min"]
    dia_cutoff_max = stats_dict["Equivalent diameter"]["cutoff_max"]

    # Extract RVE side lengths and voxel numbers info from input file
    RVEsizeX = stats_dict["RVE"]["sideX"]
    RVEsizeY = stats_dict["RVE"]["sideY"]
    RVEsizeZ = stats_dict["RVE"]["sideZ"]
    Nx = int(stats_dict["RVE"]["Nx"])
    Ny = int(stats_dict["RVE"]["Ny"])
    Nz = int(stats_dict["RVE"]["Nz"])

    if "Phase" in stats_dict.keys():
        phase_name = stats_dict["Phase"]["Name"]
        phase_number = stats_dict["Phase"]["Number"]
        VF = stats_dict["Phase"]["Volume fraction"]
    else:
        phase_name = "Material"
        phase_number = 0
        VF = 1.

    # Extract other simulation attrributes from input file
    periodicity = str(stats_dict["Simulation"]["periodicity"])
    output_units = str(stats_dict["Simulation"]["output_units"])

    # Raise ValueError if units are not specified as 'mm' or 'um'
    if output_units != 'mm' and output_units != 'um':
        raise ValueError('Output units can only be "mm" or "um"!')

    # Compute the Log-normal PDF & CDF.
    if offs is None:
        frozen_lognorm = lognorm(s=sd, scale=np.exp(mean))
    else:
        frozen_lognorm = lognorm(s=sd, loc=offs, scale=mean)

    xaxis = np.linspace(0.1,200,1000)
    ycdf = frozen_lognorm.cdf(xaxis)

    # Get the mean value for each pair of neighboring points as centers of bins
    xaxis = np.vstack([xaxis[1:], xaxis[:-1]]).mean(axis=0)

    # Based on the cutoff specified, get the restricted distribution
    index_array = np.where((xaxis >= dia_cutoff_min) & (xaxis <= dia_cutoff_max))
    eq_Dia = xaxis[index_array]          # Selected diameters within the cutoff

    # Compute the number fractions and extract them based on the cut-off
    number_fraction = np.ediff1d(ycdf)  # better use lognorm.pdf
    numFra_Dia = number_fraction[index_array]

    # Volume of each ellipsoid
    volume_array = (4/3)*np.pi*(0.5*eq_Dia)**3

    # Volume fraction for each ellipsoid
    individualK = np.multiply(numFra_Dia, volume_array)
    K = individualK/np.sum(individualK)

    # Total number of ellipsoids for packing density 65%
    num = np.divide(K*(VF*RVEsizeX*RVEsizeY*RVEsizeZ), volume_array)*0.65
    num = np.rint(num).astype(int)       # Round to the nearest integer
    totalEllipsoids = int(np.sum(num))

    # Duplicate the diameter values
    eq_Dia = np.repeat(eq_Dia, num)  # better calculate num first

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
        print("    The voxel resolutions along (X,Y,Z): ({0:.4f},{1:.4f},{2:.4f}) are not equal!"\
              .format(voxel_sizeX,voxel_sizeY, voxel_sizeZ))
        print("    Change the RVE side lengths (OR) the voxel numbers\n")
        sys.exit(0)

    # raise value error in case the grains are not voxelated well
    if voxel_sizeX >= np.amin(eq_Dia) / 3.:
        print(" ")
        print("    Grains will not be voxelated well!")
        print("    Please increase the voxel numbers (OR) decrease the RVE side lengths\n")
        sys.exit(0)
    # raise warning if large grain occur in periodic box
    if np.amax(eq_Dia) >= RVEsizeX*0.5 and periodicity:
        print("\n")
        print("    Periodic box with grains larger the half of box width.")
        print("    Check grain polygons carefully.")

    print('    Total number of particles     = {}'.format(totalEllipsoids))
    print('    RVE side lengths (X, Y, Z)    = {0}, {1}, {2}'.format(RVEsizeX, RVEsizeY, RVEsizeZ))
    print('    Number of voxels (X, Y, Z)    = {0}, {1}, {2}'.format(Nx, Ny, Nz))
    print('    Voxel resolution (X, Y, Z)    = {0:.4f}, {1:.4f}, {2:.4f}'.format(voxel_sizeX, voxel_sizeY, voxel_sizeZ))
    print('    Total number of voxels (C3D8) = {}\n'.format(Nx*Ny*Nz))

    phname = [phase_name]*totalEllipsoids
    phnum = [phase_number]*totalEllipsoids

    if stats_dict["Grain type"] == "Equiaxed":

        # Create dictionaries to store the data generated
        particle_data = {
            'Type': stats_dict["Grain type"],
            'Number': totalEllipsoids,
            'Equivalent_diameter': list(eq_Dia),
        }
        phases = {
            'Phase name':phname,
            'Phase number':phnum
        }

    elif stats_dict["Grain type"] == "Elongated":
        # Extract mean grain aspect ratio value info from dict
        sd_AR = stats_dict["Aspect ratio"]["std"]
        mean_AR = stats_dict["Aspect ratio"]["mean"]
        if "offs" in stats_dict["Aspect ratio"]:
            offs_AR = stats_dict["Aspect ratio"]["offs"]
        else:
            offs_AR = None
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
        num = totalEllipsoids
        while num>0:
            tilt = norm.rvs(scale=std_Ori, loc=mean_Ori, size=num)
            index_array = np.where((tilt >= ori_cutoff_min) & (tilt <= ori_cutoff_max))
            TA = tilt[index_array].tolist()
            tilt_angle.extend(TA)
            num = totalEllipsoids - len(tilt_angle)

        # Aspect ratio statistics
        # Sample from lognormal or gamma distribution:
        # it takes mean, std and scale of the underlying normal distribution
        finalAR = []
        num = totalEllipsoids
        while num>0:
            #ar = np.random.lognormal(mean_AR, sd_AR, num)
            if offs_AR is None:
                ar = lognorm.rvs(sd_AR, scale=np.exp(mean_AR), size=num)
            else:
                ar = lognorm.rvs(sd_AR, loc=offs_AR, scale=mean_AR, size=num)
            index_array = np.where((ar >= ar_cutoff_min) & (ar <= ar_cutoff_max))
            AR = ar[index_array].tolist()
            finalAR.extend(AR)
            num = totalEllipsoids - len(finalAR)

        finalAR = np.array(finalAR)

        # Calculate the major, minor axes lengths for pores using: (4/3)*pi*(r**3) = (4/3)*pi*(a*b*c) & b=c & a=AR*b
        minDia = eq_Dia / finalAR**(1/3)                            # Minor axis length
        majDia = minDia * finalAR                                   # Major axis length
        minDia2 = minDia.copy()                                     # Minor2 axis length (assuming spheroid)

        # Create dictionaries to store the data generated
        particle_data = {
            'Type': stats_dict["Grain type"],
            'Number': totalEllipsoids,
            'Equivalent_diameter': list(eq_Dia),
            'Major_diameter': list(majDia),
            'Minor_diameter1': list(minDia),
            'Minor_diameter2': list(minDia2),
            'Tilt angle': list(tilt_angle),
        }
        phases = {
            'Phase name': phname,
            'Phase number': phnum
        }
    else:
        raise ValueError('The value for "Grain type" must be either "Equiaxed" or "Elongated".')

    periodic = True if periodicity=='True' else False
    RVE_data = {'RVE_sizeX': RVEsizeX, 'RVE_sizeY': RVEsizeY, 'RVE_sizeZ': RVEsizeZ,
                'Voxel_numberX': Nx, 'Voxel_numberY': Ny, 'Voxel_numberZ': Nz,
                'Voxel_resolutionX': voxel_sizeX,'Voxel_resolutionY': voxel_sizeY,
                'Voxel_resolutionZ': voxel_sizeZ, 'Periodic': periodic,
                'Units': output_units}

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


    return particle_data, phases, RVE_data, simulation_data
