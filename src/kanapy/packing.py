# -*- coding: utf-8 -*-
#import os
#import sys
#import json
import random
from tqdm import tqdm

import numpy as np

from kanapy.input_output import write_dump
from kanapy.entities import Ellipsoid, Cuboid, Octree, Simulation_Box


def particle_generator(particle_data, sim_box):
    """
    Initializes ellipsoids by assigning them random positions and speeds within the simulation box.

    :param particle_data: Ellipsoid information such as such as Major, Minor, Equivalent diameters and its tilt angle. 
    :type particle_data: Python dictionary 
    :param sim_box: Simulation box representing RVE.
    :type sim_box: :class:`entities.Simulation_Box`
    
    :returns: Ellipsoids for the packing routine
    :rtype: list       
    """
    num_particles = particle_data['Number']                      # Total number of ellipsoids
    Ellipsoids = []
    for n in range(num_particles):

        iden = n+1                                                # ellipsoid 'id'
        if particle_data['Type'] == 'Equiaxed':
            a = particle_data['Equivalent_diameter'][n] / 2.
            b = particle_data['Equivalent_diameter'][n] / 2.
            c = particle_data['Equivalent_diameter'][n] / 2.
        else:    
            a = particle_data['Major_diameter'][n] / 2.               # Semi-major length
            b = particle_data['Minor_diameter1'][n] / 2.              # Semi-minor length 1
            c = particle_data['Minor_diameter2'][n] / 2.              # Semi-minor length 2

        # Random placement for ellipsoids
        x = random.uniform(c, sim_box.w - c)
        y = random.uniform(c, sim_box.h - c)
        z = random.uniform(c, sim_box.d - c)

        # Angle represents inclination of Major axis w.r.t positive x-axis        
        if particle_data['Type'] == 'Equiaxed':
            angle = np.radians(90)
        else:
            angle = np.radians(particle_data['Tilt angle'][n])         # Extract the angle        
            
        vec_a = np.array([a*np.cos(angle), a*np.sin(angle), 0.0])   # Tilt vector wrt (+ve) x axis        
        cross_a = np.cross(np.array([1, 0, 0]), vec_a)              # Do the cross product to find the quaternion axis        
        norm_cross_a = np.linalg.norm(cross_a, 2)                   # norm of the vector (Magnitude)        
        quat_axis = cross_a/norm_cross_a                            # normalize the quaternion axis

        # Find the quaternion components
        qx, qy, qz = quat_axis * np.sin(angle/2)
        qw = np.cos(angle/2)
        quat = np.array([qw, qx, qy, qz])

        # instance of Ellipsoid class
        ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat)
        ellipsoid.color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))

        # Define random speed values along the 3 axes x, y & z
        ellipsoid.speedx = np.random.uniform(low=-c/20., high=c/20.)
        ellipsoid.speedy = np.random.uniform(low=-c/20., high=c/20.)
        ellipsoid.speedz = np.random.uniform(low=-c/20., high=c/20.)
        
        Ellipsoids.append(ellipsoid)                               # adds ellipsoid to list

    return Ellipsoids


def particle_grow(sim_box, Ellipsoids, periodicity, nsteps, dump=False):
    """
    Initializes the :class:`entities.Octree` class and performs recursive subdivision with 
    collision checks and response for the ellipsoids. At each time step of the simulation it 
    increases the size of the ellipsoid by a factor, which depends on the user-defined value for total number of time steps. 

    :param sim_box: Simulation box representing RVE.
    :type sim_box: :obj:`entities.Simulation_Box`  
    :param Ellipsoids: Ellipsoids for the packing routine.
    :type Ellipsoids: list    
    :param periodicity: Status of periodicity.
    :type periodicity: boolean 
    :param nsteps:  Total simulation steps.
    :type nsteps: int       

    .. note:: :meth:`kanapy.input_output.write_dump` function is called at each time step of the simulation to
              write output (.dump) files. By default, periodic images are written to the output file, 
              but this option can be disabled within the function.         
    """
    # Reduce the size of the particles to (1/nsteps)th of its original size
    for ell in Ellipsoids:
        ell.a, ell.b, ell.c = ell.oria/nsteps, ell.orib/nsteps, ell.oric/nsteps

    # Simulation loop for particle growth and interaction steps
    for i in tqdm(range(nsteps+1)):
    
        # Initialize Octree and test for collision between ellipsoids
        tree = Octree(0, Cuboid(sim_box.left, sim_box.top, sim_box.right,
                                sim_box.bottom, sim_box.front, sim_box.back), Ellipsoids)
        tree.update()
        
        if dump:
            # Dump the ellipsoid information to be read by OVITO (Includes duplicates at periodic boundaries)
            write_dump(Ellipsoids, sim_box)

        # Delete duplicates if existing (Active only if periodicity is True)
        # find all the items which are not duplicates
        inter_ell = [ell for ell in Ellipsoids if not isinstance(ell.id, str)]
        Ellipsoids = inter_ell

        dups = []
        # Loop over the ellipsoids: move, set Bbox, & check for wall collision / PBC
        for ellipsoid in Ellipsoids:
            # Move the ellipsoid according to collision status
            ellipsoid.move()
            # grow the ellipsoid
            ellipsoid.growth(nsteps)
            # Check for wall collision or create duplicates
            ell_dups = ellipsoid.wallCollision(sim_box, periodicity)
            dups.extend(ell_dups)
            # Update the BBox of the ellipsoid
            ellipsoid.set_cub()

        # Update the actual list with duplicates
        Ellipsoids.extend(dups)
        
        # Update the simulation time
        sim_box.sim_ts += 1
        
    return Ellipsoids, sim_box


def packingRoutine(particle_data, RVE_data, simulation_data, save_files=False):
    """
    The main function that controls the particle packing routine using: :meth:`particle_grow` & :meth:`particle_generator`
    
    .. note:: Particle, RVE and simulation data are read from the JSON files generated by :meth:`kanapy.input_output.particleStatGenerator`.
              They contain the following information:
    
              * Ellipsoid attributes such as Major, Minor, Equivalent diameters and its tilt angle. 
              * RVE attributes such as RVE (Simulation domain) size, the number of voxels and the voxel resolution.
              * Simulation attributes such as total number of timesteps and periodicity.                         
    """
    print('Starting particle simulation')     
    print('    Creating simulation box of required dimensions')
    # Create an instance of simulation box
    sim_box = Simulation_Box(RVE_data['RVE_sizeX'], RVE_data['RVE_sizeY'], RVE_data['RVE_sizeZ'])
    
    print('    Creating particles from distribution statistics')
    # Create instances for particles
    Particles = particle_generator(particle_data, sim_box)
    
    # Growth of particle at each time step
    print('    Particle packing by growth simulation')
    
    if simulation_data['Periodicity'] == 'True':
        periodic_status = True
    elif simulation_data['Periodicity'] == 'False':
        periodic_status = False
    else:
        raise ValueError('Wrong value for periodicity in packingRoutine')   
        
    particles, simbox = particle_grow(sim_box, Particles, periodic_status, \
                                      simulation_data['Time steps'], dump=save_files)
    
    print('Completed particle packing')
    print('')
    
    return particles, simbox
