# -*- coding: utf-8 -*-
import random
from tqdm import tqdm
import numpy as np

from kanapy.input_output import write_dump
from kanapy.entities import Ellipsoid, Cuboid, Octree, Simulation_Box
from kanapy.collisions import collide_detect


def particle_generator(particle_data, sim_box, periodic):
    """
    Initializes ellipsoids by assigning them random positions and speeds within
    the simulation box.

    :param particle_data: Ellipsoid information such as such as Major, Minor,
          Equivalent diameters and its tilt angle. 
    :type particle_data: Python dictionary 
    :param sim_box: Simulation box representing RVE.
    :type sim_box: :class:`entities.Simulation_Box`
    
    :returns: Ellipsoids for the packing routine
    :rtype: list       
    """
    Ellipsoids = []
    id_ctr = 0
    for particle in particle_data:
        num_particles = particle['Number']  # Total number of ellipsoids
        # introduce scaling factor to reduce particle overlap for non-peridoc box
        sf = 0.5 if periodic else 0.45
        for n in range(num_particles):
            iden = id_ctr + n + 1  # ellipsoid 'id'
            if particle['Type'] == 'Equiaxed':
                a = b = c = sf * particle['Equivalent_diameter'][n]
            else:
                a = sf * particle['Major_diameter'][n]   # Semi-major length
                b = sf * particle['Minor_diameter1'][n]  # Semi-minor length 1
                c = sf * particle['Minor_diameter2'][n]  # Semi-minor length 2

            # Random placement for ellipsoids
            x = random.uniform(c, sim_box.w - c)
            y = random.uniform(c, sim_box.h - c)
            z = random.uniform(c, sim_box.d - c)

            # Angle represents inclination of Major axis w.r.t positive x-axis
            if particle['Type'] == 'Equiaxed':
                angle = 0.5*np.pi
            else:
                # Extract the angle
                angle = particle['Tilt angle'][n]
            # Tilt vector wrt (+ve) x axis
            vec_a = np.array([a*np.cos(angle), a*np.sin(angle), 0.0])
            # Do the cross product to find the quaternion axis
            cross_a = np.cross(np.array([1, 0, 0]), vec_a)
            # norm of the vector (Magnitude)
            norm_cross_a = np.linalg.norm(cross_a, 2)
            quat_axis = cross_a/norm_cross_a  # normalize the quaternion axis

            # Find the quaternion components
            qx, qy, qz = quat_axis * np.sin(angle/2)
            qw = np.cos(angle/2)
            quat = np.array([qw, qx, qy, qz])

            # instance of Ellipsoid class
            ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat, phasenum=particle['Phase'])
            ellipsoid.color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))

            # Define random speed values along the 3 axes x, y & z
            ellipsoid.speedx0 = np.random.uniform(low=-c/20., high=c/20.)
            ellipsoid.speedy0 = np.random.uniform(low=-c/20., high=c/20.)
            ellipsoid.speedz0 = np.random.uniform(low=-c/20., high=c/20.)
            ellipsoid.speedx = ellipsoid.speedx0
            ellipsoid.speedy = ellipsoid.speedy0
            ellipsoid.speedz = ellipsoid.speedz0

            Ellipsoids.append(ellipsoid)  # adds ellipsoid to list
        id_ctr += num_particles

    return Ellipsoids


def particle_grow(sim_box, Ellipsoids, periodicity, nsteps,
                  k_rep=0.0, k_att=0.0, vf=None,
                  dump=False):
    """
    Initializes the :class:`entities.Octree` class and performs recursive
    subdivision with collision checks and response for the ellipsoids. At each
    time step of the simulation it increases the size of the ellipsoid by a 
    factor, which depends on the user-defined value for total number of time
    steps.

    :param sim_box: Simulation box representing RVE.
    :type sim_box: :obj:`entities.Simulation_Box`  
    :param Ellipsoids: Ellipsoids for the packing routine.
    :type Ellipsoids: list    
    :param periodicity: Status of periodicity.
    :type periodicity: boolean 
    :param nsteps:  Total simulation steps to fill box volume with particle
         volume.
    :type nsteps: int
    :param dump: Indicate if dump files for particles are written.
    :type dump: boolean


    .. note:: :meth:`kanapy.input_output.write_dump` function is called at each
              time step of the simulation to write output (.dump) files. 
              By default, periodic images are written to the output file, 
              but this option can be disabled within the function.         
    """
    if vf is None:
        vf = 0.7
    # Reduce the size of the particles to (1/nsteps)th of its original size
    for ell in Ellipsoids:
        ell.a, ell.b, ell.c = ell.oria/nsteps, ell.orib/nsteps, ell.oric/nsteps

    # Simulation loop for particle growth and interaction steps
    end_step = int(vf*nsteps)  # grow particles only to given volume fraction
    for i in tqdm(range(end_step)):
    
        # Initialize Octree and test for collision between ellipsoids
        for ellipsoid in Ellipsoids:
            ellipsoid.branches = []
            if periodicity:
                ellipsoid.speedx = 0.
                ellipsoid.speedy = 0.
                ellipsoid.speedz = 0.
            ellipsoid.ncollision = 0
            
        tree = Octree(0, Cuboid(sim_box.left, sim_box.top, sim_box.right,
                                sim_box.bottom, sim_box.front, sim_box.back), 
                      Ellipsoids)
        tree.update()
        
        if periodicity:
            for ellipsoid in Ellipsoids:
                if ellipsoid.ncollision == 0:
                    ellipsoid.speedx = ellipsoid.speedx0
                    ellipsoid.speedy = ellipsoid.speedy0
                    ellipsoid.speedz = ellipsoid.speedz0
        if dump:
            # Dump the ellipsoid information to be read by OVITO 
            # (Includes duplicates at periodic boundaries)
            write_dump(Ellipsoids, sim_box)

        # Delete duplicates if existing (Active only if periodicity is True)
        # find all the items which are not duplicates
        inter_ell = [ell for ell in Ellipsoids if not isinstance(ell.id, str)]
        Ellipsoids = inter_ell
        
        if (not np.isclose(k_att, 0.0)) or (not np.isclose(k_rep, 0.0)):
            calculateForce(Ellipsoids, sim_box, periodicity,
                           k_rep=k_rep, k_att=k_att)

        dups = []
        ekin = 0.
        # Loop over the ellipsoids: move, set Bbox, & 
        # check for wall collision / PBC
        for ellipsoid in Ellipsoids:
            ekin += np.linalg.norm([ellipsoid.speedx, ellipsoid.speedy,
                                    ellipsoid.speedz])
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

def calculateForce(Ellipsoids, sim_box, periodicity, k_rep=0.0, k_att=0.0):
    """
    Calculate interaction force between ellipsoids.

    :param Ellipsoids: 
    :type Ellipsoids: 
    :param sim_box :
    :type sim_box:
    :param periodicity:
    :type periodicity:
    :param k_rep: optional, default is 0.0
    :type k_rep: float
    :param k_att: optional, default is 0.0
    :type k_att: float
    """

    w_half = sim_box.w/2           
    h_half = sim_box.h/2
    d_half = sim_box.d/2 
    
    
    for ell in Ellipsoids:       
        ell.force_x = 0.
        ell.force_y = 0.
        ell.force_z = 0.          
        for ell_n in Ellipsoids:
            if ell.id != ell_n.id:                
                dx = ell.x - ell_n.x
                dy = ell.y - ell_n.y
                dz = ell.z - ell_n.z  
                
                if periodicity == True:
                    if dx > w_half:
                        dx -= sim_box.w
                    if dx <= -w_half:
                        dx += sim_box.w
                    if dy > h_half:
                        dy -= sim_box.h
                    if dy <= -h_half:
                        dy += sim_box.h
                    if dz > d_half:
                        dz -= sim_box.d 
                    if dz <= -d_half:
                        dz += sim_box.d 
                        
                rSq = dx*dx + dy*dy + dz*dz
                r = np.sqrt(rSq)
                if np.isclose(r,0.):
                    continue
                r2inv = 1. / rSq
                
                # add repulsive or attractive force for dual phase systems
                if ell.phasenum == ell_n.phasenum:
                    Force = -k_rep * r2inv
                else:
                    Force = k_att * r2inv
                
                ell.force_x += Force * dx / r 
                ell.force_y += Force * dy / r
                ell.force_z += Force * dz / r
    return

def packingRoutine(particle_data, periodic, nsteps, sim_box,
                   k_rep=0.0, k_att=0.0, vf=None, save_files=False):
    """
    The main function that controls the particle packing routine using:
        :meth:`particle_grow` & :meth:`particle_generator`
    
    .. note:: Particle, RVE and simulation data are read from the JSON files
              generated by :meth:`kanapy.input_output.particleStatGenerator`.
              They contain the following information:
    
              * Ellipsoid attributes such as Major, Minor, Equivalent diameters
                and its tilt angle.  
              * RVE attributes such as RVE (Simulation domain) size, the number
                of voxels and the voxel resolution.  
              * Simulation attributes such as total number of timesteps and
                periodicity.                         
    """
    print('Starting particle simulation')
    
    print('    Creating particles from distribution statistics')
    # Create instances for particles
    Particles = particle_generator(particle_data, sim_box, periodic)
    
    # Growth of particle at each time step
    print('    Particle packing by growth simulation')
        
    particles, simbox = particle_grow(sim_box, Particles, periodic, \
                            nsteps,
                            k_rep=k_rep, k_att=k_att, vf=vf,
                            dump=save_files)
    
    # statistical evaluation of collisions
    if particles is not None:
        # check if particles are overlapping after growth
        ncoll = 0
        ekin0 = 0.
        ekin = 0.
        for E1 in particles:
            E1.ncollision = 0
            ekin0 += np.linalg.norm([E1.speedx0, E1.speedy0, E1.speedz0])
            ekin += np.linalg.norm([E1.speedx, E1.speedy, E1.speedz])
            for E2 in particles:
                if E1.id != E2.id:
                    # Distance between the centers of ellipsoids
                    dist = np.linalg.norm(np.subtract(E1.get_pos(),
                                                      E2.get_pos()))
                    # If the bounding spheres collide then check for collision
                    if dist <= (E1.a + E2.a):
                        # Check if ellipsoids overlap and update their speeds
                        # accordingly
                        if collide_detect(E1.get_coeffs(), E2.get_coeffs(),
                                    E1.get_pos(), E2.get_pos(), 
                                    E1.rotation_matrix, E2.rotation_matrix):
                            E1.ncollision += 1
                            ncoll += 1
        print('Completed particle packing')
        print(f'{ncoll} overlapping particles detected after packing')
        print(f'Kinetic energy of particles after packing: {ekin}')
        print(f'Initial kinetic energy: {ekin0}')
        print('')
    
    return particles, simbox
