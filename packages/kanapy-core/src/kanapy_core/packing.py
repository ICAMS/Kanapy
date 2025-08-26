# -*- coding: utf-8 -*-
import random
import numpy as np
from tqdm import tqdm
from .input_output import write_dump
from .entities import Ellipsoid, Cuboid, Octree
from .collisions import collision_routine


def particle_generator(particle_data, sim_box, poly):
    """
    Initializes ellipsoids by assigning them random positions and speeds within
    the simulation box.

    :param particle_data: Ellipsoid information such as Major, Minor,
          Equivalent diameters and its tilt angle. 
    :type particle_data: Python dictionary 
    :param sim_box: Simulation box representing RVE.
    :type sim_box: :class:`entities.Simulation_Box`
    :param poly: Points defining primitive polygon inside ellipsoid (optional).
    :type poly: :class: ndarry(N, 3)
    
    :returns: Ellipsoids for the packing routine
    :rtype: list       
    """
    Ellipsoids = []
    id_ctr = 0
    vell = 0.
    # introduce scaling factor to reduce particle overlap
    sf = 0.5
    for particle in particle_data:
        num_particles = particle['Number']  # Total number of ellipsoids
        for n in range(num_particles):
            iden = id_ctr + n + 1  # ellipsoid 'id'
            if particle['Type'] == 'Equiaxed':
                a = b = c = sf * particle['Equivalent_diameter'][n]
            else:
                a = sf * particle['Major_diameter'][n]  # Semi-major length
                b = sf * particle['Minor_diameter1'][n]  # Semi-minor length 1
                c = sf * particle['Minor_diameter2'][n]  # Semi-minor length 2

            # Random placement for ellipsoids
            x = random.uniform(a, sim_box.w - a)
            y = random.uniform(b, sim_box.h - b)
            z = random.uniform(c, sim_box.d - c)

            if particle['Type'] == 'Free':
                # For free particle definition, quaternion for rotation is given
                quat = particle['Quaternion'][n]
            else:
                # Angle represents inclination of Major axis w.r.t positive x-axis in xy-plane
                if particle['Type'] == 'Equiaxed':
                    angle = 0.5 * np.pi
                else:
                    # Extract the angle
                    angle = particle['Tilt angle'][n]
                # Tilt vector wrt (+ve) x-axis
                vec_a = np.array([a * np.cos(angle), a * np.sin(angle), 0.0])
                # Do the cross product to find the quaternion axis
                cross_a = np.cross(np.array([1, 0, 0]), vec_a)
                # norm of the vector (Magnitude)
                norm_cross_a = np.linalg.norm(cross_a, 2)
                quat_axis = cross_a / norm_cross_a  # normalize the quaternion axis

                # Find the quaternion components
                qx, qy, qz = quat_axis * np.sin(angle / 2)
                qw = np.cos(angle / 2)
                quat = np.array([qw, qx, qy, qz])

            # instance of Ellipsoid class
            ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat,
                                  phasenum=particle['Phase'], points=poly)
            ellipsoid.color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            vell += ellipsoid.get_volume()
            Ellipsoids.append(ellipsoid)  # adds ellipsoid to list
        id_ctr += num_particles
        print(f'    Total volume of generated ellipsoids: {vell}')

    return Ellipsoids


def particle_grow(sim_box, Ellipsoids, periodicity, nsteps,
                  k_rep=0.0, k_att=0.0, fill_factor=None,
                  dump=False, verbose=False):
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
    :param k_rep: Repulsion factor for particles
    :type k_rep: float
    :param k_att: Attraction factor for particles
    :type k_att: float
    :param fill_factor: Target volume fraction for particle filling
    :type fill_factor: float
    :param dump: Indicate if dump files for particles are written.
    :type dump: boolean
    :param verbose: Indicate if detailed output in iteration steps occurs
    :type verbose: bool


    .. note:: :meth:`kanapy.input_output.write_dump` function is called at each
              time step of the simulation to write output (.dump) files. 
              By default, periodic images are written to the output file, 
              but this option can be disabled within the function.         
    """

    def t_step(N):
        return K * N ** m

    def stop_part(ell):
        ell.speedx = 0.
        ell.speedy = 0.
        ell.speedz = 0.
        ell.force_x = 0.
        ell.force_y = 0.
        ell.force_z = 0
        ell.x *= 0.9
        ell.y *= 0.9
        ell.z *= 0.9
        ell.oldx = ell.x
        ell.oldy = ell.y
        ell.oldz = ell.z
        return

    if fill_factor is None:
        fill_factor = 0.65  # 65% should be largest packing density of ellipsoids
    # Reduce the volume of the particles to (1/nsteps)th of its original value
    end_step = int(fill_factor * nsteps) - 1  # grow particles only to given volume fraction
    m = -1 / 2.5
    K = 5
    Niter = 1
    time = 0
    while time < end_step:
        time += t_step(Niter)
        Niter += 1
    fac = nsteps ** (-1 / 3)
    vorig = 0.
    ve = 0.
    for ell in Ellipsoids:
        ell.a, ell.b, ell.c = ell.oria * fac, ell.orib * fac, ell.oric * fac
        ve += ell.get_volume()
        vorig += ell.oria * ell.orib * ell.oric * np.pi * 4 / 3
    dv = vorig / nsteps
    print(f'Volume of simulation box: {sim_box.w * sim_box.h * sim_box.d}')
    print(f'Volume of unscaled particles: {vorig}')
    print(f'Initial volume of scaled ellipsoids: {ve}, targeted final volume: {ve + end_step * dv}')
    print(f'Volume increment per time step: {dv}')

    # Simulation loop for particle growth and interaction steps
    damping = 0.4
    ncol = 0
    ndump = np.maximum(int(Niter / 1000), 1)
    time = 0
    for i in tqdm(range(1, Niter)):
        dt = t_step(i)
        time += dt
        # Initialize Octree and test for collision between ellipsoids

        ekin = 0.
        for ell in Ellipsoids:
            # apply damping force opposite to current movement
            ell.force_x = -damping * ell.speedx / dt
            ell.force_y = -damping * ell.speedy / dt
            ell.force_z = -damping * ell.speedz / dt
            ekin += ell.speedx ** 2 + ell.speedy ** 2 + ell.speedz ** 2
            ell.ncollision = 0
            ell.branches = []

        tree = Octree(0, Cuboid(sim_box.left, sim_box.top, sim_box.right,
                      sim_box.bottom, sim_box.front, sim_box.back), Ellipsoids)
        if (not np.isclose(k_att, 0.0)) or (not np.isclose(k_rep, 0.0)):
            calculateForce(Ellipsoids, sim_box, periodicity,
                           k_rep=k_rep, k_att=k_att)
        tree.update()
        #print('Tree updated', i, time)
        nc = tree.collisionsTest()
        #print('Collision Test done', nc)
        ncol += nc
        if verbose and i % 100 == 0:
            print(f'Total time {time:.1f}/{end_step} | iteration {i}/{Niter} | '
                  f'collisions in last period: {ncol} | time step: {dt:.5f} | '
                  f'kinetic energy: {ekin}')
            ncol = 0
            for ell in Ellipsoids:
                speed = np.sqrt(ell.speedx ** 2 + ell.speedy ** 2 + ell.speedz ** 2)
                if speed > 2.:
                    print('Fast particle:', ell.id, speed, ell.x, ell.y, ell.z)
                    # stop fast particle and move it closer to the center
                    stop_part(ell)
                    if ell.duplicate is not None:
                        stop_part([ell.duplicate])

        if dump and (i - 1) % ndump == 0:
            # Dump the ellipsoid information to be read by OVITO 
            # (Includes duplicates at periodic boundaries)
            write_dump(Ellipsoids, sim_box)

        # Delete duplicates if existing (Active only if periodicity is True)
        # find all the items which are not duplicates
        inter_ell = [ell for ell in Ellipsoids if not isinstance(ell.id, str)]
        Ellipsoids = inter_ell

        dups = []
        # Loop over the ellipsoids: move, set Bbox, & 
        # check for wall collision / PBC
        sc_fac = (time / nsteps) ** (1 / 3)  # scale factor for semi-axes during growth
        #print('Loop over ellipsoids:', i, time)
        for ellipsoid in Ellipsoids:
            # grow the ellipsoid
            ellipsoid.growth(sc_fac)
            # Move the ellipsoid according to collision status
            ellipsoid.move(dt)
            # Check for wall collision or create duplicates
            ell_dups = ellipsoid.wallCollision(sim_box, periodicity)
            dups.extend(ell_dups)
            # Update the BBox of the ellipsoid
            ellipsoid.set_cub()
        #print('Done:', i, time)
        # Update the actual list with duplicates
        Ellipsoids.extend(dups)

        # Update the simulation time
        sim_box.sim_ts += 1
    ve = 0.
    for ell in Ellipsoids:
        if type(ell.id) is int:
            ve += ell.get_volume()
    print(f'Actual final volume of ellipsoids: {ve}')
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

    w_half = sim_box.w / 2
    h_half = sim_box.h / 2
    d_half = sim_box.d / 2

    for ell in Ellipsoids:
        for ell_n in ell.neighborlist:
            if ell.id != ell_n.id:
                dx = ell.x - ell_n.x
                dy = ell.y - ell_n.y
                dz = ell.z - ell_n.z

                if periodicity:
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

                rSq = dx * dx + dy * dy + dz * dz
                r = np.sqrt(rSq)
                if np.isclose(r, 0.):
                    continue
                r2inv = 1. / rSq

                # add repulsive or attractive force for dual phase systems
                if ell.phasenum == ell_n.phasenum:
                    Force = k_rep * r2inv
                else:
                    Force = -k_att * r2inv

                ell.force_x += Force * dx / r
                ell.force_y += Force * dy / r
                ell.force_z += Force * dz / r
    return


def packingRoutine(particle_data, periodic, nsteps, sim_box,
                   k_rep=0.0, k_att=0.0, fill_factor=None, poly=None,
                   save_files=False, verbose=False):
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
    Particles = particle_generator(particle_data, sim_box, poly)

    # Growth of particle at each time step
    print('    Particle packing by growth simulation')

    particles, simbox = particle_grow(sim_box, Particles, periodic,
                                            nsteps,
                                            k_rep=k_rep, k_att=k_att, fill_factor=fill_factor,
                                            dump=save_files, verbose=verbose)

    # statistical evaluation of collisions
    if particles is not None:
        # check if particles are overlapping after growth
        ncoll = 0
        ekin = 0.
        for E1 in particles:
            E1.ncollision = 0
            ekin += E1.speedx ** 2 + E1.speedy ** 2 + E1.speedz ** 2
            for E2 in particles:
                id1 = E1.id if E1.duplicate is None else (E1.duplicate + len(particles))
                id2 = E2.id if E2.duplicate is None else (E2.duplicate + len(particles))
                if id2 > id1:
                    # Distance between the centers of ellipsoids
                    dist = np.linalg.norm(np.subtract(E1.get_pos(),
                                                      E2.get_pos()))
                    # If the bounding spheres collide then check for collision
                    if dist <= np.max([E1.a, E1.b, E1.c]) + np.max([E2.a, E2.b, E2.c]):
                        # Check if ellipsoids overlap and update their speeds
                        # accordingly
                        if collision_routine(E1, E2):
                            E1.ncollision += 1
                            E2.ncollision += 1
                            ncoll += 1
        print('Completed particle packing')
        print(f'{ncoll} overlapping particles detected after packing')
        print(f'Kinetic energy of particles after packing: {ekin}')
        print('')

    return particles, simbox
