# -*- coding: utf-8 -*-
import random
import numpy as np
from tqdm import tqdm
from .input_output import write_dump
from .entities import Ellipsoid, Cuboid, Octree
from .collisions import collision_routine


def particle_generator(particle_data, sim_box, poly):
    """
    Initialize a list of ellipsoids by assigning random positions, orientations,
    and sizes within the given simulation box.

    Parameters
    ----------
    particle_data : dict
        Dictionary containing ellipsoid properties such as major, minor, and
        equivalent diameters, tilt angles, and number of particles per phase.
    sim_box : entities.Simulation_Box
        Simulation box object defining the representative volume element (RVE)
        dimensions (width, height, depth).
    poly : ndarray of shape (N, 3), optional
        Points defining a primitive polygon inside the ellipsoid. Used for
        geometry construction.

    Returns
    -------
    list
        List of `Ellipsoid` objects initialized for the packing routine.

    Notes
    -----
    The function introduces a scaling factor (0.5) to reduce overlap among
    generated ellipsoids. Ellipsoids are assigned random colors and positions
    within the simulation box.
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
    Perform recursive particle growth and collision detection within the
    simulation box. Initializes an :class:`entities.Octree` instance and
    progressively grows ellipsoids while handling inter-particle and
    wall collisions.

    Parameters
    ----------
    sim_box : entities.Simulation_Box
        Simulation box representing the representative volume element (RVE).
    Ellipsoids : list
        List of `Ellipsoid` objects used in the packing routine.
    periodicity : bool
        Whether periodic boundary conditions are applied.
    nsteps : int
        Total number of simulation steps for filling the box with particles.
    k_rep : float, optional
        Repulsion factor for particle interactions (default: 0.0).
    k_att : float, optional
        Attraction factor for particle interactions (default: 0.0).
    fill_factor : float, optional
        Target volume fraction for particle filling (default: 0.65).
    dump : bool, optional
        If True, dump files for particles are written at intervals (default: False).
    verbose : bool, optional
        If True, print detailed information at iteration steps (default: False).

    Returns
    -------
    Ellipsoids : list
        Updated list of `Ellipsoid` objects after particle growth.
    sim_box : entities.Simulation_Box
        Updated simulation box object after the packing process.

    Notes
    -----
    The function calls :func:`kanapy.input_output.write_dump` at each time step
    to write `.dump` files. Periodic images are written by default but can be
    disabled inside that function.
    """

    def t_step(N):
        """
        Compute the adaptive time step based on the current iteration number

        The time step decreases as the iteration progresses, ensuring smoother
        particle growth and improved numerical stability near the final packing
        stage.

        Parameters
        ----------
        N : int
            Current iteration number

        Returns
        -------
        float
            Computed time step for the given iteration

        Notes
        -----
        The function uses the relation `t_step = K * N ** m`, where `K` and `m`
        are constants defined in the outer scope of the `particle_grow` function.
        A negative exponent `m` results in smaller time steps at later iterations.
        """
        return K * N ** m

    def stop_part(ell):
        """
        Stop the motion of a given ellipsoid and slightly shrink its position

        Sets all velocity and force components of the ellipsoid to zero, reduces
        its position coordinates by 10%, and updates the old position attributes
        to match the new coordinates.

        Parameters
        ----------
        ell : Ellipsoid
            The ellipsoid instance whose motion is to be stopped

        Returns
        -------
        None
            The function modifies the ellipsoid in-place and does not return a value

        Notes
        -----
        This function is used during the particle growth simulation to prevent
        fast-moving or unstable ellipsoids from causing numerical issues.
        """
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
    Calculate the interaction forces between ellipsoids within the simulation box

    Computes pairwise repulsive or attractive forces based on distance and phase
    between ellipsoids. Applies periodic boundary conditions if enabled.

    Parameters
    ----------
    Ellipsoids : list of Ellipsoid
        List of ellipsoid instances for which forces are calculated
    sim_box : Simulation_Box
        Simulation box representing the RVE dimensions
    periodicity : bool
        If True, applies periodic boundary conditions
    k_rep : float, optional
        Repulsion factor for particles of the same phase (default is 0.0)
    k_att : float, optional
        Attraction factor for particles of different phases (default is 0.0)
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
    Perform particle packing routine using particle generation and growth simulation

    The function first generates ellipsoids from the provided particle statistics
    and then grows them over time steps while handling collisions and interactions.
    Uses :meth:`particle_generator` to create particles and :meth:`particle_grow`
    to grow and settle them.

    Parameters
    ----------
    particle_data : list of dict
        Statistics describing particle attributes for generation
    periodic : bool
        Flag indicating if periodic boundary conditions are applied
    nsteps : int
        Total number of timesteps for the growth simulation
    sim_box : Simulation_Box
        Simulation box representing the RVE dimensions
    k_rep : float, optional
        Repulsion factor for same-phase particles (default 0.0)
    k_att : float, optional
        Attraction factor for different-phase particles (default 0.0)
    fill_factor : float, optional
        Target volume fraction for particle filling (default None, uses 0.65)
    poly : ndarray, optional
        Points defining a primitive polygon inside ellipsoids (default None)
    save_files : bool, optional
        Whether to save dump files during simulation (default False)
    verbose : bool, optional
        If True, prints detailed simulation output (default False)

    Returns
    -------
    particles : list
        List of generated and packed ellipsoids
    simbox : Simulation_Box
        Updated simulation box with current timestep information

    Notes
    -----
    Particle, RVE, and simulation data are read from JSON files generated by
    :meth:`kanapy.input_output.particleStatGenerator`. These files contain:

    * Ellipsoid attributes such as Major, Minor, Equivalent diameters, and tilt angles
    * RVE attributes such as simulation domain size, number of voxels, and voxel resolution
    * Simulation attributes such as total number of timesteps and periodicity
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
