import itertools
import numpy as np
import random
from .collisions import collision_routine
from scipy.spatial import Delaunay


def cub_oct_split(cub):
    """ 
    Splits cuboid object of the class :class:`~Cuboid` into eight smaller cuboid objects   

    :param cub: Branch cuboid object containing ellipsoids 
    :type cub: object of the class :class:`~Cuboid`
    :returns: Eight new sub-branch cuboid objects in a list
    :rtype: List 
    """
    w = cub.width / 2.0
    h = cub.height / 2.0
    d = cub.depth / 2.0

    cl = [Cuboid(cub.left, cub.top, cub.left + w,
                 cub.top + h, cub.front, cub.front + d), Cuboid(cub.left + w, cub.top, cub.left + 2. * w,
                                                                cub.top + h, cub.front, cub.front + d),
          Cuboid(cub.left, cub.top + h, cub.left + w,
                 cub.top + 2. * h, cub.front, cub.front + d), Cuboid(cub.left + w, cub.top + h, cub.left + 2. * w,
                                                                     cub.top + 2. * h, cub.front, cub.front + d),
          Cuboid(cub.left, cub.top, cub.left + w,
                 cub.top + h, cub.front + d, cub.front + 2. * d), Cuboid(cub.left + w, cub.top, cub.left + 2. * w,
                                                                         cub.top + h, cub.front + d,
                                                                         cub.front + 2. * d),
          Cuboid(cub.left, cub.top + h, cub.left + w,
                 cub.top + 2. * h, cub.front + d, cub.front + 2. * d),
          Cuboid(cub.left + w, cub.top + h, cub.left + 2. * w,
                 cub.top + 2. * h, cub.front + d, cub.front + 2. * d)]

    return cl


class Simulation_Box(object):
    """
    Represents a 3D simulation domain box

    This class defines the simulation box dimensions and boundaries,
    and keeps track of the simulation time step.

    Parameters
    ----------
    size : tuple of float
        Dimensions of the box as (width, height, depth).

    Attributes
    ----------
    w : float
        Width of the box.
    h : float
        Height of the box.
    d : float
        Depth of the box.
    sim_ts : int
        Current simulation time-step, initialized to 0.
    left : float
        Position of the left boundary, initialized to 0.
    top : float
        Position of the top boundary, initialized to 0.
    front : float
        Position of the front boundary, initialized to 0.
    right : float
        Position of the right boundary, initialized to width.
    bottom : float
        Position of the bottom boundary, initialized to height.
    back : float
        Position of the back boundary, initialized to depth.
    """

    def __init__(self, size):
        self.w = size[0]  # Width
        self.h = size[1]  # Height
        self.d = size[2]  # Depth
        self.sim_ts = 0  # Initialize simulation time-step
        self.left = 0.
        self.top = 0.
        self.front = 0.
        self.right = self.w
        self.bottom = self.h
        self.back = self.d


class Ellipsoid(object):
    """
    Creates Ellipsoid objects for each ellipsoid generated from input statistics

    Parameters
    ----------
    iden : int
        ID of the ellipsoid
    x : float
        X-coordinate of the ellipsoid center
    y : float
        Y-coordinate of the ellipsoid center
    z : float
        Z-coordinate of the ellipsoid center
    a : float
        Semi-major axis length of the ellipsoid
    b : float
        Semi-minor axis length of the ellipsoid
    c : float
        Semi-minor axis length of the ellipsoid
    quat : numpy.ndarray
        Quaternion representing the ellipsoid's orientation
    phasenum : int, optional
        Phase number of the ellipsoid, default is 0
    dup : optional
        Duplicate status used for voxelization, default is None
    points : list or None, optional
        Points used to create a Delaunay tessellation for the ellipsoid, default is None

    Attributes
    ----------
    id : int
        ID of the ellipsoid
    x, y, z : float
        Position of the ellipsoid center
    xold, yold, zold : float
        Previous positions
    a, b, c : float
        Semi-major and semi-minor axes lengths
    oria, orib, oric : float
        Original axes lengths
    speedx, speedy, speedz : float
        Velocity components, initialized to 0
    force_x, force_y, force_z : float
        Forces acting on the ellipsoid, initialized to 0
    rotation_matrix : ndarray
        Rotation matrix of the ellipsoid
    surface_points : list
        Surface points of the ellipsoid
    inside_voxels : list
        Voxels belonging to the ellipsoid
    duplicate : optional
        Duplicate status used for voxelization
    phasenum : int
        Phase number
    branches : list
        Branches associated with the ellipsoid
    neighborlist : set
        Neighbor ellipsoids
    ncollision : int
        Number of collisions
    inner : optional
        Delaunay tessellation of points if provided, else None

    Notes
    -----
    1. The orientation of each ellipsoid in the global coordinate space is defined by its
       tilt angle and axis vector, expressed in quaternion notation.

    2. Ellipsoids are initialized without a velocity; a random value is later assigned by
       `kanapy.packing.particle_generator`.

    3. An empty list for storing voxels belonging to the ellipsoid is initialized.
    """

    def __init__(self, iden, x, y, z, a, b, c, quat, phasenum=0, dup=None, points=None):
        self.id = iden
        self.x = x
        self.y = y
        self.z = z
        self.xold = x
        self.yold = y
        self.zold = z
        self.a = a
        self.b = b
        self.c = c
        self.quat = quat
        self.oria, self.orib, self.oric = a, b, c  # Store the original size of the particle
        self.speedx = 0.
        self.speedy = 0.
        self.speedz = 0.
        self.rotation_matrix = self.rotationMatrixGen()  # Initialize roatation matrix for the ellipsoid
        self.surface_points = self.surfacePointsGen()  # Initialize surface points for the ellipsoid
        self.inside_voxels = []  # List that stores voxels belonging to the ellipsoid
        self.set_cub()  # sets particle cuboid for collision testing with octree boxes
        self.duplicate = dup  # Duplicate status used for voxelization
        self.phasenum = phasenum
        self.force_x = 0.
        self.force_y = 0.
        self.force_z = 0.
        self.branches = []
        self.neighborlist = set()
        self.ncollision = 0
        if points is None:
            self.inner = None
        else:
            self.inner = self.create_poly(points)  # create a Delaunay tesselation of points

    def get_pos(self):
        """
        Return the current position of the ellipsoid

        Returns
        -------
        numpy.ndarray
            A 1D array [x, y, z] representing the current center coordinates of the ellipsoid.
        """
        return np.array([self.x, self.y, self.z])

    def get_coeffs(self):
        """
        Return the semi-axes coefficients of the ellipsoid

        Returns
        -------
        numpy.ndarray
            A 1D array [a, b, c] representing the semi-major and semi-minor axes lengths of the ellipsoid.
        """
        return np.array([self.a, self.b, self.c])

    def get_volume(self):
        """
        Return the volume of the ellipsoid

        Returns
        -------
        float
            The volume of the ellipsoid calculated as (4/3) * pi * a * b * c.
        """
        return (4 / 3) * np.pi * self.a * self.b * self.c

    def rotationMatrixGen(self):
        """
        Compute the rotation matrix of the ellipsoid from its quaternion

        Returns
        -------
        numpy.ndarray
            A 3x3 rotation matrix representing the ellipsoid's orientation.
        """

        FLOAT_EPS = np.finfo(float).resolution

        w, x, y, z = self.quat
        Nq = w * w + x * x + y * y + z * z
        if Nq < FLOAT_EPS:
            return np.eye(3)

        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z

        return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

    def surfacePointsGen(self, nang=20):
        """
        Generate points on the outer surface of the ellipsoid using its rotation matrix

        Parameters
        ----------
        nang : int, optional
            Number of points along each angular direction (default is 20).

        Returns
        -------
        numpy.ndarray
            An array of shape (nang*nang, 3) containing 3D coordinates of points on the ellipsoid surface.
        """
        # Points on the outer surface of Ellipsoid
        u = np.linspace(0, 2 * np.pi, nang)
        v = np.linspace(0, np.pi, nang)

        # Cartesian coordinates that correspond to the spherical angles:
        xval = self.a * np.outer(np.cos(u), np.sin(v))
        yval = self.b * np.outer(np.sin(u), np.sin(v))
        zval = self.c * np.outer(np.ones_like(u), np.cos(v))

        # combine the three 2D arrays element wise
        stacked_xyz = np.stack(
            (xval.ravel(), yval.ravel(), zval.ravel()), axis=1)

        # Do the dot product with rotation matrix
        return stacked_xyz.dot(self.rotation_matrix)

    def growth(self, fac):
        """
        Increase the size of the ellipsoid along its axes by a scaling factor

        Parameters
        ----------
        fac : float
            Scaling factor applied to the original axes lengths.
        """
        self.a = self.oria * fac
        self.b = self.orib * fac
        self.c = self.oric * fac
        self.set_cub()

    def Bbox(self):
        """
        Compute the axis-aligned bounding box of the ellipsoid using its surface points

        Sets the attributes:
        - bbox_xmin, bbox_xmax
        - bbox_ymin, bbox_ymax
        - bbox_zmin, bbox_zmax
        """
        # Add position vector
        self.surface_points = self.surfacePointsGen()
        new_surfPts = self.surface_points + self.get_pos()

        self.bbox_xmin, self.bbox_xmax = np.amin(
            new_surfPts[:, 0]), np.amax(new_surfPts[:, 0])
        self.bbox_ymin, self.bbox_ymax = np.amin(
            new_surfPts[:, 1]), np.amax(new_surfPts[:, 1])
        self.bbox_zmin, self.bbox_zmax = np.amin(
            new_surfPts[:, 2]), np.amax(new_surfPts[:, 2])

    def get_cub(self):
        """
        Return the cuboid object of the ellipsoid

        Returns
        -------
        Cuboid
            Cuboid object representing the ellipsoid's bounding cuboid
        """
        return self.cub

    def set_cub(self):
        """
        Initialize an object of the class Cuboid using the bounding box limits from Bbox

        This method calls the Bbox method to obtain the bounding box limits and then
        creates a Cuboid object using these limits
        """
        self.Bbox()
        self.cub = Cuboid(self.bbox_xmin, self.bbox_ymin, self.bbox_xmax,
                          self.bbox_ymax, self.bbox_zmin, self.bbox_zmax)

    def create_poly(self, points):
        """
        Create a polygon inside the ellipsoid

        This method applies a rotation to the input points using the object's rotation_matrix
        and computes the Delaunay triangulation to form a polygon

        Parameters
        ----------
        points : array_like
            Input points to construct the polygon

        Returns
        -------
        Delaunay
            A Delaunay triangulation object representing the polygon
        """
        return Delaunay(points.dot(self.rotation_matrix))

    def sync_poly(self, scale=None):
        """
        Move the center of the polygon to the center of the ellipsoid and scale the hull to fit inside the ellipsoid

        This method recenters the polygon points at the ellipsoid's center, applies rotation
        and scaling to fit the polygon inside the ellipsoid, and updates the Delaunay triangulation.

        Parameters
        ----------
        scale : float, optional
            Scaling factor to apply to the polygon. If None, the default poly_scale from kanapy is used.

        Notes
        -----
        The function assumes that self.inner contains a Delaunay object representing the polygon.
        If self.inner is None, the method does nothing.
        """
        if scale is None:
            from kanapy import poly_scale as scale
        if self.inner is None:
            return
        opts = self.inner.points
        Npts = len(opts)
        p_ctr = np.average(opts, axis=0)
        e_ctr = self.get_pos()
        pts = opts - p_ctr[None, :]  # shift center of points to origin
        pts = pts.dot(self.rotation_matrix.transpose())  # rotate points back into global coordinates
        ind0 = np.argmin(pts, axis=0)  # index of point with lowest coordinate for each Cartesian axis
        ind1 = np.argmax(pts, axis=0)  # index of point with highest coordinate for each Cartesian axis
        v_min = np.array([pts[i, j] for j, i in enumerate(ind0)])  # min. value for each Cartesian axis
        v_max = np.array([pts[i, j] for j, i in enumerate(ind1)])  # max. value for each Cartesian axis
        dim = v_max - v_min  # extension of polygon along each axis
        fac = scale * np.divide(np.array([self.a, self.b, self.c]), dim)  # scaling factors for each axis
        pts *= fac  # scale points to dimensions of ellipsoid
        pts = pts.dot(self.rotation_matrix)  # rotate back into ellipsoid frame
        pts += e_ctr[None, :]  # shift center to center of ellipsoid
        # update points Delaunay tesselation
        self.inner = Delaunay(pts)
        # check if surface points of ellipsoid are all outside polyhedron
        """self.surface_points = self.surfacePointsGen() + e_ctr[None, :]
        if any(self.inner.find_simplex(self.surface_points) >= 0):
            logging.error(f'Polyhedron too large for ellipsoid {self.id}. Reduce scale.')"""

    def move(self, dt):
        """
        Move the ellipsoid by updating its position vector using the Verlet integration method

        This method updates the ellipsoid's position and velocity components based on
        the current force and previous positions, and then updates the Cuboid object.

        Parameters
        ----------
        dt : float
            Time step used in the Verlet integration

        Notes
        -----
        The Cuboid object of the ellipsoid must be updated every time the ellipsoid moves
        """
        xx = 2.0 * self.x - self.xold + self.force_x * dt * dt
        yy = 2.0 * self.y - self.yold + self.force_y * dt * dt
        zz = 2.0 * self.z - self.zold + self.force_z * dt * dt
        self.speedx = (xx - self.xold) / (2.0 * dt)
        self.speedy = (yy - self.yold) / (2.0 * dt)
        self.speedz = (zz - self.zold) / (2.0 * dt)
        self.xold = self.x
        self.yold = self.y
        self.zold = self.z
        self.x = xx
        self.y = yy
        self.z = zz
        self.set_cub()

    def gravity_effect(self, value):
        """
        Move the ellipsoid downwards to mimic the effect of gravity acting on it

        Parameters
        ----------
        value : float
            User defined value for downward movement

        Notes
        -----
        The Cuboid object of the ellipsoid must be updated every time it moves
        """
        self.x += 0
        self.y -= value
        self.z += 0
        self.set_cub()

    def wallCollision(self, sim_box, periodicity):
        """
        Evaluate whether the ellipsoid collides with the boundaries of the simulation box

        This method checks if the ellipsoid intersects with the simulation box walls.
        If periodicity is enabled, duplicates of the ellipsoid are created on opposite faces.
        If periodicity is disabled, the ellipsoid bounces back from the walls.

        Parameters
        ----------
        sim_box : Simulation_Box
            Simulation box object containing boundary limits
        periodicity : bool
            Status of periodicity

        Returns
        -------
        list
            List of ellipsoid duplicates if periodicity is enabled, otherwise an empty list

        Notes
        -----
        The Cuboid object of the ellipsoid must be updated every time the ellipsoid moves
        """
        duplicates = []
        # for periodic boundaries
        if periodicity:
            # Check if particle's center is inside or outside the box
            if sim_box.right > self.x > sim_box.left and \
                    sim_box.bottom > self.y > sim_box.top and \
                    sim_box.back > self.z > sim_box.front:
                # If inside: Check which face the particle collides with
                left = self.bbox_xmin < sim_box.left
                top = self.bbox_ymin < sim_box.top
                front = self.bbox_zmin < sim_box.front
                right = self.bbox_xmax > sim_box.right
                bottom = self.bbox_ymax > sim_box.bottom
                back = self.bbox_zmax > sim_box.back

            else:
                # It is outside: Move the particle to the opposite side
                if self.x > sim_box.right:
                    diff = self.x - sim_box.right
                    self.x = sim_box.left + diff
                elif self.x < sim_box.left:
                    diff = abs(sim_box.left - self.x)
                    self.x = sim_box.right - diff

                if self.y > sim_box.bottom:
                    diff = self.y - sim_box.bottom
                    self.y = sim_box.top + diff
                elif self.y < sim_box.top:
                    diff = abs(sim_box.top - self.y)
                    self.y = sim_box.bottom - diff

                if self.z > sim_box.back:
                    diff = self.z - sim_box.back
                    self.z = sim_box.front + diff
                elif self.z < sim_box.front:
                    diff = abs(sim_box.front - self.z)
                    self.z = sim_box.back - diff

                self.set_cub()  # update the bounding box due to its movement

                # Now its inside: Check which face the particle collides with
                left = self.bbox_xmin < sim_box.left
                top = self.bbox_ymin < sim_box.top
                front = self.bbox_zmin < sim_box.front
                right = self.bbox_xmax > sim_box.right
                bottom = self.bbox_ymax > sim_box.bottom
                back = self.bbox_zmax > sim_box.back

            sim_width = abs(sim_box.right - sim_box.left)
            sim_height = abs(sim_box.bottom - sim_box.top)
            sim_depth = abs(sim_box.back - sim_box.front)

            # If it collides with any three faces: Create 7 duplicates
            if sum([left, top, right, bottom, front, back]) == 3:
                p1 = None
                if left and top and front:
                    p1 = Ellipsoid(str(self.id) + '_RTF', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_RBF', self.x + sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_LBF', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_LTB', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_RTB', self.x + sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_RBB', self.x + sim_width, self.y +
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_LBB', self.x, self.y +
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and top and front:
                    p1 = Ellipsoid(str(self.id) + '_RBF', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LBF', self.x - sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_LTF', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_RTB', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_RBB', self.x, self.y +
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_LBB', self.x - sim_width, self.y +
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_LTB', self.x - sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and bottom and front:
                    p1 = Ellipsoid(str(self.id) + '_LBF', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LTF', self.x - sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RTF', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_RBB', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_LBB', self.x - sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_LTB', self.x - sim_width, self.y -
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_RTB', self.x, self.y -
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif left and bottom and front:
                    p1 = Ellipsoid(str(self.id) + '_LTF', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_RTF', self.x + sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RBF', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_LBB', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_LTB', self.x, self.y -
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_RTB', self.x + sim_width, self.y -
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_RBB', self.x + sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif left and top and back:
                    p1 = Ellipsoid(str(self.id) + '_RTB', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_RBB', self.x + sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_LBB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_LTF', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_RTF', self.x + sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_RBF', self.x + sim_width, self.y +
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_LBF', self.x, self.y +
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and top and back:
                    p1 = Ellipsoid(str(self.id) + '_RBB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LBB', self.x - sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_LTB', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_RTF', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_RBF', self.x, self.y +
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_LBF', self.x - sim_width, self.y +
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_LTF', self.x - sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and bottom and back:
                    p1 = Ellipsoid(str(self.id) + '_LBB', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LTB', self.x - sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RTB', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_RBF', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_LBF', self.x - sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_LTF', self.x - sim_width, self.y -
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_RTF', self.x, self.y -
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif left and bottom and back:
                    p1 = Ellipsoid(str(self.id) + '_LTB', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_RTB', self.x + sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RBB', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id) + '_LBF', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id) + '_LTF', self.x, self.y -
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id) + '_RTF', self.x + sim_width, self.y -
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id) + '_RBF', self.x + sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.extend([p1, p2, p3, p4, p5, p6, p7])

            # If it collides with any two faces: Create 3 duplicates
            elif sum([left, top, right, bottom, front, back]) == 2:
                p1 = None
                if left and top:
                    p1 = Ellipsoid(str(self.id) + '_RT', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RB', self.x + sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and top:
                    p1 = Ellipsoid(str(self.id) + '_LT', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LB', self.x - sim_width, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif left and bottom:
                    p1 = Ellipsoid(str(self.id) + '_RB', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_LT', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_RT', self.x + sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif right and bottom:
                    p1 = Ellipsoid(str(self.id) + '_LB', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_RT', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_LT', self.x - sim_width, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif front and top:
                    p1 = Ellipsoid(str(self.id) + '_FB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_BT', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_BB', self.x, self.y +
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif front and bottom:
                    p1 = Ellipsoid(str(self.id) + '_FT', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_BT', self.x, self.y -
                                   sim_height, self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_BB', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif back and top:
                    p1 = Ellipsoid(str(self.id) + '_BB', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_FT', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_FB', self.x, self.y +
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif back and bottom:
                    p1 = Ellipsoid(str(self.id) + '_BT', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_FT', self.x, self.y -
                                   sim_height, self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_FB', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif front and right:
                    p1 = Ellipsoid(str(self.id) + '_FL', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_BR', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_BL', self.x - sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif front and left:
                    p1 = Ellipsoid(str(self.id) + '_FR', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_BL', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_BR', self.x + sim_width, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif back and right:
                    p1 = Ellipsoid(str(self.id) + '_BL', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_FR', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_FL', self.x - sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif back and left:
                    p1 = Ellipsoid(str(self.id) + '_BR', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id) + '_FL', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id) + '_FR', self.x + sim_width, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.extend([p1, p2, p3])

            # If it collides with any single face: Create 1 duplicate
            elif sum([left, top, right, bottom, front, back]) == 1:
                p1 = None
                if left:
                    p1 = Ellipsoid(str(self.id) + '_R', self.x + sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right:
                    p1 = Ellipsoid(str(self.id) + '_L', self.x - sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif bottom:
                    p1 = Ellipsoid(str(self.id) + '_T', self.x, self.y -
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif top:
                    p1 = Ellipsoid(str(self.id) + '_B', self.x, self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif front:
                    p1 = Ellipsoid(str(self.id) + '_Ba', self.x, self.y,
                                   self.z + sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                elif back:
                    p1 = Ellipsoid(str(self.id) + '_F', self.x, self.y,
                                   self.z - sim_depth, self.a, self.b, self.c, self.quat, dup=self.id,
                                   phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.append(p1)
        else:
            # no periodicity
            if self.bbox_xmin < sim_box.left:
                diff = sim_box.left - self.bbox_xmin
                # move the ellipsoid in opposite direction after bouncing
                self.x += diff
                self.xold = self.x

            if self.bbox_ymin < sim_box.top:
                diff = sim_box.top - self.bbox_ymin
                self.y += diff
                self.yold = self.y

            if self.bbox_zmin < sim_box.front:
                diff = sim_box.front - self.bbox_zmin
                self.z += diff
                self.zold = self.z

            if self.bbox_xmax > sim_box.right:
                diff = self.bbox_xmax - sim_box.right
                self.x -= diff
                self.xold = self.x

            if self.bbox_ymax > sim_box.bottom:
                diff = self.bbox_ymax - sim_box.bottom
                self.y -= diff
                self.yold = self.y

            if self.bbox_zmax > sim_box.back:
                diff = self.bbox_zmax - sim_box.back
                self.z -= diff
                self.zold = self.z
        return duplicates


class Cuboid(object):
    """
    Create Cuboid objects for ellipsoids and Octree sub-branches

    Parameters
    ----------
    left : float
        Bounding box minimum along x
    top : float
        Bounding box minimum along y
    right : float
        Bounding box maximum along x
    bottom : float
        Bounding box maximum along y
    front : float
        Bounding box minimum along z
    back : float
        Bounding box maximum along z
    """

    def __init__(self, left, top, right, bottom, front, back):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.front = front
        self.back = back

        self.width = abs(self.right - self.left)
        self.height = abs(self.bottom - self.top)
        self.depth = abs(self.back - self.front)

    def intersect(self, other):
        """
        Evaluate whether the Cuboid object of the ellipsoid intersects with the Cuboid object of the Octree sub-branch

        Parameters
        ----------
        other: Cuboid
            Sub-branch cuboid object of the octree

        Returns
        -------
        bool
            True if intersection occurs, False otherwise
        """

        # six conditions that guarantee non-overlapping of cuboids
        cond1 = self.left > other.right
        cond2 = self.right < other.left
        cond3 = self.top > other.bottom
        cond4 = self.bottom < other.top
        cond5 = self.front > other.back
        cond6 = self.back < other.front

        return not (cond1 or cond2 or cond3 or cond4 or cond5 or cond6)


class Octree(object):
    """
    Create an Octree object for tree trunk and its sub-branches

    Parameters
    ----------
    level: int
        Current level of the Octree, 0 for the trunk
    cub: Cuboid
        Cuboid object of the tree trunk or sub-branches, should be entire simulation box for the trunk
    particles: list
        List of particles within the tree trunk or sub-branches, contains all ellipsoids in the simulation domain for the trunk

    Notes
    -----
    1. level is set to zero for the trunk of the Octree
    2. cub should be entire simulation box for the tree trunk
    3. particles list contains all the ellipsoids in the simulation domain for the tree trunk
    """

    def __init__(self, level, cub, particles=[]):

        self.maxlevel = 3  # max number of subdivisions
        self.level = level  # current level of subdivision
        self.maxparticles = 10  # max number of particles without subdivision
        self.cub = cub  # cuboid object
        self.particles = particles  # list of particles
        self.branches = []  # empty list that is filled with 8 branches if subdivided

    def get_cub(self):
        """
        Return the cuboid object of the octree sub-branch

        Returns
        -------
        Cuboid
            Cuboid object corresponding to the octree sub-branch
        """
        return self.cub

    def subdivide(self):
        """
        Divide the Octree sub-branch into eight further sub-branches and initialize each as an Octree object

        Notes
        -----
        Each newly created sub-branch is appended to the branches list of the current Octree
        """
        for cub in cub_oct_split(self.cub):
            branch = Octree(self.level + 1, cub, [])
            self.branches.append(branch)

    def add_particle(self, particle):
        """
        Update the particle list of the Octree sub-branch

        Parameters
        ----------
        particle: object
            Particle object to be added to the sub-branch
        """
        self.particles.append(particle)

    def subdivide_particles(self):
        """
        Evaluate which ellipsoids belong to each Octree sub-branch by checking intersections

        Notes
        -----
        Calls the intersect method of each sub-branch cuboid with each particle cuboid
        and adds the particle to the sub-branch if an intersection occurs
        """
        for particle, branch in itertools.product(self.particles, self.branches):
            if branch.get_cub().intersect(particle.get_cub()):
                branch.add_particle(particle)

    def make_neighborlist(self):
        """
        Find the neighbor list for each particle in the Octree sub-branch

        Notes
        -----
        Initializes the neighborlist of each particle as a set and updates it with particles
        from all branches that the particle belongs to
        """
        for particle in self.particles:
            particle.neighborlist = set()
            for branch in particle.branches:
                particle.neighborlist.update(branch.particles)

    def collisionsTest(self):
        """
        Test for collisions between all ellipsoids in the Octree sub-branch

        Returns
        -------
        int
            Number of collisions detected

        Notes
        -----
        Uses bounding sphere check as a preliminary filter and then calls the
        collision_routine to determine actual overlaps. Updates the collision count
        for each ellipsoid
        """
        self.make_neighborlist()
        ncoll = 0
        count = 0
        colp = []
        ppar = []
        timecd = 0.
        timecr = 0.
        third = 1.0 / 3.0
        for E1 in self.particles:
            for E2 in E1.neighborlist:
                id1 = E1.id if E1.duplicate is None else (E1.duplicate + len(self.particles))
                id2 = E2.id if E2.duplicate is None else (E2.duplicate + len(self.particles))
                if id2 > id1:
                    # Distance between the centers of ellipsoids
                    dist = np.linalg.norm(np.subtract(E1.get_pos(), E2.get_pos()))
                    psize = 0.5 * (E1.get_volume() ** third + E2.get_volume() ** third)
                    # If the bounding spheres collide then check for collision
                    if dist <= psize:
                        # Check if ellipsoids overlap and update their speeds accordingly
                        count += 1
                        ppar.append([dist, psize])
                        if collision_routine(E1, E2):
                            colp.append([dist, psize])
                            ncoll += 1
                            E1.ncollision += 1
                            E2.ncollision += 1
        # print(f'Checked {count} pairs with {ncoll} collisions.')
        # print(f'collision time: {timecd}, force time: {timecr}')
        # print(f'Pair params:', ppar)
        # print(f'Collisions:', colp)
        return ncoll

    def update(self):
        """
        Update the Octree and begin recursive subdivision or particle assignment

        Notes
        -----
        If the number of particles exceeds the maximum allowed and the current
        level is below the maximum, the sub-branch is subdivided and particles
        are distributed among the new branches. Otherwise, particles are
        associated with this branch
        """
        if len(self.particles) > self.maxparticles and self.level <= self.maxlevel:
            self.subdivide()
            self.subdivide_particles()
            random.shuffle(self.branches)
            for branch in self.branches:
                branch.update()
        else:
            for particle in self.particles:
                particle.branches.append(self)
