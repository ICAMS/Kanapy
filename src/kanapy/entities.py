# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import random

from kanapy.collision_detect_react import collision_routine


def cub_oct_split(cub):
    """ 
    Splits cuboid object of the class :class:`~Cuboid` into eight smaller cuboid objects   

    :param cub: Branch cuboid object containing ellipsoids 
    :type cub: object of the class :class:`~Cuboid`
    :returns: Eight new sub-branch cuboid objects in a list
    :rtype: List 
    """
    w = cub.width/2.0
    h = cub.height/2.0
    d = cub.depth/2.0

    cl = []
    cl.append(Cuboid(cub.left,   cub.top,   cub.left + w,
                     cub.top + h,    cub.front,   cub.front + d))
    cl.append(Cuboid(cub.left+w, cub.top,   cub.left+2.*w,
                     cub.top + h,    cub.front,   cub.front + d))
    cl.append(Cuboid(cub.left,   cub.top+h, cub.left + w,
                     cub.top + 2.*h, cub.front,   cub.front + d))
    cl.append(Cuboid(cub.left+w, cub.top+h, cub.left+2.*w,
                     cub.top+2.*h,   cub.front,   cub.front + d))
    cl.append(Cuboid(cub.left,   cub.top,   cub.left + w,
                     cub.top + h,    cub.front+d, cub.front + 2.*d))
    cl.append(Cuboid(cub.left+w, cub.top,   cub.left+2.*w,
                     cub.top + h,    cub.front+d, cub.front + 2.*d))
    cl.append(Cuboid(cub.left,   cub.top+h, cub.left + w,
                     cub.top + 2.*h, cub.front+d, cub.front + 2.*d))
    cl.append(Cuboid(cub.left+w, cub.top+h, cub.left+2.*w,
                     cub.top+2.*h,   cub.front+d, cub.front + 2.*d))

    return cl


class Simulation_Box(object):
    """
    Creates :class:`~Simulation_Box` objects for the defined simulation domain.

    :param w: width
    :param h: height
    :param d: depth of the simulation domain
    :type w: float
    :type h: float
    :type d: float   
    """

    def __init__(self, w, h, d):
        self.w = w           # Width
        self.h = h           # Height
        self.d = d           # Depth
        self.sim_ts = 0      # Initialize simulation time-step
        self.left = 0
        self.top = 0
        self.front = 0
        self.right = w
        self.bottom = h
        self.back = d


class Ellipsoid(object):
    """
    Creates :class:`~Ellipsoid` objects for each ellipsoid generated from input statistics. 

    :param iden: ID of the ellipsoid
    :type iden: integer
    :param center: Position :math:`(x, y, z)` of the ellipsoid center in the simulation domain      
    :type center: floats    
    :param coefficient: Semi-major and semin-minor axes lengths :math:`(a, b, c)` of the ellipsoid    
    :type coefficient: floats   
    :param quat: Quaternion representing ellipsoid's axis and tilt angle with respect 
                 to the positive x-axis       
    :type quat: numpy array    

    .. note:: 1. The orientations of ellipsoid :math:`i` in the global coordinate space is defined by its 
                 tilt angle and axis vector and expressed in quaternion notation as,

                 .. image:: /figs/quaternion_ell.png                        
                    :width: 210px
                    :height: 40px
                    :align: center   

              2. Ellipsoids are initilaized without a value for its velocity,
                 and is later assigned a random value by :mod:`kanapy.packing.particle_generator`.

              3. An empty list for storing voxels belonging to the ellipsoid is initialized.                   
    """

    def __init__(self, iden, x, y, z, a, b, c, quat, phasenum=0, dup=None):
        self.id = iden
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.quat = quat        
        self.oria, self.orib, self.oric = a, b, c   # Store the original size of the particle 
        self.speedx0 = 0.
        self.speedy0 = 0.
        self.speedz0 = 0.
        self.speedx = 0.
        self.speedy = 0.
        self.speedz = 0.
        self.rotationMatrixGen()   # Initialize roatation matrix for the ellipsoid
        self.surfacePointsGen()    # Initialize surface points for the ellipsoid
        self.inside_voxels = []    # List that stores voxels belonging to the ellipsoid
        self.set_cub()             # sets particle cuboid for collision testing with octree boxes
        self.duplicate = dup      # Duplicate status used for voxelization
        self.phasenum = phasenum
        self.force_x = 0.
        self.force_y = 0.
        self.force_z = 0.
        self.q = 2*(self.phasenum)-1
        self.branches = []
        self.neighborlist = set()
        
    def get_pos(self):
        """
        Returns the position array of the ellipsoid

        :rtype: numpy array
        """
        return np.array([self.x, self.y, self.z])

    def get_coeffs(self):
        """
        Returns the coefficients array of the ellipsoid

        :rtype: numpy array
        """
        return np.array([self.a, self.b, self.c])

    def get_volume(self):
        """
        Returns the volume of the ellipsoid

        :rtype: float
        """
        return (4/3)*np.pi*self.a*self.b*self.c

    def rotationMatrixGen(self):
        """
        Evaluates the rotation matrix for the ellipsoid using the quaternion

        :rtype: numpy array
        """

        FLOAT_EPS = np.finfo(float).eps

        w, x, y, z = self.quat
        Nq = w*w + x*x + y*y + z*z

        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X
        wY = w*Y
        wZ = w*Z
        xX = x*X
        xY = x*Y
        xZ = x*Z
        yY = y*Y
        yZ = y*Z
        zZ = z*Z

        if Nq < FLOAT_EPS:
            self.rotation_matrix = np.eye(3)
        else:
            self.rotation_matrix = np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                                             [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                                             [xZ-wY, yZ+wX, 1.0-(xX+yY)]])

        # Rotation matrix has to be transposed as OVITO uses the transposed matrix for visualization.
        #self.rotation_matrix = self.rotation_matrix.T          # Not required, it's consistent!!!

    def surfacePointsGen(self):
        """
        Generates points on the outer surface of the ellipsoid using the rotation matrix from :meth:`rotationMatrixGen`

        :rtype: numpy array
        """
        # Points on the outer surface of Ellipsoid
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        # Cartesian coordinates that correspond to the spherical angles:
        xval = self.a * np.outer(np.cos(u), np.sin(v))
        yval = self.b * np.outer(np.sin(u), np.sin(v))
        zval = self.c * np.outer(np.ones_like(u), np.cos(v))

        # combine the three 2D arrays element wise
        stacked_xyz = np.stack(
            (xval.ravel(), yval.ravel(), zval.ravel()), axis=1)

        # Do the dot product with rotation matrix
        self.surface_points = stacked_xyz.dot(self.rotation_matrix)

    def growth(self, factor):
        """
        Increases the size of the ellipsoid along its axes governed by the user defined value

        :param factor: Parameter that controls the total simulation time 
        :type factor: float

        .. note:: Increment value is determined as the ratio of particle size to the total simulation time       
        """
        self.a += self.oria/factor
        self.b += self.orib/factor
        self.c += self.oric/factor
        self.set_cub()

    def Bbox(self):
        """
        Generates the bounding box limits along x, y, z directions using the surface points from :meth:`surfacePointsGen`

        :rtype: numpy array
        """
        # Add position vector
        self.surfacePointsGen()
        new_surfPts = self.surface_points + self.get_pos()

        self.bbox_xmin, self.bbox_xmax = np.amin(
            new_surfPts[:, 0]), np.amax(new_surfPts[:, 0])
        self.bbox_ymin, self.bbox_ymax = np.amin(
            new_surfPts[:, 1]), np.amax(new_surfPts[:, 1])
        self.bbox_zmin, self.bbox_zmax = np.amin(
            new_surfPts[:, 2]), np.amax(new_surfPts[:, 2])

    def get_cub(self):
        """
        Returns the cuboid object of the ellipsoid

        :rtype: object of the class :class:`~Cuboid` 
        """
        return self.cub

    def set_cub(self):
        """
        Initializes an object of the class :class:`~Cuboid` using the bounding box limits from :meth:`Bbox`         
        """
        self.Bbox()
        self.cub = Cuboid(self.bbox_xmin, self.bbox_ymin, self.bbox_xmax,
                          self.bbox_ymax, self.bbox_zmin, self.bbox_zmax)

    def move(self):
        """
        Moves the ellipsoid by updating its position vector according to Eulerian integration method 

        .. note:: The :class:`~Cuboid` object of the ellipsoid has to be updated everytime it moves
        """
        self.x += self.speedx
        self.y += self.speedy
        self.z += self.speedz
        
        self.speedx += self.force_x
        self.speedy += self.force_y 
        self.speedz += self.force_z 
        self.set_cub()

    def gravity_effect(self, value):
        """
        Moves the ellipsoid downwards to mimick the effect of gravity acting on it  

        :param value: User defined value for downward movement 
        :type value: float

        .. note:: The :class:`~Cuboid` object of the ellipsoid has to be updated everytime it moves
        """
        self.x += 0
        self.y -= value
        self.z += 0
        self.set_cub()
        
    def wallCollision(self, sim_box, periodicity):
        """
        Evaluates whether the ellipsoid collides with the boundaries of the simulation box.

        * If periodicity is enabled -> Creates duplicates of the ellipsoid on opposite faces of the box
        * If periodicity is disabled -> Mimicks the bouncing back effect.

        :param sim_box: Simulation box
        :type sim_box: object of the class :class:`~Simulation_Box`
        :param periodicity: Status of periodicity
        :type periodicity: boolean
        :returns: if periodic - ellipsoid duplicates, else None
        :rtype: list

        .. note:: The :class:`~Cuboid` object of the ellipsoid has to be updated everytime it moves
        """
        duplicates = []
        # for periodic boundaries
        if periodicity:
            # Check if particle's center is inside or outside the box
            if sim_box.right > self.x > sim_box.left and sim_box.bottom > self.y > sim_box.top and sim_box.back > self.z > sim_box.front:
                # If inside: Check which face the particle collides with
                left = True if self.bbox_xmin < sim_box.left else False
                top = True if self.bbox_ymin < sim_box.top else False
                front = True if self.bbox_zmin < sim_box.front else False
                right = True if self.bbox_xmax > sim_box.right else False
                bottom = True if self.bbox_ymax > sim_box.bottom else False
                back = True if self.bbox_zmax > sim_box.back else False
                
            else:
                # Its outside: Move the particle to the opposite side
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

                self.set_cub()             # update the bounding box due to its movement

                # Now its inside: Check which face the particle collides with
                left = True if self.bbox_xmin < sim_box.left else False
                top = True if self.bbox_ymin < sim_box.top else False
                front = True if self.bbox_zmin < sim_box.front else False
                right = True if self.bbox_xmax > sim_box.right else False
                bottom = True if self.bbox_ymax > sim_box.bottom else False
                back = True if self.bbox_zmax > sim_box.back else False

            sim_width = abs(sim_box.right - sim_box.left)
            sim_height = abs(sim_box.bottom - sim_box.top)
            sim_depth = abs(sim_box.back - sim_box.front)
            
            # If it collides with any three faces: Create 7 duplicates
            if sum([left, top, right, bottom, front, back]) == 3:
                p1 = None
                if left and top and front:
                    p1 = Ellipsoid(str(self.id)+'_RTF', self.x+sim_width, self.y,
                                   self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_RBF', self.x+sim_width, self.y +
                                   sim_height, self.z,self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_LBF', self.x,           self.y +
                                   sim_height, self.z, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_LTB', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_RTB', self.x+sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_RBB', self.x+sim_width, self.y +
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_LBB', self.x,           self.y +
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and top and front:
                    p1 = Ellipsoid(str(self.id)+'_RBF', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LBF', self.x-sim_width, self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_LTF', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_RTB', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_RBB', self.x,           self.y +
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_LBB', self.x-sim_width, self.y +
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_LTB', self.x-sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and bottom and front:
                    p1 = Ellipsoid(str(self.id)+'_LBF', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LTF', self.x-sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RTF', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_RBB', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_LBB', self.x-sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_LTB', self.x-sim_width, self.y -
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_RTB', self.x,           self.y -
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif left and bottom and front:
                    p1 = Ellipsoid(str(self.id)+'_LTF', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_RTF', self.x+sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RBF', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_LBB', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_LTB', self.x,           self.y -
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_RTB', self.x+sim_width, self.y -
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_RBB', self.x+sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif left and top and back:
                    p1 = Ellipsoid(str(self.id)+'_RTB', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_RBB', self.x+sim_width, self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_LBB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_LTF', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_RTF', self.x+sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_RBF', self.x+sim_width, self.y +
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_LBF', self.x,           self.y +
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and top and back:
                    p1 = Ellipsoid(str(self.id)+'_RBB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LBB', self.x-sim_width, self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_LTB', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_RTF', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_RBF', self.x,           self.y +
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_LBF', self.x-sim_width, self.y +
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_LTF', self.x-sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and bottom and back:
                    p1 = Ellipsoid(str(self.id)+'_LBB', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LTB', self.x-sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RTB', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_RBF', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_LBF', self.x-sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_LTF', self.x-sim_width, self.y -
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_RTF', self.x,           self.y -
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif left and bottom and back:
                    p1 = Ellipsoid(str(self.id)+'_LTB', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_RTB', self.x+sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RBB', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p4 = Ellipsoid(str(self.id)+'_LBF', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p5 = Ellipsoid(str(self.id)+'_LTF', self.x,           self.y -
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p6 = Ellipsoid(str(self.id)+'_RTF', self.x+sim_width, self.y -
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p7 = Ellipsoid(str(self.id)+'_RBF', self.x+sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.extend([p1, p2, p3, p4, p5, p6, p7])

            # If it collides with any two faces: Create 3 duplicates
            elif sum([left, top, right, bottom, front, back]) == 2:
                p1 = None
                if left and top:
                    p1 = Ellipsoid(str(self.id)+'_RT', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RB', self.x+sim_width, self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and top:
                    p1 = Ellipsoid(str(self.id)+'_LT', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LB', self.x-sim_width, self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif left and bottom:
                    p1 = Ellipsoid(str(self.id)+'_RB', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_LT', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_RT', self.x+sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right and bottom:
                    p1 = Ellipsoid(str(self.id)+'_LB', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_RT', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_LT', self.x-sim_width, self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif front and top:
                    p1 = Ellipsoid(str(self.id)+'_FB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_BT', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_BB', self.x,           self.y +
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif front and bottom:
                    p1 = Ellipsoid(str(self.id)+'_FT', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_BT', self.x,           self.y -
                                   sim_height, self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_BB', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif back and top:
                    p1 = Ellipsoid(str(self.id)+'_BB', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_FT', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_FB', self.x,           self.y +
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif back and bottom:
                    p1 = Ellipsoid(str(self.id)+'_BT', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_FT', self.x,           self.y -
                                   sim_height, self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_FB', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif front and right:
                    p1 = Ellipsoid(str(self.id)+'_FL', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_BR', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_BL', self.x-sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif front and left:
                    p1 = Ellipsoid(str(self.id)+'_FR', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_BL', self.x,           self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_BR', self.x+sim_width, self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif back and right:
                    p1 = Ellipsoid(str(self.id)+'_BL', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_FR', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_FL', self.x-sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif back and left:
                    p1 = Ellipsoid(str(self.id)+'_BR', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p2 = Ellipsoid(str(self.id)+'_FL', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                    p3 = Ellipsoid(str(self.id)+'_FR', self.x+sim_width, self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.extend([p1, p2, p3])

            # If it collides with any single face: Create 1 duplicate
            elif sum([left, top, right, bottom, front, back]) == 1:
                p1 = None
                if left:
                    p1 = Ellipsoid(str(self.id)+'_R', self.x+sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif right:
                    p1 = Ellipsoid(str(self.id)+'_L', self.x-sim_width, self.y,
                                   self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif bottom:
                    p1 = Ellipsoid(str(self.id)+'_T', self.x,           self.y -
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif top:
                    p1 = Ellipsoid(str(self.id)+'_B', self.x,           self.y +
                                   sim_height, self.z,           self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif front:
                    p1 = Ellipsoid(str(self.id)+'_Ba', self.x,          self.y,
                                   self.z+sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                elif back:
                    p1 = Ellipsoid(str(self.id)+'_F', self.x,           self.y,
                                   self.z-sim_depth, self.a, self.b, self.c, self.quat, dup=self.id, phasenum=self.phasenum)
                if p1 is not None:
                    duplicates.append(p1)
        else:
            # no periodicity
            if self.bbox_xmin < sim_box.left:
                diff = sim_box.left - self.bbox_xmin
                # move the ellipsoid in opposite direction after bouncing
                self.x += diff
                self.speedx *= -1

            if self.bbox_ymin < sim_box.top:
                diff = sim_box.top - self.bbox_ymin
                self.y += diff
                self.speedy *= -1

            if self.bbox_zmin < sim_box.front:
                diff = sim_box.front - self.bbox_zmin
                self.z += diff
                self.speedz *= -1

            if self.bbox_xmax > sim_box.right:
                diff = self.bbox_xmax - sim_box.right
                self.x -= diff
                self.speedx *= -1

            if self.bbox_ymax > sim_box.bottom:
                diff = self.bbox_ymax - sim_box.bottom
                self.y -= diff
                self.speedy *= -1

            if self.bbox_zmax > sim_box.back:
                diff = self.bbox_zmax - sim_box.back
                self.z -= diff
                self.speedz *= -1
        return duplicates

class Cuboid(object):
    """
    Creates :class:`Cuboid` objects for ellipsoids and Octree sub-branches.

    :param left: Bounding box minimum along x
    :param top: Bounding box minimum along y
    :param right: Bounding box maximum along x
    :param bottom: Bounding box maximum along y
    :param front: Bounding box minimum along z
    :param back: Bounding box maximum along z
    :type left: float
    :type top: float
    :type right: float
    :type bottom: float
    :type front: float
    :type back: float   
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
        Evaluates whether the :class:`Cuboid` object of the ellipsoid intersects with the :class:`Cuboid` object of the :class:`Octree` sub-branch.

        :param other: Sub-branch cuboid object of the octree
        :type other: object of the class :class:`Cuboid`
        :returns: if intersection - **True**, else **False**
        :rtype: boolean
        """

        # six conditions that guarantee non-overlapping of cuboids
        cond1 = self.left > other.right
        cond2 = self.right < other.left
        cond3 = self.top > other.bottom
        cond4 = self.bottom < other.top
        cond5 = self.front > other.back
        cond6 = self.back < other.front

        if cond1 or cond2 or cond3 or cond4 or cond5 or cond6:
            return False
        return True


class Octree(object):
    """
    Creates :class:`~Octree` objects for tree trunk and its sub-branches.

    :param level: Current level of the Octree
    :param cub: Cuboid object of the tree trunk / sub-branches
    :param particles: Particles within the tree trunk / sub-branches
    :type level: int
    :type cub: object of the class :class:`~Cuboid`
    :type particles: list  

    .. note:: 1. **level** is set to zero for the trunk of the Octree.
              2. **cub** should be entire simulation box for the tree trunk.
              3. **particles** list contains all the ellipsoids in the simulation domain for the tree trunk.
    """

    def __init__(self, level, cub, particles=[]):

        self.maxlevel = 3           # max number of subdivisions
        self.level = level          # current level of subdivision
        self.maxparticles = 10      # max number of particles without subdivision
        self.cub = cub              # cuboid object
        self.particles = particles  # list of particles
        self.branches = []          # empty list that is filled with 8 branches if subdivided

    def get_cub(self):
        """
        Returns the cuboid object of the octree sub-branch        

        :rtype: object :obj:`Cuboid`                
        """
        return self.cub

    def subdivide(self):
        """
        Divides the given Octree sub-branch into eight further sub-branches and 
        initializes each newly created sub-branch as an :class:`~Octree` object             
        """
        for cub in cub_oct_split(self.cub):
            branch = Octree(self.level+1, cub, [])
            self.branches.append(branch)

    def add_particle(self, particle):
        """
        Update the particle list of the Octree sub-branch            
        """
        self.particles.append(particle)

    def subdivide_particles(self):
        """
        Evaluates whether ellipsoids belong to a particular Octree sub-branch
        by calling :meth:`intersect` on each ellipsoid.           
        """
        for particle, branch in itertools.product(self.particles, self.branches):
            if branch.get_cub().intersect(particle.get_cub()):
                branch.add_particle(particle)

    # def collisionsTest(self):
    #     """
    #     Tests for collision between all ellipsoids in the particle list of a particular Octree
    #     sub-branch           
    #     """
    #     for i, E1 in enumerate(self.particles):
    #         for E2 in self.particles[i+1:]:

    #             # Distance between the centers of ellipsoids
    #             dist = np.sqrt(np.sum(np.square(np.subtract(E1.get_pos(), E2.get_pos()))))
                
    #             # If the bounding spheres collide then check for collision
    #             if dist <= E1.a + E2.a:
    #                 # Check if ellipsoids overlap and update their speeds accordingly
    #                 collision_routine(E1, E2)
    
    def neighborlistMake(self):
        """
        Finds the neighborlist for each particle
        """
        for particle in self.particles:
            for branch in particle.branches:
                        particle.neighborlist.update(branch.particles)
                        
    def collisionsTest(self):
        """
        Tests for collision between all ellipsoids in the particle list of octree         
        """
        self.neighborlistMake()
        for i, E1 in enumerate(self.particles):
            for E2 in E1.neighborlist:
                id1 = E1.id if E1.duplicate == None else (E1.duplicate + len(self.particles))
                id2 = E2.id if E2.duplicate == None else (E2.duplicate + len(self.particles))
                if id2 > id1:
                    # Distance between the centers of ellipsoids
                    dist = np.sqrt(np.sum(np.square(np.subtract(E1.get_pos(), E2.get_pos()))))
    
                    # If the bounding spheres collide then check for collision
                    if dist <= (E1.a + E2.a):
    
                        # Check if ellipsoids overlap and update their speeds accordingly
                        collision_routine(E1, E2)

    def update(self):
        """
        Updates the Octree and begins the recursive process of subdividing or collision testing           
        """
        if len(self.particles) > self.maxparticles and self.level <= self.maxlevel:
            self.subdivide()
            self.subdivide_particles()
            random.shuffle(self.branches)
            for branch in self.branches:
                branch.update()
        else:
            # if len(self.particles) > 1:
            #     self.collisionsTest()
            for particle in self.particles:
                particle.branches.append(self)

                

