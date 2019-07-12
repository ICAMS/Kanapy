import pytest
import numpy as np

import kanapy
from src.kanapy.entities import *


@pytest.fixture
def CuboidBox(mocker):
    cb = mocker.MagicMock()

    # Define attributes to mocker object
    cb.left, cb.top, cb.front = 0, 0, 0
    cb.right, cb.bottom, cb.back = 3, 3, 3
    cb.width, cb.height, cb.depth = 3, 3, 3

    return cb


def test_cub_oct_split(CuboidBox):

    simbox = CuboidBox
    cub_list = cub_oct_split(simbox)

    assert len(cub_list) == 8


# Test functions for Class Simulation_Box
def test_Simulation_Box_init():

    # Inititalize the simulation box
    sbox = Simulation_Box(3, 3, 3)

    assert sbox.w == 3
    assert sbox.h == 3
    assert sbox.d == 3
    assert sbox.sim_ts == 0
    assert sbox.left == 0
    assert sbox.top == 0
    assert sbox.front == 0
    assert sbox.right == 3
    assert sbox.bottom == 3
    assert sbox.back == 3


# Test functions for Class Cuboid
class TestCuboid():

    def test_Cuboid_init(self):

        # Initialize the Cuboid
        self.cbox = Cuboid(0, 0, 3, 3, 0, 3)

        assert self.cbox.left == 0
        assert self.cbox.top == 0
        assert self.cbox.right == 3
        assert self.cbox.bottom == 3
        assert self.cbox.front == 0
        assert self.cbox.back == 3

        assert self.cbox.width == 3
        assert self.cbox.height == 3
        assert self.cbox.depth == 3

    def test_Cuboid_intersect(self):

        # Initialize cuboid to be tested for intersection
        self.cub = Cuboid(0, 0, 3, 3, 0, 3)

        # Other cuboid instances
        other1 = Cuboid(1.5, 1.5, 4.5, 4.5, 1.5, 4.5)
        other2 = Cuboid(4, 4, 7, 7, 4, 7)

        assert self.cub.intersect(other1) == True
        assert self.cub.intersect(other2) == False


# Test functions for Class Ellipsoid
@pytest.fixture
def rot_surf():

    # Define attributes
    a, b, c = 2.0, 1.5, 1.5                                     # Semi axes lengths    
    vec_a = np.array([a*np.cos(90), a*np.sin(90), 0.0])         # Tilt vector wrt (+ve) x axis    
    cross_a = np.cross(np.array([1, 0, 0]), vec_a)              # Find the quaternion axis    
    norm_cross_a = np.linalg.norm(cross_a, 2)                   # norm of the vector (Magnitude)    
    quat_axis = cross_a/norm_cross_a                            # normalize the quaternion axis

    # Find the quaternion components
    qx, qy, qz = quat_axis * np.sin(90/2)
    qw = np.cos(90/2)
    quat = np.array([qw, qx, qy, qz])

    # Generate rotation matrix
    Nq = qw*qw + qx*qx + qy*qy + qz*qz

    s = 2.0/Nq
    X = qx*s
    Y = qy*s
    Z = qz*s
    wX = qw*X
    wY = qw*Y
    wZ = qw*Z
    xX = qx*X
    xY = qx*Y
    xZ = qx*Z
    yY = qy*Y
    yZ = qy*Z
    zZ = qz*Z

    rotation_matrix = np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                                [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                                [xZ-wY, yZ+wX, 1.0-(xX+yY)]])

    # Rotation matrix has to be transposed as OVITO uses the transposed matrix for visualization.
    rotation_matrix = rotation_matrix.T

    # Points on the outer surface of Ellipsoid
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    # Cartesian coordinates that correspond to the spherical angles:
    xval = a * np.outer(np.cos(u), np.sin(v))
    yval = b * np.outer(np.sin(u), np.sin(v))
    zval = c * np.outer(np.ones_like(u), np.cos(v))

    # combine the three 2D arrays element wise
    stacked_xyz = np.stack((xval.ravel(), yval.ravel(), zval.ravel()), axis=1)

    # assign attributes to the mocker objects
    surface_points = stacked_xyz.dot(rotation_matrix)

    return quat, rotation_matrix, surface_points


class TestEllipsoid():

    def test_init(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        assert type(self.ell.id) == int
        assert self.ell.x == 1
        assert self.ell.y == 0.5
        assert self.ell.z == 0.75
        assert self.ell.a == 2.0
        assert self.ell.b == 1.5
        assert self.ell.c == 1.5
        assert np.array_equal(self.ell.quat, rot_surf[0])
        assert self.ell.oria == 2.0
        assert self.ell.orib == 1.5
        assert self.ell.oric == 1.5

        assert self.ell.speedx == 0.
        assert self.ell.speedy == 0.
        assert self.ell.speedz == 0.

        assert np.array_equal(self.ell.rotation_matrix, rot_surf[1])
        assert np.array_equal(self.ell.surface_points, rot_surf[2])
        assert len(self.ell.inside_voxels) == 0
        assert isinstance(self.ell.cub, Cuboid)

    def test_get(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        assert np.array_equal(self.ell.get_pos(), np.array([1, 0.5, 0.75]))
        assert np.array_equal(self.ell.get_coeffs(), np.array([2.0, 1.5, 1.5]))
        assert self.ell.get_volume() == (4/3)*np.pi*2.0*1.5*1.5
        assert isinstance(self.ell.get_cub(), Cuboid)

    def test_growth(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        self.ell.growth(1000)

        assert self.ell.a == 2.0 + (2.0/1000)
        assert self.ell.b == 1.5 + (1.5/1000)
        assert self.ell.c == 1.5 + (1.5/1000)

    def test_Bbox(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        self.ell.Bbox()

        surface_pts_new = rot_surf[2] + np.array([1, 0.5, 0.75])

        assert self.ell.bbox_xmin == np.amin(surface_pts_new[:, 0])
        assert self.ell.bbox_xmax == np.amax(surface_pts_new[:, 0])
        assert self.ell.bbox_ymin == np.amin(surface_pts_new[:, 1])
        assert self.ell.bbox_ymax == np.amax(surface_pts_new[:, 1])
        assert self.ell.bbox_zmin == np.amin(surface_pts_new[:, 2])
        assert self.ell.bbox_zmax == np.amax(surface_pts_new[:, 2])

    def test_move(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])
        self.ell.speedx, self.ell.speedy, self.ell.speedz = -0.02, 0.075, -0.05

        self.ell.move()

        assert self.ell.x == 1 + (-0.02)
        assert self.ell.y == 0.5 + (0.075)
        assert self.ell.z == 0.75 + (-0.05)

    def test_gravity(self, rot_surf):

        # Initialize the Ellipsoid
        self.ell = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        self.ell.gravity_effect(0.6)

        assert self.ell.x == 1
        assert self.ell.y == 0.5 - (0.6)
        assert self.ell.z == 0.75

    @pytest.mark.parametrize("position, duplicates", [(np.array([0.2, 0.5, 0.75]), 7), (np.array([0.2, 5.0, 0.75]), 3), (np.array([0.2, 5.0, 5.0]), 1), (np.array([5.0, 5.0, 5.0]), 0)])
    def test_wallCollision_Periodic(self, rot_surf, position, duplicates):

        sbox = Simulation_Box(10, 10, 10)
        self.ell = Ellipsoid(
            1, position[0], position[1], position[2], 2.0, 1.5, 1.5, rot_surf[0])

        # Test for periodicity == TRUE
        dups = self.ell.wallCollision(sbox, True)
        assert len(dups) == duplicates

    @pytest.mark.parametrize("position, speeds", [(np.array([0.2, 0.5, 0.75]), [0.02, -0.075, 0.05]), (np.array([9.8, 9.5, 9.25]), [0.02, -0.075, 0.05])])
    def test_wallCollision_Regular(self, rot_surf, position, speeds):

        sbox = Simulation_Box(10, 10, 10)
        self.ell = Ellipsoid(
            1, position[0], position[1], position[2], 2.0, 1.5, 1.5, rot_surf[0])
        self.ell.speedx, self.ell.speedy, self.ell.speedz = -0.02, 0.075, -0.05

        # Test for periodicity == FALSE
        self.ell.wallCollision(sbox, False)
        assert self.ell.speedx == speeds[0]
        assert self.ell.speedy == speeds[1]
        assert self.ell.speedz == speeds[2]

        assert isinstance(self.ell.get_cub(), Cuboid)


# Test functions for Class Octree
class TestOctree():

    def test_init(self):

        cbox = Cuboid(0, 0, 10, 10, 0, 10)              # Initialize the cuboid
        self.tree = Octree(0, cbox, particles=[])       # Initialize the octree

        assert self.tree.level == 0
        assert isinstance(self.tree.cub, Cuboid)
        assert len(self.tree.particles) == 0
        assert len(self.tree.branches) == 0

    def test_subdivide(self):

        cbox = Cuboid(0, 0, 10, 10, 0, 10)              # Initialize the Cuboid
        self.tree = Octree(0, cbox, particles=[])       # Initialize the Octree

        self.tree.subdivide()
        assert len(self.tree.branches) == 8

        for br in self.tree.branches:
            assert isinstance(br, Octree)
            assert br.level == 1

    def test_add_particle(self):

        cbox = Cuboid(0, 0, 10, 10, 0, 10)              # Initialize the Cuboid
        self.tree = Octree(0, cbox, particles=[])       # Initialize the Octree

        self.tree.add_particle('par')
        assert len(self.tree.particles) == 1

    def test_subdivide_particles(self, rot_surf):

        cbox = Cuboid(0, 0, 10, 10, 0, 10)              # Initialize the Cuboid

        # Initialize the Ellipsoids
        ell1 = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])
        ell2 = Ellipsoid(2, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])

        # Initialize the Octree
        self.tree = Octree(0, cbox, particles=[ell1, ell2])

        self.tree.subdivide_particles()
        assert len(self.tree.particles) == 2

    def test_collisions(self, rot_surf):

        cbox = Cuboid(0, 0, 10, 10, 0, 10)              # Initialize the Cuboid

        # Initialize the Ellipsoids
        ell1 = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rot_surf[0])
        ell2 = Ellipsoid(2, 1.9, 1.68, 2.6, 2.0, 1.5, 1.5, rot_surf[0])
        ell1.speedx, ell1.speedy, ell1.speedz = -0.02, 0.075, -0.05
        ell2.speedx, ell2.speedy, ell2.speedz = 0.5, -0.025, -0.36

        # Initialize the Octree
        self.tree = Octree(0, cbox, particles=[ell1, ell2])
        self.tree.subdivide_particles()
        self.tree.collisionsTest()

        assert round(ell1.speedx, 6) == -0.035037
        assert round(ell1.speedy, 6) == -0.045938
        assert round(ell1.speedz, 6) == -0.072021

        assert round(ell2.speedx, 6) == 0.233994
        assert round(ell2.speedy, 6) == 0.306793
        assert round(ell2.speedz, 6) == 0.480988

    def test_update1(self, mocker, rot_surf):

        cbox = Cuboid(0, 0, 40, 40, 0, 40)              # Initialize the Cuboid

        # Initialize the Ellipsoids
        ell1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell2 = Ellipsoid(2, 15, 15, 15, 2.0, 2.0, 2.0, rot_surf[0])
        ell3 = Ellipsoid(3, 25, 5, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell4 = Ellipsoid(4, 35, 15, 15, 2.0, 2.0, 2.0, rot_surf[0])
        ell5 = Ellipsoid(5, 25, 25, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell6 = Ellipsoid(6, 35, 35, 15, 2.0, 2.0, 2.0, rot_surf[0])
        ell7 = Ellipsoid(7, 5, 25, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell8 = Ellipsoid(8, 5, 25, 25, 2.0, 2.0, 2.0, rot_surf[0])
        ell9 = Ellipsoid(9, 25, 25, 25, 2.0, 2.0, 2.0, rot_surf[0])
        ell10 = Ellipsoid(10, 25, 5, 25, 2.0, 2.0, 2.0, rot_surf[0])
        ell11 = Ellipsoid(11, 15, 5, 25, 2.0, 2.0, 2.0, rot_surf[0])

        # Initialize the Octree
        self.tree = Octree(0, cbox, particles=[
                           ell1, ell2, ell3, ell4, ell5, ell6, ell7, ell8, ell9, ell10, ell11])

        # Spy on the function's calling in update method.
        mocker.spy(self.tree, 'subdivide')
        mocker.spy(self.tree, 'subdivide_particles')

        self.tree.update()
        assert self.tree.subdivide.call_count == 1
        assert self.tree.subdivide_particles.call_count == 1

    def test_update2(self, mocker, rot_surf):

        # Initialize the Cuboid
        cbox = Cuboid(0, 0, 40, 40, 0, 40)

        # Initialize the Ellipsoids
        ell1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell2 = Ellipsoid(2, 15, 15, 15, 2.0, 2.0, 2.0, rot_surf[0])
        ell3 = Ellipsoid(3, 25, 5, 5, 2.0, 2.0, 2.0, rot_surf[0])
        ell4 = Ellipsoid(4, 35, 15, 15, 2.0, 2.0, 2.0, rot_surf[0])

        # Initialize the Octree
        self.tree = Octree(0, cbox, particles=[ell1, ell2, ell3, ell4])

        # Spy on the function being called in update method
        mocker.spy(self.tree, 'collisionsTest')

        self.tree.update()
        assert self.tree.collisionsTest.call_count == 1


if __name__ == "__main__":
    test_cub_oct_split()
    test_Simulation_Box_init()
    TestCuboid()
    TestEllipsoid()
    TestOctree()
