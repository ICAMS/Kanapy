#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import mock
import pytest
import kanapy
from kanapy.packing import *
from kanapy.entities import Ellipsoid, Simulation_Box


@pytest.fixture
def par_sim(mocker):
    parDict = {'Type': 'Elongated',
               'Number': 3,
               'Major_diameter': np.array([5.2, 3.6, 2.4]),
               'Minor_diameter1': np.array([2.15, 3.6, 1.15]),
               'Minor_diameter2': np.array([2.15, 3.6, 1.15]),
               'Tilt angle': np.array([92, 89.3, 85]),
               'Phase': [0, 0, 0]}

    sb = mocker.MagicMock()
    # Define attributes to mocker object
    sb.w, sb.h, sb.d = 3, 3, 3
    sb.sim_ts = 0
    sb.left, sb.top, sb.front = 0, 0, 0
    sb.right, sb.bottom, sb.back = 3, 3, 3

    return parDict, sb


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


def test_particle_generator(par_sim):
    periodic = False
    ell_list = particle_generator([par_sim[0]], par_sim[1], periodic)

    for ell in ell_list:
        assert isinstance(ell, Ellipsoid)


def test_particle_grow(rot_surf):
    rs = rot_surf[0]
    ell1 = Ellipsoid(1, 1, 0.5, 0.75, 2.0, 1.5, 1.5, rs)
    ell2 = Ellipsoid(2, 1.9, 1.68, 2.6, 2.0, 1.5, 1.5, rs)
    ell1.speedx0, ell1.speedy0, ell1.speedz0 = -0.02, 0.075, -0.05
    ell2.speedx0, ell2.speedy0, ell2.speedz0 = 0.5, -0.025, -0.36

    ells = [ell1, ell2]
    sb = Simulation_Box((10, 10, 10))

    particle_grow(sb, ells, True, 10, dump=True)
    
    assert np.isclose(ell1.x, 0.86)
    assert np.isclose(ell1.y, 1.025)
    assert np.isclose(ell2.z, 0.08)

    
def test_packingRoutine():
    pd = {'Type': 'Equiaxed',
          'Number': 2,
          'Equivalent_diameter': [1.651, 1.651],
          'Major_diameter': [2.0, 2.0],
          'Minor_diameter1': [1.5, 1.5],
          'Minor_diameter2': [1.5, 1.5],
          'Tilt angle': [86, 92],
          'Phase': [0, 0]}
    sb = Simulation_Box((3, 3, 3))

    # Test if the 'particle_generator' function is called once
    with mock.patch('kanapy.packing.particle_generator') as mocked_method:
        packingRoutine([pd], True, 2, sb, save_files=False)
        assert mocked_method.call_count == 1

    # Test if the 'particle_grow' function is called once
    with mock.patch('kanapy.packing.particle_grow', return_value=(None, None)) as mocked_method:
        packingRoutine([pd], True, 2, sb, save_files=False)
        assert mocked_method.call_count == 1
