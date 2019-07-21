#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from unittest import mock

import kanapy
from src.kanapy.collision_detect_react import *
from kanapy.base import collideDetect
from src.kanapy.entities import Ellipsoid


@pytest.fixture
def ellip(mocker):

    # Define attributes to mocker object
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

    # Define the mocker objects
    el1 = mocker.MagicMock()
    el2 = mocker.MagicMock()

    # assign attributes to the mocker objects
    el1.rotation_matrix = rotation_matrix
    el1.surface_points = stacked_xyz.dot(rotation_matrix)
    el1.id = 1
    el1.a, el1.b, el1.c = 2.0, 1.5, 1.5
    el1.x, el1.y, el1.z = 1, 0.5, 0.75
    el1.speedx, el1.speedy, el1.speedz = -0.02, 0.075, -0.05
    el1.quat = quat
    el1.inside_voxels = []
    el1.get_pos.return_value = np.array([1, 0.5, 0.75])
    el1.get_coeffs.return_value = np.array([2.0, 1.5, 1.5])
    el1.get_volume.return_value = (4/3)*np.pi*a*b*c

    el2.rotation_matrix = rotation_matrix
    el2.surface_points = stacked_xyz.dot(rotation_matrix)
    el2.id = 2
    el2.a, el2.b, el2.c = 2.0, 1.5, 1.5
    el2.x, el2.y, el2.z = 1.9, 1.68, 2.6
    el2.speedx, el2.speedy, el2.speedz = 0.5, -0.025, -0.36
    el2.quat = quat
    el2.inside_voxels = []
    el2.get_pos.return_value = np.array([1.9, 1.68, 2.6])
    el2.get_coeffs.return_value = np.array([2.0, 1.5, 1.5])
    el2.get_volume.return_value = (4/3)*np.pi*a*b*c

    return [el1, el2]


def test_collideDetect(ellip):

    status = collideDetect(ellip[0].get_coeffs(), ellip[1].get_coeffs(),
                           ellip[0].get_pos(), ellip[1].get_pos(),
                           ellip[0].rotation_matrix, ellip[1].rotation_matrix)

    assert status == True


def test_collision_react():

    # Initialize the Ellipsoids
    el1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, np.array(
        [0.52532199, 0., -0., 0.85090352]))
    el2 = Ellipsoid(2, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0,
                    np.array([0.52532199, 0., -0., 0.85090352]))

    el1.speedx, el1.speedy, el1.speedz = 0.1, 0.075, -0.05
    el2.speedx, el2.speedy, el2.speedz = -0.1, -0.025, -0.36
        
    # Test different conditions
    # Condition: xdiff > 0 && zdiff > 0          
    collision_react(el1, el2)
    collision_react(el2, el1)
    
    assert round(el1.speedx, 6) == -0.077728
    assert round(el1.speedy, 6) == -0.077728
    assert round(el1.speedz, 6) == -0.077728
    
    assert round(el2.speedx, 6) == 0.216198
    assert round(el2.speedy, 6) == 0.216198
    assert round(el2.speedz, 6) == 0.216198    

    # Condition: xdiff > 0 && zdiff < 0
    el2.z = 4.5   
    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == -0.077728
    assert round(el1.speedy, 6) == -0.077728
    assert round(el1.speedz, 6) == 0.077728
    
    assert round(el2.speedx, 6) == 0.216198
    assert round(el2.speedy, 6) == 0.216198
    assert round(el2.speedz, 6) == -0.216198          

    # Condition: xdiff < 0 && zdiff > 0
    el2.x = 4.5
    el2.z  = 5.5  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == 0.077728
    assert round(el1.speedy, 6) == -0.077728
    assert round(el1.speedz, 6) == -0.077728
    
    assert round(el2.speedx, 6) == -0.216198
    assert round(el2.speedy, 6) == 0.216198
    assert round(el2.speedz, 6) == 0.216198          

    # Condition: xdiff < 0 && zdiff < 0
    el2.x = 4.5
    el2.z  = 4.5  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == 0.077728
    assert round(el1.speedy, 6) == -0.077728
    assert round(el1.speedz, 6) == 0.077728
    
    assert round(el2.speedx, 6) == -0.216198
    assert round(el2.speedy, 6) == 0.216198
    assert round(el2.speedz, 6) == -0.216198          

    # Condition: xdiff = 0 && zdiff = 0
    el2.x = 5.0
    el2.z  = 5.0  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == 0.000038
    assert round(el1.speedy, 6) == 0.134629
    assert round(el1.speedz, 6) == 0.0
    
    assert round(el2.speedx, 6) == 0.000106
    assert round(el2.speedy, 6) == -0.374466
    assert round(el2.speedz, 6) == 0.0          

    # Condition: xdiff = 0 && zdiff > 0
    el2.x = 5.0
    el2.z  = 5.5  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == 0.0
    assert round(el1.speedy, 6) == 0.095197
    assert round(el1.speedz, 6) == -0.095197
    
    assert round(el2.speedx, 6) == 0.0
    assert round(el2.speedy, 6) == -0.264788
    assert round(el2.speedz, 6) == 0.264788          

    # Condition: xdiff = 0 && zdiff < 0    
    el2.x = 5.0
    el2.z  = 4.5  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == 0.0
    assert round(el1.speedy, 6) == 0.095197
    assert round(el1.speedz, 6) == 0.095197
    
    assert round(el2.speedx, 6) == 0.0
    assert round(el2.speedy, 6) == -0.264788
    assert round(el2.speedz, 6) == -0.264788          

    # Condition: xdiff < 0 && zdiff = 0    
    el2.x = 4.5
    el2.z  = 5.0  

    collision_react(el1, el2)
    collision_react(el2, el1)   
    
    assert round(el1.speedx, 6) == 0.095197
    assert round(el1.speedy, 6) == 0.095197
    assert round(el1.speedz, 6) == 0.0
    
    assert round(el2.speedx, 6) == -0.264788
    assert round(el2.speedy, 6) == -0.264788
    assert round(el2.speedz, 6) == 0.0          


    # Condition: xdiff > 0 && zdiff = 0    
    el2.x = 5.5
    el2.z  = 5.0  

    collision_react(el1, el2)
    collision_react(el2, el1)   

    assert round(el1.speedx, 6) == -0.095197
    assert round(el1.speedy, 6) == 0.095197
    assert round(el1.speedz, 6) == 0.0
    
    assert round(el2.speedx, 6) == 0.264788
    assert round(el2.speedy, 6) == -0.264788
    assert round(el2.speedz, 6) == 0.0          
                            

def test_collision_routine_sphere():

    # Initialize the Ellipsoids
    el1 = Ellipsoid(1, 5, 5, 5, 2.0, 2.0, 2.0, np.array(
        [0.52532199, 0., -0., 0.85090352]))
    el2 = Ellipsoid(2, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0,
                    np.array([0.52532199, 0., -0., 0.85090352]))

    el1.speedx, el1.speedy, el1.speedz = 0.1, 0.075, -0.05
    el2.speedx, el2.speedy, el2.speedz = -0.1, -0.025, -0.36

    # Test if the 'collision_react' function is called twice
    with mock.patch('src.kanapy.collision_detect_react.collision_react') as mocked_method:
        collision_routine(el1, el2)
        assert mocked_method.call_count == 2


def test_collision_routine_ellipsoid(ellip):

    # Test if the 'collision_react' function is called twice
    with mock.patch('src.kanapy.collision_detect_react.collision_react') as mocked_method:
        collision_routine(ellip[0], ellip[1])
        assert mocked_method.call_count == 2
        
        
