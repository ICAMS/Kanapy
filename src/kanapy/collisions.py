# -*- coding: utf-8 -*-
import numpy as np
import kanapy.base as kbase

def collision_routine(E1, E2, damp=0):
    """
    Calls the c++ method :meth:`kanapy.base.collideDetect` to determine whether the given two ellipsoid objects overlap using 
    the Algebraic separation condition developed by W. Wang et al. A detailed description is provided
    therein.

    Also calls the :meth:`collision_react` to evaluate the response after collision.     

    :param E1: Ellipsoid :math:`i` 
    :type E1: object :obj:`Ellipsoid`
    :param E2: Ellipsoid :math:`j`
    :type E2: object :obj:`Ellipsoid`

    .. note:: 1. If both the particles to be tested for overlap are spheres, then the bounding sphere hierarchy is sufficient to 
                 determine whether they overlap.
              2. Else, if either of them is an ellipsoid, then their coefficients, positions & rotation matrices are used
                 to determine whether they overlap.                             
    """

    # call the c++ method
    overlap_status = kbase.collideDetect(E1.get_coeffs(), E2.get_coeffs(), 
                                   E1.get_pos(), E2.get_pos(), 
                                   E1.rotation_matrix, E2.rotation_matrix)

    if overlap_status:
        collision_react(E1, E2, damp=damp)                  # calculates resultant speed for E1            
        collision_react(E2, E1, damp=damp)                  # calculates resultant speed for E2

    return overlap_status


def collision_react(ell1, ell2, damp=0.):
    r"""
    Evaluates and modifies the magnitude and direction of the ellipsoid's velocity after collision.    

    :param ell1: Ellipsoid :math:`i`
    :type ell1: object :obj:`Ellipsoid`
    :param ell2: Ellipsoid :math:`j`
    :type ell2: object :obj:`Ellipsoid`

    .. note:: Consider two ellipsoids :math:`i, j` at collision. Let them occupy certain positions in space
              defined by the position vectors :math:`\mathbf{r}^{i}, \mathbf{r}^{j}` and have certain 
              velocities represented by :math:`\mathbf{v}^{i}, \mathbf{v}^{j}` respectively. The objective
              is to  find the velocity vectors after collision. The elevation angle :math:`\phi` between
              the ellipsoids is determined by,      

              .. image:: /figs/elevation_ell.png                        
                :width: 200px
                :height: 45px
                :align: center                 

              where :math:`dx, dy, dz` are defined as the distance between the two ellipsoid centers along :math:`x, y, z` directions given by,

              .. image:: /figs/dist_ell.png                        
                  :width: 110px
                  :height: 75px
                  :align: center

              Depending on the magnitudes of :math:`dx, dz` as projected on the :math:`x-z` plane, the angle :math:`\Theta` is computed. 
              The angles :math:`\Theta` and :math:`\phi` determine the in-plane and out-of-plane directions along which the ellipsoid :math:`i` 
              would bounce back after collision. Thus, the updated velocity vector components along the :math:`x, y, z` directions are determined by,

              .. image:: /figs/velocities_ell.png                        
                  :width: 180px
                  :height: 80px
                  :align: center                        
    """
    ell1_speed = np.linalg.norm([ell1.speedx0, ell1.speedy0, ell1.speedz0])*(1. - damp)
    x_diff = ell2.x - ell1.x
    y_diff = ell2.y - ell1.y
    z_diff = ell2.z - ell1.z
    elevation_angle = np.arctan2(y_diff, np.linalg.norm([x_diff, z_diff]))
    angle = np.arctan2(z_diff, x_diff)
                
    x_speed = -ell1_speed * np.cos(angle)*np.cos(elevation_angle)
    y_speed = -ell1_speed * np.sin(elevation_angle)
    z_speed = -ell1_speed * np.sin(angle)*np.cos(elevation_angle)

    # Assign new speeds 
    ell1.speedx += x_speed
    ell1.speedy += y_speed
    ell1.speedz += z_speed

    return
