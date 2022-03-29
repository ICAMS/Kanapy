# -*- coding: utf-8 -*-
import math
import numpy as np
import kanapy.base as kbase

def collision_routine(E1, E2):
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

    # If spheres:
    if E1.a == E1.b == E1.c and E2.a == E2.b == E2.c:        
        collision_react(E1, E2)                       # calculates resultant speed for E1        
        collision_react(E2, E1)                       # calculates resultant speed for E2
        return

    # Elif ellipsoids
    else:
        # call the c++ method
        overlap_status = kbase.collideDetect(E1.get_coeffs(), E2.get_coeffs(), 
                                       E1.get_pos(), E2.get_pos(), 
                                       E1.rotation_matrix, E2.rotation_matrix)

        if overlap_status:            
            collision_react(E1, E2)                  # calculates resultant speed for E1            
            collision_react(E2, E1)                  # calculates resultant speed for E2

        return


def collision_react(E1, E2):
    r"""
    Evaluates and modifies the magnitude and direction of the ellipsoid's velocity after collision.    

    :param E1: Ellipsoid :math:`i`
    :type E1: object :obj:`Ellipsoid`
    :param E2: Ellipsoid :math:`j`
    :type E2: object :obj:`Ellipsoid`

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
    E1Speed = np.sqrt((E1.speedx0**2)+(E1.speedy0**2)+(E1.speedz0**2))
    XDiff = -(E1.x - E2.x)
    YDiff = -(E1.y - E2.y)
    ZDiff = -(E1.z - E2.z)

    # To avoid zero-Division error
    if XDiff == 0 and ZDiff == 0:
        ElevationAngle = np.arctan(YDiff/(math.sqrt((0.0001**2)+(0.0001**2))))
    else:
        ElevationAngle = np.arctan(YDiff/(math.sqrt((XDiff**2)+(ZDiff**2))))

    if XDiff != 0 and ZDiff != 0:
        if XDiff > 0:
            if ZDiff > 0:
                Angle = np.arctan(ZDiff/XDiff)
            elif ZDiff < 0:
                Angle = np.arctan(ZDiff/XDiff)
        elif XDiff < 0:
            if ZDiff > 0:
                Angle = np.pi + np.arctan(ZDiff/XDiff)
            elif ZDiff < 0:
                Angle = -np.pi + np.arctan(ZDiff/XDiff)
        XSpeed = -E1Speed * math.cos(Angle)*math.cos(ElevationAngle)
        YSpeed = -E1Speed * math.sin(ElevationAngle)
        ZSpeed = -E1Speed * math.sin(Angle)*math.cos(ElevationAngle)

    else:
        if XDiff == 0 and ZDiff == 0:
            Angle = 0
        if XDiff == 0 and ZDiff != 0:
            if ZDiff > 0:
                Angle = -np.pi/2.0
            else:
                Angle = np.pi/2.0
        if XDiff != 0 and ZDiff == 0:
            if XDiff < 0:
                Angle = 0.0
            else:
                Angle = np.pi
        XSpeed = E1Speed * math.cos(Angle)*math.cos(ElevationAngle)
        YSpeed = E1Speed * math.sin(ElevationAngle)
        ZSpeed = E1Speed * math.sin(Angle)*math.cos(ElevationAngle)

    # Assign new speeds 
    
    E1.speedx += XSpeed
    E1.speedy += YSpeed
    E1.speedz += ZSpeed

    
    # if E1.phasenum == E2.phasenum:
    #     E1.speedx += 2*XSpeed
    #     E1.speedy += 2*YSpeed
    #     E1.speedz += 2*ZSpeed
    # else:
    #     E1.speedx += XSpeed
    #     E1.speedy += YSpeed
    #     E1.speedz += ZSpeed

    return
