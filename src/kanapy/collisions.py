# -*- coding: utf-8 -*-
import numpy as np


def collision_routine(E1, E2, damp=0):
    """
    Calls the method :meth:'collide_detect' to determine whether the given two ellipsoid objects overlap using
    the Algebraic separation condition developed by W. Wang et al. A detailed description is provided
    therein.

    Also calls the :meth:`collision_react` to evaluate the response after collision.     

    :param E1: Ellipsoid :math:`i` 
    :type E1: object :obj:`Ellipsoid`
    :param E2: Ellipsoid :math:`j`
    :type E2: object :obj:`Ellipsoid`
    :param damp: Damping factor
    :type damp: float

    .. note:: 1. If both the particles to be tested for overlap are spheres, then the bounding sphere hierarchy is
                 sufficient to determine whether they overlap.
              2. Else, if either of them is an ellipsoid, then their coefficients, positions & rotation matrices are
                 used to determine whether they overlap.
    """

    # call the collision detection algorithm
    overlap_status = collide_detect(E1.get_coeffs(), E2.get_coeffs(),
                                    E1.get_pos(), E2.get_pos(),
                                    E1.rotation_matrix, E2.rotation_matrix)

    # check if inner polyhedra overlap if present
    """if overlap_status and E1.inner is not None and E2.inner is not None:
        E1.sync_poly()
        E2.sync_poly()
        if all(E1.inner.find_simplex(E2.inner.points) < 0) and\
           all(E2.inner.find_simplex(E1.inner.points) < 0):
            overlap_status = False"""

    if overlap_status:
        collision_react(E1, E2, damp=damp)  # calculates resultant speed for E1
        collision_react(E2, E1, damp=damp)  # calculates resultant speed for E2

    return overlap_status


def collision_react(ell1, ell2, damp=0.):
    r"""
    Evaluates and modifies the magnitude and direction of the ellipsoid's velocity after collision.    

    :param ell1: Ellipsoid :math:`i`
    :type ell1: object :obj:`Ellipsoid`
    :param ell2: Ellipsoid :math:`j`
    :type ell2: object :obj:`Ellipsoid`
    :param damp: Damping factor
    :type damp: float

    .. note:: Consider two ellipsoids :math:`i, j` at collision. Let them occupy certain positions in space
              defined by the position vectors :math:`\mathbf{r}^{i}, \mathbf{r}^{j}` and have certain 
              velocities represented by :math:`\mathbf{v}^{i}, \mathbf{v}^{j}` respectively. The objective
              is to  find the velocity vectors after collision. The elevation angle :math:`\phi` between
              the ellipsoids is determined by,      

              .. image:: /figs/elevation_ell.png                        
                :width: 200px
                :height: 45px
                :align: center                 

              where :math:`dx, dy, dz` are defined as the distance between the two ellipsoid centers along
                    :math:`x, y, z` directions given by,

              .. image:: /figs/dist_ell.png                        
                  :width: 110px
                  :height: 75px
                  :align: center

              Depending on the magnitudes of :math:`dx, dz` as projected on the :math:`x-z` plane, the angle
              :math:`\Theta` is computed.
              The angles :math:`\Theta` and :math:`\phi` determine the in-plane and out-of-plane directions along which
              the ellipsoid :math:`i` would bounce back after collision. Thus, the updated velocity vector components
              along the :math:`x, y, z` directions are determined by,

              .. image:: /figs/velocities_ell.png                        
                  :width: 180px
                  :height: 80px
                  :align: center                        
    """
    ell1_speed = np.linalg.norm([ell1.speedx0, ell1.speedy0, ell1.speedz0]) * (1. - damp)
    x_diff = ell2.x - ell1.x
    y_diff = ell2.y - ell1.y
    z_diff = ell2.z - ell1.z
    elevation_angle = np.arctan2(y_diff, np.linalg.norm([x_diff, z_diff]))
    angle = np.arctan2(z_diff, x_diff)

    x_speed = -ell1_speed * np.cos(angle) * np.cos(elevation_angle)
    y_speed = -ell1_speed * np.sin(elevation_angle)
    z_speed = -ell1_speed * np.sin(angle) * np.cos(elevation_angle)

    # Assign new speeds 
    ell1.speedx += x_speed
    ell1.speedy += y_speed
    ell1.speedz += z_speed

    return


def collide_detect(coef_i, coef_j, r_i, r_j, A_i, A_j):
    r"""Implementation of Algebraic separation condition developed by
    W. Wang et al. 2001 for overlap detection between two static ellipsoids.
    Kudos to ChatGPT for support with translation from C++ code.

    :param coef_i: Coefficients of ellipsoid :math:`i`
    :type coef_i: numpy array
    :param coef_j: Coefficients of ellipsoid :math:`j`
    :type coef_j: numpy array
    :param r_i: Position of ellipsoid :math:`i`
    :type r_i: numpy array
    :param r_j: Position of ellipsoid :math:`j`
    :type r_j: numpy array
    :param A_i: Rotation matrix of ellipsoid :math:`i`
    :type A_i: numpy array
    :param A_j: Rotation matrix of ellipsoid :math:`j`
    :type A_j: numpy array
    :returns: **True** if ellipsoids :math:`i, j` overlap, else **False**
    :rtype: boolean
    """
    SMALL = 1.e-12

    # Initialize Matrices A & B with zeros
    A = np.zeros((4, 4), dtype=float)
    B = np.zeros((4, 4), dtype=float)

    A[0, 0] = 1 / (coef_i[0] ** 2)
    A[1, 1] = 1 / (coef_i[1] ** 2)
    A[2, 2] = 1 / (coef_i[2] ** 2)
    A[3, 3] = -1

    B[0, 0] = 1 / (coef_j[0] ** 2)
    B[1, 1] = 1 / (coef_j[1] ** 2)
    B[2, 2] = 1 / (coef_j[2] ** 2)
    B[3, 3] = -1

    # Rigid body transformations
    T_i = np.zeros((4, 4), dtype=float)
    T_j = np.zeros((4, 4), dtype=float)

    T_i[:3, :3] = A_i
    T_i[:3, 3] = r_i
    T_i[3, 3] = 1.0

    T_j[:3, :3] = A_j
    T_j[:3, 3] = r_j
    T_j[3, 3] = 1.0

    # Copy the arrays for future operations
    Ma = np.tile(T_i, (1, 1))
    Mb = np.tile(T_j, (1, 1))

    # aij of matrix A in det(lambda*A - Ma'*(Mb^-1)'*B*(Mb^-1)*Ma).
    # bij of matrix b = Ma'*(Mb^-1)'*B*(Mb^-1)*Ma
    aux = np.linalg.inv(Mb) @ Ma
    bm = aux.T @ B @ aux

    # Coefficients of the Characteristic Polynomial.
    T0 = (-A[0, 0] * A[1, 1] * A[2, 2])
    T1 = (A[0, 0] * A[1, 1] * bm[2, 2] + A[0, 0] * A[2, 2] * bm[1, 1] + A[1, 1] * A[2, 2] * bm[0, 0] - A[0, 0] * A[
        1, 1] * A[2, 2] * bm[3, 3])
    T2 = (A[0, 0] * bm[1, 2] * bm[2, 1] - A[0, 0] * bm[1, 1] * bm[2, 2] - A[1, 1] * bm[0, 0] * bm[2, 2] + A[1, 1] * bm[
        0, 2] * bm[2, 0] -
          A[2, 2] * bm[0, 0] * bm[1, 1] + A[2, 2] * bm[0, 1] * bm[1, 0] + A[0, 0] * A[1, 1] * bm[2, 2] * bm[3, 3] - A[
              0, 0] * A[1, 1] * bm[2, 3] * bm[3, 2] +
          A[0, 0] * A[2, 2] * bm[1, 1] * bm[3, 3] - A[0, 0] * A[2, 2] * bm[1, 3] * bm[3, 1] + A[1, 1] * A[2, 2] * bm[
              0, 0] * bm[3, 3] -
          A[1, 1] * A[2, 2] * bm[0, 3] * bm[3, 0])
    T3 = (bm[0, 0] * bm[1, 1] * bm[2, 2] - bm[0, 0] * bm[1, 2] * bm[2, 1] - bm[0, 1] * bm[1, 0] * bm[2, 2] + bm[0, 1] *
          bm[1, 2] * bm[2, 0] +
          bm[0, 2] * bm[1, 0] * bm[2, 1] - bm[0, 2] * bm[1, 1] * bm[2, 0] - A[0, 0] * bm[1, 1] * bm[2, 2] * bm[3, 3] +
          A[0, 0] * bm[1, 1] * bm[2, 3] * bm[3, 2] +
          A[0, 0] * bm[1, 2] * bm[2, 1] * bm[3, 3] - A[0, 0] * bm[1, 2] * bm[2, 3] * bm[3, 1] - A[0, 0] * bm[1, 3] * bm[
              2, 1] * bm[3, 2] +
          A[0, 0] * bm[1, 3] * bm[2, 2] * bm[3, 1] - A[1, 1] * bm[0, 0] * bm[2, 2] * bm[3, 3] + A[1, 1] * bm[0, 0] * bm[
              2, 3] * bm[3, 2] +
          A[1, 1] * bm[0, 2] * bm[2, 0] * bm[3, 3] - A[1, 1] * bm[0, 2] * bm[2, 3] * bm[3, 0] - A[1, 1] * bm[0, 3] * bm[
              2, 0] * bm[3, 2] +
          A[1, 1] * bm[0, 3] * bm[2, 2] * bm[3, 0] - A[2, 2] * bm[0, 0] * bm[1, 1] * bm[3, 3] + A[2, 2] * bm[0, 0] * bm[
              1, 3] * bm[3, 1] +
          A[2, 2] * bm[0, 1] * bm[1, 0] * bm[3, 3] - A[2, 2] * bm[0, 1] * bm[1, 3] * bm[3, 0] - A[2, 2] * bm[0, 3] * bm[
              1, 0] * bm[3, 1] +
          A[2, 2] * bm[0, 3] * bm[1, 1] * bm[3, 0])
    T4 = (bm[0, 0] * bm[1, 1] * bm[2, 2] * bm[3, 3] - bm[0, 0] * bm[1, 1] * bm[2, 3] * bm[3, 2] - bm[0, 0] * bm[1, 2] *
          bm[2, 1] * bm[3, 3] +
          bm[0, 0] * bm[1, 2] * bm[2, 3] * bm[3, 1] + bm[0, 0] * bm[1, 3] * bm[2, 1] * bm[3, 2] - bm[0, 0] * bm[1, 3] *
          bm[2, 2] * bm[3, 1] -
          bm[0, 1] * bm[1, 0] * bm[2, 2] * bm[3, 3] + bm[0, 1] * bm[1, 0] * bm[2, 3] * bm[3, 2] + bm[0, 1] * bm[1, 2] *
          bm[2, 0] * bm[3, 3] -
          bm[0, 1] * bm[1, 2] * bm[2, 3] * bm[3, 0] - bm[0, 1] * bm[1, 3] * bm[2, 0] * bm[3, 2] + bm[0, 1] * bm[1, 3] *
          bm[2, 2] * bm[3, 0] +
          bm[0, 2] * bm[1, 0] * bm[2, 1] * bm[3, 3] - bm[0, 2] * bm[1, 0] * bm[2, 3] * bm[3, 1] - bm[0, 2] * bm[1, 1] *
          bm[2, 0] * bm[3, 3] +
          bm[0, 2] * bm[1, 1] * bm[2, 3] * bm[3, 0] + bm[0, 2] * bm[1, 3] * bm[2, 0] * bm[3, 1] - bm[0, 2] * bm[1, 3] *
          bm[2, 1] * bm[3, 0] -
          bm[0, 3] * bm[1, 0] * bm[2, 1] * bm[3, 2] + bm[0, 3] * bm[1, 0] * bm[2, 2] * bm[3, 1] + bm[0, 3] * bm[1, 1] *
          bm[2, 0] * bm[3, 2] -
          bm[0, 3] * bm[1, 1] * bm[2, 2] * bm[3, 0] - bm[0, 3] * bm[1, 2] * bm[2, 0] * bm[3, 1] + bm[0, 3] * bm[1, 2] *
          bm[2, 1] * bm[3, 0])

    # Roots of the characteristic_polynomial (lambda0, ... , lambda4).
    cp = np.array([T0, T1, T2, T3, T4], dtype=float)

    # Solve the polynomial
    roots = np.roots(cp)

    # Find the real roots where imaginary part doesn't exist
    real_roots = [root.real for root in roots if abs(root.imag) < SMALL]

    # Count number of real negative roots
    count_neg = sum(1 for root in real_roots if root < -SMALL)

    # Sort the real roots in ascending order
    real_roots.sort()

    # Algebraic separation conditions to determine overlapping
    if count_neg == 2:
        if abs(real_roots[0] - real_roots[1]) > SMALL:
            return False
        else:
            return True
    else:
        return True

# Example usage:
# coef_i = np.array([a_i, b_i, c_i])
# coef_j = np.array([a_j, b_j, c_j])
# r_i = np.array([x_i, y_i, z_i])
# r_j = np.array([x_j, y_j, z_j])
# A_i = np.array([[A11_i, A12_i, A13_i], [A21_i, A22_i, A23_i], [A31_i, A32_i, A33_i]])
# A_j = np.array([[A11_j, A12_j, A13_j], [A21_j, A22_j, A23_j], [A31_j, A32_j, A33_j]])
# result = collide_detect(coef_i, coef_j, r_i, r_j, A_i, A_j)
