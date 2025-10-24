# -*- coding: utf-8 -*-
import numpy as np


def collision_routine(E1, E2):
    """
    Detect and handle collision between two ellipsoid objects

    Uses the algebraic separation condition by W. Wang et al. to detect overlap
    between two ellipsoids, and if a collision occurs, computes the contact
    response via the `collision_react` method.

    Parameters
    ----------
    E1 : Ellipsoid
        First ellipsoid object (particle i)
    E2 : Ellipsoid
        Second ellipsoid object (particle j)

    Returns
    -------
    bool
        True if overlap (collision) is detected, False otherwise

    Notes
    -----
    - If both particles are spheres, the bounding sphere hierarchy alone
      suffices to detect collision.
    - If either particle is an ellipsoid, their coefficients, positions,
      and rotation matrices are used in the overlap test.
    """

    # call the collision detection algorithm
    overlap_status = collide_detect(E1.get_coeffs(), E2.get_coeffs(),
                                    E1.get_pos(), E2.get_pos(),
                                    E1.rotation_matrix, E2.rotation_matrix)
    if overlap_status:
        collision_react(E1, E2)  # calculates resultant contact forces of both particles
    return overlap_status


def collision_react(ell1, ell2):
    """
    Evaluates and modifies the magnitude and direction of the ellipsoid's velocity after collision.

    Parameters
    ----------
    ell1 : Ellipsoid
        Ellipsoid i
    ell2 : Ellipsoid
        Ellipsoid j

    Notes
    -----
    Consider two ellipsoids i, j at collision. Let them occupy certain positions in space
    defined by the position vectors r^i, r^j and have certain velocities represented by v^i, v^j
    respectively. The objective is to find the velocity vectors after collision. The elevation angle φ
    between the ellipsoids is determined by:

    .. image:: /figs/elevation_ell.png
        :width: 200px
        :height: 45px
        :align: center

    where dx, dy, dz are defined as the distance between the two ellipsoid centers along
    x, y, z directions given by:

    .. image:: /figs/dist_ell.png
        :width: 110px
        :height: 75px
        :align: center

    Depending on the magnitudes of dx, dz as projected on the x-z plane, the angle Θ is computed.
    The angles Θ and φ determine the in-plane and out-of-plane directions along which
    ellipsoid i would bounce back after collision. Thus, the updated velocity vector components
    along the x, y, z directions are determined by:

    .. image:: /figs/velocities_ell.png
        :width: 180px
        :height: 80px
        :align: center
    """
    cdir = ell1.get_pos() - ell2.get_pos()
    dst = np.linalg.norm(cdir)
    eqd1 = ell1.get_volume()**(1/3)
    eqd2 = ell2.get_volume()**(1/3)
    val = (0.5*(eqd1 + eqd2) / dst)**3
    val = np.minimum(5.0, val)

    ell1.force_x += val * cdir[0] / dst
    ell1.force_y += val * cdir[1] / dst
    ell1.force_z += val * cdir[2] / dst
    ell2.force_x -= val * cdir[0] / dst
    ell2.force_y -= val * cdir[1] / dst
    ell2.force_z -= val * cdir[2] / dst
    return


def collide_detect(coef_i, coef_j, r_i, r_j, A_i, A_j):
    """
    Determines overlap between two static ellipsoids using the Algebraic Separation Condition
    developed by W. Wang et al., 2001. This function implements the method for two ellipsoids
    with given coefficients, positions, and rotation matrices.

    Parameters
    ----------
    coef_i : numpy array
        Coefficients of ellipsoid i
    coef_j : numpy array
        Coefficients of ellipsoid j
    r_i : numpy array
        Position vector of ellipsoid i
    r_j : numpy array
        Position vector of ellipsoid j
    A_i : numpy array
        Rotation matrix of ellipsoid i
    A_j : numpy array
        Rotation matrix of ellipsoid j

    Returns
    -------
    bool
        True if ellipsoids i and j overlap, False otherwise

    Notes
    -----
    The method calculates the characteristic polynomial based on the ellipsoid coefficients
    and rigid body transformations. Roots of this polynomial are used with the Algebraic
    Separation Condition to determine whether the ellipsoids intersect. Real roots with
    negative values are analyzed to establish the overlap.
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
