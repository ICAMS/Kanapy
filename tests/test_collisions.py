import pytest
import numpy as np
from kanapy.core.entities import Ellipsoid
from kanapy.core.collisions import collision_routine, collision_react, collide_detect

# ------------------------------
# Test different ellipsoid scenarios
# ------------------------------
test_cases = [
    # Regular collision
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 0.8, 0, 0, 1, 1, 1, np.array([1,0,0,0]))),

    # Light touch collision
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 1.95, 0, 0, 1, 1, 1, np.array([1,0,0,0]))),

    # Boundary contact
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 2.0, 0, 0, 1, 1, 1, np.array([1,0,0,0]))),

    # Full overlap
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0]))),

    # Extreme size difference
    ((1, 0, 0, 0, 0.5, 0.5, 0.5, np.array([1,0,0,0])),
     (2, 1.0, 0, 0, 2, 2, 2, np.array([1,0,0,0]))),

    # Rotated ellipsoid
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 0.5, 0, 0, 1, 1, 1, np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]))),

    # Zero initial forces
    ((1, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0])),
     (2, 0, 0, 0, 1, 1, 1, np.array([1,0,0,0]))),
]

# ------------------------------
# Test collision_routine
# ------------------------------
@pytest.mark.parametrize("ell1_params, ell2_params", test_cases)
def test_collision_routine(ell1_params, ell2_params):
    ell1 = Ellipsoid(*ell1_params)
    ell2 = Ellipsoid(*ell2_params)

    overlap = collision_routine(ell1, ell2)
    assert isinstance(overlap, bool)

    if overlap:
        assert ell1.force_x != 0 or ell1.force_y != 0 or ell1.force_z != 0
        assert ell2.force_x != 0 or ell2.force_y != 0 or ell2.force_z != 0

# ------------------------------
# Test collision_react
# ------------------------------
@pytest.mark.parametrize("ell1_params, ell2_params", test_cases)
def test_collision_react(ell1_params, ell2_params):
    ell1 = Ellipsoid(*ell1_params)
    ell2 = Ellipsoid(*ell2_params)

    f1 = (ell1.force_x, ell1.force_y, ell1.force_z)
    f2 = (ell2.force_x, ell2.force_y, ell2.force_z)

    collision_react(ell1, ell2)
    assert (ell1.force_x, ell1.force_y, ell1.force_z) != f1
    assert (ell2.force_x, ell2.force_y, ell2.force_z) != f2

# ------------------------------
# Test collide_detect
# ------------------------------
@pytest.mark.parametrize("ell1_params, ell2_params", test_cases)
def test_collide_detect(ell1_params, ell2_params):
    ell1 = Ellipsoid(*ell1_params)
    ell2 = Ellipsoid(*ell2_params)

    coef_i = ell1.get_coeffs()
    coef_j = ell2.get_coeffs()
    r_i = ell1.get_pos()
    r_j = ell2.get_pos()
    A_i = ell1.rotation_matrix
    A_j = ell2.rotation_matrix

    overlap = collide_detect(coef_i, coef_j, r_i, r_j, A_i, A_j)
    assert isinstance(overlap, bool)
