"""Fixed-size positional relaxation of packed ellipsoids."""
from itertools import product
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from .collisions import collide_detect


def validate_relaxation_steps(steps):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('relaxation_steps must be a nonnegative integer')


def relax_particles(particles, box, periodic, *, max_steps=2000, verbose=False):
    """Remove overlaps by damped pair-separation corrections, without growth.

    Candidate pairs come from a bounding-sphere tree. All potentially contacting
    lattice images are tested, including self-images; no minimum-image-only
    assumption is made. A pair correction follows the normal of the ellipsoid
    contact function. Corrections are averaged and limited before
    simultaneous application, and contacts are recomputed after every step.

    Returns refreshed particles/images and a convergence report. An infeasible
    or unconverged packing is returned with an explicit warning, never shrunk.
    """
    validate_relaxation_steps(max_steps)
    if max_steps == 0:
        return particles, dict(enabled=False, converged=None, steps=0, stop_reason='disabled')
    originals = [p for p in particles if p.duplicate is None]
    lengths = np.array([box.w, box.h, box.d], float)
    origin = np.array([box.left, box.top, box.front], float)
    axes = np.array([[p.a, p.b, p.c] for p in originals]).reshape(-1, 3)
    rotations = np.array([p.rotation_matrix for p in originals]).reshape(-1, 3, 3)
    extent = np.sqrt(np.einsum('ni,nij->nj', axes**2, rotations**2))
    shapes = np.array([r.T @ np.diag(a**2) @ r for a, r in zip(axes, rotations)])
    radii = axes.max(axis=1) if len(axes) else np.empty(0)
    centers = np.array([p.get_pos() for p in originals]).reshape(-1, 3)
    impossible_walls = not periodic and bool(np.any(2*extent > lengths))
    if periodic:
        centers = origin + (centers-origin) % lengths
    elif not impossible_walls:
        centers = np.clip(centers, origin+extent, origin+lengths-extent)

    def contacts():
        if not len(originals):
            return []
        local = (centers-origin) % lengths if periodic else centers-origin
        tree = cKDTree(local, boxsize=lengths if periodic else None)
        pairs = sorted(tree.query_pairs(2*radii.max()))
        if periodic:
            pairs += [(i, i) for i in range(len(originals))]
        found = []
        for i, j in pairs:
            delta = centers[j]-centers[i]
            radius = radii[i]+radii[j]
            if periodic:
                lower = np.ceil((-radius-delta)/lengths).astype(int)
                upper = np.floor((radius-delta)/lengths).astype(int)
                shifts = product(*(range(a, b+1) for a, b in zip(lower, upper)))
            else:
                shifts = [(0, 0, 0)]
            for shift in shifts:
                if i == j:
                    nonzero = [v for v in shift if v]
                    if not nonzero or nonzero[0] < 0:
                        continue
                d = delta + np.asarray(shift)*lengths
                if np.dot(d, d) > radius**2:
                    continue
                if collide_detect(axes[i], axes[j], np.zeros(3), d, rotations[i], rotations[j]):
                    found.append((i, j, d))
        return found

    history = []
    steps = 0
    while True:
        overlapping = contacts()
        history.append(len(overlapping))
        if not overlapping or steps >= max_steps or impossible_walls:
            break
        correction = np.zeros_like(centers)
        degree = np.zeros(len(originals))
        for i, j, d in overlapping:
            if i == j:
                continue  # Translating a particle cannot separate its own images.
            distance = np.linalg.norm(d)
            direction = d/distance if distance > 0 else np.ones(3)/np.sqrt(3)
            # The maximum is the contact value for unit centre separation.
            # For fixed shapes it scales with separation squared.
            def objective(t):
                return -t*(1-t)*direction @ np.linalg.solve(
                    (1-t)*shapes[i]+t*shapes[j], direction)
            optimum = minimize_scalar(objective, bounds=(0., 1.), method='bounded',
                                      options={'xatol': 1e-12})
            t = optimum.x
            root = np.sqrt(-optimum.fun)
            # Gradient of sqrt(contact value) with respect to centre separation.
            # Unlike a centre-line force this accounts for the contact normal
            # of anisotropic particles. For spheres it gives the exact gap.
            normal = t*(1-t)*np.linalg.solve((1-t)*shapes[i]+t*shapes[j], direction)/root
            deficit = max(0., 1+1e-7-distance*root)
            delta = .5*deficit*normal/np.dot(normal, normal)
            correction[i] -= delta
            correction[j] += delta
            degree[i] += 1
            degree[j] += 1
        correction /= np.maximum(degree, 1)[:, None]
        size = np.linalg.norm(correction, axis=1)
        cap = .1*axes.min(axis=1)
        correction *= np.minimum(1., cap/np.maximum(size, np.finfo(float).tiny))[:, None]
        previous = centers.copy()
        centers += correction
        if periodic:
            centers = origin + (centers-origin) % lengths
        else:
            centers = np.clip(centers, origin+extent, origin+lengths-extent)
        steps += 1
        box.sim_ts += 1
        if verbose and steps % 50 == 0:
            print(f'Relaxation step {steps}: {len(overlapping)} overlapping image contacts')
        if np.array_equal(centers, previous):
            overlapping = contacts()
            break

    converged = not overlapping and not impossible_walls
    reason = ('complete' if converged else 'particle_larger_than_box' if impossible_walls
              else 'step_limit' if steps >= max_steps else 'stalled')
    report = dict(enabled=True, converged=converged, steps=steps,
                  initial_contacts=history[0], remaining_contacts=len(overlapping),
                  contact_history=history, stop_reason=reason)
    # Discard growth momentum and stale forces; rebuild images at the final size.
    images = []
    for p, center in zip(originals, centers):
        p.x, p.y, p.z = center
        p.xold, p.yold, p.zold = center
        p.speedx = p.speedy = p.speedz = 0.
        p.force_x = p.force_y = p.force_z = 0.
        p.ncollision = 0
        p.branches = []
        p.neighborlist = set()
        p.set_cub()
        if periodic:
            images.extend(p.wallCollision(box, True))
    print(f'Final relaxation: {steps} steps, {history[0]} -> {len(overlapping)} '
          f'overlapping image contacts ({reason})')
    if not converged:
        warnings.warn(f'Packing relaxation did not converge: {reason}; '
                      f'{len(overlapping)} overlapping image contacts remain.', RuntimeWarning,
                      stacklevel=2)
    return originals + images, report
