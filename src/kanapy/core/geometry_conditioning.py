"""Volume-controlled, damped Lloyd conditioning of an existing APD.

This changes seed positions, not crystallographic orientations or metrics. It is
not curvature flow and does not guarantee connected cells or a minimum radius.
"""
from copy import deepcopy
import warnings

import numpy as np

from .power_diagram import AnisotropicPowerDiagram


def _measure(diagram, points):
    """Sample volumes and seed-relative first moments using metric-nearest images."""
    n = len(diagram.centers)
    counts = np.zeros(n, dtype=int)
    moments = np.zeros((n, 3))
    energy = 0.
    for start in range(0, len(points), 4096):
        sample = points[start:start+4096]
        costs = diagram.costs(sample, weighted=False)
        labels = np.argmin(costs-diagram.weights, axis=1)
        counts += np.bincount(labels, minlength=n)
        energy += costs[np.arange(len(sample)), labels].sum()
        for i in np.unique(labels):
            owned = sample[labels == i]
            if diagram.periodic:
                tree, transform = diagram._image_trees[i]
                # Coordinatewise minimum-image wrapping is wrong for rotated
                # anisotropy. Use exactly the metric image selected by costs().
                indices = tree.query(owned @ transform)[1]
                centers = np.linalg.solve(transform.T, tree.data[indices].T).T
                delta = owned-centers
            else:
                delta = owned-diagram.centers[i]
            moments[i] += delta.sum(axis=0)
    offsets = np.divide(moments, counts[:, None], out=np.zeros_like(moments),
                        where=counts[:, None] != 0)
    return counts*diagram.volume/len(points), offsets, float(energy/len(points))


def condition_apd(diagram, *, iterations=5, damping=.5, max_displacement=None,
                  max_volume_change=.1, n_samples=32768, validation_samples=16384,
                  seed=19, fit_tolerance=.03, fit_maxiter=300,
                  centroid_tolerance=1e-3, backtracking_steps=8):
    """Return an independent conditioned APD and a serializable report.

    Damped seed-to-centroid moves alternate with volume-weight refits. Targets
    are the input diagram's sampled volumes, preserving the existing geometry
    rather than silently imposing different particle targets. A separate Sobol
    sample gates relative volume changes against the input. All accepted moves
    must also decrease sampled unweighted transport energy. Rejected candidates
    are backtracked and never committed.

    max_displacement is a cumulative seed-travel budget in coordinate units;
    None uses one quarter of each grain's initial sampled volume cube root.
    centroid_tolerance is relative to that grain length. Matrices/IDs are fixed.
    Periodic centroids use metric-nearest seed images. Validation is quadrature,
    not a guarantee on subsequently reconstructed polyhedral volumes.
    """
    for name, value in [('iterations', iterations), ('fit_maxiter', fit_maxiter),
                        ('backtracking_steps', backtracking_steps)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, value, upper in [('damping', damping, 1.), ('max_volume_change', max_volume_change, 1.),
                               ('fit_tolerance', fit_tolerance, 1.)]:
        if not np.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f'{name} must be in (0, {upper}]')
    if not np.isfinite(centroid_tolerance) or centroid_tolerance < 0:
        raise ValueError('centroid_tolerance must be finite and nonnegative')
    if max_displacement is not None and (not np.isfinite(max_displacement) or max_displacement < 0):
        raise ValueError('max_displacement must be finite and nonnegative')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    # _sample_points validates positive power-of-two quadrature sizes.
    sample = diagram._sample_points(n_samples, seed)
    validation = diagram._sample_points(validation_samples, seed+1)
    volumes, offsets, energy = _measure(diagram, sample)
    baseline, _, _ = _measure(diagram, validation)
    if np.any(volumes == 0) or np.any(baseline == 0):
        raise ValueError('Conditioning quadrature misses a grain; increase sample counts')
    current = deepcopy(diagram)
    lengths = np.cbrt(volumes)
    limits = .25*lengths if max_displacement is None else np.full(len(volumes), max_displacement)
    travel = np.zeros(len(volumes))
    initial_energy = energy
    initial_offsets = offsets.copy()
    history = []
    reason = 'iteration_limit'
    for iteration in range(iterations):
        norms = np.linalg.norm(offsets, axis=1)
        if np.max(norms/lengths) <= centroid_tolerance:
            reason = 'centroid_tolerance'; break
        remaining = np.maximum(0, limits-travel)
        step = damping*offsets
        step *= np.minimum(1., np.divide(remaining, np.linalg.norm(step, axis=1),
                           out=np.ones_like(remaining), where=norms != 0))[:, None]
        if np.max(np.linalg.norm(step, axis=1)) <= np.finfo(float).eps*np.linalg.norm(diagram.box_size):
            reason = 'displacement_budget'; break
        accepted = False
        for attempt in range(backtracking_steps):
            delta = step*(.5**attempt)
            candidate = AnisotropicPowerDiagram(current.centers+delta, current.matrices,
                current.box_size, volumes, current.grain_ids, current.periodic)
            candidate.weights = current.weights.copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fit = candidate.fit_volumes(n_samples=n_samples, seed=seed,
                    tolerance=fit_tolerance, maxiter=fit_maxiter)
            _, proposed_offsets, proposed_energy = _measure(candidate, sample)
            checked, _, _ = _measure(candidate, validation)
            error = np.abs(checked/baseline-1)
            if (fit['converged'] and np.all(checked > 0)
                    and error.max() <= max_volume_change
                    and proposed_energy < energy-1e-12*max(np.finfo(float).tiny, abs(energy))):
                candidate._ellipsoids = [(center.copy(), axes.copy(), rotation.copy())
                    for center, (_, axes, rotation) in zip(candidate.centers, diagram._ellipsoids)]
                current, offsets, energy = candidate, proposed_offsets, proposed_energy
                travel += np.linalg.norm(delta, axis=1)
                accepted = True
                break
        history.append(dict(iteration=iteration+1, accepted=accepted, attempts=attempt+1,
            sampled_energy=energy, candidate_energy=proposed_energy,
            validation_max_relative_volume_change=float(error.max()),
            fit_max_relative_error=float(fit['max_relative_error']),
            fit_converged=bool(fit['converged']),
            rejection_reason=None if accepted else (
                'volume_fit' if not fit['converged'] else
                'validation_volume_budget' if error.max() > max_volume_change or np.any(checked <= 0)
                else 'transport_energy'),
            max_seed_travel=float(travel.max())))
        if not accepted:
            reason = 'no_admissible_step'; break
    final_volumes, _, _ = _measure(current, validation)
    report = dict(method='damped_lloyd', stop_reason=reason, history=history,
        parameters=dict(iterations=int(iterations), damping=float(damping),
            max_volume_change=float(max_volume_change), fit_tolerance=float(fit_tolerance),
            fit_maxiter=int(fit_maxiter), centroid_tolerance=float(centroid_tolerance),
            backtracking_steps=int(backtracking_steps)),
        accepted_iterations=sum(h['accepted'] for h in history),
        n_samples=int(n_samples), validation_samples=int(validation_samples), seed=int(seed),
        volume_targets='initial sampled APD volumes',
        initial_training_energy=initial_energy, final_training_energy=energy,
        initial_centers=diagram.centers.tolist(), final_centers=current.centers.tolist(),
        initial_max_relative_centroid_offset=float(np.max(np.linalg.norm(initial_offsets,axis=1)/lengths)),
        final_max_relative_centroid_offset=float(np.max(np.linalg.norm(offsets,axis=1)/lengths)),
        initial_validation_volumes={int(g):float(v) for g,v in zip(diagram.grain_ids,baseline)},
        final_validation_volumes={int(g):float(v) for g,v in zip(diagram.grain_ids,final_volumes)},
        relative_volume_change={int(g):float(v/b-1) for g,v,b in zip(diagram.grain_ids,final_volumes,baseline)},
        seed_travel={int(g):float(v) for g,v in zip(diagram.grain_ids,travel)},
        displacement_limits={int(g):float(v) for g,v in zip(diagram.grain_ids,limits)},
        curvature_bound_enforced=False, connectivity_guaranteed=False)
    return current, report
