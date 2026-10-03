"""Generic single-phase grain growth using FiPy (an optional dependency)."""
import numpy as np


def grain_growth(grains, *, spacing=(1., 1., 1.), periodic=False,
                 mobility=10., kappa=2., dt=0.002, max_steps=2000,
                 volume_change=0.1):
    """Return annealed labels and a diagnostics dictionary, without mutating input.

    One Allen-Cahn field per grain evolves with bulk energy
    sum_i (eta_i**4/4 - eta_i**2/2) + 1.5*sum_{i<j} eta_i**2*eta_j**2
    and gradient energy kappa/2*sum_i |grad eta_i|**2. Lengths are scaled
    by the smallest voxel spacing; mobility and time are dimensionless.
    Diffusion and nonnegative reaction sinks are implicit, with all coupling
    coefficients frozen at the previous time level (no grain-order bias).

    Stop at the FIRST completed step with max_i |V_i/V_i_initial - 1|
    >= volume_change. Volumes are counts of argmax-labelled voxels, including
    extinct grains. Discrete voxels/time steps can overshoot the threshold.
    max_steps is a safety limit, not evidence that the volume target was met.
    Nonperiodic boundaries have zero flux. Memory scales as grains * voxels.
    """
    labels = np.asarray(grains)
    if (labels.ndim != 3 or labels.size == 0 or
            not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('grains must be a nonempty 3D integer array')
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError('periodic must be a boolean')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError('spacing must contain three positive finite values')
    for name, value in [('mobility', mobility), ('kappa', kappa), ('dt', dt),
                        ('volume_change', volume_change)]:
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if volume_change > 1:
        raise ValueError('volume_change must be at most 1')
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if mobility * dt > 0.05:
        raise ValueError('Use mobility * dt <= 0.05 for temporal resolution')
    try:
        from fipy import CellVariable, Grid3D, PeriodicGrid3D
        from fipy import TransientTerm, DiffusionTerm, ImplicitSourceTerm
        from fipy.solvers.scipy import LinearLUSolver
    except ImportError as exc:
        raise ImportError('Grain growth requires FiPy: python -m pip install fipy') from exc
    ids, initial = np.unique(labels, return_counts=True)
    result = labels.copy()
    counts = initial.copy()
    history = [0.]
    reason = 'single_grain' if len(ids) == 1 else 'max_steps'
    step = 0
    if len(ids) > 1:
        h = spacing / spacing.min()
        grid = (PeriodicGrid3D if periodic else Grid3D)(
            nx=labels.shape[0], ny=labels.shape[1], nz=labels.shape[2],
            dx=h[0], dy=h[1], dz=h[2])
        # FiPy has x-fastest indexing; Kanapy voxel IDs use C order.
        fields = [CellVariable(mesh=grid, value=(labels == gid).astype(float).ravel(order='F'),
                               hasOld=True) for gid in ids]
        sink = CellVariable(mesh=grid, value=0.)
        solver = LinearLUSolver(tolerance=1e-10)
        for step in range(1, max_steps + 1):
            for field in fields:
                field.updateOld()
            total = sum(np.asarray(field.old.value)**2 for field in fields)
            best = np.full(labels.size, -np.inf)
            winners = np.zeros(labels.size, dtype=int)
            for i, field in enumerate(fields):
                old = np.asarray(field.old.value)
                sink.setValue(mobility * (old**2 + 3.*(total-old**2)))
                equation = (TransientTerm() == DiffusionTerm(coeff=mobility*kappa)
                            - ImplicitSourceTerm(coeff=sink) + mobility*field.old)
                equation.solve(var=field, dt=dt, solver=solver)
                values = np.asarray(field.value)
                if not np.all(np.isfinite(values)):
                    raise RuntimeError('Nonfinite phase-field solution; reduce dt')
                take = values > best
                best[take] = values[take]
                winners[take] = i
            counts = np.bincount(winners, minlength=len(ids))
            change = float(np.max(np.abs(counts / initial - 1.)))
            history.append(change)
            result = ids[winners].reshape(labels.shape, order='F')
            if change >= volume_change:
                reason = 'volume_change'
                break
    cell_volume = float(np.prod(spacing))
    return result, dict(reason=reason, steps=step, time=step*dt,
                        max_volume_change=history[-1], threshold=volume_change,
                        grain_ids=ids, initial_volumes=initial*cell_volume,
                        final_volumes=counts*cell_volume,
                        max_volume_change_history=np.asarray(history))
