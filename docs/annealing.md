# Phase-field annealing after voxelization

Install the optional dependency in the Kanapy environment with
`python -m pip install 'kanapy[annealing]'` (or `python -m pip install fipy`).

```python
microstructure.voxelize()
report = microstructure.anneal()
print(report['reason'], report['max_volume_change'])
microstructure.plot_voxels()
# Optional geometric smoothing of the annealed voxel boundaries:
microstructure.smoothen()
```

`Microstructure.anneal()` evolves one continuous Allen–Cahn field per grain
using FiPy. The first version supports fully dense single-phase structures.
It uses isotropic boundary properties with dimensionless mobility 10,
gradient coefficient 2, time step 0.002, and at most 2000 steps. These are
shape-relaxation parameters, not a calibrated temperature/time prescription.
Spacing ratios are retained; the shortest voxel edge defines the length unit.
Increasing mobility speeds the time scale, not the final equilibrium.

The bulk free energy is
`sum(eta_i**4/4 - eta_i**2/2) + 1.5*sum(i<j, eta_i**2*eta_j**2)`;
the gradient term is `kappa/2*sum(|grad eta_i|**2)`.
Diffusion and reaction sinks are implicit, while coupling coefficients and
the growth source use the previous time level. Every grain uses the same old
state. Initial fields are binary grain indicators; they develop diffuse
interfaces during evolution. There is no imposed sum-to-one constraint.

The stopping rule is `max(abs(V / V_initial - 1)) >= 0.1`, checked after
each step over every initial grain. Volumes use voxel counts after argmax
assignment, not integrals of diffuse fields. The first crossing is retained;
voxel quantization and finite steps may overshoot 10%, especially for small
grains. `reason='max_steps'` means the threshold was not reached and emits a
warning through the API. A one-grain domain returns immediately.

Options can be overridden, for example:

```python
report = microstructure.anneal(mobility=10., dt=0.001,
                              max_steps=4000, volume_change=0.1)
```

Use `mobility * dt <= 0.05`. Check time-step and grid convergence for
quantitative studies. Memory grows with the number of grains times voxels.
Interfaces need several voxels to resolve, and very small grains can vanish.
Periodicity is inherited from the voxelization APD or RVE; override it with
`periodic=True/False`. Nonperiodic boundaries have zero normal flux.

The result updates grain labels, grain/phase dictionaries, particle voxel
memberships, grain counts, and surviving orientations. Smoothed nodes,
statistics and old APD geometry are invalidated. `generate_grains()` currently
reconstructs APD geometry from packed particles, so it is blocked after
annealing to avoid silently recreating the original grain shapes. Use the
voxel representation or `smoothen()` instead. Revoxelizing creates a fresh mesh.

For array-only use, `kanapy.core.annealing.grain_growth(labels, ...)` returns
`(new_labels, report)` without modifying its input. The report contains initial
and final volumes, original grain IDs, stopping reason, step count, simulation
time, and the maximum relative volume-change history. Volumes use supplied
spacing units cubed; time is dimensionless.
