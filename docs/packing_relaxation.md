# Final packing relaxation

`Microstructure.pack()` now performs fixed-size positional relaxation after particle growth reaches its requested fill factor. The default maximum is 2,000 relaxation steps:

```python
ms.pack(fill_factor=0.5, relaxation_steps=2000)
print(ms.simbox.packing_relaxation)
```

Use `relaxation_steps=0` to disable this stage. The same option is available on `packingRoutine()` and `particle_grow()`.

The relaxation holds semiaxes and orientations fixed and removes growth momentum. It applies bounded, averaged position corrections along the ellipsoid contact-function normals. These are geometric relaxation iterations, not physical time integration; optional attraction and long-range repulsion used during growth are not applied during relaxation.

A bounding-sphere spatial tree selects candidates, followed by analytical ellipsoid collision checks. For periodic boxes, every potentially contacting relative lattice image is checked, not only the nearest centre image. Corrections act on original particles, positions are wrapped, and periodic copies are rebuilt on completion. Nonperiodic particles are constrained using exact rotated ellipsoid extents. The stage stops once no detected overlaps remain. An already separated packing retains its positions.

The report on `ms.simbox.packing_relaxation` contains:

- `enabled`, `converged`, and `steps`;
- `initial_contacts` and `remaining_contacts`, counted as distinct parent-pair/relative-image contacts, including self-images;
- `contact_history`, the contact counts observed during relaxation;
- `stop_reason`: `complete`, `step_limit`, `stalled`, `particle_larger_than_box`, or `disabled`.

Contacts may increase temporarily as one overlap is removed and another pair touches. Reaching the limit or an infeasible configuration emits a `RuntimeWarning` and returns the current packing with `converged=False`. It never shrinks particles to obtain apparent convergence. Translation cannot eliminate self-image overlap, and fixed-size relaxation is not guaranteed to find a feasible packing at arbitrary density. Increasing `relaxation_steps` may help a slowly converging configuration.

The final packing statistics now use a read-only contact check, so they do not reintroduce forces after relaxation. When dump output is requested, a final dump includes the relaxed state.

Validation for the five-ellipsoid example: the saved overlapping packing converged from six contacts to zero in 1,906 steps. An independent scalar contact-function check found all tested pair contact values above one; semiaxes and orientations were preserved. A fresh notebook packing was already separated after growth and needed zero relaxation steps. These are separate configurations, not a comparison of the same stochastic packing run. Packing relaxation does not by itself guarantee a manifold downstream grain-growth surface.
