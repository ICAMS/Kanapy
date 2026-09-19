# APD voxelization

`Microstructure.voxelize(particles=None, dim=None)` now uses the anisotropic
power diagram (APD). Its original positional arguments remain valid.

Each voxel receives the grain ID minimizing
`(x - center).T @ A @ (x - center) - weight`, evaluated at the voxel center.
The metric includes the packed ellipsoid's semiaxes and rotation. Periodic
RVEs use the minimum cost over periodic images; duplicates share the original
grain ID. Exact ties follow original-particle order.

Weights are fitted to relative packed particle volumes using continuous Sobol
quadrature by default. Final voxel fractions have discretization error, and
small grains can disappear. Connectivity is not imposed.

```python
ms.voxelize(fit_options={"n_samples": 32768, "tolerance": 0.03})
ms.voxelize(fit_volumes=False)  # zero additive weights
ms.voxelize(weights=my_weights, chunk_size=4096)
```

`periodic` can override RVE periodicity. Explicit weights are ordered by original
particles, excluding duplicates, and bypass fitting. The diagram is available
as `ms.mesh.apd`; its coordinates are relative to the minimum mesh coordinates.

The low-level `kanapy.core.voxelization.voxelizationRoutine` retains
`(Ellipsoids, mesh, nphases, prec_vf=None)`. `prec_vf` is ignored with a
`DeprecationWarning`, since APD fills the entire box. Polygon interiors are not
used by the APD metric.

For the historical growth/polygon method, including matrix, precipitate and
porosity fractions, use `ms.voxelize_legacy()` or
`ms.voxelize(method="legacy")`. Its implementation and helpers reside in
`kanapy.core.voxelization_legacy`, with `_legacy` appended to function names.
The historical unit tests reside in `tests/test_voxelization_legacy.py`.
