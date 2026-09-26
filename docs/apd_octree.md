# Adaptive APD background: octree preparation

This implements the octree stage before tetrahedralization. It samples the
continuous APD and does not change voxel assignments, APD weights, or geometry.

```python
import matplotlib.pyplot as plt

apd = ms.mesh.apd  # or ms.geometry['APD'], or an AnisotropicPowerDiagram
octree = apd.background_octree(resolution=2, max_depth=4, max_cells=1_000_000)
print(octree.summary())

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
for ax, color in zip(axes, ['level', 'grain', 'boundary']):
    octree.plot_slice(axis='z', color_by=color, ax=ax)
plt.show()
```

`resolution` is the root cell count per axis (an integer or three integers).
Each subdivision creates eight children. For example, resolution 2 and depth 4
give a finest spacing of `box_size / 32`. Root cells and children are rectangular
when domain lengths and cell counts do not yield cubes. This is an octree forest
with multiple roots. `max_cells` limits total leaf count and raises an exception
before a subdivision exceeds the budget; no partial result is silently returned.
`batch_size` bounds the number of points in each cost evaluation.

Inspect `indices`, `levels`, `lower`, `upper`, `sizes`, `centers`, `labels`,
`boundary_cells`, and `sampled_boundary`. The bounds cover the full box without
overlap; centre labels are diagnostic samples, not a new voxelization.

## Refinement criterion

A cell is retained as interior only if cost bounds establish that its centre
winner dominates throughout the entire box. Otherwise it subdivides until the
depth limit. Nonperiodic diagrams bound the quadratic difference between each
competitor and the centre winner. Periodic diagrams use the triangle inequality
for distance to a grain's lattice images in its anisotropic metric; these bounds
remain valid across changes of the nearest image. Floating-point margins bias
ambiguous decisions towards refinement. These are conservative geometric bounds,
not formal interval-arithmetic certificates.

This avoids relying on corner labels, which can miss an enclosed small grain.
Bounds can over-refine, particularly for periodic diagrams or tied cost functions.
Reaching the depth limit does not guarantee all small features are resolved.

In boundary plots, **interior** means a single-grain bound succeeded;
**candidate** means a boundary cannot be excluded; **sampled GB** means different
grain IDs were found among the 27 corner, edge-midpoint, face-centre and centre
samples. Sampling is a visualization aid, not the refinement criterion.
Depth-limited candidates are **not** labelled as non-manifold or unsolved topology:
ordinary resolved interfaces also remain candidates.

Slices accept `axis='x'`, `'y'`, or `'z'`, and a physical `position` (default:
midplane). They plot actual leaf rectangles; grain colours use leaf centre labels
and do not depict a reconstructed smooth grain interface. Plotting returns an
Axes without calling `show()`.

## Relationship to build_grain_geometry

`build_grain_geometry` still uses its uniform tetrahedral background. The octree
is available independently through `diagram.background_octree(...)` or
`kanapy.core.apd_octree.build_background_octree(...)`. It is not yet a drop-in
replacement: leaves have hanging nodes, are not 2:1 balanced, and periodic face
subdivisions are not paired. The existing point locator also assumes a uniform
Freudenthal grid. Connecting the octree to geometry reconstruction requires the
conforming tetrahedral stage and an adaptive point locator.

## Proposed topology and tetrahedral stages

For step (2), distinguish sampling contacts from contacts of the continuous APD.
The regular-grid 2x2x2 occupancy lookup cannot be applied directly across hanging
nodes. Use a common local refinement or a conforming surface-link representation
to inspect each grain around an adaptive-grid junction. Balance neighbouring
cells, propagate periodic seam refinements, and check the continuous APD at each
suspect point before deciding to refine. Re-coarsening is appropriate only when
it passes the same geometric error and topology checks; otherwise it may hide a
thin grain or a genuine contact. Use a bounded repair loop and preserve a list
of unresolved locations, involved grain IDs, depth and diagnostic reason.

For step (3), choosing cube diagonals alone does not remove a stair-stepped
interface. Use the continuous cost equality `f_i(x) = f_j(x)` to locate shared
interface points, and its gradient `grad(f_i - f_j)` as a local normal away from
singularities and periodic branch changes. Constrain triple lines and junctions
jointly rather than projecting independently onto grain-pair surfaces. Construct
shared faces once, resolve hanging nodes with conforming transition templates,
and pair periodic faces. Validate surface links, shared-face conformity, positive
tetrahedral volumes and element quality after fitting. A genuine singularity
cannot always be repaired without changing the intended geometry; such cases
must remain explicitly unresolved. The current APD tetrahedral clipping already
interpolates continuous costs, so its roughness is interpolation error rather
than simply exposed voxel faces.
