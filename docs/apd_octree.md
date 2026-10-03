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

## Cleanup of thin assigned grain regions

```python
thin = octree.thin_grain_cells(max_width=2)  # inspect without changing labels
report = octree.clean_thin_grains(apd, max_width=2, max_passes=3)
print(report['changed_cells'], report['remaining_thin_cells'])
print(report['volumes_before'], report['volumes_after'])
octree.plot_slice(axis='z', color_by='grain')
```

Cleanup changes only `octree.labels`, in place. A thin feature is a run of one
or two finest-level boundary-candidate cells along any coordinate axis, bounded
at both ends by other grains. All cells in the run must belong to the refined
GB zone. Domain exterior is not another grain. Face neighbours are found by
dyadic ancestor lookup, including coarse neighbours and periodic wrapping,
without expanding the octree into a dense grid. This directional definition
does not measure arbitrary oblique thickness or certify manifoldness.

The replacement is the lowest centre-cost APD grain among current face neighbours,
excluding the current grain. Cost ties follow APD grain order. Changes are applied
sequentially, with cheaper initial cost penalties processed first in each pass;
thickness and neighbouring labels are rechecked before each change. Only cells
detected as thin at the start are eligible, and each may change once per call.
This prevents repeated erosion or label oscillation within a cleanup call.
The remaining-thin mask includes newly formed thin regions, which are reported
rather than automatically added to the original cleanup set. Repeated explicit
calls can therefore cause additional changes.

The report includes initial, changed and remaining boolean leaf masks, per-cell
old/new labels and centre-cost increases, before/after grain volumes, and any
eliminated grains. Small grains can disappear, junctions can shift, and grain
connectivity or manifoldness is not guaranteed. Volumes refer to assigned whole
leaf cells, not exact continuous APD regions.

Cell bounds, levels, APD weights, and existing voxel data remain unchanged.
`boundary_cells` and `sampled_boundary` continue to describe the original APD;
the boundary plot is therefore not a recomputed boundary of the cleaned labels.
The cleaned labels are not yet consumed by `build_grain_geometry`: evaluating
the original APD again will recover its original cost winners.

Run the before/after example with:

```sh
PYTHONPATH=src python examples/RVE_generation/apd_octree_preview.py --clean-thin
```

## Relationship to build_grain_geometry

`build_grain_geometry` still uses its uniform tetrahedral background. The octree
is available independently through `diagram.background_octree(...)` or
`kanapy.core.apd_octree.build_background_octree(...)`. It is not yet a drop-in
replacement: leaves have hanging nodes, are not 2:1 balanced, and periodic face
subdivisions are not paired. The existing point locator also assumes a uniform
Freudenthal grid. Connecting the octree to geometry reconstruction requires the
conforming tetrahedral stage and an adaptive point locator.

## Non-manifold regions of the labelled octree

```python
mask = octree.nonmanifold_grain_boundary_cells()
report = octree.nonmanifold_grain_boundary_cells(return_report=True)
print(report['nonmanifold_cells'], report['nonmanifold_vertices'])
for contact in report['contacts']:
    print(contact['point'], contact['grain_id'], contact['cell_indices'])
octree.plot_slice(axis='z', color_by='nonmanifold')
```

The mask has one entry per leaf. Indices in the report are **zero-based octree
leaf indices**, not the one-based IDs of the original voxel mesh. The equivalent
standalone function is
`kanapy.core.apd_octree.nonmanifold_octree_grain_boundary_cells(octree, ...)`.

This uses the same per-grain surface-link definition as
`nonmanifold_grain_boundary_voxels`: an edge or vertex self-contact is flagged;
ordinary junctions between distinct grains are accepted if each individual
grain's local surface is manifold. It checks the **current assigned labels**,
including cleanup edits, and finds the GB zone from current face contacts.
The original APD `boundary_cells` flags do not restrict this check.

At every leaf corner, dyadic ancestor lookup finds the leaves occupying its
eight surrounding octants. A coarse leaf may occupy several octants at a hanging
node. The local occupancy is checked using the shared 256-pattern surface-link
table. No dense finest-resolution volume is built, no balancing is required,
and the octree is not modified. Periodic vertices are identified across seams;
nonperiodic exterior closes shells but does not count as another grain.

The report contains `cell_mask`, `cell_indices`, `nonmanifold_cells`,
`nonmanifold_vertices`, and `contacts`. Each contact describes one offending
grain at one vertex: `vertex_index` is in the finest integer lattice, `point`
is in physical coordinates, and `cell_indices` lists incident flagged leaves of
that grain. Periodic points use the half-open box. Reports deduplicate contacts
but do not group them into connected regions; a non-manifold edge can have
observations at both endpoints. With `return_report=False`, only the mask is
retained. `chunk_size` controls temporary corner batches (default 65536).

This diagnoses the cubical surfaces of the assigned leaves, not the continuous
APD or global grain connectivity. It performs no repair. The slice plot colours
flagged leaves intersecting the plane; the critical vertex itself may lie above
or below that plane within such a leaf.

## Proposed topology and tetrahedral stages

For step (2), distinguish sampling contacts from contacts of the continuous APD.
The implemented octree detector handles hanging nodes through local octant
ownership; directly indexing neighbouring leaf IDs as a regular 2x2x2 grid would
not do so. For subsequent repair and meshing, balance neighbouring
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
