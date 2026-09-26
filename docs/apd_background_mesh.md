# Step 1: APD background mesh

For adaptive octree sampling before tetrahedralization, see
[the octree preparation stage](apd_octree.md). The tetrahedral workflow below
continues to use a uniform background.

Given an existing, optionally volume-fitted `AnisotropicPowerDiagram`:

```python
background = apd.background_mesh(resolution=(12, 16, 20), batch_size=8192)
print(background.summary())
```

Resolution counts Cartesian cells along each axis. Each cell is split into six
positively oriented tetrahedra with matching face diagonals between cells.
Coordinates use the APD domain `[0, box_size]`. No weights are fitted or changed.

Inspect these intermediate arrays:

- `points`: `(N, 3)` vertex coordinates.
- `tetrahedra`: `(M, 4)` zero-based vertex connectivity.
- `costs`: `(N, G)` weighted APD costs for all grains, not interpolated voxel labels.
- `grain_ids`: correspondence between cost columns and original grain IDs.
- `labels`: winning grain ID at each vertex; ties follow grain order.
- `faces`: unique triangular faces, with sorted zero-based vertex IDs.
- `face_tetrahedra`: two incident tetrahedron indices per face, with `-1` outside.
- `boundary_ids`: 0 internally; 1–6 for xmin, xmax, ymin, ymax, zmin, zmax.
- `signed_volumes`: signed volume of every tetrahedron.

`faces` are unoriented; later clipping must establish orientations from incident
cells. `summary()` reports counts, minimum and total tetrahedral volume, box
volume and positivity. Full face conformity is independently checked in
`tests/test_apd_mesh.py`, including a noncubic box with unequal grid spacings.

Cost evaluation is batched, but the complete cost array is retained for the next
clipping stage: its storage is O(N G). Mesh topology also requires memory
proportional to the number of tetrahedra. Begin with a modest resolution.

This is a reconstruction background, not a grain-conforming FE mesh. Periodic
APD costs can be sampled, but periodic boundary node pairing is not implemented.
Uniform sampling does not guarantee detection of arbitrarily small features.

For a voxelized microstructure with an attached diagram, use
`ms.mesh.apd.background_mesh(...)`. The resulting object remains separate from
`ms.mesh`, preserving the existing voxel representation.

## Step 2: inspect a local grain partition

```python
parts = background.partition_tetrahedron(0)
for part in parts:
    print(part.grain_id, part.volume, len(part.vertices), len(part.faces))
assert np.isclose(sum(p.volume for p in parts), background.signed_volumes[0])
```

Import NumPy as `np` for the check above. The standalone
`kanapy.core.apd_mesh.partition_tetrahedron(points, costs, grain_ids=None)`
accepts four corner coordinates and an array with four rows and one column per
grain. Costs are linearly interpolated within that tetrahedron; the routine does
not evaluate or change the continuous APD.

The partition is defined by intersecting the tetrahedron with pairwise grain
cost inequalities. By default, conservative pruning first removes strictly
dominated grains, and a single surviving candidate returns the original
tetrahedron directly. For multiple surviving candidates, the algorithm
enumerates intersections of triples of supporting planes in barycentric
coordinates and retains feasible vertices. This is equivalent to halfspace
clipping; incremental polygon clipping is not implemented. The exhaustive
reference path is available with `optimize=False`. Both paths can find regions
whose grain wins at none of the original corners.

Each returned `LocalGrainPolyhedron` contains:

- `grain_id`, physical `vertices`, four-component `barycentric` coordinates,
  outward-oriented polygon `faces` (local vertex indices), and `volume`.
- `constraints`: rows `[a0, a1, a2, b]` specifying
  `a @ barycentric[1:] + b <= 0`.
- `constraint_sources`: `('tetrahedron', k)` means the face opposite corner k;
  `('grain', id)` means the equality against that competing grain.
- `vertex_constraints` and `face_constraints`: indices into those constraint
  rows, preserving multiple coincident supporting constraints where applicable.

Exactly identical interpolated cost functions are assigned to the first grain
in input order. Other adjacent grain regions share their boundary closures.
Empty and lower-dimensional regions are omitted. Their constraints can still
appear on the boundaries of retained full-dimensional regions.

The verification suite checks analytical planar cuts, triple and four-grain
junctions, an interior-only grain, duplicates, dominated grains, random point
ownership, and affine coordinate changes. It independently reconstructs volume
from outward face triangles and checks closure via oriented edge incidence.

Limitations at this stage:

- Connectivity is local: vertices/faces in different tetrahedra are not welded.
- The unoptimized reference implementation has roughly O(G^5) work for G grains;
  use individual tetrahedra for inspection before any large-scale assembly.
- `tolerance` defaults to 1e-10 in normalized barycentric constraint calculations.
  Very thin cells, near-degenerate intersections, or ill-conditioned tetrahedra
  need additional numerical treatment. This is not exact-arithmetic geometry.
- Volume coverage should be checked at each inspected tetrahedron; features near
  the numerical tolerance may be unresolved.

The local routine does not share vertices or faces across tetrahedra; use the
step-3 assembler below for global connectivity.

## Step 3: assemble a globally shared partition

```python
partition = background.assemble()
print(partition.summary())
print(partition.grain_volumes)
```

Start with a small background mesh: assembly prunes local competitors, directly
returns strictly uncut tetrahedra, and uses the exhaustive step-2 partitioner
for the remaining candidates. It does not refit or modify the APD,
change background costs, triangulate interfaces, or generate FE tetrahedra.

The returned `APDPolyhedralMesh` stores:

- `points`, `vertex_keys`: global coordinates and topological identities.
- `faces`: shared polygon loops, each stored once, oriented outward from its
  first incident region.
- `face_regions`: first and second region indices; the second is -1 outside.
- `boundary_ids`: 0 internally, or the same 1–6 box-plane codes as step 1.
- `region_faces`, `region_face_signs`: face indices and orientations (+1/-1)
  reconstructing each region's closed outward shell.
- `region_grain_ids`, `region_tetrahedra`, `region_volumes`: grain ownership,
  parent background tetrahedron, and volume of every local fragment.
- `interface_faces`: indices of interior faces separating different grains.
- `grain_volumes`: volumes summed over fragments of each surviving grain.
- `timings`, `candidate_statistics`: stage timings and local candidate counts
  described below; also included in `summary()`.

Same-grain faces across background tetrahedra are retained for the complete
partition, but excluded from `interface_faces`. Regions are not yet merged into
complete grains, and periodic boundaries are not paired.

Vertex keys combine the IDs of the supporting background simplex's vertices
with the indices of tied minimum-cost columns. This avoids global coordinate
rounding as the identity mechanism. Matching coordinates are checked within a
scale-dependent tolerance. Output vertices, regions and faces have deterministic
ordering; `tetrahedron_order` can optionally supply a permutation for verification.

Assembly checks local volume coverage, paired internal faces with opposite
orientations, closed region shells, and independently reconstructed shell volumes.
It raises `ValueError` on inconsistent vertex keys, unmatched internal faces,
nonmanifold faces or volume failures. Near-degenerate cases requiring additional
face subdivision/overlay are rejected rather than silently repaired. Tolerance
still limits feature resolution; this is not a certified exact-arithmetic mesh.

The step-3 tests include processing-order invariance, analytical grain volumes,
random-point ownership, noncubic grids, duplicate grains, deliberate face damage,
and decreasing curved-interface volume error under background refinement.


## Performance controls and diagnostics

Optimization is enabled by default in both `background.partition_tetrahedron(i)`
and `background.assemble()`. Pass `optimize=False` to either method (or the
standalone functions) to run the exhaustive reference for comparison.

Before constructing polyhedra, the optimized path compares every grain against
the corner-winning grains. A candidate is removed only when one competitor has
strictly smaller affine costs at all four corners, with a numerical margin.
Because barycentric coordinates are nonnegative, this proves dominance
throughout the tetrahedron. Grains that win only in the interior are preserved.
Ties and face-touching competitors are retained. If only one candidate remains,
the original tetrahedron is returned directly, avoiding plane intersections and
ConvexHull. Strictly redundant constraints removed by pruning are absent from
local metadata; active boundary constraints are retained.

The expensive enumeration depends on the surviving local candidate count K,
roughly O(K^5), rather than the total G. The conservative filter costs O(G W)
for W distinct corner winners (at most four). Worst-case K can still equal G.
No incremental polyhedron-clipping algorithm has been introduced.

```python
partition = background.assemble()
print(partition.timings)
print(partition.candidate_statistics)
reference = background.assemble(optimize=False)  # use a small mesh for comparison
```

`timings` contains elapsed seconds for `setup`, `local_partition`,
`topology_collection` (including local coverage checks), `vertex_welding`,
`face_assembly`, `validation`, and `total`. `pruning` is a **subset** of
`local_partition`, not an additional stage. Total also includes loop overhead.
All existing topology, orientation and volume validation remains enabled.

`candidate_statistics` reports the input grain count, a histogram mapping
surviving candidate counts to numbers of tetrahedra, mean/max candidates,
`uncut_shortcuts`, and whether optimization was enabled. An optional
`diagnostics={}` argument to local partitioning receives its candidate counts,
shortcut flag and pruning time. Assembly timings do not include prior APD
fitting or background cost evaluation.


### Measured RVE performance

For the small periodic RVE in `examples/RVE_generation/work_rve.py`, the user
reported the following optimized assembly times with background resolution
`(12, 16, 20)`, corresponding to 23,040 tetrahedra:

| Stage | Seconds | Share of total |
| --- | ---: | ---: |
| Setup | 0.004 | <0.1% |
| Local partitioning | 9.966 | 70.1% |
| Topology collection | 1.474 | 10.4% |
| Vertex welding | 0.180 | 1.3% |
| Face assembly | 0.606 | 4.3% |
| Validation | 1.965 | 13.8% |
| **Total** | **14.210** | **100%** |

Pruning took **0.297 s**, already included in local partitioning. Stage totals
can differ slightly from the reported total because of loop overhead and
rounding. Relative to the earlier reported assembly time of approximately eight
minutes, this is approximately a **34× speedup**. These are user-reported example
measurements, not a controlled benchmark or a performance guarantee; the grain
population, candidate counts, resolution and hardware affect runtime.

Local partitioning remains the main cost. The candidate histogram helps explain
whether that time is spent on many small candidate sets or a few larger ones.
Validation remains enabled and should be included when comparing timings.
Incremental clipping is deferred; it would be a future optimization if the
remaining plane-triple enumeration becomes limiting.

For a reproducible comparison, reuse the same background object (and therefore
identical sampled costs and fitted weights) for optimized and reference runs.
On a modest mesh, compare grain volumes and shared topology as well as time.
Do not rerun random RVE generation or refit weights between the two runs.

## Step 4: grain-boundary and junction complex

```python
boundary = partition.boundary_complex()
print(boundary.summary())
residuals = boundary.residuals(apd)
fig, axes = boundary.plot(apd)  # interfaces, junctions, equality residuals
# import matplotlib.pyplot as plt
# plt.show()
```

Use the same APD coordinates and fitted weights that generated the background.
Extraction removes same-grain internal background faces and retains grain-pair
interfaces and exterior box faces. Shared polygons are stored once, without
projection or retriangulation. `points` retains the partition's global vertex
numbering (including unused interior points); `source_faces` maps the extracted
faces to the partition. Neither input object is changed.

Each `face_grains` entry is an ordered `(first, second)` grain pair, or
`(grain, None)` outside the box. Polygon normals point out of the first grain.
`boundary_ids` keeps the step-1 box-plane codes. `patches` groups faces by grain
pair, box-plane ID and edge-connected component. Separate components of the
same grain-pair interface remain separate patches.

`grain_shells[grain]` contains connected, closed boundary shells. Each shell is a
sequence of `(face_index, sign)` entries, where +1 uses the stored loop and -1
reverses it. Adjacent grains therefore use the same interface with opposite
orientations. Exterior faces close shells where grains meet the box. Directed
edge incidence is checked on each shell; open or nonmanifold shells raise
`ValueError`. Disconnected components and cavity boundary shells are retained
separately; no connected-grain assumption is imposed.

Junction topology is derived from grain incidence:

- `junction_edges` are edges incident to at least three distinct grains;
  `junction_edge_grains` records their grain sets.
- `junction_curves` are ordered vertex polylines with a constant incident grain
  set. Curves split at degree changes, grain-set changes, vertices involving
  four or more grains, and box contacts. Closed loops repeat the first vertex.
- `junction_vertices` includes curve endpoints/branch points and vertices with
  at least four incident grains. `vertex_grains` supplies vertex incidence.
- `boundary_junction_vertices` identifies junction points on the exterior.
- `boundary_trace_edges` stores two-grain interface intersections with the box;
  these are not classified as three-grain junctions.

`residuals(apd)` returns per-face `equality` and `dominance` arrays. For each
internal face, costs are sampled at its vertices, edge midpoints and centroid.
Equality is the maximum absolute difference of the two incident-grain costs.
Dominance is the maximum excess of the larger incident cost over the global
minimum cost. Both have units of length squared; exterior faces contain NaN.
These are sampled continuous-APD diagnostics, not distance errors or certified
bounds over entire polygons. Nonzero residuals are expected for the curved APD
because extraction uses the linearly interpolated approximation.

`plot(apd)` returns a Matplotlib figure and three axes without calling `show()`.
It displays interface patches, junction curves and equality residuals separately.
No APD argument is required to inspect just the interfaces and junctions.

Verification covers closed single-grain and two-grain shells, shared face
orientation, an analytical triple line, a four-grain vertex compared with the
existing junction extractor, disconnected grain/interface components, and
residual reduction under refinement of a curved interface. Periodic topology
pairing, geometric projection, adaptive refinement and volume meshing remain
later steps. Point-/edge-touching nonmanifold grain configurations may be
rejected by shell validation rather than repaired.

## Shared boundary triangulation and STL export

The surface portion of step 5 is available independently of volume meshing:

```python
boundary = partition.boundary_complex()
surface = boundary.triangulate()  # internal grain boundaries only
ms.write_stl('grain_boundaries.stl', boundary=surface)
# Include exterior box faces when inspecting closed per-grain shells:
closed_surface = boundary.triangulate(include_exterior=True)
ms.write_stl('grain_shells.stl', boundary=closed_surface, include_exterior=True)
# Alternatively triangulate and export directly:
ms.write_stl('boundaries.stl', boundary=boundary)
```

Triangulation traverses shared polygons once, not once per adjacent grain.
Existing triangles are retained. Larger convex polygons are triangulated by
connecting their centroid to every perimeter edge. This preserves collinear
perimeter vertices used by neighboring faces and junctions. The same triangles
serve both grains, with opposite orientation when assembling a grain's shell.
No grain-by-grain surface copies are created. Polygon centroids are added once
per selected face; projection onto curved APD interfaces is not performed.

`APDBoundaryTriangles` exposes `points`, zero-based `triangles`, `source_faces`
(indices into the boundary complex), `face_grains`, `boundary_ids`, `areas`,
`area_vectors`, and unit `normals`. Normals point out of the first grain in each
pair. The planar two-grain regression verifies that the exported interface area
is counted once, not doubled, and that triangle coordinates are unique.

`Microstructure.write_stl(file=None, path='./', *, boundary=None,
include_exterior=False)` now exports this shared APD surface in ASCII STL.
`boundary` can supply a boundary complex or triangulated surface; when omitted,
the boundary from `generate_grains()` is used. Export never refits or rebuilds
the APD. The default filename is
`self.name + '.stl'`; the output directory must already exist. Exterior box faces
are excluded by default. For a pretriangulated surface, `include_exterior=True`
retains all available faces but cannot restore faces omitted at triangulation.

The old `data`, `phases`, and `phase_num` options and particle/legacy-geometry
export have been removed. Replace old grain-export calls with the explicit APD
boundary argument above. STL writes each triangle once with its unit normal but
cannot store grain-pair metadata, shared-node indices, or junction topology.
Retain the Python surface object for subsequent meshing. Internal interface
networks are not generally closed manifold solids; adding exterior faces closes
individual grain shells, not the entire network as a single manifold surface.

Surface tests check polygon-area preservation, consistent normals, matching
oriented edges on closed grain shells, duplicate rejection, and STL coordinate
and normal round trips. Volume tetrahedralization is still deferred.

## High-level grain geometry, plotting and statistics

```python
# After ms.pack(); voxelization is optional.
ms.generate_grains(resolution=(12, 16, 20), batch_size=8192, optimize=True)
ms.plot_grains()
ms.write_stl('grain_boundaries.stl')
ms.write_centers()
```

`generate_grains` now runs the APD background, partition, boundary extraction and
shared surface triangulation pipeline. `resolution` defaults to 10 cells per
axis and is independent of voxel resolution. It reuses `ms.mesh.apd` without refitting when available. Otherwise it builds
an APD from original packed ellipsoids, excludes periodic duplicates, uses RVE
size and periodicity, and fits weights to relative particle volumes. The new
diagram is accessible as `ms.geometry['APD']`; no voxel mesh is created.
Geometry is committed only after verification succeeds. Missing ellipsoids
raise a clear error when there is no existing APD to use. Imported voxel-only
structures therefore require ellipsoids or an attached APD. Voxel assignments,
orientation sets and voxel grain counts remain unchanged. For legacy matrix or
porosity inputs the new APD fills the entire box; it does not reproduce the
legacy matrix volume fraction.

`ms.geometry` retains a dictionary interface for downstream consumers. Its
`APD`, `Background`, `Partition`, `Boundary` and `Surface` entries expose all
intermediate objects. `Points` and `Facets` contain shared surface coordinates
and triangles, including the box exterior. `Grains` maps original grain IDs to
records; `GBarea` lists `[grain1, grain2, shared_area]` once per grain pair,
summing disconnected patches. `PhaseVolumes` contains integrated phase volumes.

Grain records contain `Volume`, `Center`, `Covariance`, `SemiAxes`, `Axes`,
`eqDia`, `Area`, `Phase`, `Shells`, and outward `Simplices`. Centers and covariance
are obtained by integration over oriented closed surface triangles, not by
averaging surface samples or constructing a convex hull. `eqDia` uses actual
polyhedral volume; shape semi-axes are `sqrt(5 * covariance_eigenvalues)`, ordered
largest first. These are moment-equivalent ellipsoid axes, not enclosing axes.
Disconnected components and cavities contribute to the volume integrals.
`majDia` and `minDia` summarize twice the largest semi-axis and twice the mean of
the other two. Statistics use these moments and volume-equivalent diameters.

Plotting draws shared triangles once, colored by the first adjacent grain or its
phase. Polygon ANG slices evaluate the same piecewise-affine background costs,
so disconnected/nonconvex grains are not filled by a convex hull. Orientation
dictionaries use original grain IDs, including sparse IDs. STL defaults to
`geometry['Boundary']` when no explicit boundary is supplied; explicit inputs
remain supported. Centroid CSV export uses volume-integrated grain centers.

APD and voxel grain populations can differ at finite resolutions. Geometry
phase filtering uses geometry's own metadata. These geometry operations do not
create an FE volume mesh. Periodic moments describe all grain fragments within
the fundamental box without unwrapping; they should not be interpreted as the
shape of a reconstructed periodic grain. Periodic face pairing remains deferred.
