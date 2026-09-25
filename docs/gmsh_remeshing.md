# Remeshing grain surfaces with Gmsh

Install the optional backend in your Kanapy environment:

```sh
pip install 'kanapy[gmsh]'
```

Generate a boundary complex, then request a target edge length in the same units
as the microstructure:

```python
ms.generate_grains(resolution=10, periodic_images=False)
result = ms.remesh_grains(mesh_size=2.0)
ms.write_stl('remeshed.stl', boundary=result.surface, include_exterior=True)
print(result.report)
```

`ms.geometry['Remeshed']` stores the result only after validation succeeds.
Reference geometry, grain statistics, and voxel data remain unchanged. The result
contains a shared `APDBoundaryTriangles` surface: `points`, `triangles`,
`face_grains`, `boundary_ids`, and `source_faces`. STL export discards those labels;
keep the Python result when using the mesh for further processing.

You can also supply the boundary complex directly:

```python
from kanapy.core.gmsh_remeshing import remesh_grain_surface

result = remesh_grain_surface(ms.geometry['Boundary'], mesh_size=2.0)
```

Each polygon becomes one Gmsh plane surface. Adjacent polygons use the same
points and curves, so grain interfaces are meshed once and junction nodes are
shared. By default, compound surfaces allow new triangles to cross old polygon
edges within each labelled patch. Strongly turning or closed patches are split
into connected directional charts for parametrization. Chart seams and junction
segments remain constrained. `source_faces` records Gmsh's classification on an
original boundary polygon; compound triangles need not lie entirely inside it.

Compound remeshing approximates the original faceted surface and can change
areas and grain volumes. It does not fit a smooth APD surface. The report includes
per-grain volumes before and after remeshing and their relative changes. Validation
checks finite, nondegenerate, unique triangles, retained patches, closed oriented
grain shells, and positive grain volumes. These checks do not certify absence of
self-intersections or guarantee a particular minimum element angle.

For exact retention of the polygonal geometry, or a patch that Gmsh cannot
parametrize, use:

```python
result = ms.remesh_grains(mesh_size=2.0, compound=False)
```

This regenerates triangles inside each polygon while preserving its perimeter.
Small original polygon edges can therefore limit the attainable element quality
and size. `mesh_size` is a target, not a strict edge-length bound.

For periodic APDs, the API enables periodic matching automatically. It requires
box-clipped boundary data with matching opposite polygon footprints; use
`periodic_images=False` when generating grains. Periodic exterior patches retain
polygon edges so Gmsh can mesh them as translated copies. The result's
`periodic_nodes` contains `(slave_indices, master_indices, translation)` arrays
in the returned surface's compact point numbering. For a bare boundary, specify
`periodic=True, box_size=[Lx, Ly, Lz]` explicitly. The box origin must be zero.
Passing `periodic=False` explicitly requests unconstrained box surfaces.
Unwrapped whole-grain and regularized triangle surfaces are not accepted by this
polygon-boundary entry point.

The backend generates surface triangles only. Gmsh is imported lazily and runs
without its GUI. Calls must be serial because Gmsh has global state. Existing
Gmsh models, the current model, and options changed by Kanapy are restored even
if meshing fails.
