"""Gmsh surface remeshing of a labelled, shared APD boundary complex.

Gmsh is optional and is imported only when remeshing is requested. No STL
round-trip is used: polygon, interface and junction incidence stays explicit.
"""
from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .apd_boundary import APDBoundaryComplex, APDBoundaryTriangles, _components


@dataclass
class GmshRemeshingResult:
    """Independent surface candidate and diagnostics.

    ``periodic_nodes`` contains (slave, master, translation) tuples, using the
    returned surface's zero-based point indices. ``source_faces`` on the surface
    records Gmsh's classification on input polygons; compound triangles can
    cross polygon edges, so this is provenance, not exact containment.
    """
    surface: APDBoundaryTriangles
    report: dict
    periodic_nodes: tuple = ()


def _shell_volumes(surface):
    """Check directed edge cancellation and compute each grain's volume."""
    if (not np.all(np.isfinite(surface.points)) or
            np.any(surface.areas <= 0)):
        raise ValueError('Remeshing produced nonfinite or degenerate triangles')
    if len({tuple(sorted(t)) for t in surface.triangles}) != len(surface.triangles):
        raise ValueError('Remeshing produced duplicate interface triangles')
    edges = defaultdict(lambda: defaultdict(list))
    volumes = defaultdict(float)
    origin = surface.points.mean(axis=0)
    for tri, pair in zip(surface.triangles, surface.face_grains):
        for grain, oriented in ((pair[0], tri), (pair[1], tri[::-1])):
            if grain is None:
                continue
            xyz = surface.points[oriented] - origin
            volumes[grain] += np.linalg.det(xyz) / 6
            for a, b in zip(oriented, np.roll(oriented, -1)):
                edges[grain][tuple(sorted((a, b)))].append((a, b))
    for grain, shell in edges.items():
        if any(len(e) != 2 or e[0] != e[1][::-1] for e in shell.values()):
            raise ValueError(f'Remeshing left an open or nonmanifold shell for grain {grain}')
        if volumes[grain] <= 0:
            raise ValueError(f'Remeshing produced nonpositive volume for grain {grain}')
    return dict(volumes)


def _periodic_faces(boundary, box, tolerance):
    """Match polygon footprints under box translations, before calling Gmsh."""
    pairs = []
    for axis in range(3):
        low = np.flatnonzero(boundary.boundary_ids == 2 * axis + 1)
        high = set(np.flatnonzero(boundary.boundary_ids == 2 * axis + 2))
        shift = np.eye(3)[axis] * box[axis]
        for master in low:
            xyz = boundary.points[boundary.faces[master]] + shift
            candidates = []
            for slave in high:
                other = boundary.points[boundary.faces[slave]]
                if len(xyz) == len(other):
                    distances = np.linalg.norm(xyz[:, None] - other[None, :], axis=2)
                    if (np.all(distances.min(axis=0) <= tolerance) and
                            np.all(distances.min(axis=1) <= tolerance)):
                        candidates.append(slave)
            if len(candidates) != 1:
                raise ValueError('Periodic box polygons do not match under translation; '
                                 'use a boundary with matching opposite faces')
            slave = candidates[0]
            high.remove(slave)
            pairs.append((int(slave), int(master), shift))
        if high or not len(low):
            raise ValueError('Periodic remeshing requires all six matching box faces')
    return pairs


def _compound_groups(boundary, normals):
    """Split strongly turning/closed patches into connected directional charts.

    A closed surface cannot have a single planar parametrization. Keeping seams
    between normal-direction sectors gives Gmsh smaller, open charts without
    discarding the original interface labels.
    """
    normals = np.asarray(normals)
    adjacency = defaultdict(set)
    edges = defaultdict(list)
    for fi, face in enumerate(boundary.faces):
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((a, b)))].append(fi)
    for incident in edges.values():
        for fi in incident:
            adjacency[fi].update(incident)
    for patch in boundary.patches:
        direction = normals[patch.faces].sum(axis=0)
        length = np.linalg.norm(direction)
        if length > 0 and np.all(normals[patch.faces] @ (direction / length) > .25):
            yield patch.boundary_id, patch.faces
            continue
        sectors = defaultdict(list)
        for fi in patch.faces:
            axis = int(np.argmax(np.abs(normals[fi])))
            sectors[axis, normals[fi, axis] > 0].append(fi)
        for faces in sectors.values():
            for group in _components(faces, adjacency):
                yield patch.boundary_id, group


def remesh_grain_surface(geometry, mesh_size, *, compound=True, periodic=None,
                         box_size=None, algorithm=6, tolerance=1e-8):
    """Generate a new triangular mesh from shared boundary polygons.

    Parameters
    ----------
    geometry : dict or APDBoundaryComplex
        APD geometry containing ``Boundary``, or the boundary itself. Includes
        exterior faces so every grain is closed. Regularized/whole-grain triangle
        soups are not accepted as substitutes for a labelled polygon complex.
    mesh_size : float
        Target edge length in the coordinate units of the boundary.
    compound : bool
        Remesh across polygon edges within each connected labelled patch.
        Strongly turning/closed patches are split into directional charts.
        Gmsh must be able to parametrize each chart. For difficult patches,
        use False to remesh each polygon while preserving its perimeter.
        Periodic box patches retain their polygon edges for exact pairing.
    periodic : bool or None
        None follows geometry['APD'].periodic (False for a bare boundary).
        True constrains opposite box polygons and returns node correspondences.
    box_size : array-like, optional
        Required with periodic=True for a bare boundary; box origin is zero.
    algorithm : int
        Gmsh 2D meshing algorithm (6: Frontal-Delaunay).
    tolerance : float
        Relative coordinate tolerance for periodic matching.

    Returns a GmshRemeshingResult without mutating the input. Surface geometry is
    derived from the polygonal boundary, not projected onto the continuous APD.
    No volume elements are generated. Gmsh uses global state: call serially.
    Existing Gmsh models and the options changed here are restored on exit.
    """
    if not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError('mesh_size must be finite and positive')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if not isinstance(compound, (bool, np.bool_)):
        raise ValueError('compound must be a boolean')
    if periodic is not None and not isinstance(periodic, (bool, np.bool_)):
        raise ValueError('periodic must be None or a boolean')
    if isinstance(geometry, dict):
        if geometry.get('PeriodicImageGeometry'):
            raise ValueError('Remeshing requires box-clipped Boundary data; generate '
                             'grains with periodic_images=False first')
        boundary = geometry.get('Boundary')
        diagram = geometry.get('APD')
        if periodic is None:
            periodic = bool(getattr(diagram, 'periodic', False))
        if box_size is None:
            box_size = getattr(diagram, 'box_size', None)
    else:
        boundary = geometry
    if not isinstance(boundary, APDBoundaryComplex):
        raise ValueError('Remeshing requires an APDBoundaryComplex (generate_grains first)')
    if not len(boundary.faces):
        raise ValueError('Cannot remesh an empty boundary')
    reference = boundary.triangulate(include_exterior=True)
    before = _shell_volumes(reference)
    scale = np.ptp(boundary.points, axis=0).max()
    periodic_pairs = []
    if periodic:
        box = np.asarray(box_size, dtype=float)
        if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
            raise ValueError('Periodic remeshing requires three positive box_size values')
        periodic_pairs = _periodic_faces(boundary, box, tolerance * scale)
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise ImportError('Gmsh remeshing requires the optional dependency: '
                          'pip install "kanapy[gmsh]"') from exc

    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize([], readConfigFiles=False)
    previous = gmsh.model.getCurrent()
    name = 'kanapy_remesh_' + uuid4().hex
    settings = {'General.Terminal': 0, 'Mesh.Algorithm': algorithm,
                'Mesh.ElementOrder': 1, 'Mesh.RecombineAll': 0,
                'Mesh.MeshSizeMin': mesh_size, 'Mesh.MeshSizeMax': mesh_size,
                'Mesh.MeshSizeFactor': 1, 'Mesh.CompoundClassify': 1,
                'Mesh.MeshOnlyVisible': 0, 'Mesh.MeshOnlyEmpty': 0}
    saved = {}
    created = False
    try:
        for key, value in settings.items():
            saved[key] = gmsh.option.getNumber(key)
            gmsh.option.setNumber(key, value)
        gmsh.model.add(name)
        created = True
        geo = gmsh.model.geo
        used = np.unique(np.concatenate(boundary.faces))
        vertices = {int(v): geo.addPoint(*boundary.points[v], mesh_size) for v in used}
        edges, surfaces, normals = {}, [], []
        for face in boundary.faces:
            lines = []
            for a, b in zip(face, np.roll(face, -1)):
                a, b = int(a), int(b)
                key = tuple(sorted((a, b)))
                if key not in edges:
                    edges[key] = geo.addLine(vertices[key[0]], vertices[key[1]])
                lines.append(edges[key] if a < b else -edges[key])
            surfaces.append(geo.addPlaneSurface([geo.addCurveLoop(lines)]))
            xyz = boundary.points[face]
            normal = np.cross(xyz - xyz[0], np.roll(xyz, -1, axis=0) - xyz[0]).sum(axis=0)
            normal /= np.linalg.norm(normal)
            if np.max(np.abs((xyz - xyz[0]) @ normal)) > tolerance * scale:
                raise ValueError('Boundary polygons must be planar')
            normals.append(normal)
        geo.synchronize()
        if compound:
            for boundary_id, faces in _compound_groups(boundary, normals):
                if len(faces) > 1 and not (periodic and boundary_id):
                    gmsh.model.mesh.setCompound(2, [surfaces[f] for f in faces])
        for slave, master, shift in periodic_pairs:
            transform = np.eye(4)
            transform[:3, 3] = shift
            gmsh.model.mesh.setPeriodic(2, [surfaces[slave]], [surfaces[master]],
                                        transform.ravel().tolist())
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates).reshape(-1, 3)
        node_index = {int(tag): i for i, tag in enumerate(tags)}
        triangles, sources, pairs, bids = [], [], [], []
        for fi, tag in enumerate(surfaces):
            types, _, blocks = gmsh.model.mesh.getElements(2, tag)
            for kind, block in zip(types, blocks):
                if kind != 2:
                    raise ValueError('Gmsh returned non-linear or non-triangular elements')
                tri = np.array([node_index[int(n)] for n in block]).reshape(-1, 3)
                xyz = points[tri]
                reverse = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]) @ normals[fi] < 0
                tri[reverse] = tri[reverse, ::-1]
                triangles.extend(tri)
                sources.extend([fi] * len(tri))
                pairs.extend([boundary.face_grains[fi]] * len(tri))
                bids.extend([boundary.boundary_ids[fi]] * len(tri))
        if not triangles or set(pairs) != set(boundary.face_grains):
            raise ValueError('Gmsh did not mesh every grain interface')
        represented = set(sources)
        if any(not represented.intersection(patch.faces) for patch in boundary.patches):
            raise ValueError('Gmsh omitted a connected boundary patch')
        triangles = np.asarray(triangles, dtype=int)
        # Compound meshing leaves unused CAD/auxiliary nodes; expose only the mesh.
        used = np.unique(triangles)
        compact = np.full(len(points), -1, dtype=int)
        compact[used] = np.arange(len(used))
        surface = APDBoundaryTriangles(points[used], compact[triangles],
            np.asarray(sources, dtype=int), tuple(pairs), np.asarray(bids, dtype=np.int8))
        after = _shell_volumes(surface)
        periodic_nodes = []
        for slave, master, shift in periodic_pairs:
            _, slaves, masters, _ = gmsh.model.mesh.getPeriodicNodes(2, surfaces[slave])
            si = np.array([compact[node_index[int(n)]] for n in slaves], dtype=int)
            mi = np.array([compact[node_index[int(n)]] for n in masters], dtype=int)
            if not len(si) or np.any(si < 0) or np.any(mi < 0):
                raise ValueError('Gmsh returned incomplete periodic node correspondence')
            if not np.allclose(surface.points[si] - surface.points[mi], shift,
                               atol=tolerance * scale, rtol=0):
                raise ValueError('Remeshed periodic nodes do not match')
            periodic_nodes.append((si, mi, shift.copy()))
        report = dict(backend='gmsh', mesh_size=float(mesh_size), compound=bool(compound),
                      periodic=bool(periodic), triangles_before=len(reference.triangles),
                      triangles_after=len(surface.triangles), points_after=len(surface.points),
                      grain_volumes_before=before, grain_volumes_after=after,
                      relative_volume_changes={g: (after[g]-v)/v for g, v in before.items()})
        return GmshRemeshingResult(surface, report, tuple(periodic_nodes))
    finally:
        if created:
            gmsh.model.setCurrent(name)
            gmsh.model.remove()
        for key, value in saved.items():
            gmsh.option.setNumber(key, value)
        if previous in gmsh.model.list():
            gmsh.model.setCurrent(previous)
        if owned:
            gmsh.finalize()
