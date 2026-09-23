"""Background meshes for APD reconstruction; not grain-conforming volume meshes."""
from dataclasses import dataclass, field
from time import perf_counter
from itertools import permutations

import numpy as np


@dataclass
class APDBackgroundMesh:
    """Zero-based connectivity on [0, box_size], with a snapshot of APD costs.

    ``costs[v, g]`` corresponds to ``grain_ids[g]``. Exact ties in ``labels``
    use grain order. Faces are sorted vertex triples; ``face_tetrahedra`` gives
    their two incident tetrahedra (-1 for the exterior). ``boundary_ids`` is
    zero internally and 1..6 for xmin, xmax, ymin, ymax, zmin, zmax respectively.
    Periodic APD costs are supported, but opposite boundary nodes are not welded
    or paired. Stored costs require O(vertices * grains) memory despite batching.
    """
    points: np.ndarray
    tetrahedra: np.ndarray
    costs: np.ndarray
    grain_ids: np.ndarray
    faces: np.ndarray
    face_tetrahedra: np.ndarray
    boundary_ids: np.ndarray
    resolution: tuple
    box_size: np.ndarray

    @property
    def labels(self):
        if not self.costs.shape[1]:
            raise ValueError('Element-growth backgrounds have no nodal costs; use ElementGrainIDs')
        return self.grain_ids[np.argmin(self.costs, axis=1)]

    @property
    def signed_volumes(self):
        corners = self.points[self.tetrahedra]
        return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6

    def assemble(self, *, tolerance=1e-10, tetrahedron_order=None, optimize=True):
        """Return the globally shared, verified local-polyhedron partition."""
        return assemble_polyhedral_mesh(self, tolerance=tolerance,
                                        tetrahedron_order=tetrahedron_order, optimize=optimize)

    def partition_tetrahedron(self, index, *, tolerance=1e-10, optimize=True, diagnostics=None):
        """Inspect local grain polyhedra without modifying the background mesh."""
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(self.tetrahedra)):
            raise ValueError('index must identify a background tetrahedron')
        corners = self.tetrahedra[index]
        return partition_tetrahedron(self.points[corners], self.costs[corners],
                                     self.grain_ids, tolerance=tolerance,
                                     optimize=optimize, diagnostics=diagnostics)

    def summary(self):
        """Return basic counts and geometric checks for intermediate inspection."""
        volumes = self.signed_volumes
        return dict(vertices=len(self.points), tetrahedra=len(self.tetrahedra),
                    interior_faces=int(np.count_nonzero(self.boundary_ids == 0)),
                    boundary_faces=int(np.count_nonzero(self.boundary_ids)),
                    minimum_volume=float(volumes.min()),
                    total_volume=float(volumes.sum()),
                    box_volume=float(np.prod(self.box_size)),
                    all_positive=bool(np.all(volumes > 0)))


def build_background_mesh(diagram, resolution=10, *, batch_size=8192):
    """Build six positively oriented Freudenthal tetrahedra per Cartesian cell.

    ``resolution`` is a positive integer or three positive integers counting
    cells, not vertices. Costs include the diagram's existing weights; no fitting
    or mutation of the diagram occurs. All grains are evaluated at every vertex.
    The mesh represents geometry sampling only, not APD interfaces or grain
    volumes. Its arrays can be inspected independently before later clipping.
    """
    raw = np.asarray(resolution)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or raw.dtype.kind not in 'iu' or np.any(raw < 1):
        raise ValueError('resolution must be a positive integer or three positive integers')
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')
    shape = tuple(int(v) for v in raw)
    box = np.array(diagram.box_size, dtype=float, copy=True)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
        raise ValueError('box_size must contain three finite positive lengths')
    axes = [np.linspace(0, length, n + 1) for length, n in zip(box, shape)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    stride = np.array([(shape[1] + 1) * (shape[2] + 1), shape[2] + 1, 1])
    origins = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'),
                       axis=-1).reshape(-1, 3) @ stride
    paths = []
    for perm in permutations(range(3)):
        path = np.vstack([np.zeros(3, dtype=int), np.cumsum(np.eye(3, dtype=int)[list(perm)], axis=0)])
        if np.linalg.det(path[1:] - path[0]) < 0:
            path[[1, 2]] = path[[2, 1]]
        paths.append(path @ stride)
    tetrahedra = (origins[:, None, None] + np.array(paths)[None, :, :]).reshape(-1, 4)
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    occurrences = np.sort(tetrahedra[:, local_faces].reshape(-1, 3), axis=1)
    faces, inverse, counts = np.unique(occurrences, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError('Nonmanifold background face')
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    neighbors = np.full((len(faces), 2), -1, dtype=int)
    neighbors[:, 0] = order[starts] // 4
    interior = counts == 2
    neighbors[interior, 1] = order[starts[interior] + 1] // 4
    boundary_ids = np.zeros(len(faces), dtype=np.int8)
    exterior = np.flatnonzero(~interior)
    coordinates = points[faces[exterior]]
    for axis in range(3):
        for side, value in enumerate((0., box[axis])):
            on_plane = np.all(coordinates[:, :, axis] == value, axis=1)
            boundary_ids[exterior[on_plane]] = 2 * axis + side + 1
    if np.any(boundary_ids[exterior] == 0):
        raise RuntimeError('Unmatched face inside the background mesh')
    costs = np.empty((len(points), len(diagram.grain_ids)))
    for start in range(0, len(points), batch_size):
        costs[start:start + batch_size] = diagram.costs(points[start:start + batch_size])
    return APDBackgroundMesh(points, tetrahedra, costs, diagram.grain_ids.copy(),
                             faces, neighbors, boundary_ids, shape, box)


@dataclass
class LocalGrainPolyhedron:
    """One full-dimensional grain region in a single tetrahedron.

    Faces are outward-oriented polygon loops indexing ``vertices``. Constraint
    rows encode ``A @ barycentric[:, 1:] + b <= 0``. ``constraint_sources``
    identifies each row as ('tetrahedron', opposite local corner) or
    ('grain', competing grain ID). Vertex/face constraints index those rows;
    coincident supporting planes are retained as multiple metadata entries.
    """
    grain_id: object
    vertices: np.ndarray
    barycentric: np.ndarray
    faces: tuple
    volume: float
    constraints: np.ndarray
    constraint_sources: tuple
    vertex_constraints: tuple
    face_constraints: tuple


def partition_tetrahedron(points, costs, grain_ids=None, *, tolerance=1e-10,
                          optimize=True, diagnostics=None):
    """Partition a tetrahedron using the lower envelope of affine vertex costs.

    ``points`` has shape (4, 3), ``costs`` (4, grains). All grain constraints
    participate, including grains that win at no corner. Exactly identical
    affine costs belong to the first grain; boundary closures otherwise overlap
    only on zero-volume contacts. Empty/lower-dimensional regions are omitted.

    By default strict affine dominance prunes candidates and single-candidate
    tetrahedra are returned directly. Set optimize=False for the exhaustive
    reference path. Remaining candidates use plane-triple enumeration (roughly
    O(candidates**5)); no incremental clipping or global welding is done.
    Optional diagnostics is a dict populated with candidate counts, shortcut
    use and pruning time. Removed strictly redundant constraints are not stored.
    ``tolerance`` controls normalized halfspace feasibility, vertex merging and
    numerical rank. Features at this tolerance may be unresolved. Returned
    geometry approximates interpolated costs, not the continuous curved APD.
    """
    from itertools import combinations
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError('points must be finite with shape (4, 3)')
    if (costs.ndim != 2 or costs.shape[0] != 4 or costs.shape[1] < 1
            or not np.all(np.isfinite(costs))):
        raise ValueError('costs must be finite with shape (4, grains)')
    if not np.isfinite(tolerance) or not 0 < tolerance < 1e-3:
        raise ValueError('tolerance must be positive and less than 1e-3')
    transform = points[1:] - points[0]
    singular = np.linalg.svd(transform, compute_uv=False)
    if singular[0] == 0 or singular[-1] <= tolerance * singular[0]:
        raise ValueError('tetrahedron must be nondegenerate at the requested tolerance')
    n = costs.shape[1]
    ids = np.arange(1, n + 1) if grain_ids is None else np.asarray(grain_ids)
    if ids.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError('grain_ids must have one unique ID per cost column')
    if optimize:
        started = perf_counter()
        # A single affine competitor strictly below a grain at every corner
        # proves that grain irrelevant everywhere, including the interior.
        # Keep near ties to avoid losing active-constraint metadata.
        winners = np.unique(np.argmin(costs, axis=1))
        dominated = np.zeros(n, dtype=bool)
        for winner in winners:
            delta = costs - costs[:, winner, None]
            margin = 32 * tolerance * np.max(np.abs(delta), axis=0)
            dominated |= np.all(delta > margin[None, :], axis=0)
        candidates = np.flatnonzero(~dominated)
        if diagnostics is not None:
            diagnostics.update(original_grains=n, candidates=len(candidates),
                               shortcut=len(candidates) == 1,
                               pruning_seconds=perf_counter()-started)
        if len(candidates) == 1:
            # No grain plane touches this region: preserve the original tetrahedron.
            planes = np.array([[1., 1., 1., -1.], [-1., 0., 0., 0.],
                               [0., -1., 0., 0.], [0., 0., -1., 0.]])
            planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
            faces = []
            for opposite in range(4):
                face = np.array([v for v in range(4) if v != opposite])
                xyz = points[face]
                if np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]) @ (points[opposite]-xyz[0]) > 0:
                    face = face[::-1]
                faces.append(face)
            return [LocalGrainPolyhedron(
                ids[candidates[0]], points.copy(), np.eye(4), tuple(faces),
                float(abs(np.linalg.det(transform))/6), planes,
                tuple(('tetrahedron', k) for k in range(4)),
                tuple(tuple(k for k in range(4) if k != v) for v in range(4)),
                tuple((k,) for k in range(4)))]
        return partition_tetrahedron(points, costs[:, candidates], ids[candidates],
                                     tolerance=tolerance, optimize=False)
    if diagnostics is not None:
        diagnostics.update(original_grains=n, candidates=n, shortcut=False,
                           pruning_seconds=0.)
    result = []
    for i in range(n):
        # Equal affine functions have positive-volume ties: keep only the first.
        if any(np.array_equal(costs[:, i], costs[:, j]) for j in range(i)):
            continue
        rows = [[1., 1., 1., -1.], [-1., 0., 0., 0.],
                [0., -1., 0., 0.], [0., 0., -1., 0.]]
        sources = [('tetrahedron', k) for k in range(4)]
        impossible = False
        for j in range(n):
            if i == j:
                continue
            diff = costs[:, i] - costs[:, j]
            gradient = diff[1:] - diff[0]
            norm = np.linalg.norm(gradient)
            if norm == 0:
                if diff[0] > 0:
                    impossible = True
                    break
                continue
            rows.append([*gradient, diff[0]])
            sources.append(('grain', ids[j]))
        if impossible:
            continue
        planes = np.asarray(rows)
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        a, b = planes[:, :3], planes[:, 3]
        vertices = []
        for triple in combinations(range(len(planes)), 3):
            matrix = a[list(triple)]
            if abs(np.linalg.det(matrix)) <= tolerance:
                continue
            vertex = np.linalg.solve(matrix, -b[list(triple)])
            if np.max(a @ vertex + b) > tolerance:
                continue
            if not any(np.linalg.norm(vertex - old) <= tolerance for old in vertices):
                vertices.append(vertex)
        if len(vertices) < 4:
            continue
        local = np.array(vertices)
        if np.linalg.svd(local - local.mean(axis=0), compute_uv=False)[-1] <= tolerance:
            continue
        hull = ConvexHull(local)
        barycentric = np.column_stack([1 - local.sum(axis=1), local])
        physical = points[0] + local @ transform
        active = np.abs(local @ a.T + b) <= 4 * tolerance
        face_map = {}
        for constraint in range(len(planes)):
            indices = tuple(np.flatnonzero(active[:, constraint]))
            if len(indices) >= 3:
                face_map.setdefault(indices, []).append(constraint)
        faces, face_constraints = [], []
        for indices, constraints in face_map.items():
            idx = np.array(indices)
            xyz = physical[idx]
            # Transform a barycentric-coordinate outward normal to physical space.
            normal = np.linalg.solve(transform, a[constraints[0]])
            normal /= np.linalg.norm(normal)
            center = xyz.mean(axis=0)
            u = xyz[0] - center
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            order = np.argsort(np.arctan2((xyz-center) @ v, (xyz-center) @ u))
            polygon = idx[order]
            area_vector = np.sum(np.cross(physical[polygon] - center,
                                         np.roll(physical[polygon], -1, axis=0) - center), axis=0)
            if np.linalg.norm(area_vector) == 0:
                continue
            faces.append(polygon)
            face_constraints.append(tuple(constraints))
        result.append(LocalGrainPolyhedron(
            ids[i], physical, barycentric, tuple(faces),
            float(hull.volume * abs(np.linalg.det(transform))), planes,
            tuple(sources), tuple(tuple(np.flatnonzero(row)) for row in active),
            tuple(face_constraints)))
    return result


@dataclass
class APDPolyhedralMesh:
    """Shared polyhedral partition of the interpolated APD (zero-based indices).

    ``faces`` are polygon loops, outward from ``face_regions[:, 0]``.
    The second incident region is -1 on the box boundary. ``region_faces``
    and ``region_face_signs`` reconstruct outward shells for each region.
    A region is one grain fragment in one background tetrahedron. Same-grain
    background faces remain in the partition; ``interface_faces`` selects
    actual grain boundaries. Periodic boundary pairing is not performed.
    """
    points: np.ndarray
    vertex_keys: tuple
    faces: tuple
    face_regions: np.ndarray
    boundary_ids: np.ndarray
    region_faces: tuple
    region_face_signs: tuple
    region_grain_ids: np.ndarray
    region_tetrahedra: np.ndarray
    region_volumes: np.ndarray
    timings: dict = field(default_factory=dict)
    candidate_statistics: dict = field(default_factory=dict)

    def boundary_complex(self):
        """Extract shared interface patches, grain shells and junction curves."""
        from .apd_boundary import extract_boundary_complex
        return extract_boundary_complex(self)

    @property
    def interface_faces(self):
        interior = np.flatnonzero(self.face_regions[:, 1] >= 0)
        adjacent = self.face_regions[interior]
        return interior[self.region_grain_ids[adjacent[:, 0]] !=
                        self.region_grain_ids[adjacent[:, 1]]]

    @property
    def grain_volumes(self):
        return {gid: float(self.region_volumes[self.region_grain_ids == gid].sum())
                for gid in np.unique(self.region_grain_ids)}

    def summary(self):
        return dict(vertices=len(self.points), faces=len(self.faces),
                    regions=len(self.region_volumes),
                    interface_faces=len(self.interface_faces),
                    boundary_faces=int(np.count_nonzero(self.boundary_ids)),
                    total_volume=float(self.region_volumes.sum()),
                    grain_volumes=self.grain_volumes, timings=self.timings.copy(),
                    candidate_statistics=self.candidate_statistics.copy())


def assemble_polyhedral_mesh(background, *, tolerance=1e-10, tetrahedron_order=None,
                             optimize=True):
    """Assemble and verify all local partitions without changing the background.

    Vertex identities combine the supporting background simplex and the set of
    tied minimum-cost columns. Coordinates are checked, never used as rounded
    global keys. Output ordering is deterministic even when processing order is
    reversed. Unmatched internal faces, inconsistent vertex identities, coverage
    failures and nonmanifold faces raise ValueError rather than return a cracked
    mesh. Degenerate subdivisions needing face overlays are not repaired here.
    Conservative pruning and uncut shortcuts are enabled by default; optimize=False
    retains the exhaustive reference. Timings (seconds) and candidate statistics
    are returned on the mesh, without printing. Validation is never skipped.
    """
    total_started = perf_counter()
    timings = dict(local_partition=0., topology_collection=0., pruning=0.)
    histogram = {}
    shortcuts = 0
    count = len(background.tetrahedra)
    order = np.arange(count) if tetrahedron_order is None else np.asarray(tetrahedron_order)
    if (order.shape != (count,) or order.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(order), np.arange(count))):
        raise ValueError('tetrahedron_order must be a permutation of all tetrahedra')
    if not np.isfinite(tolerance) or not 0 < tolerance < 1e-3:
        raise ValueError('tolerance must be positive and less than 1e-3')
    scale = float(np.max(background.box_size))
    # A common scale keeps tie identification consistent on shared simplices.
    relative_costs = background.costs - background.costs[:, :1]
    cost_scale = float(np.max(np.abs(relative_costs)))
    tie_tol = 8 * tolerance * cost_scale
    records = {}
    occurrences = {}
    volumes = background.signed_volumes
    grain_index = {gid: i for i, gid in enumerate(background.grain_ids)}
    timings["setup"] = perf_counter()-total_started
    for ti in order:
        corners = background.tetrahedra[ti]
        local_stats = {}
        started = perf_counter()
        parts = background.partition_tetrahedron(int(ti), tolerance=tolerance,
                                                  optimize=optimize, diagnostics=local_stats)
        timings['local_partition'] += perf_counter()-started
        timings['pruning'] += local_stats['pruning_seconds']
        k = local_stats['candidates']
        histogram[k] = histogram.get(k, 0) + 1
        shortcuts += int(local_stats['shortcut'])
        started = perf_counter()
        if not np.isclose(sum(p.volume for p in parts), volumes[ti],
                          rtol=100*tolerance, atol=0):
            raise ValueError(f'Local volume coverage failed in tetrahedron {ti}')
        for part in parts:
            region_key = (int(ti), grain_index[part.grain_id])
            keys = []
            for bary, xyz in zip(part.barycentric, part.vertices):
                support_mask = bary > 4*tolerance
                support = tuple(sorted(int(v) for v in corners[support_mask]))
                if not support:
                    raise ValueError('Unresolved background vertex support')
                # Remove numerical off-simplex components before computing ties.
                clean = np.where(support_mask, bary, 0.)
                clean /= clean.sum()
                values = clean @ relative_costs[corners]
                tied = tuple(np.flatnonzero(values - values.min() <= tie_tol))
                key = (support, tied)
                keys.append(key)
                occurrences.setdefault(key, []).append((region_key, xyz.copy()))
            if len(set(keys)) != len(keys):
                raise ValueError('Distinct local vertices have an unresolved topology key')
            records[region_key] = (part, keys)
        timings['topology_collection'] += perf_counter()-started
    started = perf_counter()
    vertex_keys = tuple(sorted(occurrences))
    vertex_index = {key: i for i, key in enumerate(vertex_keys)}
    points = []
    for key in vertex_keys:
        entries = sorted(occurrences[key], key=lambda item: item[0])
        xyz = entries[0][1]
        if any(np.linalg.norm(other-xyz) > 32*tolerance*scale for _, other in entries):
            raise ValueError('Inconsistent coordinates for shared vertex topology')
        points.append(xyz)
    points = np.asarray(points)
    timings['vertex_welding'] = perf_counter()-started
    started = perf_counter()
    region_keys = sorted(records)
    face_entries = {}
    for ri, key in enumerate(region_keys):
        part, keys = records[key]
        for face in part.faces:
            loop = tuple(vertex_index[keys[v]] for v in face)
            face_entries.setdefault(tuple(sorted(loop)), []).append((ri, loop))
    faces, neighbors, boundary_ids = [], [], []
    region_faces = [[] for _ in region_keys]
    region_signs = [[] for _ in region_keys]
    for fi, key in enumerate(sorted(face_entries)):
        entries = face_entries[key]
        if len(entries) not in (1, 2):
            raise ValueError('Nonmanifold polyhedral face')
        ri, loop = entries[0]
        # Canonical cyclic origin; preserve the first region's outward orientation.
        pivot = loop.index(min(loop))
        loop = loop[pivot:] + loop[:pivot]
        boundary = 0
        if len(entries) == 2:
            other_ri, other = entries[1]
            reverse = tuple(reversed(other))
            pivot = reverse.index(loop[0])
            if reverse[pivot:] + reverse[:pivot] != loop or ri == other_ri:
                raise ValueError('Shared face orientations or polygon loops disagree')
        else:
            other_ri = -1
            # Use background support, not a distance threshold, to identify box faces.
            supporting_nodes = sorted({v for vi in loop for v in vertex_keys[vi][0]})
            xyz = background.points[supporting_nodes]
            for axis in range(3):
                for side, value in enumerate((0., background.box_size[axis])):
                    if np.all(xyz[:, axis] == value):
                        boundary = 2*axis + side + 1
            if boundary == 0:
                raise ValueError('Unmatched internal face; local subdivisions need reconciliation')
        faces.append(np.array(loop, dtype=int))
        neighbors.append((ri, other_ri))
        boundary_ids.append(boundary)
        region_faces[ri].append(fi)
        region_signs[ri].append(1)
        if other_ri >= 0:
            region_faces[other_ri].append(fi)
            region_signs[other_ri].append(-1)
    result = APDPolyhedralMesh(
        points, vertex_keys, tuple(faces), np.array(neighbors, dtype=int),
        np.array(boundary_ids, dtype=np.int8),
        tuple(np.array(f, dtype=int) for f in region_faces),
        tuple(np.array(s, dtype=int) for s in region_signs),
        np.array([records[k][0].grain_id for k in region_keys]),
        np.array([k[0] for k in region_keys]),
        np.array([records[k][0].volume for k in region_keys]))
    timings['face_assembly'] = perf_counter()-started
    started = perf_counter()
    # Independently verify closed outward shells and volume after welding.
    for ri, (face_ids, signs) in enumerate(zip(result.region_faces, result.region_face_signs)):
        edges = {}
        vertex_ids = np.unique(np.concatenate([faces[f] for f in face_ids]))
        center = points[vertex_ids].mean(axis=0)
        volume = 0.
        for fi, sign in zip(face_ids, signs):
            loop = faces[fi] if sign == 1 else faces[fi][::-1]
            for a, b in zip(loop, np.roll(loop, -1)):
                edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
            xyz = points[loop] - center
            for j in range(1, len(xyz)-1):
                volume += np.linalg.det(xyz[[0, j, j+1]]) / 6
        if not all(len(e) == 2 and e[0] == e[1][::-1] for e in edges.values()):
            raise ValueError('Polyhedral region shell is not closed')
        if volume <= 0 or not np.isclose(volume, result.region_volumes[ri], rtol=100*tolerance, atol=0):
            raise ValueError('Polyhedral region volume changed during assembly')
    timings['validation'] = perf_counter()-started
    timings['total'] = perf_counter()-total_started
    result.timings = timings
    result.candidate_statistics = dict(
        histogram=dict(sorted(histogram.items())), original_grains=len(background.grain_ids),
        mean_candidates=sum(k*v for k, v in histogram.items())/count,
        max_candidates=max(histogram), uncut_shortcuts=shortcuts, optimized=bool(optimize))
    return result
