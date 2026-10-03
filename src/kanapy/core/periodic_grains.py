"""Lift closed grains from a periodic box into a whole-grain Euclidean cluster.

No convexification or refitting: periodic fragments are joined by lattice
translations and artificial box facets are discarded. Winding parents can be
split into compact entities using retained partition faces, with explicit
material/orientation inheritance through parent IDs.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from .apd_boundary import APDBoundaryTriangles


@dataclass
class PeriodicGrainGeometry:
    surface: APDBoundaryTriangles
    box_size: np.ndarray
    vertex_classes: np.ndarray
    vertex_shifts: np.ndarray
    source_triangles: np.ndarray
    paired_faces: np.ndarray
    pair_translations: np.ndarray
    phase_by_grain: dict
    report: dict
    grain_parent_ids: dict = field(default_factory=dict)
    grain_orientations: dict = field(default_factory=dict)

    def as_geometry(self):
        """Plotting/statistics view; moments describe these lifted whole grains."""
        surface = self.surface
        grains = {}
        areas = surface.areas
        for gid, phase in self.phase_by_grain.items():
            ids = [i for i, pair in enumerate(surface.face_grains) if gid in pair]
            tri = surface.triangles[ids].copy()
            signs = np.array([1 if surface.face_grains[i][0] == gid else -1 for i in ids])
            tri[signs < 0] = tri[signs < 0, ::-1]
            vertices = np.unique(tri)
            origin = surface.points[vertices].mean(axis=0)
            xyz = surface.points[tri]-origin
            signed = np.linalg.det(xyz)/6
            volume = signed.sum()
            sums = xyz.sum(axis=1)
            mean = np.einsum('n,ni->i', signed, sums)/(4*volume)
            second = np.einsum('n,nij->ij', signed,
                np.einsum('nki,nkj->nij', xyz, xyz)+np.einsum('ni,nj->nij', sums, sums))/(20*volume)
            covariance = second-np.outer(mean, mean)
            eigenvalues, axes = np.linalg.eigh(covariance)
            if volume <= 0 or np.any(eigenvalues <= 0):
                raise ValueError(f'Invalid lifted moments for grain {gid}')
            semi = np.sqrt(5*eigenvalues[::-1])
            grains[gid] = dict(Phase=phase, ParentGrain=self.grain_parent_ids.get(gid, gid),
                Volume=float(volume), Center=origin+mean,
                Covariance=covariance, SemiAxes=semi, Axes=axes[:, ::-1],
                eqDia=float((6*volume/np.pi)**(1/3)), majDia=float(2*semi[0]),
                minDia=float(2*np.mean(semi[1:])), Area=float(areas[ids].sum()),
                Vertices=vertices, Points=surface.points[vertices], Simplices=tri.tolist(),
                TriangleIndices=np.array(ids))
        return dict(Representation='PeriodicWholeGrains', Surface=surface,
                    Points=surface.points, Facets=surface.triangles, Grains=grains,
                    Ngrains=len(grains), PhaseVolumes={p: sum(g['Volume'] for g in grains.values()
                         if g['Phase'] == p) for p in set(self.phase_by_grain.values())})


class WindingGrainError(ValueError):
    def __init__(self, grain_ids, axes=(0, 1, 2)):
        self.grain_ids = tuple(sorted(map(int, grain_ids)))
        self.axes = set(map(int, axes))
        super().__init__(f'Periodic grain(s) {self.grain_ids} wind around the box; a finite uncut compact hull does not exist')


def _split_entities(geometry, winding):
    """Partition winding parents into periodic bands of existing background cells.

    Only labels of existing convex fragments change. New cut faces are already
    shared partition faces, so no gaps, overlap, or volume changes are introduced.
    Cuts are internal grid planes, never the prescribed RVE box planes.
    """
    partition, background = geometry.get('Partition'), geometry.get('Background')
    if partition is None or background is None:
        raise ValueError('Splitting winding grains requires the retained APD Partition and Background')
    resolution = np.array(background.resolution, dtype=int)
    if np.any(resolution < 3):
        raise ValueError('Splitting winding grains requires background resolution >= 3 on each axis')
    box = np.asarray(background.box_size)
    low = np.maximum(1, resolution//4)
    high = np.minimum(resolution-1, low+np.maximum(1, resolution//2))
    tetra_points = background.points[background.tetrahedra[partition.region_tetrahedra]]
    cells = np.rint(tetra_points.min(axis=1)/box*resolution).astype(int)
    labels = partition.region_grain_ids.copy()
    parent_ids = {int(g): int(g) for g in geometry['Grains']}
    centers = {int(g): np.array(c) for g, c in zip(geometry['APD'].grain_ids, geometry['APD'].centers)}
    next_id = int(labels.max())+1
    for gid in sorted(winding):
        axis_weights = np.array([2**axis if axis in winding[gid] else 0 for axis in range(3)])
        bands = ((cells >= low) & (cells < high)).astype(int) @ axis_weights
        parent_regions = np.flatnonzero(partition.region_grain_ids == gid)
        for k, band in enumerate(sorted(set(bands[parent_regions]))):
            ids = parent_regions[bands[parent_regions] == band]
            entity = int(gid) if k == 0 else next_id
            if k: next_id += 1
            labels[ids] = entity; parent_ids[entity] = int(gid)
            midpoint = (cells[ids]+.5)/resolution*box
            weights = partition.region_volumes[ids]
            circular = np.sum(weights[:, None]*np.exp(2j*np.pi*midpoint/box), axis=0)
            centers[entity] = (np.angle(circular) % (2*np.pi))*box/(2*np.pi)
    split_partition = replace(partition, region_grain_ids=labels)
    boundary = split_partition.boundary_complex()
    surface = boundary.triangulate(include_exterior=True)
    grains = {int(g): dict(Volume=v, Phase=geometry['Grains'][parent_ids[int(g)]]['Phase'])
              for g, v in split_partition.grain_volumes.items()}
    return dict(geometry, Partition=split_partition, Boundary=boundary, Surface=surface,
                Grains=grains, EntityCenters=centers, GrainParents=parent_ids,
                SplitParents=sorted(map(int, winding)),
                SplitPlanes=np.vstack([low, high])/resolution*box)


def unwrap_periodic_grains(geometry, *, tolerance=1e-10, split_winding=True):
    """Join periodic fragments; optionally split winding parents into entities.

    Splits preserve total parent volume and phase and record parent identity for
    orientation inheritance. Set split_winding=False to reject winding parents.
    """
    if not isinstance(split_winding, (bool, np.bool_)):
        raise ValueError('split_winding must be a boolean')
    if geometry.get('PeriodicImageGeometry'):
        return geometry['WholeGrains']
    winding = {}
    working = geometry
    while True:
        try:
            result = _unwrap_periodic_grains(working, tolerance=tolerance)
            break
        except WindingGrainError as exc:
            if not split_winding: raise
            parents = working.get('GrainParents', {})
            found = {parents.get(g, g) for g in exc.grain_ids}
            changed = False
            for parent in found:
                axes = winding.setdefault(parent, set())
                if not exc.axes <= axes:
                    axes.update(exc.axes); changed = True
            if not changed:
                raise ValueError('Winding entities remain after splitting; increase reconstruction resolution') from exc
            working = _split_entities(geometry, winding)
    result.grain_parent_ids = working.get('GrainParents', {g: g for g in result.phase_by_grain}).copy()
    split_faces = set()
    for fi, (a, b) in enumerate(result.surface.face_grains):
        if b is not None and result.grain_parent_ids[a] == result.grain_parent_ids[b]: split_faces.add(fi)
    for a, b in result.paired_faces:
        ga, gb = result.surface.face_grains[a][0], result.surface.face_grains[b][0]
        if result.grain_parent_ids[ga] == result.grain_parent_ids[gb]: split_faces.update([int(a), int(b)])
    parent_volumes = defaultdict(float)
    for g, volume in result.report['grain_volumes'].items(): parent_volumes[result.grain_parent_ids[g]] += volume
    for parent, volume in parent_volumes.items():
        if not np.isclose(volume, geometry['Grains'][parent]['Volume'], rtol=1e-7):
            raise ValueError(f'Splitting changed volume of parent grain {parent}')
    result.report.update(split_winding=bool(split_winding), split_parent_grains=sorted(winding),
        grain_parent_ids=result.grain_parent_ids, split_face_ids=sorted(split_faces),
        split_axes={int(g): sorted(axes) for g, axes in winding.items()},
        parent_grain_volumes=dict(parent_volumes), source_triangles_reference='split partition surface' if winding else 'reference surface',
        split_planes=working.get('SplitPlanes', np.empty((0, 3))).tolist())
    return result


def _unwrap_periodic_grains(geometry, *, tolerance=1e-10):
    """Join periodic fragments into closed, uncut grain shells near their seeds.

    Coordinates can lie outside the original box. The cluster has a jagged
    boundary made of real grain interfaces; its opposite translated patches
    are recorded in paired_faces/pair_translations. Interior interfaces are
    stored once. Source geometry remains unchanged. No independent hull fitting
    is used. Disconnected components/cavities are retained. Noncontractible
    (winding) grain components and incompatible seam triangulations raise.
    """
    from .surface_validation import validate_surface
    diagram = geometry.get('APD')
    if diagram is None or not diagram.periodic:
        raise ValueError('Whole-grain unwrapping requires a periodic APD')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    box = np.asarray(diagram.box_size, dtype=float)
    source = geometry['Surface']
    ids = np.flatnonzero(source.boundary_ids == 0)
    used = np.unique(source.triangles)
    tol = tolerance*np.linalg.norm(box)
    wrapped = np.mod(source.points[used], box)
    wrapped[np.isclose(wrapped, box, atol=tol, rtol=0)] = 0
    wrapped[np.abs(wrapped) < tol] = 0
    parent = np.arange(len(used))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return int(i)
    for a, b in sorted(cKDTree(wrapped).query_pairs(tol)):
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(len(used))])
    unique, inverse = np.unique(roots, return_inverse=True)
    canonical = wrapped[unique]
    if np.any(np.linalg.norm(wrapped-canonical[inverse], axis=1) > tol):
        raise ValueError('Ambiguous periodic vertex welding; reduce tolerance')
    classes = np.full(len(source.points), -1, dtype=int); classes[used] = inverse
    lattice = np.zeros_like(source.points, dtype=int)
    lattice[used] = np.rint((source.points[used]-canonical[inverse])/box).astype(int)
    if not np.allclose(source.points[used], canonical[inverse]+lattice[used]*box, atol=tol, rtol=0):
        raise ValueError('Periodic vertex coordinates do not match within tolerance')
    face_pairs = list(source.face_grains)
    seam_faces = defaultdict(list)
    for fi in np.flatnonzero(source.boundary_ids):
        bid = int(source.boundary_ids[fi])
        axis = (bid-1)//2
        tangent = [k for k in range(3) if k != axis]
        key = tuple(sorted((int(classes[v]), *map(int, lattice[v, tangent]))
                           for v in source.triangles[fi]))
        seam_faces[(axis, key)].append(int(fi))
    real_seams = []
    for group in seam_faces.values():
        if len(group) != 2:
            raise ValueError('Opposite box triangulations do not match; cannot join periodic fragments')
        a, b = group
        if abs(int(source.boundary_ids[a])-int(source.boundary_ids[b])) != 1:
            raise ValueError('Invalid opposite-face pairing')
        ga, gb = source.face_grains[a][0], source.face_grains[b][0]
        if ga != gb:
            face_pairs[a] = (ga, gb)
            real_seams.append(a)
    ids = np.concatenate([ids, np.array(real_seams, dtype=int)])
    if not len(ids):
        raise WindingGrainError(geometry['Grains'])
    by_grain = defaultdict(list)
    for fi in ids:
        a, b = face_pairs[fi]
        if b is None:
            raise ValueError('Internal periodic triangles require two grain IDs')
        by_grain[int(a)].append((int(fi), 1))
        by_grain[int(b)].append((int(fi), -1))
    expected = set(map(int, geometry['Grains']))
    if set(by_grain) != expected:
        raise WindingGrainError(expected-set(by_grain))
    centers = geometry.get('EntityCenters', {int(g): c for g, c in zip(diagram.grain_ids, diagram.centers)})
    point_keys, point_ids, records = [], {}, defaultdict(list)
    for gid in sorted(by_grain):
        faces = by_grain[gid]
        adjacency = defaultdict(list)
        for fi, _ in faces:
            verts = source.triangles[fi]
            for a, b in zip(verts, np.roll(verts, -1)):
                ca, cb = int(classes[a]), int(classes[b])
                shift = lattice[b]-lattice[a]
                adjacency[ca].append((cb, shift))
                adjacency[cb].append((ca, -shift))
        shifts, components = {}, {}
        component = 0
        for start in sorted(adjacency):
            if start in shifts: continue
            shifts[start] = np.zeros(3, dtype=int); components[start] = component
            queue = deque([start])
            while queue:
                a = queue.popleft()
                for b, delta in adjacency[a]:
                    proposed = shifts[a]+delta
                    if b in shifts:
                        if not np.array_equal(shifts[b], proposed):
                            raise WindingGrainError([gid], np.flatnonzero(shifts[b] != proposed))
                    else:
                        shifts[b] = proposed; components[b] = component; queue.append(b)
            component += 1
        # Choose one lattice image per connected shell near the original seed.
        for c in range(component):
            local_faces = [(fi, sign) for fi, sign in faces if components[int(classes[source.triangles[fi, 0]])] == c]
            xyz = np.array([[canonical[classes[v]]+shifts[int(classes[v])]*box
                             for v in source.triangles[fi][::sign]] for fi, sign in local_faces])
            origin = xyz.mean(axis=(0, 1))
            signed = np.linalg.det(xyz-origin)/6
            if abs(signed.sum()) <= np.finfo(float).eps*np.prod(box):
                raise ValueError(f'Grain {gid} has an open/zero-volume lifted shell')
            centroid = origin+np.einsum('n,ni->i', signed, (xyz-origin).sum(axis=1))/(4*signed.sum())
            offset = np.rint((centers[gid]-centroid)/box).astype(int)
            for v in shifts:
                if components[v] == c: shifts[v] += offset
        for fi, sign in faces:
            nodes = []
            for v in source.triangles[fi][::sign]:
                cv = int(classes[v]); key = (cv, *map(int, shifts[cv]))
                if key not in point_ids:
                    point_ids[key] = len(point_keys); point_keys.append(key)
                nodes.append(point_ids[key])
            records[fi].append((gid, tuple(nodes)))
    keys = np.array(point_keys, dtype=int)
    points = canonical[keys[:, 0]]+keys[:, 1:]*box
    triangles, pairs, sources, paired, translations = [], [], [], [], []
    for fi, rec in sorted(records.items()):
        if len(rec) != 2:
            raise ValueError('Periodic interface must have two grain sides')
        (a, ta), (b, tb) = rec
        if set(ta) == set(tb):
            triangles.append(ta); pairs.append((a, b)); sources.append(fi)
        else:
            # The two sides are translated copies with reversed orientation.
            ia = {int(keys[v, 0]): v for v in ta}; ib = {int(keys[v, 0]): v for v in tb}
            if ia.keys() != ib.keys() or len(ia) != 3:
                raise ValueError('Incompatible periodic interface triangulation')
            delta = np.array([keys[ib[c], 1:]-keys[ia[c], 1:] for c in ia])
            if not np.all(delta == delta[0]):
                raise ValueError('Periodic face does not have a single lattice translation')
            paired.append((len(triangles), len(triangles)+1)); translations.append(delta[0])
            triangles.extend([ta, tb]); pairs.extend([(a, None), (b, None)]); sources.extend([fi, fi])
    surface = APDBoundaryTriangles(points, np.array(triangles, dtype=int),
                    np.full(len(triangles), -1, dtype=int), tuple(pairs), np.zeros(len(triangles), dtype=np.int8))
    stats = validate_surface(points, surface.triangles, pairs)
    for gid, volume in stats['grain_volumes'].items():
        if not np.isclose(volume, geometry['Grains'][gid]['Volume'], rtol=1e-7, atol=1e-10*np.prod(box)):
            raise ValueError(f'Unwrapping changed the volume of grain {gid}')
    result = PeriodicGrainGeometry(surface, box.copy(), keys[:, 0], keys[:, 1:],
        np.array(sources, dtype=int), np.array(paired, dtype=int).reshape(-1, 2),
        np.array(translations, dtype=int).reshape(-1, 3),
        {int(g): int(rec['Phase']) for g, rec in geometry['Grains'].items()},
        dict(whole_grains=True, artificial_box_faces=0, paired_faces=len(paired),
             reference_box_volume=float(np.prod(box)), grain_volumes=stats['grain_volumes']))
    return result
