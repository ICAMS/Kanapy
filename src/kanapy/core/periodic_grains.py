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
    from .surface_regularization import _validate
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
    stats = _validate(points, surface.triangles, pairs)
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


def regularize_periodic_grains(geometry, **options):
    """Whole-grain regularization with coupled motion of periodic vertex copies.

    Interior topology operations use the existing regularizer. Paired exterior
    patch connectivity is retained, then periodic vertex orbits move together
    without box-plane constraints. This preserves translated interface meshes;
    it does not independently remesh opposite periodic patches.
    """
    from scipy.optimize import minimize
    from .surface_regularization import regularize_grain_surface, _validate, _quality
    from ._surface_remeshing import CollisionGuard, minimum_angles, classify_bad_triangles

    whole = unwrap_periodic_grains(geometry, split_winding=options.pop('split_winding', True))
    frequency = dict(zip(*np.unique(whole.vertex_classes, return_counts=True)))
    fixed = np.array([i for i, c in enumerate(whole.vertex_classes) if frequency[c] > 1], dtype=int)
    working = dict(geometry, Surface=whole.surface, PeriodicImageGeometry=False, Grains={g: dict(Phase=p)
                          for g, p in whole.phase_by_grain.items()})
    controls = dict(options)
    required_quality = controls.pop('min_quality', None)
    # Interior topology repair uses geometry only; the synchronized stage below
    # applies the optional physical-area term with periodic multiplicities and
    # excludes artificial same-parent split interfaces.
    controls['junction_energy_weight'] = 0.
    if required_quality is not None and (not np.isfinite(required_quality) or not 0 < required_quality <= 1):
        raise ValueError('min_quality must be in (0, 1]')
    result = regularize_grain_surface(working, periodic=False, min_quality=None,
                                     _fixed_vertices=fixed, **controls)
    surface = result.surface
    # Paired topology was protected in the interior pass, so each paired face
    # must survive with its own unique lineage.
    face_lookup = {orig[0]: i for i, orig in enumerate(result.triangle_origins) if len(orig) == 1}
    try:
        paired = np.array([[face_lookup[int(a)], face_lookup[int(b)]] for a, b in whole.paired_faces], dtype=int).reshape(-1, 2)
    except KeyError as exc:
        raise ValueError('Interior remeshing changed protected periodic face topology') from exc
    classes, shifts = [], []
    for orig in result.vertex_origins:
        v = min(orig, key=lambda x: int(whole.vertex_classes[x]))
        classes.append(whole.vertex_classes[v]); shifts.append(whole.vertex_shifts[v])
    classes, shifts = np.array(classes, dtype=int), np.array(shifts, dtype=int)
    by_class = defaultdict(list)
    for v, c in enumerate(classes): by_class[int(c)].append(v)
    incident = defaultdict(set)
    for fi, tri in enumerate(surface.triangles):
        for v in tri: incident[int(v)].add(fi)
    original = whole.surface.points
    max_move = result.report['max_displacement']
    target = options.get('target_angle', 5.)
    iterations = options.get('iterations', 3)
    budget = options.get('max_volume_change', .1)
    max_moves = options.get('max_repairs_per_sweep', 100)
    ref_volumes = result.report['before']['grain_volumes']
    origin = surface.points.mean(axis=0)
    stats = _validate(surface.points, surface.triangles, surface.face_grains)
    volumes = dict(stats['grain_volumes'])
    history, total_moves, collisions = [], 0, 0
    stopped = 'iteration_limit'
    stale = 0
    best_bad = int(np.sum(minimum_angles(surface.points[surface.triangles]) < target))
    best_quality = stats['min_quality']
    physical_weights = np.ones(len(surface.triangles))
    physical_weights[paired.ravel()] = .5
    for fi, (a, b) in enumerate(surface.face_grains):
        if b is not None and whole.grain_parent_ids[a] == whole.grain_parent_ids[b]:
            physical_weights[fi] = 0
    for a, b in paired:
        if whole.grain_parent_ids[surface.face_grains[a][0]] == whole.grain_parent_ids[surface.face_grains[b][0]]:
            physical_weights[[a, b]] = 0
    for sweep in range(iterations):
        if max_move == 0:
            stopped = 'zero_displacement_budget'; break
        guard = CollisionGuard(surface.points, surface.triangles)
        periodic_guard = _PeriodicCollisionGuard(surface, classes, shifts, whole.box_size, max_move)
        angles = minimum_angles(surface.points[surface.triangles])
        order = sorted(by_class, key=lambda c: (min(angles[list(set().union(*(incident[v] for v in by_class[c])))]), c))
        touched = set(); moved = 0
        for c in order:
            if moved >= max_moves: break
            nodes = by_class[c]
            ids = np.array(sorted(set().union(*(incident[v] for v in nodes))))
            if touched.intersection(ids): continue
            local = surface.triangles[ids]
            old_xyz = surface.points[local].copy()
            old_angles = minimum_angles(old_xyz)
            if old_angles.min() >= target: continue
            oldq, oldn = _quality(old_xyz)
            area0 = np.dot(np.linalg.norm(oldn, axis=1), physical_weights[ids])
            mask = np.isin(local, nodes)
            # A single displacement acts on the entire lattice orbit. Relative
            # offsets to every represented reference vertex bound that movement.
            offsets = np.concatenate([original[list(result.vertex_origins[v])]-surface.points[v] for v in nodes])
            lower = np.max(offsets-max_move, axis=0)/max_move
            upper = np.min(offsets+max_move, axis=0)/max_move
            if np.any(lower >= upper): continue
            def objective(x):
                delta = max_move*np.asarray(x)
                distance = np.linalg.norm(offsets-delta, axis=1).max()/max_move
                xyz = old_xyz.copy(); xyz[mask] += delta
                q, normals = _quality(xyz)
                if (distance > 1+1e-10 or not np.isfinite(q).all() or np.any(q <= 0)
                        or np.any(np.einsum('ij,ij->i', normals, oldn) <= 0)):
                    return 1e6+1e3*max(0, distance-1)
                a = minimum_angles(xyz)
                energy = (options.get('junction_energy_weight', 0.)*(
                    np.dot(np.linalg.norm(normals, axis=1), physical_weights[ids])/area0-1)
                    if area0 > 0 else 0.)
                return float(np.sum(np.maximum(0, 1-a/target)**2)+.001*np.sum((1-q)**2)+energy)
            solution = minimize(objective, np.zeros(3), method='Powell', bounds=list(zip(lower, upper)),
                                options={'maxiter': 20, 'maxfev': 160, 'xtol': 1e-4, 'ftol': 1e-6})
            for fraction in (1., .5, .25):
                delta = fraction*max_move*solution.x
                xyz = old_xyz.copy(); xyz[mask] += delta
                q, normals = _quality(xyz); a = minimum_angles(xyz)
                if (np.linalg.norm(offsets-delta, axis=1).max() > max_move
                        or q.min() <= oldq.min()+1e-9 or a.min() < old_angles.min()-1e-8
                        or np.sum(a < target) > np.sum(old_angles < target)
                        or np.any(np.einsum('ij,ij->i', normals, oldn) <= 0)):
                    continue
                changes = defaultdict(float)
                dv = (np.linalg.det(xyz-origin)-np.linalg.det(old_xyz-origin))/6
                for d, fi in zip(dv, ids):
                    ga, gb = surface.face_grains[fi]
                    changes[ga] += d
                    if gb is not None: changes[gb] -= d
                if any(abs(volumes[g]+d-ref_volumes[g])/ref_volumes[g] > budget+1e-12 for g, d in changes.items()):
                    continue
                if not guard.allows(surface.points, surface.triangles, ids, local, xyz): continue
                if not periodic_guard.allows(ids, xyz): continue
                surface.points[nodes] += delta
                guard.update(surface.points, surface.triangles, ids)
                for g, d in changes.items(): volumes[g] += d
                touched.update(ids); moved += 1
                break
        total_moves += moved; collisions += guard.rejections+periodic_guard.rejections
        stats = _validate(surface.points, surface.triangles, surface.face_grains)
        if not np.isclose(sum(stats['grain_volumes'].values()), np.prod(whole.box_size), rtol=1e-8):
            raise ValueError('Periodic regularization changed the cell volume')
        history.append(dict(sweep=sweep+1, moved_vertex_orbits=moved,
                            min_angle_degrees=stats['min_angle_degrees'],
                            triangles_below_5_degrees=stats['triangles_below_5_degrees']))
        if stats['min_angle_degrees'] >= target: stopped = 'target_angle_reached'; break
        if moved == 0: stopped = 'no_admissible_periodic_moves'; break
        bad_count = int(np.sum(minimum_angles(surface.points[surface.triangles]) < target))
        if bad_count < best_bad or stats['min_quality'] > best_quality+max(1e-8, best_quality*1e-3):
            stale = 0
            best_bad = min(best_bad, bad_count); best_quality = max(best_quality, stats['min_quality'])
        else:
            stale += 1
        if options.get('stagnation_sweeps') is not None and stale >= options['stagnation_sweeps']:
            stopped = 'quality_stagnation'; break
    # Explicitly check translated copies and reversed triangle orientation.
    area_vectors = surface.area_vectors
    for (a, b), shift in zip(paired, whole.pair_translations):
        ta, tb = surface.triangles[a], surface.triangles[b]
        pb = {int(classes[v]): surface.points[v] for v in tb}
        if not np.allclose(np.array([pb[int(classes[v])] for v in ta])-surface.points[ta],
                           shift*whole.box_size, atol=1e-9*np.linalg.norm(whole.box_size), rtol=0):
            raise ValueError('Periodic face correspondence was lost')
        if np.dot(area_vectors[a], area_vectors[b]) >= 0:
            raise ValueError('Periodic boundary copies have inconsistent orientation')
    stats = _validate(surface.points, surface.triangles, surface.face_grains)
    if required_quality is not None and stats['min_quality'] < required_quality:
        raise ValueError(f"Regularized minimum quality {stats['min_quality']:.6g} does not meet requested {required_quality:.6g}")
    report = result.report
    report.update(periodic=True, whole_grains=True, artificial_box_faces=0,
        interior_stop_reason=report['stop_reason'], stop_reason=stopped,
        periodic_face_pairs=len(paired), periodic_vertex_orbits=len(by_class),
        synchronized_vertex_moves=total_moves, periodic_history=history,
        junction_energy_weight=float(options.get('junction_energy_weight', 0.)),
        periodic_stop_reason=stopped, after=stats,
        periodic_collision_scope='synchronized moves checked against translated images; input/interior pass not globally certified',
        periodic_face_topology_preserved=True, periodic_collision_rejections=collisions,
        quality_target_met=None if required_quality is None else True,
        relative_grain_volume_change={g: v/ref_volumes[g]-1 for g, v in stats['grain_volumes'].items()},
        bad_triangles=classify_bad_triangles(surface.points, surface.triangles,
                         surface.face_grains, surface.boundary_ids, target))
    report['phase_volumes_after'] = {p: sum(v for g, v in stats['grain_volumes'].items()
        if whole.phase_by_grain[g] == p) for p in set(whole.phase_by_grain.values())}
    report['max_vertex_displacement'] = float(max(np.linalg.norm(
        original[list(orig)]-point, axis=1).max() for point, orig in zip(surface.points, result.vertex_origins)))
    sources = np.array([whole.source_triangles[orig[0]] if len(orig) == 1 else -1 for orig in result.triangle_origins])
    parent_volumes = defaultdict(float)
    for g, volume in stats['grain_volumes'].items(): parent_volumes[whole.grain_parent_ids[g]] += volume
    report.update(grain_parent_ids=whole.grain_parent_ids.copy(),
                  split_parent_grains=whole.report['split_parent_grains'],
                  parent_grain_volumes=dict(parent_volumes), split_planes=whole.report['split_planes'],
                  split_axes=whole.report['split_axes'])
    split_faces = set()
    for fi, (a, b) in enumerate(surface.face_grains):
        if b is not None and whole.grain_parent_ids[a] == whole.grain_parent_ids[b]: split_faces.add(fi)
    for a, b in paired:
        if whole.grain_parent_ids[surface.face_grains[a][0]] == whole.grain_parent_ids[surface.face_grains[b][0]]:
            split_faces.update([int(a), int(b)])
    report['construction'] = whole.report.get('construction', 'legacy_fragment_unwrapping')
    if geometry.get('PeriodicImageGeometry'):
        report['split_face_ids'] = []
        report['self_image_face_ids'] = sorted(split_faces)
    else:
        report['split_face_ids'] = sorted(split_faces)
    result.periodic_geometry = PeriodicGrainGeometry(surface, whole.box_size, classes, shifts,
        sources, paired, whole.pair_translations.copy(), whole.phase_by_grain, report,
        whole.grain_parent_ids.copy())
    return result


class _PeriodicCollisionGuard:
    """Screen candidate faces against all intersecting translated mesh images.

    A toroidal center tree is a broad phase only. Exact integer AABB ranges
    enumerate every image that can intersect a candidate; shared lattice nodes
    identify allowed contacts. Tree movement padding covers a whole sweep.
    """
    def __init__(self, surface, classes, shifts, box, movement):
        self.surface, self.classes, self.shifts, self.box = surface, classes, shifts, box
        xyz = surface.points[surface.triangles]
        centers = xyz.mean(axis=1)
        self.tree = cKDTree(np.mod(centers, box), boxsize=box)
        self.radius = float(np.linalg.norm(xyz-centers[:, None, :], axis=2).max())+2*movement
        self.tolerance = 1e-11*np.linalg.norm(box)
        self.rejections = 0

    def allows(self, ids, xyz):
        from itertools import product
        from ._surface_remeshing import triangles_conflict
        proposed = dict(zip(map(int, ids), xyz))
        for fi, a in zip(ids, xyz):
            tri_a = self.surface.triangles[fi]
            keys_a = [(int(self.classes[v]), *map(int, self.shifts[v])) for v in tri_a]
            center = a.mean(axis=0)
            radius = np.linalg.norm(a-center, axis=1).max()+self.radius+self.tolerance
            candidates = self.tree.query_ball_point(np.mod(center, self.box), radius)
            lo, hi = a.min(axis=0)-self.tolerance, a.max(axis=0)+self.tolerance
            for j in candidates:
                tri_b = self.surface.triangles[j]
                b = proposed.get(j)
                if b is None: b = self.surface.points[tri_b]
                lower = np.ceil((lo-b.max(axis=0))/self.box).astype(int)
                upper = np.floor((hi-b.min(axis=0))/self.box).astype(int)
                for offset in product(*(range(l, h+1) for l, h in zip(lower, upper))):
                    offset = np.array(offset, dtype=int)
                    keys_b = [(int(self.classes[v]), *map(int, self.shifts[v]+offset)) for v in tri_b]
                    if set(keys_a) == set(keys_b): continue  # same physical facet
                    if triangles_conflict(a, b+offset*self.box, keys_a, keys_b, self.tolerance):
                        self.rejections += 1
                        return False
        return True
