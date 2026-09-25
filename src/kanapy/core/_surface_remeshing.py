"""Local cavity triangulation and numerical collision guards for shared surfaces."""
from functools import lru_cache

import numpy as np


def minimum_angles(xyz):
    angles = []
    for i in range(3):
        a, b = xyz[:, (i+1) % 3]-xyz[:, i], xyz[:, (i+2) % 3]-xyz[:, i]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                           np.einsum('ij,ij->i', a, b))))
    return np.min(angles, axis=0)


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _intersection_points(a, b, tol):
    """Triangle intersection vertices, including coplanar contacts."""
    na = np.cross(a[1]-a[0], a[2]-a[0])
    nb = np.cross(b[1]-b[0], b[2]-b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if min(la, lb) == 0:
        return [a.mean(axis=0)]  # invalid candidates are rejected before this
    na, nb = na/la, nb/lb
    da, db = (a-b[0]) @ nb, (b-a[0]) @ na
    if (np.all(da > tol) or np.all(da < -tol) or
            np.all(db > tol) or np.all(db < -tol)):
        return []
    direction = np.cross(na, nb)
    if np.linalg.norm(direction) > 1e-10:
        def cut(x, d):
            out = [p for p, s in zip(x, d) if abs(s) <= tol]
            for i in range(3):
                j = (i+1) % 3
                if (d[i] < -tol and d[j] > tol) or (d[j] < -tol and d[i] > tol):
                    out.append(x[i]+d[i]/(d[i]-d[j])*(x[j]-x[i]))
            return out
        ca, cb = cut(a, da), cut(b, db)
        if not ca or not cb:
            return []
        direction /= np.linalg.norm(direction)
        origin = ca[0]
        sa, sb = (np.asarray(ca)-origin) @ direction, (np.asarray(cb)-origin) @ direction
        lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
        return [] if lo > hi+tol else [origin+lo*direction, origin+hi*direction]
    if np.max(np.abs(db)) > tol:
        return []
    axes = [i for i in range(3) if i != np.argmax(np.abs(na))]
    aa, bb = a[:, axes], b[:, axes]
    scale = max(np.linalg.norm(np.ptp(np.vstack([aa, bb]), axis=0)), tol)
    eps = tol*scale
    def inside(p, tri):
        s = [_cross2(tri[(i+1) % 3]-tri[i], p-tri[i]) for i in range(3)]
        return min(s) >= -eps or max(s) <= eps
    out = [p for p, pp in zip(a, aa) if inside(pp, bb)]
    out += [p for p, pp in zip(b, bb) if inside(pp, aa)]
    for i in range(3):
        for j in range(3):
            u, v = aa[(i+1) % 3]-aa[i], bb[(j+1) % 3]-bb[j]
            det = _cross2(u, v)
            if abs(det) <= eps:
                continue  # collinear overlap endpoints handled by containment
            delta = bb[j]-aa[i]
            t, s = _cross2(delta, v)/det, _cross2(delta, u)/det
            if -tol/scale <= t <= 1+tol/scale and -tol/scale <= s <= 1+tol/scale:
                out.append(a[i]+t*(a[(i+1) % 3]-a[i]))
    return out


def triangles_conflict(a, b, ids_a, ids_b, tolerance):
    """Allow only the shared topological vertex/edge, not other intersection."""
    intersection = _intersection_points(a, b, tolerance)
    if not intersection:
        return False
    common = sorted(set(ids_a) & set(ids_b))
    if not common or len(common) == 3:
        return True
    shared = np.array([a[list(ids_a).index(v)] for v in common])
    for p in intersection:
        if len(common) == 1:
            distance = np.linalg.norm(p-shared[0])
        else:
            edge = shared[1]-shared[0]
            t = np.clip(np.dot(p-shared[0], edge)/np.dot(edge, edge), 0, 1)
            distance = np.linalg.norm(p-shared[0]-t*edge)
        if distance > 4*tolerance:
            return True
    return False


class CollisionGuard:
    """Sweep-local broad phase; changed faces are updated after every acceptance."""
    def __init__(self, points, triangles):
        xyz = points[triangles]
        self.lower, self.upper = xyz.min(axis=1), xyz.max(axis=1)
        self.tolerance = max(float(np.linalg.norm(np.ptp(xyz.reshape(-1, 3), axis=0)))*1e-11,
                             np.finfo(float).tiny)
        self.active = np.ones(len(triangles), dtype=bool)
        self.rejections = 0

    def allows(self, points, triangles, removed, replacement, xyz=None):
        xyz = points[replacement] if xyz is None else xyz
        excluded = set(map(int, removed))
        tol = self.tolerance
        for i, (tri, a) in enumerate(zip(replacement, xyz)):
            lo, hi = a.min(axis=0)-tol, a.max(axis=0)+tol
            candidates = np.flatnonzero(self.active & np.all(self.lower <= hi, axis=1)
                                        & np.all(self.upper >= lo, axis=1))
            for j in candidates:
                if int(j) not in excluded and triangles_conflict(a, points[triangles[j]], tri, triangles[j], tol):
                    self.rejections += 1
                    return False
            for b, other in zip(xyz[:i], replacement[:i]):
                if (np.all(b.min(axis=0) <= hi) and np.all(b.max(axis=0) >= lo)
                        and triangles_conflict(a, b, tri, other, tol)):
                    self.rejections += 1
                    return False
        return True

    def update(self, points, triangles, kept, removed=()):
        self.active[list(removed)] = False
        xyz = points[triangles[list(kept)]]
        self.lower[list(kept)], self.upper[list(kept)] = xyz.min(axis=1), xyz.max(axis=1)


def cavity_triangulation(points, triangles, quality, max_deviation):
    """Remove the interior of a disk patch and maximize its worst triangle quality.

    Dynamic programming considers all valid triangulations of its projected
    simple boundary polygon (at most 18 vertices). Boundary vertices/edges stay
    fixed. Reject folded, non-disk, holed or excessively nonplanar patches.
    """
    edges = {}
    for tri in triangles:
        for a, b in zip(tri, np.roll(tri, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b)))
    if any(len(es) > 2 or (len(es) == 2 and es[0] != es[1][::-1]) for es in edges.values()):
        return None
    perimeter = [es[0] for es in edges.values() if len(es) == 1]
    nxt = dict(perimeter)
    if len(nxt) != len(perimeter) or not 3 <= len(nxt) <= 18:
        return None
    loop = [min(nxt)]
    while nxt.get(loop[-1]) != loop[0]:
        v = nxt.get(loop[-1])
        if v is None or v in loop:
            return None
        loop.append(v)
    if len(loop) != len(nxt):
        return None
    used = set(triangles.ravel())
    if len(used)-len(edges)+len(triangles) != 1:
        return None
    xyz = points[triangles]
    _, normals = quality(xyz)
    normal = normals.sum(axis=0)
    length = np.linalg.norm(normal)
    if length == 0 or np.any(normals @ normal <= 0):
        return None
    normal /= length
    origin = points[loop].mean(axis=0)
    scale = np.linalg.norm(np.ptp(points[list(used)], axis=0))
    tol = max(scale*1e-11, np.finfo(float).tiny)
    # A full range bound also bounds the separation between the old and new
    # height graphs over a valid common projection.
    heights = (points[list(used)]-origin) @ normal
    if np.ptp(heights) > max_deviation+tol:
        return None
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(axis, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    xy = (points[loop]-origin) @ np.array([u, v]).T
    n = len(loop)
    eps = scale*tol
    def crosses(a, b, c, d):
        s1, s2 = _cross2(b-a, c-a), _cross2(b-a, d-a)
        s3, s4 = _cross2(d-c, a-c), _cross2(d-c, b-c)
        return s1*s2 < -eps**2 and s3*s4 < -eps**2
    for i in range(n):
        for j in range(i+1, n):
            if len({i, (i+1) % n, j, (j+1) % n}) == 4 and crosses(xy[i], xy[(i+1) % n], xy[j], xy[(j+1) % n]):
                return None
    def inside(p):
        winding = 0
        for a, b in zip(xy, np.roll(xy, -1, axis=0)):
            cross = _cross2(b-a, p-a)
            if a[1] <= p[1] < b[1] and cross > 0: winding += 1
            if b[1] <= p[1] < a[1] and cross < 0: winding -= 1
        return winding != 0
    @lru_cache(None)
    def diagonal(i, j):
        if j-i == 1 or (i == 0 and j == n-1):
            return True
        for k in range(n):
            l = (k+1) % n
            if len({i, j, k, l}) == 4 and crosses(xy[i], xy[j], xy[k], xy[l]):
                return False
        # Reject a diagonal passing through another boundary vertex.
        a, b = xy[i], xy[j]
        for k in range(n):
            if k not in (i, j) and abs(_cross2(b-a, xy[k]-a)) <= eps and np.dot(xy[k]-a, xy[k]-b) < 0:
                return False
        return inside((a+b)/2)
    @lru_cache(None)
    def solve(i, j):
        if j-i < 2:
            return (float('inf'), ())
        if not diagonal(i, j):
            return (-1., ())
        best = (-1., ())
        for k in range(i+1, j):
            if _cross2(xy[k]-xy[i], xy[j]-xy[i]) <= eps:
                continue
            left, right = solve(i, k), solve(k, j)
            tri = (loop[i], loop[k], loop[j])
            q = float(quality(points[np.array([tri])])[0][0])
            score = min(left[0], right[0], q)
            if score > best[0]:
                best = (score, left[1]+right[1]+(tri,))
        return best
    score, result = solve(0, n-1)
    if score <= 0:
        return None
    new = np.array(result, dtype=int)
    # Projection must cover exactly the same disk, without folded old facets.
    old_area = .5*sum(normals @ normal)
    _, new_normals = quality(points[new])
    if not np.isclose(.5*sum(new_normals @ normal), old_area, rtol=1e-9, atol=scale*tol):
        return None
    return new


def repair_pass(points, triangles, pairs, boundary_ids, origins, memberships,
                original, quality, max_displacement, target_angle,
                patch_retriangulation, simplify_junctions, junction_energy_weight,
                max_repairs_per_sweep, reference_volumes, max_volume_change,
                fixed_vertices=()):
    """Quality-driven local replacements and off-surface junction optimization.

    Topology/labels and box contacts survive; every proposed replacement is
    screened against the current complex. The caller checks volume budgets and
    can roll back the entire sweep.
    """
    from collections import defaultdict
    from scipy.optimize import minimize

    guard = CollisionGuard(points, triangles)
    incident = defaultdict(set)
    edge_faces = defaultdict(set)
    for fi, tri in enumerate(triangles):
        for v in tri:
            incident[int(v)].add(fi)
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].add(fi)
    angles = minimum_angles(points[triangles])
    bad_faces = np.flatnonzero(angles < target_angle)
    counts = dict(retriangulated_patches=0, removed_patch_vertices=0,
                  simplified_junction_vertices=0, collision_rejections=0,
                  repair_volume_rejections=0)
    signatures = {v: {(pairs[i], int(boundary_ids[i])) for i in fs}
                  for v, fs in incident.items()}
    touched = set(fixed_vertices)
    origin = (points.min(axis=0)+points.max(axis=0))/2
    current_volumes = defaultdict(float)
    for signed, (a, b) in zip(np.linalg.det(points[triangles]-origin)/6, pairs):
        current_volumes[a] += signed
        if b is not None: current_volumes[b] -= signed
    def volume_changes(ids, xyz, new_pairs):
        changes = defaultdict(float)
        for factor, coords, labels in [(-1, points[triangles[ids]], [pairs[i] for i in ids]),
                                        (1, xyz, new_pairs)]:
            for signed, (a, b) in zip(factor*np.linalg.det(coords-origin)/6, labels):
                changes[a] += signed
                if b is not None: changes[b] -= signed
        for g, delta in changes.items():
            ref = reference_volumes[g]
            if abs(current_volumes[g]+delta-ref)/ref > max_volume_change+1e-12:
                counts['repair_volume_rejections'] += 1
                return None
        return changes
    def improves(old_xyz, new_xyz):
        oldq, _ = quality(old_xyz); newq, _ = quality(new_xyz)
        olda, newa = minimum_angles(old_xyz), minimum_angles(new_xyz)
        return (np.isfinite(newq).all() and newq.min() > oldq.min()+1e-9
                and newa.min() >= olda.min()-1e-8
                and np.sum(newa < target_angle) <= np.sum(olda < target_angle))

    if patch_retriangulation:
        seeds = sorted({int(v) for fi in bad_faces for v in triangles[fi]},
                       key=lambda v: (min(angles[list(incident[v])]), v))
        for v in seeds:
            if counts['retriangulated_patches'] >= max_repairs_per_sweep:
                break
            if v in touched or len(signatures[v]) != 1:
                continue
            star = incident[v]
            # A single star or two neighboring stars form a cavity; the latter
            # replaces a larger configuration without requiring intermediate
            # single-edge improvements.
            neighbors = sorted(set(triangles[list(star)].ravel())-{v})
            cavities = [star] + [star | incident[int(w)] for w in neighbors
                                if signatures[int(w)] == signatures[v]]
            for ids_set in cavities:
                ids = np.array(sorted(ids_set))
                vertices = set(triangles[ids].ravel())
                if vertices & touched:
                    continue
                new = cavity_triangulation(points, triangles[ids], quality, max_displacement)
                if new is None or len(new) > len(ids) or not improves(points[triangles[ids]], points[new]):
                    continue
                # An introduced diagonal cannot already belong to another patch.
                edges_old = defaultdict(int)
                for tri in triangles[ids]:
                    for a, b in zip(tri, np.roll(tri, -1)):
                        edges_old[tuple(sorted((int(a), int(b))))] += 1
                valid = True
                for tri in new:
                    for a, b in zip(tri, np.roll(tri, -1)):
                        edge = tuple(sorted((int(a), int(b))))
                        if edges_old.get(edge) != 1 and edge_faces.get(edge, set())-ids_set:
                            valid = False
                changes = volume_changes(ids, points[new], [pairs[ids[0]]]*len(new))
                if not valid or changes is None or not guard.allows(points, triangles, ids, new):
                    continue
                for g, delta in changes.items(): current_volumes[g] += delta
                kept, removed = ids[:len(new)], ids[len(new):]
                lineage = set().union(*(origins[i] for i in ids))
                triangles[kept] = new
                for i in kept:
                    origins[i] = lineage.copy()
                guard.update(points, triangles, kept, removed)
                removed_vertices = vertices-set(new.ravel())
                for w in removed_vertices:
                    memberships.pop(int(w), None)
                touched.update(vertices)
                counts['retriangulated_patches'] += 1
                counts['removed_patch_vertices'] += len(removed_vertices)
                break

    if simplify_junctions and max_displacement > 0:
        seeds = sorted({int(v) for fi in bad_faces for v in triangles[fi]},
                       key=lambda v: (min(angles[list(incident[v])]), v))
        for v in seeds:
            if counts['simplified_junction_vertices'] >= max_repairs_per_sweep:
                break
            ids = np.array(sorted(incident[v]))
            if not guard.active[ids].all():
                continue
            local = triangles[ids]
            vertices = set(local.ravel())
            if vertices & touched or len(signatures[v]) < 2:
                continue
            # Only physical box constraints are fixed. Grain-pair tangent planes
            # no longer pin the junction to the original APD intersection.
            fixed = {(bid-1)//2 for _, bid in signatures[v] if bid}
            free = [axis for axis in range(3) if axis not in fixed]
            if not free:
                continue
            old_xyz = points[local].copy()
            oldq, oldn = quality(old_xyz)
            if minimum_angles(old_xyz).min() >= target_angle:
                continue
            reference = original[sorted(memberships[v])]
            # Coordinates scaled to the movement budget improve conditioning.
            center = points[v].copy()
            lower = np.max(reference[:, free]-max_displacement, axis=0)
            upper = np.min(reference[:, free]+max_displacement, axis=0)
            if np.any(lower >= upper):
                continue
            bounds = list(zip((lower-center[free])/max_displacement,
                              (upper-center[free])/max_displacement))
            mask = local == v
            internal = np.array([boundary_ids[i] == 0 for i in ids])
            internal_area0 = np.linalg.norm(oldn[internal], axis=1).sum()
            def candidate_data(x):
                p = center.copy(); p[free] += max_displacement*np.asarray(x)
                xyz = old_xyz.copy(); xyz[mask] = p
                return p, xyz
            def objective(x):
                p, xyz = candidate_data(x)
                q, normals = quality(xyz)
                distance = np.linalg.norm(reference-p, axis=1).max()/max_displacement
                if (distance > 1+1e-10 or not np.isfinite(q).all() or np.any(q <= 0)
                        or np.any(np.einsum('ij,ij->i', normals, oldn) <= 0)):
                    return 1e6+1e3*max(0, distance-1)
                a = minimum_angles(xyz)
                deficit = np.maximum(0, 1-a/target_angle)
                score = np.sum(deficit**2)+.001*np.sum((1-q)**2)
                # Optional equal-isotropic-energy area term, counted once per
                # shared interior triangle, excluding artificial box surfaces.
                if junction_energy_weight and internal_area0 > 0:
                    score += junction_energy_weight*(
                        np.linalg.norm(normals[internal], axis=1).sum()/internal_area0-1)
                return float(score)
            solution = minimize(objective, np.zeros(len(free)), method='Powell', bounds=bounds,
                                options={'maxiter': 20, 'maxfev': 160, 'xtol': 1e-4, 'ftol': 1e-6})
            # A partial optimizer result is acceptable only after all independent
            # geometric and quality gates pass; success alone is not sufficient.
            for fraction in (1., .5, .25):
                p, xyz = candidate_data(fraction*solution.x)
                _, normals = quality(xyz)
                changes = volume_changes(ids, xyz, [pairs[i] for i in ids])
                if (changes is None or np.linalg.norm(reference-p, axis=1).max() > max_displacement
                        or not improves(old_xyz, xyz)
                        or np.any(np.einsum('ij,ij->i', normals, oldn) <= 0)
                        or not guard.allows(points, triangles, ids, local, xyz)):
                    continue
                for g, delta in changes.items(): current_volumes[g] += delta
                points[v] = p
                guard.update(points, triangles, ids)
                touched.update(vertices)
                counts['simplified_junction_vertices'] += 1
                break
    counts['collision_rejections'] = guard.rejections
    active = guard.active
    return (triangles[active], [p for p, keep in zip(pairs, active) if keep],
            boundary_ids[active], [o for o, keep in zip(origins, active) if keep], counts)


def classify_bad_triangles(points, triangles, pairs, boundary_ids, target_angle):
    """Serializable locations/constraints for the unresolved low-angle tail."""
    from collections import defaultdict
    grains, box = defaultdict(set), defaultdict(set)
    for tri, pair, bid in zip(triangles, pairs, boundary_ids):
        for v in tri:
            grains[int(v)].update(g for g in pair if g is not None)
            if bid: box[int(v)].add(int(bid))
    angles = minimum_angles(points[triangles])
    result = []
    for fi in np.flatnonzero(angles < target_angle):
        tri = triangles[fi]
        features = []
        if any(len(grains[int(v)]) >= 4 for v in tri): features.append('higher_order_junction')
        if any(len(grains[int(v)]) == 3 for v in tri): features.append('triple_line')
        if any(box[int(v)] for v in tri): features.append('box_contact')
        result.append(dict(triangle=int(fi), grain_pair=[None if g is None else int(g) for g in pairs[fi]],
                           center=points[tri].mean(axis=0).tolist(), min_angle_degrees=float(angles[fi]),
                           features=features or ['interface_interior']))
    return result
