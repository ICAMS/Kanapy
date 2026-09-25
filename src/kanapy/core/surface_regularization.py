"""Regularization and local repair of a labeled, shared APD triangle complex.

This is a surface preprocessing stage, not a tetrahedral mesher. Operations are
shared across grains; reference geometry is never changed. No Gmsh dependency.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .apd_boundary import APDBoundaryTriangles
from ._surface_remeshing import minimum_angles, repair_pass, classify_bad_triangles


def _quality(xyz):
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    lengths2 = np.sum((xyz - np.roll(xyz, -1, axis=1)) ** 2, axis=2)
    return 2 * np.sqrt(3) * area2 / lengths2.sum(axis=1), cross


def _diagnostics(points, triangles, pairs):
    xyz = points[triangles]
    quality, cross = _quality(xyz)
    angles = []
    for i in range(3):
        a, b = xyz[:, (i+1) % 3]-xyz[:, i], xyz[:, (i+2) % 3]-xyz[:, i]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                           np.einsum('ij,ij->i', a, b))))
    angles = np.min(angles, axis=0)
    origin = (points.min(axis=0)+points.max(axis=0))/2
    signed = np.linalg.det(xyz-origin)/6
    volumes = defaultdict(float)
    for v, (a, b) in zip(signed, pairs):
        volumes[a] += v
        if b is not None:
            volumes[b] -= v
    return dict(triangles=len(triangles), vertices=len(np.unique(triangles)),
                min_quality=float(quality.min()), median_quality=float(np.median(quality)),
                quality_p01=float(np.quantile(quality, .01)),
                min_angle_degrees=float(angles.min()),
                triangles_below_5_degrees=int(np.sum(angles < 5)),
                grain_volumes={int(g): float(v) for g, v in volumes.items()})


def _validate(points, triangles, pairs):
    if not len(triangles) or not np.isfinite(points).all():
        raise ValueError('Expected a finite, nonempty closed surface complex')
    q, _ = _quality(points[triangles])
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('Degenerate surface triangle')
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError('Duplicate surface triangle')
    # Each individual grain must be closed even though the union has junctions.
    edges = defaultdict(list)
    for tri, pair in zip(triangles, pairs):
        for gid, sign in ((pair[0], 1), (pair[1], -1)):
            if gid is None:
                continue
            for a, b in zip(tri, np.roll(tri, -1)):
                edges[(gid, min(a, b), max(a, b))].append(sign*(1 if a < b else -1))
    if any(len(signs) != 2 or sum(signs) != 0 for signs in edges.values()):
        raise ValueError('Each grain requires a closed, consistently oriented manifold shell')
    stats = _diagnostics(points, triangles, pairs)
    if any(v <= 0 for v in stats['grain_volumes'].values()):
        raise ValueError('Nonpositive grain volume')
    return stats


@dataclass
class RegularizedGrainSurface:
    """Separate working surface and audit, with no stale APD polygon mappings.

    ``source_faces`` in ``surface`` is -1: remeshed triangles need not belong to
    any one original polygon. ``triangle_origins`` records contributing original
    triangle indices instead. ``vertex_origins`` records the original vertex
    cluster represented by each output vertex. Grain IDs and phases are retained.
    """
    surface: APDBoundaryTriangles
    triangle_origins: tuple
    vertex_origins: tuple
    phase_by_grain: dict
    report: dict
    periodic_geometry: object = None


def regularize_grain_surface(geometry, *, target_edge_length=None,
                             max_displacement=None, iterations=3,
                             max_volume_change=0.1, min_quality=None,
                             periodic=None, patch_retriangulation=False,
                             simplify_junctions=False, target_angle=5.,
                             junction_energy_weight=0., max_repairs_per_sweep=100,
                             stagnation_sweeps=None, split_winding=True, _fixed_vertices=None):
    """Collapse, flip and tangentially relax a shared APD triangulation.

    Parameters
    ----------
    geometry : dict
        Result of ``build_grain_geometry``, with closed ``Surface`` including
        box faces. Reference data is not modified.
    target_edge_length : float, optional
        Edges shorter than half this length are collapse candidates. Default:
        median input edge length. This is a coarsening scale, not a promise of
        uniform sizing (long edges are not split).
    max_displacement : float, optional
        Maximum distance of a merged vertex from *every* original vertex it
        represents; default 0.25 times target_edge_length. Also bounds the
        nonplanarity allowed for edge flips. In coordinate units.
    iterations : int, default 3
        Number of collapse/flip/relocation sweeps. Disjoint neighborhoods per sweep.
    max_volume_change : float, default 0.1
        Maximum relative per-grain volume change from the reference (10%). A
        configurable conservative budget, not a universal statistical criterion.
        A sweep exceeding this budget is rolled back and processing stops.
    min_quality : float, optional
        Required minimum 4*sqrt(3)*area/sum(edge_length**2). If unmet, raise
        ValueError without committing a result. Without it return diagnostics.
    periodic : bool or None, default None
        None detects APD periodicity. True uses constructed image cells or joins
        legacy periodic fragments, synchronizing translated interface copies
        without box-plane constraints. False requires legacy box-clipped geometry
        (build with periodic_images=False). Legacy winding grains are split into
        compact entities unless split_winding=False. The result exposes its
        periodic_geometry and preserves connectivity of paired exterior patches.

    split_winding : bool, default True
        For legacy periodic fragments, split winding parents along internal grid
        planes into compact entities, retaining parent IDs and phases. Requires
        the original Partition/Background and at least 3 cells per axis.
        False reports winding grains without changing entity count.
    patch_retriangulation : bool, default False
        Replace single/two-vertex disk cavities near triangles below target_angle
        by a max-min-quality boundary triangulation. Removes interior vertices;
        preserves cavity boundaries and labels. Screen replacements for collisions.
    simplify_junctions : bool, default False
        Allow bounded quality-driven motion of triple lines, higher-order
        junctions and box traces off their original APD interfaces. Box planes
        remain exact. Shared incident triangles move together and are screened
        for collisions. Existing short-edge collapse also simplifies junctions.
    target_angle : float, default 5
        Minimum angle in degrees driving the additional repair operations.
        This is a target, not a guarantee; unresolved triangles are reported.
    junction_energy_weight : float, default 0
        Optional dimensionless weight on relative shared internal area in the
        junction objective (equal isotropic energy assumption). Not a prescribed
        angle, anisotropic energy law, or simulated annealing model.
    max_repairs_per_sweep : int, default 100
        Limit accepted cavity replacements and junction moves, each, per sweep.
    stagnation_sweeps : int, optional
        Stop after this many sweeps without reducing the count below target_angle
        or materially increasing the minimum quality. Disabled by default.

    Notes
    -----
    Retains grain and phase identities and the local topology of each grain.
    Junction/box feature signatures restrict collapses; high-order junctions
    and box corners are fixed in the original operations. simplify_junctions
    permits higher-order junction motion while keeping box corners fixed.
    Moves must preserve incident triangle orientation
    and not decrease the local minimum quality. No projection onto the APD is
    performed. Surface volume changes are measured, not constrained to zero.

    The result is a candidate for volume meshing, not an FEM quality certificate.
    Checks cover shell edge incidence, duplicates, local foldovers, volumes and
    quality. New cavity replacements and junction moves have numerical collision
    screening against the current mesh, including intersections beyond shared
    edges/vertices. This does not certify the input or the original operations
    against global self-intersections. Periodic correspondence is not certified.
    This stage can leave small constrained edges;
    it does not remove grains, reconnect junctions or guarantee a minimum angle.
    """
    for name, value in [('patch_retriangulation', patch_retriangulation),
                        ('simplify_junctions', simplify_junctions), ('split_winding', split_winding)]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean')
    if not np.isfinite(target_angle) or not 0 < target_angle < 60:
        raise ValueError('target_angle must be in (0, 60) degrees')
    if not np.isfinite(junction_energy_weight) or junction_energy_weight < 0:
        raise ValueError('junction_energy_weight must be finite and nonnegative')
    if (not isinstance(max_repairs_per_sweep, (int, np.integer))
            or isinstance(max_repairs_per_sweep, bool) or max_repairs_per_sweep < 1):
        raise ValueError('max_repairs_per_sweep must be a positive integer')
    if stagnation_sweeps is not None and (not isinstance(stagnation_sweeps, (int, np.integer))
            or isinstance(stagnation_sweeps, bool) or stagnation_sweeps < 1):
        raise ValueError('stagnation_sweeps must be None or a positive integer')
    if periodic is None:
        periodic = bool(getattr(geometry.get('APD'), 'periodic', False))
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError('periodic must be None or a boolean')
    if geometry.get('PeriodicImageGeometry') and not periodic:
        raise ValueError('Image-cell geometry requires coupled periodic regularization; '
                         'rebuild with periodic_images=False for the box-clipped path')
    if periodic:
        from .periodic_grains import regularize_periodic_grains
        result = regularize_periodic_grains(geometry, target_edge_length=target_edge_length,
            max_displacement=max_displacement, iterations=iterations,
            max_volume_change=max_volume_change, min_quality=min_quality,
            patch_retriangulation=patch_retriangulation, simplify_junctions=simplify_junctions,
            target_angle=target_angle, junction_energy_weight=junction_energy_weight,
            max_repairs_per_sweep=max_repairs_per_sweep, stagnation_sweeps=stagnation_sweeps,
            split_winding=split_winding)
        return result
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    if not np.isfinite(max_volume_change) or not 0 <= max_volume_change < 1:
        raise ValueError('max_volume_change must be in [0, 1)')
    if min_quality is not None and (not np.isfinite(min_quality) or not 0 < min_quality <= 1):
        raise ValueError('min_quality must be in (0, 1]')
    if geometry.get('Representation') != 'APD':
        raise ValueError('regularize_grain_surface requires APD grain geometry')
    source = geometry['Surface']
    points = source.points.copy()
    triangles = source.triangles.copy()
    pairs = list(source.face_grains)
    boundary_ids = source.boundary_ids.copy()
    before = _validate(points, triangles, pairs)
    original = points.copy()
    memberships = {int(v): {int(v)} for v in np.unique(triangles)}
    origins = [{i} for i in range(len(triangles))]
    lengths = np.linalg.norm(points[triangles]-points[np.roll(triangles, -1, axis=1)], axis=2)
    if target_edge_length is None:
        target_edge_length = float(np.median(lengths))
    if not np.isfinite(target_edge_length) or target_edge_length <= 0:
        raise ValueError('target_edge_length must be finite and positive')
    if max_displacement is None:
        max_displacement = .25*target_edge_length
    if not np.isfinite(max_displacement) or max_displacement < 0:
        raise ValueError('max_displacement must be finite and nonnegative')
    fixed_vertices = set() if _fixed_vertices is None else set(map(int, _fixed_vertices))
    accepted_collapses = accepted_flips = accepted_relocations = 0
    stopped = 'iteration_limit'
    repair_totals = dict(retriangulated_patches=0, removed_patch_vertices=0,
                         simplified_junction_vertices=0, collision_rejections=0,
                         repair_volume_rejections=0)
    history = []
    stale = 0
    best_bad = int(np.sum(minimum_angles(points[triangles]) < target_angle))
    best_quality = before['min_quality']
    for sweep in range(iterations):
        saved = (points.copy(), triangles.copy(), pairs.copy(), boundary_ids.copy(),
                 [x.copy() for x in origins], {v: x.copy() for v, x in memberships.items()})
        incident = defaultdict(set)
        edge_faces = defaultdict(list)
        for fi, tri in enumerate(triangles):
            for v in tri:
                incident[int(v)].add(fi)
            for a, b in zip(tri, np.roll(tri, -1)):
                edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
        signatures = {v: frozenset((pairs[i], int(boundary_ids[i])) for i in fs)
                      for v, fs in incident.items()}
        grain_sets = {v: {g for pair, _ in sig for g in pair if g is not None}
                      for v, sig in signatures.items()}
        alive = np.ones(len(triangles), dtype=bool)
        touched = fixed_vertices.copy()
        nc = nf = 0
        edges = sorted(edge_faces, key=lambda e: (np.linalg.norm(points[e[0]]-points[e[1]]), e))
        for a, b in edges:
            if np.linalg.norm(points[a]-points[b]) >= .5*target_edge_length:
                break
            neighborhood = incident[a] | incident[b]
            vertices = set(triangles[list(neighborhood)].ravel())
            if vertices & touched:
                continue
            # An unconstrained endpoint may collapse onto a more constrained
            # endpoint, but never pull a junction/corner off its feature.
            if signatures[a] < signatures[b]:
                a, b = b, a
            if not signatures[b] <= signatures[a]:
                continue
            same_feature = signatures[a] == signatures[b]
            # Freeze high-order junctions and box corners. Ordinary triple lines
            # and box traces can be simplified coherently across all sheets.
            box_ids = {bid for _, bid in signatures[a] if bid}
            if same_feature and (len(grain_sets[a]) >= 4 or len(box_ids) >= 3):
                continue
            # Manifold link condition on each grain shell (including link edges).
            valid = True
            for gid in grain_sets[a]:
                links = []
                for v in (a, b):
                    link_edges = {tuple(sorted(int(x) for x in triangles[i] if x != v))
                                  for i in incident[v] if gid in pairs[i]}
                    links.append(({x for e in link_edges for x in e}, link_edges))
                opposite = {int(x) for i in edge_faces[tuple(sorted((a, b)))] if gid in pairs[i]
                            for x in triangles[i] if x not in (a, b)}
                if links[0][0] & links[1][0] != opposite or links[0][1] & links[1][1]:
                    valid = False
                    break
            if not valid:
                continue
            members = memberships[a] | memberships[b]
            candidate = (points[a]+points[b])/2 if same_feature else points[a].copy()
            if np.max(np.linalg.norm(original[list(members)]-candidate, axis=1)) > max_displacement:
                continue
            ids = np.array(sorted(neighborhood))
            old = triangles[ids]
            new = np.where(old == b, a, old)
            keep = np.array([len(set(t)) == 3 for t in new])
            if not keep.any():
                continue
            xyz = points[new[keep]].copy()
            xyz[new[keep] == a] = candidate
            newq, newn = _quality(xyz)
            oldq, oldn = _quality(points[old])
            if (np.any(newq <= 0) or newq.min() < oldq.min()-1e-12 or
                    np.any(np.einsum('ij,ij->i', newn, oldn[keep]) <= 0)):
                continue
            triangles[ids[keep]] = new[keep]
            alive[ids[~keep]] = False
            points[a] = candidate
            memberships[a] = members
            del memberships[b]
            for i in ids[keep]:
                origins[i] |= set().union(*(origins[j] for j in ids[~keep]))
            touched.update(vertices)
            nc += 1
        # Flip only within one labeled sheet; both triangles use opposite edge
        # directions. Disjoint neighborhoods keep the cached incidence valid.
        for (a, b), fs in sorted(edge_faces.items()):
            if len(fs) != 2 or pairs[fs[0]] != pairs[fs[1]] or boundary_ids[fs[0]] != boundary_ids[fs[1]]:
                continue
            i, j = fs
            if not alive[i] or not alive[j]:
                continue
            old = triangles[[i, j]]
            vertices = set(old.ravel())
            if vertices & touched or len(vertices) != 4:
                continue
            c = next(int(v) for v in old[0] if v not in (a, b))
            d = next(int(v) for v in old[1] if v not in (a, b))
            if tuple(sorted((c, d))) in edge_faces:
                continue
            # Orient a,b as in the first triangle.
            k = list(old[0]).index(a)
            u, v = (a, b) if old[0][(k+1) % 3] == b else (b, a)
            new = np.array([[c, d, v], [d, c, u]])
            oldq, oldn = _quality(points[old])
            newq, newn = _quality(points[new])
            normal = oldn.sum(axis=0)
            if (newq.min() <= oldq.min()+1e-10 or
                    np.any(newn @ normal <= 0) or np.any(newn @ oldn.T <= 0)):
                continue
            # Bound geometric change by the opposite-vertex plane distances.
            heights = [abs(np.dot(points[d]-points[a], oldn[0]))/np.linalg.norm(oldn[0]),
                       abs(np.dot(points[c]-points[a], oldn[1]))/np.linalg.norm(oldn[1])]
            if max(heights) > max_displacement:
                continue
            triangles[[i, j]] = new
            lineage = origins[i] | origins[j]
            origins[i], origins[j] = lineage.copy(), lineage.copy()
            touched.update(vertices)
            nf += 1
        triangles = triangles[alive]
        pairs = [p for p, keep in zip(pairs, alive) if keep]
        boundary_ids = boundary_ids[alive]
        origins = [x for x, keep in zip(origins, alive) if keep]
        # Tangential relocation redistributes vertices left in thin strips.
        # Constrain motion to the tangent common to all incident labeled sheets;
        # box normals consequently keep boundary vertices on their exact planes.
        incident = defaultdict(list)
        for fi, tri in enumerate(triangles):
            for v in tri:
                incident[int(v)].append(fi)
        q, _ = _quality(points[triangles])
        order = sorted(incident, key=lambda v: (min(q[incident[v]]), v))
        touched = fixed_vertices.copy()
        nr = 0
        for v in order:
            ids = np.array(incident[v])
            local = triangles[ids]
            vertices = set(local.ravel())
            if vertices & touched:
                continue
            sheets = defaultdict(list)
            for k, fi in enumerate(ids):
                sheets[(pairs[fi], int(boundary_ids[fi]))].append(k)
            grains = {g for pair, _ in sheets for g in pair if g is not None}
            if len(grains) >= 4:
                continue
            oldq, oldn = _quality(points[local])
            normals = np.array([oldn[ks].sum(axis=0) for ks in sheets.values()])
            norm = np.linalg.norm(normals, axis=1)
            if np.any(norm == 0):
                continue
            _, singular, vh = np.linalg.svd(normals/norm[:, None], full_matrices=True)
            tangent = vh[np.sum(singular > 1e-8):]
            if not len(tangent):
                continue
            neighbors = sorted(vertices-{v})
            delta = points[neighbors].mean(axis=0)-points[v]
            delta = tangent.T @ (tangent @ delta)
            for fraction in (1., .5, .25):
                candidate = points[v]+fraction*delta
                if np.max(np.linalg.norm(original[list(memberships[v])]-candidate, axis=1)) > max_displacement:
                    continue
                xyz = points[local].copy()
                xyz[local == v] = candidate
                newq, newn = _quality(xyz)
                if (newq.min() > oldq.min()+1e-10 and
                        np.all(np.einsum('ij,ij->i', newn, oldn) > 0)):
                    points[v] = candidate
                    touched.update(vertices)
                    nr += 1
                    break
        repair_counts = dict.fromkeys(repair_totals, 0)
        if patch_retriangulation or simplify_junctions:
            triangles, pairs, boundary_ids, origins, repair_counts = repair_pass(
                points, triangles, pairs, boundary_ids, origins, memberships,
                original, _quality, max_displacement, target_angle,
                patch_retriangulation, simplify_junctions, junction_energy_weight,
                max_repairs_per_sweep, before['grain_volumes'], max_volume_change,
                fixed_vertices=fixed_vertices)
        after = _validate(points, triangles, pairs)
        bad_count = int(np.sum(minimum_angles(points[triangles]) < target_angle))
        entry = dict(sweep=sweep+1, accepted=True, triangles=len(triangles),
                     min_quality=after['min_quality'], median_quality=after['median_quality'],
                     quality_p01=after['quality_p01'],
                     min_angle_degrees=after['min_angle_degrees'], below_target_angle=bad_count,
                     collapsed_edges=nc, flipped_edges=nf, relocated_vertices=nr, **repair_counts)
        history.append(entry)
        drift = {g: (v-before['grain_volumes'][g])/before['grain_volumes'][g]
                 for g, v in after['grain_volumes'].items()}
        total = sum(before['grain_volumes'].values())
        if (set(after['grain_volumes']) != set(before['grain_volumes']) or
                max(map(abs, drift.values())) > max_volume_change+1e-12 or
                not np.isclose(sum(after['grain_volumes'].values()), total, rtol=1e-10)):
            points, triangles, pairs, boundary_ids, origins, memberships = saved
            entry['accepted'] = False
            stopped = 'volume_budget'
            break
        accepted_collapses += nc
        accepted_flips += nf
        accepted_relocations += nr
        for key in repair_totals:
            repair_totals[key] += repair_counts[key]
        if ((patch_retriangulation or simplify_junctions) and bad_count == 0):
            stopped = 'target_angle_reached'
            break
        if bad_count < best_bad or after['min_quality'] > best_quality+max(1e-8, best_quality*1e-3):
            stale = 0
            best_bad = min(best_bad, bad_count)
            best_quality = max(best_quality, after['min_quality'])
        else:
            stale += 1
        if stagnation_sweeps is not None and stale >= stagnation_sweeps:
            stopped = 'quality_stagnation'
            break
        if nc+nf+nr+repair_counts['retriangulated_patches']+repair_counts['simplified_junction_vertices'] == 0:
            stopped = 'no_admissible_operations'
            break
    used, inv = np.unique(triangles, return_inverse=True)
    surface = APDBoundaryTriangles(points[used], inv.reshape(-1, 3),
                                  np.full(len(triangles), -1, dtype=int),
                                  tuple(pairs), boundary_ids)
    after = _validate(surface.points, surface.triangles, surface.face_grains)
    if min_quality is not None and after['min_quality'] < min_quality:
        raise ValueError(f"Regularized minimum quality {after['min_quality']:.6g} "
                         f"does not meet requested {min_quality:.6g}")
    phases = {int(g): int(record['Phase']) for g, record in geometry['Grains'].items()}
    def phase_volumes(stats):
        result = defaultdict(float)
        for g, vol in stats['grain_volumes'].items():
            result[phases[g]] += vol
        return dict(result)
    report = dict(before=before, after=after, collapsed_edges=accepted_collapses,
                  flipped_edges=accepted_flips, relocated_vertices=accepted_relocations,
                  stop_reason=stopped, sweeps=sweep+1,
                  target_edge_length=float(target_edge_length), max_displacement=float(max_displacement),
                  max_volume_change=float(max_volume_change),
                  max_vertex_displacement=float(max(
                      np.linalg.norm(original[list(memberships[int(v)])]-points[v], axis=1).max()
                      for v in used)),
                  relative_grain_volume_change={g: v/before['grain_volumes'][g]-1
                                               for g, v in after['grain_volumes'].items()},
                  phase_volumes_before=phase_volumes(before), phase_volumes_after=phase_volumes(after),
                  periodic=False, global_self_intersections_checked=False,
                  patch_retriangulation=bool(patch_retriangulation),
                  simplify_junctions=bool(simplify_junctions), target_angle=float(target_angle),
                  junction_energy_weight=float(junction_energy_weight),
                  max_repairs_per_sweep=int(max_repairs_per_sweep),
                  stagnation_sweeps=None if stagnation_sweeps is None else int(stagnation_sweeps),
                  new_operations_collision_checked=bool(patch_retriangulation or simplify_junctions),
                  history=history, **repair_totals,
                  bad_triangles=classify_bad_triangles(surface.points, surface.triangles,
                      surface.face_grains, surface.boundary_ids, target_angle),
                  fem_ready=False,
                  quality_target_met=None if min_quality is None else True)
    result = RegularizedGrainSurface(surface, tuple(tuple(sorted(x)) for x in origins),
                                   tuple(tuple(sorted(memberships[int(v)])) for v in used),
                                   phases, report)
    return result
