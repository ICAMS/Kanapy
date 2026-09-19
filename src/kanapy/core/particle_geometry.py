"""Triangulated ellipsoid interfaces for particles in a continuous matrix."""
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .apd_boundary import APDBoundaryTriangles


def build_particle_geometry(particles, box_size, resolution=10, *, periodic=False,
                            origin=None, tolerance=1e-10):
    """Mesh packed ellipsoids, retaining matrix space instead of fitting an APD.

    Resolution controls angular sampling (at least eight samples per direction).
    Convex particle shells are clipped to the box; periodic images share their
    original grain ID. Volumes and moments describe the clipped polyhedra.
    The matrix is implicit, with phase 0 and grain 0 reserved for it. Particles
    are meshed independently: overlapping particles are not Boolean-unioned.
    """
    if any(getattr(p, 'inner', None) is not None for p in particles):
        raise ValueError('Particle surface meshing does not support particles with inner structure.')
    originals = [p for p in particles if p.duplicate is None]
    if not originals:
        raise ValueError('No ellipsoids available. Run pack() before generate_grains.')
    raw = np.asarray(resolution)
    if raw.shape not in ((), (3,)) or raw.dtype.kind not in 'iu' or np.any(raw < 1):
        raise ValueError('resolution must be a positive integer or three positive integers')
    nang = max(8, 2 * int(raw.max()) + 1)
    box = np.asarray(box_size, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
        raise ValueError('box_size must contain three finite positive lengths')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    eps = tolerance * np.max(box)
    box_planes = np.vstack([np.column_stack([-np.eye(3), np.zeros(3)]),
                            np.column_stack([np.eye(3), -box])])
    points, triangles, pairs, boundaries = [], [], [], []
    grain_indices, grain_shells, phases = {}, {}, {}
    for particle in originals:
        gid = int(particle.id)
        if gid <= 0 or gid in phases or particle.phasenum <= 0:
            raise ValueError('Matrix particles require unique positive grain IDs and positive phase IDs.')
        phases[gid] = int(particle.phasenum)
        center = particle.get_pos() - origin
        local = particle.surfacePointsGen(nang)
        extent = np.sqrt(np.sum((np.array([particle.a, particle.b, particle.c])[:, None]
                                 * particle.rotation_matrix)**2, axis=0))
        shifts = [range(int(np.ceil((-center[k]-extent[k])/box[k])),
                        int(np.floor((box[k]-center[k]+extent[k])/box[k]))+1)
                  for k in range(3)] if periodic else [(0,)] * 3
        grain_indices[gid], grain_shells[gid] = [], []
        for shift in product(*shifts):
            xyz = local + center + np.asarray(shift)*box
            if np.any(xyz.max(axis=0) <= 0) or np.any(xyz.min(axis=0) >= box):
                continue
            hull = ConvexHull(xyz)
            if np.any(xyz < 0) or np.any(xyz > box):
                planes = np.vstack([hull.equations, box_planes])
                # Chebyshev center supplies a strictly interior point for Qhull.
                result = linprog([0., 0., 0., -1.],
                                 A_ub=np.column_stack([planes[:, :3], np.ones(len(planes))]),
                                 b_ub=-planes[:, 3], bounds=[(None, None)]*3+[(0, None)],
                                 method='highs')
                if not result.success or result.x[3] <= eps:
                    continue
                xyz = HalfspaceIntersection(planes, result.x[:3]).intersections
                xyz = np.clip(xyz, 0, box)
                hull = ConvexHull(xyz)
            faces = hull.simplices.copy()
            normals = np.cross(xyz[faces[:, 1]]-xyz[faces[:, 0]],
                               xyz[faces[:, 2]]-xyz[faces[:, 0]])
            flip = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
            faces[flip] = faces[flip, ::-1]
            first = len(triangles)
            for face in faces:
                boundary = 0
                for axis in range(3):
                    for side in (0, 1):
                        if np.all(np.abs(xyz[face, axis]-side*box[axis]) <= eps):
                            boundary = 2*axis+side+1
                pairs.append((gid, None if boundary else 0))
                boundaries.append(boundary)
            triangles.extend(faces + len(points))
            points.extend(xyz + origin)
            indices = list(range(first, len(triangles)))
            grain_indices[gid].extend(indices)
            grain_shells[gid].append(tuple((i, 1) for i in indices))
    surface = APDBoundaryTriangles(
        np.asarray(points).reshape(-1, 3), np.asarray(triangles, dtype=int).reshape(-1, 3),
        np.arange(len(triangles)), tuple(pairs), np.asarray(boundaries, dtype=np.int8))
    areas = surface.areas
    grains, shared, phase_volumes = {}, [], {}
    reference = origin + box/2
    for gid, indices in grain_indices.items():
        if not indices:
            continue
        faces = surface.triangles[indices]
        xyz = surface.points[faces] - reference
        volumes = np.linalg.det(xyz)/6
        volume = volumes.sum()
        sums = xyz.sum(axis=1)
        mean = np.einsum('n,ni->i', volumes, sums)/(4*volume)
        second = np.einsum('n,nij->ij', volumes,
                          np.einsum('nki,nkj->nij', xyz, xyz)
                          + np.einsum('ni,nj->nij', sums, sums))/(20*volume)
        covariance = second - np.outer(mean, mean)
        values, axes = np.linalg.eigh(covariance)
        if volume <= 0 or np.any(values <= 0):
            raise ValueError(f'Invalid particle surface moments for grain {gid}')
        semiaxes = np.sqrt(5*values[::-1])
        vertices = np.unique(faces)
        grains[gid] = dict(Phase=phases[gid], Volume=float(volume), Center=reference+mean,
                           Covariance=covariance, SemiAxes=semiaxes, Axes=axes[:, ::-1],
                           eqDia=float((6*volume/np.pi)**(1/3)), majDia=float(2*semiaxes[0]),
                           minDia=float(2*np.mean(semiaxes[1:])),
                           Area=float(areas[indices].sum()), Vertices=vertices,
                           Points=surface.points[vertices], Simplices=faces.tolist(),
                           TriangleIndices=np.asarray(indices), Shells=tuple(grain_shells[gid]))
        area = sum(areas[i] for i in indices if boundaries[i] == 0)
        shared.append([0, gid, float(area)])
        phase_volumes[phases[gid]] = phase_volumes.get(phases[gid], 0.) + volume
    phase_volumes[0] = float(np.prod(box) - sum(phase_volumes.values()))
    return dict(Representation='Particles', Boundary=surface, Surface=surface,
                Points=surface.points, Facets=surface.triangles, Grains=grains,
                GBarea=shared, Ngrains=len(grains), PhaseVolumes=phase_volumes)
