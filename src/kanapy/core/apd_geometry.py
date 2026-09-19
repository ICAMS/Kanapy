"""Grain geometry and volume moments from shared APD boundary triangles."""
import numpy as np


def build_grain_geometry(diagram, phase_by_grain, resolution=10, *, batch_size=8192,
                         optimize=True, tolerance=1e-10):
    """Build a polyhedral APD partition and per-grain geometry without voxel hulls.

    Covariance integrates the uniform volume measure over closed oriented shells,
    including disconnected components and cavities. Shape axes are those of a
    moment-equivalent ellipsoid, sqrt(5 * covariance eigenvalues). For periodic
    grains these describe fragments in the fundamental box, not an unwrapped
    grain. No volume FE mesh or periodic face pairing is generated.
    """
    missing = set(diagram.grain_ids) - set(phase_by_grain)
    if missing:
        raise ValueError(f'Missing phase numbers for APD grains: {sorted(missing)}')
    background = diagram.background_mesh(resolution, batch_size=batch_size)
    partition = background.assemble(tolerance=tolerance, optimize=optimize)
    boundary = partition.boundary_complex()
    surface = boundary.triangulate(include_exterior=True)
    grains, shared = {}, {}
    areas = surface.areas
    origin = diagram.box_size / 2
    for gid, expected_volume in partition.grain_volumes.items():
        indices = [i for i, pair in enumerate(surface.face_grains) if gid in pair]
        signs = np.array([1 if surface.face_grains[i][0] == gid else -1 for i in indices])
        triangles = surface.triangles[indices].copy()
        triangles[signs < 0] = triangles[signs < 0, ::-1]
        xyz = surface.points[triangles] - origin
        volumes = np.linalg.det(xyz) / 6
        volume = volumes.sum()
        if not np.isclose(volume, expected_volume, rtol=1e-7, atol=0):
            raise ValueError(f'Boundary volume disagrees with APD partition for grain {gid}')
        sums = xyz.sum(axis=1)
        mean = np.einsum('n,ni->i', volumes, sums) / (4*volume)
        second = np.einsum('n,nij->ij', volumes,
                          np.einsum('nki,nkj->nij', xyz, xyz) +
                          np.einsum('ni,nj->nij', sums, sums)) / (20*volume)
        covariance = second - np.outer(mean, mean)
        eigenvalues, axes = np.linalg.eigh(covariance)
        if np.any(eigenvalues <= 0):
            raise ValueError(f'Non-positive volume covariance for grain {gid}')
        semiaxes = np.sqrt(5*eigenvalues[::-1])
        vertices = np.unique(triangles)
        grains[gid] = dict(Phase=int(phase_by_grain[gid]), Volume=float(volume),
                           Center=origin+mean, Covariance=covariance,
                           SemiAxes=semiaxes, Axes=axes[:, ::-1],
                           eqDia=float((6*volume/np.pi)**(1/3)),
                           majDia=float(2*semiaxes[0]), minDia=float(2*np.mean(semiaxes[1:])),
                           Area=float(areas[indices].sum()),
                           Vertices=vertices, Points=surface.points[vertices],
                           Simplices=triangles.tolist(), TriangleIndices=np.array(indices),
                           Shells=boundary.grain_shells[gid])
    for i, pair in enumerate(surface.face_grains):
        if pair[1] is not None:
            shared[pair] = shared.get(pair, 0.) + areas[i]
    return dict(Representation='APD', APD=diagram, Background=background,
                Partition=partition, Boundary=boundary, Surface=surface,
                Points=surface.points, Facets=surface.triangles, Grains=grains,
                GBarea=[[a, b, area] for (a, b), area in sorted(shared.items())],
                Ngrains=len(grains), PhaseVolumes={ip: sum(g['Volume'] for g in grains.values()
                                                         if g['Phase'] == ip)
                                                  for ip in set(phase_by_grain.values())})


def label_geometry_points(geometry, points):
    """Label points by the same piecewise-affine costs used for APD meshing.

    Locate each point in the regular Freudenthal background using fractional
    cell coordinates. This avoids convexifying nonconvex or disconnected grains.
    Points must be inside the closed APD box. Exact ties follow diagram order.
    """
    background = geometry['Background']
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('points must be finite with shape (n, 3)')
    if np.any(points < 0) or np.any(points > background.box_size):
        raise ValueError('Slice points must lie inside the APD box')
    shape = np.asarray(background.resolution)
    scaled = points / background.box_size * shape
    cell = np.minimum(np.floor(scaled).astype(int), shape-1)
    frac = scaled-cell
    perm = np.argsort(-frac, axis=1, kind='stable')
    sorted_frac = np.take_along_axis(frac, perm, axis=1)
    bary = np.column_stack([1-sorted_frac[:, 0], sorted_frac[:, 0]-sorted_frac[:, 1],
                            sorted_frac[:, 1]-sorted_frac[:, 2], sorted_frac[:, 2]])
    paths = np.concatenate([np.zeros((len(points),1,3), dtype=int),
                            np.cumsum(np.eye(3, dtype=int)[perm], axis=1)], axis=1)
    nodes = paths + cell[:, None, :]
    ids = np.ravel_multi_index(nodes.reshape(-1,3).T, tuple(shape+1)).reshape(-1,4)
    values = np.einsum('nk,nkg->ng', bary, background.costs[ids])
    return background.grain_ids[np.argmin(values, axis=1)]
