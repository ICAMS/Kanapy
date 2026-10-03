"""Validation and diagnostics for shared, oriented grain boundary surfaces."""
from collections import defaultdict

import numpy as np


def _quality(xyz):
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    lengths2 = np.sum((xyz - np.roll(xyz, -1, axis=1)) ** 2, axis=2)
    return 2 * np.sqrt(3) * area2 / lengths2.sum(axis=1), cross


def _diagnostics(points, triangles, pairs):
    xyz = points[triangles]
    quality, _ = _quality(xyz)
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


def validate_surface(points, triangles, pairs):
    """Check closed grain shells and return quality and signed-volume statistics.

    Each triangle is oriented outward from the first grain in its pair and
    inward to the second (None denotes the exterior). Shared interfaces are
    stored once. This checks degeneracy, duplicates, oriented edge incidence,
    and positive grain volumes; it does not test geometric self-intersection.
    """
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
