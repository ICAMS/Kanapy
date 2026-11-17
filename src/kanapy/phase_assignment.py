"""
Choose original measurements for filling omitted phase pixels.
"""

import numpy as np
from scipy.spatial import cKDTree


def nearest_phase_sources(xy, phase_ids, donors, query_points):
    """
    Select nearest donors, using phase counts to resolve equal distances.

    Only measurements at the shortest distance vote. Tied phase counts use
    the smallest phase ID. The original nearest donor is kept if it belongs
    to the winning phase; otherwise the smallest original source index wins.
    Single phase inputs retain the existing k one assignment exactly.

    Parameters
    ----------
    xy : numpy.ndarray
        Original measurement coordinates with shape (n, 2).
    phase_ids : numpy.ndarray
        Original phase ID for each measurement.
    donors : numpy.ndarray
        Indices of retained measurements with valid orientations.
    query_points : array_like
        Coordinates to fill, with shape (m, 2).

    Returns
    -------
    numpy.ndarray
        Original donor indices for copying phase, orientation, and confidence.

    Raises
    ------
    ValueError
        If no donor measurements are provided.
    """
    donors = np.asarray(donors, dtype=int)
    if donors.size == 0:
        raise ValueError("Phase assignment requires at least one donor.")
    query_points = np.asarray(query_points, dtype=float)
    donor_xy = np.asarray(xy)[donors]
    donor_phases = np.asarray(phase_ids)[donors]
    tree = cKDTree(donor_xy)
    nearest_distances, nearest = tree.query(query_points, k=1, workers=-1)
    sources = donors[nearest].copy()
    if query_points.shape[0] == 0 or np.unique(donor_phases).size == 1:
        return sources

    second_distances = tree.query(query_points, k=[2], workers=-1)[0][:, 0]
    # permit only floating point roundoff, scaled to the actual distance
    tolerance = 64.0 * np.finfo(float).eps * nearest_distances
    tied = np.flatnonzero(second_distances - nearest_distances <= tolerance)
    for start in range(0, tied.size, 4096):
        batch = tied[start:start + 4096]
        radii = np.nextafter(nearest_distances[batch] + tolerance[batch], np.inf)
        candidates = tree.query_ball_point(
            query_points[batch], radii, workers=-1, return_sorted=True,
        )
        for target, indices in zip(batch, candidates):
            indices = np.asarray(indices, dtype=int)
            distances = np.linalg.norm(donor_xy[indices] - query_points[target], axis=1)
            indices = indices[
                np.abs(distances - nearest_distances[target]) <= tolerance[target]
            ]
            phases, counts = np.unique(donor_phases[indices], return_counts=True)
            winner = phases[np.argmax(counts)]
            if donor_phases[nearest[target]] == winner:
                continue
            winning_sources = donors[indices[donor_phases[indices] == winner]]
            sources[target] = np.min(winning_sources)
    return sources
