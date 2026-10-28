"""
Utilities for resampling staggered hex EBSD maps to rectangular grids.
"""

import numpy as np
from scipy.spatial import cKDTree


def get_equal_area_square_spacing(x_step, y_step):
    """
    Return square grid spacing that preserves hexagonal sample density.

    A staggered hexagonal scan contributes one sample per rectangular area
    ``x_step * y_step``. The returned square spacing gives each resampled
    pixel the same physical area.

    Parameters
    ----------
    x_step : float
        Source scan step along each staggered row.
    y_step : float
        Source scan step between neighboring rows.

    Returns
    -------
    float
        Isotropic square grid spacing in the source coordinate unit.

    Raises
    ------
    ValueError
        If either source step is not positive.
    """
    x_step = float(x_step)
    y_step = float(y_step)

    if x_step <= 0.0 or y_step <= 0.0:
        raise ValueError("Hexagonal scan steps must be positive.")

    return float(np.sqrt(x_step * y_step))


def make_regular_grid(xy, dx_out=None, dy_out=None, nx=None, ny=None, bounds=None):
    """
    Create a regular Cartesian query grid.

    Explicit spacing values are preserved exactly. The final coordinate may
    stop short of the upper bound when the extent is not an integer multiple
    of the requested spacing.

    Parameters
    ----------
    xy : ndarray
        Source coordinates with shape ``(N, 2)``.
    dx_out : float, optional
        Target grid spacing in the x direction.
    dy_out : float, optional
        Target grid spacing in the y direction.
    nx : int, optional
        Number of target grid points in the x direction.
    ny : int, optional
        Number of target grid points in the y direction.
    bounds : tuple, optional
        Bounds as ``(x_min, x_max, y_min, y_max)``.

    Returns
    -------
    tuple
        ``(x_grid, y_grid, query_points)`` for the regular grid.
    """
    if bounds is None:
        x_min, y_min = np.min(xy, axis=0)
        x_max, y_max = np.max(xy, axis=0)
    else:
        x_min, x_max, y_min, y_max = bounds

    if (nx is None or ny is None) and (dx_out is None or dy_out is None):
        raise ValueError("Provide either (nx, ny) or (dx_out, dy_out).")

    if nx is None or ny is None:
        nx = int(np.floor((x_max - x_min) / dx_out)) + 1
        ny = int(np.floor((y_max - y_min) / dy_out)) + 1
        xg = x_min + np.arange(nx, dtype=float) * float(dx_out)
        yg = y_min + np.arange(ny, dtype=float) * float(dy_out)
    else:
        xg = np.linspace(x_min, x_max, nx, dtype=float)
        yg = np.linspace(y_min, y_max, ny, dtype=float)

    x_grid, y_grid = np.meshgrid(xg, yg, indexing="xy")
    query_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    return x_grid, y_grid, query_points


def _idw_weights(dist, p=2.0, eps=1e-12):
    """
    Compute inverse distance weights.

    Parameters
    ----------
    dist : ndarray
        Distance array with shape ``(M, k)``.
    p : float, optional
        Inverse distance power. Default is 2.0.
    eps : float, optional
        Minimum distance used to avoid division by zero.

    Returns
    -------
    ndarray
        Normalized weights with shape ``(M, k)``.
    """
    weights = 1.0 / np.maximum(dist, eps) ** p
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    return weights / weights_sum


def resample_phase_majority(phase_src, idx):
    """
    Resample phase IDs by nearest-neighbor majority vote.

    Parameters
    ----------
    phase_src : ndarray
        Source phase IDs with shape ``(N,)``.
    idx : ndarray
        Neighbor indices for each target grid point with shape ``(M, k)``.

    Returns
    -------
    ndarray
        Resampled phase IDs with shape ``(M,)``.
    """
    neighbors = phase_src[idx]
    labels = neighbors.astype(np.int64)
    label_min = labels.min()
    labels0 = labels - label_min
    n_labels = labels0.max() + 1
    n_points = labels0.shape[0]

    flat = labels0 + (np.arange(n_points)[:, None] * n_labels)
    counts = np.bincount(flat.ravel(), minlength=n_points * n_labels)
    counts = counts.reshape(n_points, n_labels)
    winner = counts.argmax(axis=1) + label_min

    return winner.astype(phase_src.dtype, copy=False)


def resample_scalar_idw(values_src, idx, dist, p=2.0, fill_value=np.nan):
    """
    Resample scalar values by inverse distance weighting.

    Parameters
    ----------
    values_src : ndarray
        Source scalar values with shape ``(N,)``.
    idx : ndarray
        Neighbor indices for each target grid point.
    dist : ndarray
        Neighbor distances for each target grid point.
    p : float, optional
        Inverse distance power. Default is 2.0.
    fill_value : float, optional
        Retained for compatibility with earlier callers. All query points are
        assigned an interpolated value, so this value is unused.

    Returns
    -------
    ndarray
        Resampled scalar values with shape ``(M,)``.
    """
    values = values_src[idx]
    output = np.empty((idx.shape[0],), dtype=float)

    exact = dist[:, 0] == 0.0
    if np.any(exact):
        output[exact] = values[exact, 0]

    remaining = ~exact
    if np.any(remaining):
        weights = _idw_weights(dist[remaining], p=p)
        output[remaining] = np.sum(weights * values[remaining], axis=1)

    return output


def _normalize_quat(quat, eps=1e-15):
    """
    Normalize quaternion data.

    Parameters
    ----------
    quat : ndarray
        Quaternion array with quaternion components on the last axis.
    eps : float, optional
        Minimum norm used to avoid division by zero.

    Returns
    -------
    ndarray
        Normalized quaternion array.
    """
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, eps)


def quat_markley_mean(quats, weights=None):
    """
    Compute a weighted Markley mean quaternion.

    Parameters
    ----------
    quats : ndarray
        Unit quaternion array with shape ``(k, 4)``.
    weights : ndarray, optional
        Optional weights with shape ``(k,)``.

    Returns
    -------
    ndarray
        Mean quaternion with shape ``(4,)``.
    """
    if weights is None:
        weights = np.ones((quats.shape[0],), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    accumulator = (quats * weights[:, None]).T @ quats
    _, eigvecs = np.linalg.eigh(accumulator)
    mean_quat = eigvecs[:, -1]

    if mean_quat[0] < 0.0:
        mean_quat = -mean_quat

    return mean_quat


def resample_quat_knn_markley(quat_src, idx, dist, p=2.0):
    """
    Resample quaternion data by k-nearest neighbors and Markley averaging.

    Parameters
    ----------
    quat_src : ndarray
        Source quaternions with shape ``(N, 4)``.
    idx : ndarray
        Neighbor indices for each target grid point.
    dist : ndarray
        Neighbor distances for each target grid point.
    p : float, optional
        Inverse distance power. Default is 2.0.

    Returns
    -------
    ndarray
        Resampled quaternions with shape ``(M, 4)``.
    """
    quat_src = _normalize_quat(np.asarray(quat_src, dtype=float))
    idx = np.asarray(idx)
    dist = np.asarray(dist)

    if idx.ndim == 1:
        idx = idx[:, None]
    if dist.ndim == 1:
        dist = dist[:, None]

    output = np.empty((idx.shape[0], 4), dtype=float)

    exact = dist[:, 0] == 0.0
    if np.any(exact):
        output[exact] = quat_src[idx[exact, 0]]

    remaining = ~exact
    if np.any(remaining):
        weights = _idw_weights(dist[remaining], p=p)
        neighbors = quat_src[idx[remaining]]

        reference = neighbors[:, :1, :]
        signs = np.sign(np.sum(neighbors * reference, axis=2, keepdims=True))
        signs[signs == 0.0] = 1.0
        aligned = neighbors * signs

        # Vectorised Markley mean: build one (4,4) weighted outer-product
        # accumulator per query point, then extract the leading eigenvector
        # of all M matrices in a single batched eigh call instead of a loop.
        # Sign is canonicalised so the w-component (index 0) is non-negative
        # (q and -q encode the same rotation, but a consistent sign avoids
        # downstream issues in misorientation calculations).
        W = weights[:, :, None] * aligned
        accumulators = np.einsum('mki,mkj->mij', W, aligned)
        _, eigvecs = np.linalg.eigh(accumulators)
        averaged = eigvecs[:, :, -1].copy()
        averaged[averaged[:, 0] < 0.0] *= -1

        output[remaining] = _normalize_quat(averaged)

    return output


def resample_ebsd_to_rect_grid(
    xy,
    phase,
    quat,
    iq,
    dx_out=None,
    dy_out=None,
    nx=None,
    ny=None,
    bounds=None,
    k_phase=1,
    k_iq=1,
    k_quat=1,
    p_iq=2.0,
    p_quat=2.0,
):
    """
    Resample staggered EBSD data to a regular rectangular grid.

    Parameters
    ----------
    xy : ndarray
        Source coordinates with shape ``(N, 2)``.
    phase : ndarray
        Source phase IDs with shape ``(N,)``.
    quat : ndarray
        Source quaternions with shape ``(N, 4)``.
    iq : ndarray
        Source scalar image quality or confidence values with shape ``(N,)``.
    dx_out : float, optional
        Target grid spacing in the x direction.
    dy_out : float, optional
        Target grid spacing in the y direction.
    nx : int, optional
        Number of target grid points in the x direction.
    ny : int, optional
        Number of target grid points in the y direction.
    bounds : tuple, optional
        Bounds as ``(x_min, x_max, y_min, y_max)``.
    k_phase : int, optional
        Number of neighbors for phase resampling. Default is 1.
    k_iq : int, optional
        Number of neighbors for scalar resampling. Default is 1.
    k_quat : int, optional
        Number of neighbors for quaternion resampling. Default is 1.
    p_iq : float, optional
        Inverse distance power for scalar resampling. Default is 2.0.
    p_quat : float, optional
        Inverse distance power for quaternion resampling. Default is 2.0.

    Returns
    -------
    tuple
        ``(x_grid, y_grid, phase_grid, iq_grid, quat_grid)`` on the target grid.
    """
    x_grid, y_grid, query_points = make_regular_grid(
        xy, dx_out=dx_out, dy_out=dy_out, nx=nx, ny=ny, bounds=bounds
    )

    tree = cKDTree(xy)

    dist_phase, idx_phase = tree.query(query_points, k=k_phase, workers=-1)
    if k_phase == 1:
        phase_grid = phase[idx_phase]
    else:
        phase_grid = resample_phase_majority(phase, idx_phase)

    dist_iq, idx_iq = tree.query(query_points, k=k_iq, workers=-1)
    if k_iq == 1:
        iq_grid = iq[idx_iq].astype(float)
    else:
        iq_grid = resample_scalar_idw(iq.astype(float), idx_iq, dist_iq, p=p_iq)

    dist_quat, idx_quat = tree.query(query_points, k=k_quat, workers=-1)
    if k_quat == 1:
        quat_grid = _normalize_quat(quat[idx_quat].astype(float))
    else:
        quat_grid = resample_quat_knn_markley(quat, idx_quat, dist_quat, p=p_quat)

    ny_grid, nx_grid = x_grid.shape
    return (
        x_grid,
        y_grid,
        phase_grid.reshape(ny_grid, nx_grid),
        iq_grid.reshape(ny_grid, nx_grid),
        quat_grid.reshape(ny_grid, nx_grid, 4),
    )
