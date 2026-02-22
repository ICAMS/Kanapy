import numpy as np
from scipy.spatial import cKDTree

def make_regular_grid(xy, dx_out=None, dy_out=None, nx=None, ny=None, bounds=None):
    """
    Create a regular Cartesian grid and return:
      Xg, Yg : (ny, nx) meshgrid
      pts_g  : (ny*nx, 2) flattened query points
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

    xg = np.linspace(x_min, x_max, nx, dtype=float)
    yg = np.linspace(y_min, y_max, ny, dtype=float)
    Xg, Yg = np.meshgrid(xg, yg, indexing="xy")
    pts_g = np.column_stack([Xg.ravel(), Yg.ravel()])
    return Xg, Yg, pts_g


def _idw_weights(dist, p=2.0, eps=1e-12):
    # dist: (M, k)
    # If a query point coincides with a source, handle separately outside.
    w = 1.0 / np.maximum(dist, eps) ** p
    w_sum = np.sum(w, axis=1, keepdims=True)
    return w / w_sum


def resample_phase_majority(phase_src, idx):
    """
    phase_src: (N,)
    idx: (M, k) neighbor indices for each grid point
    returns (M,) int
    """
    neigh = phase_src[idx]  # (M, k)
    # Majority vote without Python loops:
    # For general labels, use per-row bincount via offset trick.
    labels = neigh.astype(np.int64)
    lab_min = labels.min()
    labels0 = labels - lab_min
    L = labels0.max() + 1
    M, k = labels0.shape
    # Build counts matrix (M, L) via flat bincount
    flat = labels0 + (np.arange(M)[:, None] * L)
    counts = np.bincount(flat.ravel(), minlength=M * L).reshape(M, L)
    winner = counts.argmax(axis=1) + lab_min
    return winner.astype(phase_src.dtype, copy=False)


def resample_scalar_idw(values_src, idx, dist, p=2.0, fill_value=np.nan):
    """
    IDW resampling for scalars (IQ etc.)
    values_src: (N,)
    idx/dist: (M, k)
    """
    v = values_src[idx]  # (M, k)
    out = np.empty((idx.shape[0],), dtype=float)

    # If any dist == 0, take the exact source value (avoid inf weights)
    zero = dist[:, 0] == 0.0
    if np.any(zero):
        out[zero] = v[zero, 0]

    nonzero = ~zero
    if np.any(nonzero):
        w = _idw_weights(dist[nonzero], p=p)
        out[nonzero] = np.sum(w * v[nonzero], axis=1)

    # Optional: could mask points too far away (outside convex hull)
    # Here: leave as computed; user can apply max_dist filter if desired.
    return out


def _normalize_quat(q, eps=1e-15):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def quat_markley_mean(quats, weights=None):
    """
    Markley mean for unit quaternions.
    quats: (k, 4) array, assumed roughly aligned in sign already.
    weights: (k,) optional.
    returns (4,)
    """
    Q = quats
    if weights is None:
        w = np.ones((Q.shape[0],), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    # Accumulate symmetric 4x4 matrix
    A = (Q * w[:, None]).T @ Q  # 4x4
    # Largest eigenvector
    evals, evecs = np.linalg.eigh(A)
    q_mean = evecs[:, np.argmax(evals)]
    # Ensure deterministic sign (optional): force scalar part >= 0
    if q_mean[0] < 0:
        q_mean = -q_mean
    return q_mean


def resample_quat_knn_markley(quat_src, idx, dist, p=2.0):
    """
    Quaternion resampling using k-NN, IDW weights, and Markley mean.
    quat_src: (N,4)
    idx/dist: (M,k)
    returns (M,4)
    """
    Qs = _normalize_quat(quat_src.astype(float, copy=False))
    M, k = idx.shape
    out = np.empty((M, 4), dtype=float)

    # Exact hits: take nearest neighbor quaternion
    zero = dist[:, 0] == 0.0
    if np.any(zero):
        out[zero] = Qs[idx[zero, 0]]

    nonzero = ~zero
    if np.any(nonzero):
        w = _idw_weights(dist[nonzero], p=p)  # (M', k)
        neigh = Qs[idx[nonzero]]             # (M', k, 4)

        # Sign alignment: align all neighbors to the first neighbor
        ref = neigh[:, :1, :]                # (M',1,4)
        sgn = np.sign(np.sum(neigh * ref, axis=2, keepdims=True))  # (M',k,1)
        sgn[sgn == 0] = 1.0
        neigh_aligned = neigh * sgn

        # Markley mean per grid point: small loop over M' is OK (k is small)
        # If M is huge (multi-million), consider numba or chunking.
        out_non = np.empty((neigh_aligned.shape[0], 4), dtype=float)
        for i in range(neigh_aligned.shape[0]):
            out_non[i] = quat_markley_mean(neigh_aligned[i], weights=w[i])

        out[nonzero] = _normalize_quat(out_non)

    return out


def resample_ebsd_to_rect_grid(
    xy, phase, quat, iq,
    dx_out=None, dy_out=None, nx=None, ny=None, bounds=None,
    k_phase=7, k_iq=7, k_quat=9,
    p_iq=2.0, p_quat=2.0
):
    """
    Returns:
      Xg, Yg: (ny, nx)
      phase_g: (ny, nx)
      iq_g: (ny, nx)
      quat_g: (ny, nx, 4)
    """
    Xg, Yg, pts_g = make_regular_grid(xy, dx_out=dx_out, dy_out=dy_out, nx=nx, ny=ny, bounds=bounds)

    tree = cKDTree(xy)

    # Phase: majority vote k-NN
    dist_p, idx_p = tree.query(pts_g, k=k_phase, workers=-1)
    if k_phase == 1:
        phase_g = phase[idx_p]
    else:
        phase_g = resample_phase_majority(phase, idx_p)

    # IQ: IDW k-NN
    dist_i, idx_i = tree.query(pts_g, k=k_iq, workers=-1)
    if k_iq == 1:
        iq_g = iq[idx_i].astype(float)
    else:
        iq_g = resample_scalar_idw(iq.astype(float), idx_i, dist_i, p=p_iq)

    # Quat: Markley mean with IDW weights
    dist_q, idx_q = tree.query(pts_g, k=k_quat, workers=-1)
    if k_quat == 1:
        quat_g = _normalize_quat(quat[idx_q].astype(float))
    else:
        quat_g = resample_quat_knn_markley(quat, idx_q, dist_q, p=p_quat)

    ny_, nx_ = Xg.shape
    return (
        Xg, Yg,
        phase_g.reshape(ny_, nx_),
        iq_g.reshape(ny_, nx_),
        quat_g.reshape(ny_, nx_, 4),
    )