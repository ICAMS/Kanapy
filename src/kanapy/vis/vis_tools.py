
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union, Mapping
try:
    import pyvista as pv
except ImportError as e:
    raise ImportError(
        "Optional dependency 'pyvista' is required for advanced visualization tools. "
        "Install via: pip install kanapy[vis]"
    ) from e


def plot_ori(
    initial: Mapping[str, Any],     # {'O','t_all','n','phase','pole_vector'}
    regridded: Mapping[str, Any],   # same keys
    *,
    figsize: Tuple[float, float] = (12.0, 8.0),
    show: bool = True,
) -> plt.Figure:
    """
    Plot orientation diagnostics (PF/PDF) for initial vs regridded state.

    Parameters
    ----------
    initial : mapping
        Dictionary with keys:
        - "t_all"        : symmetrized poles projected by orientations (O.inv().outer(t))
        - "n"            : number of orientations (int)
        - "phase"        : orix Phase object (used by stereographic plotting)
        - "pole_vector"  : tuple[int,int,int] e.g. (1,0,0)
    regridded : mapping
        Same structure as `initial`.
    figsize : tuple of float, optional
        Figure size passed to Matplotlib.
    show : bool, optional
        If True, displays the figure. If False, closes it (headless-friendly).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure (PF/PDF panels).
    """
    # -----------------------------
    # Validate required keys
    # -----------------------------
    required = ("t_all", "n", "phase", "pole_vector")
    for name, d in (("initial", initial), ("regridded", regridded)):
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"{name} is missing required keys: {missing}")

    t0_all = initial["t_all"]
    tR_all = regridded["t_all"]

    n0 = int(initial["n"])
    nR = int(regridded["n"])

    phase0 = initial["phase"]
    pole0 = tuple(initial["pole_vector"])
    poleR = tuple(regridded["pole_vector"])

    if pole0 != poleR:
        raise ValueError(f"pole_vector mismatch: initial={pole0}, regridded={poleR}")

    # Nice label like <100>
    hkl_str = f"{pole0[0]}{pole0[1]}{pole0[2]}"

    # -----------------------------
    # Size/alpha scaling (stable)
    # -----------------------------
    def size_alpha(n: int) -> Tuple[float, float]:
        scf = 1.0 / np.sqrt(max(int(n), 1))
        size = float(np.clip(250.0 * scf, 0.25, 25.0))
        alpha = float(np.clip(4.0 * scf, 0.05, 0.50))
        return size, alpha

    size0, alpha0 = size_alpha(n0)
    sizeR, alphaR = size_alpha(nR)

    # -----------------------------
    # Layout: 2 rows x 2 cols
    # -----------------------------
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig, hspace=0.30, wspace=0.10)

    # Row 0: PF
    ax_pf_i   = fig.add_subplot(gs[0, 0], projection="stereographic")
    ax_pf_r   = fig.add_subplot(gs[0, 1], projection="stereographic")

    # Row 1: PDF
    ax_pdf_i  = fig.add_subplot(gs[1, 0], projection="stereographic")
    ax_pdf_r  = fig.add_subplot(gs[1, 1], projection="stereographic")

    # -----------------------------
    # PF: scatter poles
    # -----------------------------
    ax_pf_i.scatter(t0_all, s=size0, alpha=alpha0)
    ax_pf_i.set_labels("X", "Y", None)
    ax_pf_i.set_title(rf"{phase0.name} PF $\langle{hkl_str}\rangle$ (initial)")

    ax_pf_r.scatter(tR_all, s=sizeR, alpha=alphaR)
    ax_pf_r.set_labels("X", "Y", None)
    ax_pf_r.set_title(rf"{phase0.name} PF $\langle{hkl_str}\rangle$ (regridded)")

    # -----------------------------
    # PDF: density function
    # -----------------------------
    ax_pdf_i.pole_density_function(t0_all)
    ax_pdf_i.set_labels("X", "Y", None)
    ax_pdf_i.set_title(rf"{phase0.name} PDF $\langle{hkl_str}\rangle$ (initial)")

    ax_pdf_r.pole_density_function(tR_all)
    ax_pdf_r.set_labels("X", "Y", None)
    ax_pdf_r.set_title(rf"{phase0.name} PDF $\langle{hkl_str}\rangle$ (regridded)")

    # Optional: small column headers (cleaner than repeating words in titles)
    #fig.text(0.25, 0.99, "Initial",   ha="center", va="top", fontsize=12)
    #fig.text(0.75, 0.99, "Regridded", ha="center", va="top", fontsize=12)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_snapshots_voxel(
        json_path: Union[str, Path],
        snapshot_ids: Sequence[int] = (0, 1, 5, 10, 20),
        *,
        color_by: ColorMode = "grain_id",
        ipf_rgb_key: str = "IPFcolor_(0 0 1)",
        # --- subplot integration (match compare_two_snapshots_voxel) ---
        plotter: pv.Plotter | None = None,
        row: int = 0,
        col0: int = 0,
        show: bool = True,
        window_size: Tuple[int, int] = (1800, 900),
        show_edges: bool = False,
        lighting: bool = False,
):
    """
    Plot multiple snapshots side-by-side as voxel volumes using PyVista ImageData.

    Notes
    -----
    - Voxels are placed by voxel_index [i,j,k] (1-based).
    - For a NumPy volume vol[i,j,k], VTK ImageData expects flattening with order="F".
    - `grain_id` is rendered via deterministic per-id RGB for strong categorical separation.
    - `ipf_rgb` uses per-voxel RGB stored in JSON.

    Parameters
    ----------
    json_path
        Path to JSON containing data["microstructure"] list of snapshots.
    snapshot_ids
        Snapshot indices to plot (one row).
    color_by
        "grain_id" or "ipf_rgb".
    ipf_rgb_key
        Voxel key containing RGB triplet, e.g. "IPFcolor_(1 0 0)".
    plotter, row, col0
        If plotter is provided, draws into subplot (row, col0+i).
        If plotter is None, creates a new plotter sized (1, len(snapshot_ids)).
    show
        If True, calls plotter.show().
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    micro = data.get("microstructure", None)
    if not isinstance(micro, list) or len(micro) == 0:
        raise ValueError("Expected data['microstructure'] to be a non-empty list of snapshots.")

    def resolve_index(i: int) -> int:
        ii = i if i >= 0 else len(micro) + i
        if ii < 0 or ii >= len(micro):
            raise IndexError(f"Snapshot index {i} resolves to {ii}, out of range [0,{len(micro) - 1}].")
        return ii

    resolved = [resolve_index(i) for i in snapshot_ids]

    # Deterministic per grain_id RGB (stable across runs and snapshots)
    def gid_to_rgb(gid: int) -> np.ndarray:
        rng = np.random.default_rng(int(gid))
        return rng.uniform(0.20, 0.95, size=3).astype(float)

    def build_grid(snap: dict) -> pv.ImageData:
        vox = snap.get("voxels", [])
        if not isinstance(vox, list) or len(vox) == 0:
            raise ValueError("Snapshot has no voxels.")

        ijk = np.asarray([v["voxel_index"] for v in vox], dtype=int)  # 1-based
        Nx, Ny, Nz = ijk.max(axis=0).tolist()

        grid = pv.ImageData()
        grid.dimensions = (Nx + 1, Ny + 1, Nz + 1)
        grid.spacing = snap.get("grid", {}).get("grid_spacing", [1.0, 1.0, 1.0])

        if color_by == "grain_id":
            gid = np.asarray([v["grain_id"] for v in vox], dtype=int)

            vol_gid = np.zeros((Nx, Ny, Nz), dtype=int)
            vol_gid[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1] = gid

            vol_rgb = np.zeros((Nx, Ny, Nz, 3), dtype=float)
            unique_g = np.unique(gid)
            rgb_lut = {int(g): gid_to_rgb(int(g)) for g in unique_g}
            rgb_vals = np.vstack([rgb_lut[int(g)] for g in gid])  # (Nvox,3)
            vol_rgb[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1, :] = rgb_vals

            # IMPORTANT: VTK ordering for vol[i,j,k] is order="F"
            grid.cell_data["grain_id"] = vol_gid.ravel(order="F")
            grid.cell_data["RGB"] = vol_rgb.reshape(-1, 3, order="F")

        elif color_by == "ipf_rgb":
            rgb = np.asarray([v[ipf_rgb_key] for v in vox], dtype=float)
            if rgb.ndim != 2 or rgb.shape[1] != 3:
                raise ValueError(f"Voxel field '{ipf_rgb_key}' must be an RGB triplet per voxel.")

            if np.nanmax(rgb) > 1.0:
                rgb = rgb / 255.0

            vol_rgb = np.zeros((Nx, Ny, Nz, 3), dtype=float)
            vol_rgb[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1, :] = rgb

            grid.cell_data["RGB"] = vol_rgb.reshape(-1, 3, order="F")

        else:
            raise ValueError("color_by must be 'grain_id' or 'ipf_rgb'.")

        return grid

    grids = [build_grid(micro[i]) for i in resolved]
    titles = [f"Snapshot {i} (t={micro[i].get('time', 'NA')})" for i in resolved]

    own_plotter = False
    if plotter is None:
        plotter = pv.Plotter(shape=(1, len(grids)), window_size=window_size)
        row = 0
        col0 = 0
        own_plotter = True

    for j, (grid, title) in enumerate(zip(grids, titles)):
        plotter.subplot(row, col0 + j)
        if color_by == "ipf_rgb":
            plotter.add_text(f"{title}\ncolor_by={color_by} ({ipf_rgb_key})", font_size=10)
        else:
            plotter.add_text(f"{title}\ncolor_by={color_by}", font_size=10)
        plotter.add_mesh(
            grid,
            scalars="RGB",
            rgb=True,
            lighting=lighting,
            show_edges=show_edges,
        )
        plotter.show_axes()

    if own_plotter:
        plotter.link_views()

    if show:
        plotter.show()

    return plotter, resolved

def compare_two_snapshots_voxel(
        json_path: Union[str, Path],
        snap_a: int,
        snap_b: int,
        *,
        color_by: ColorBy = "grain_id",
        prefer_undeformed_for_b: bool = True,
        plotter: pv.Plotter | None = None,
        row: int = 0,
        col0: int = 0,
        show: bool = False,
        window_size: Tuple[int, int] = (1800, 900),
        show_edges: bool = False,
        lighting: bool = False,
):
    """
    Compare two snapshots side-by-side as voxel volumes (ImageData).

    For color_by="grain_id", this renders with a deterministic RGB per grain_id
    (same mapping as plot_snapshots_voxel) to guarantee consistent colors across
    different rows/panels.

    Assumptions
    ----------
    - voxel_index is 1-based [i, j, k]
    - Your JSON voxel ordering is C-order (k fastest): [1,1,1], [1,1,2], ...
      Therefore, we flatten volumes with order="C" to match that convention.

    Returns
    -------
    plotter : pv.Plotter
    (ia, ib) : tuple[int, int]
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    micro = data.get("microstructure", [])
    if not isinstance(micro, list) or len(micro) == 0:
        raise ValueError("Expected data['microstructure'] to be a non-empty list of snapshots.")

    # ----------------------------
    # Define title helper
    # ----------------------------
    def _title_for_snap(idx: int, snap: dict) -> str:
        t = snap.get("time", "NA")
        status = str(snap.get("grid", {}).get("status", "")).strip().lower()

        if status == "undeformed":
            return f"Snapshot {idx} (t={t}, regridded)"
        return f"Snapshot {idx} (t={t})"

    # ----------------------------
    # Resolve index helper
    # ----------------------------
    def resolve_index(i: int) -> int:
        ii = i if i >= 0 else len(micro) + i
        if ii < 0 or ii >= len(micro):
            raise IndexError(f"Snapshot index {i} resolves to {ii}, out of range [0,{len(micro) - 1}].")
        return ii

    ia = resolve_index(snap_a)
    ib_raw = resolve_index(snap_b)

    # ----------------------------
    # Ensure B is the UNDEFORMED snapshot at same time (regridded)
    # ----------------------------
    def is_undeformed(snap: Mapping) -> bool:
        st = str(snap.get("grid", {}).get("status", "")).strip().lower()
        return st == "undeformed"

    if prefer_undeformed_for_b:
        tB = micro[ib_raw].get("time", None)
        if tB is not None:
            candidates = [
                i for i, s in enumerate(micro)
                if s.get("time", None) == tB and is_undeformed(s)
            ]
            ib = candidates[-1] if candidates else ib_raw
        else:
            ib = ib_raw
    else:
        ib = ib_raw

    snapA = micro[ia]
    snapB = micro[ib]

    # ----------------------------
    # Deterministic per-grain RGB (same as plot_snapshots_voxel)
    # ----------------------------
    def gid_to_rgb(gid: int) -> np.ndarray:
        rng = np.random.default_rng(int(gid))
        return rng.uniform(0.20, 0.95, size=3).astype(float)

    # ----------------------------
    # Scalar extraction
    # ----------------------------
    def extract_scalar(vox: list[dict]) -> np.ndarray:
        if color_by == "grain_id":
            return np.asarray([v["grain_id"] for v in vox], dtype=np.int64)

        if color_by == "voxel_id":
            if "voxel_id" not in vox[0]:
                raise KeyError("color_by='voxel_id' requested but voxel_id not present in this snapshot.")
            return np.asarray([v["voxel_id"] for v in vox], dtype=np.int64)

        if color_by == "euler_phi1":
            return np.asarray([v["orientation"][0] for v in vox], dtype=float)

        raise ValueError(f"Unsupported color_by={color_by!r}")

    # ----------------------------
    # Build grid
    # ----------------------------
    def build_grid(snap: dict, field_name: str) -> pv.ImageData:
        vox = snap.get("voxels", [])
        if not isinstance(vox, list) or len(vox) == 0:
            raise ValueError("Snapshot has no voxels.")

        ijk = np.asarray([v["voxel_index"] for v in vox], dtype=np.int64)  # 1-based [i,j,k]
        vals = extract_scalar(vox)

        Nx, Ny, Nz = ijk.max(axis=0).tolist()

        grid = pv.ImageData()
        grid.dimensions = (Nx + 1, Ny + 1, Nz + 1)
        grid.spacing = snap.get("grid", {}).get("grid_spacing", [1.0, 1.0, 1.0])

        # Always keep the scalar field (useful for debugging)
        vol = np.zeros((Nx, Ny, Nz), dtype=vals.dtype)
        vol[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1] = vals
        grid.cell_data[field_name] = vol.ravel(order="F")

        # If grain_id: also build RGB field (same as plot_snapshots_voxel)
        if color_by == "grain_id":
            vol_rgb = np.zeros((Nx, Ny, Nz, 3), dtype=float)
            unique_g = np.unique(vals)
            rgb_lut = {int(g): gid_to_rgb(int(g)) for g in unique_g}
            rgb_vals = np.vstack([rgb_lut[int(g)] for g in vals])  # (Nvox,3)
            vol_rgb[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1, :] = rgb_vals
            grid.cell_data["RGB"] = vol_rgb.reshape(-1, 3, order="F")

        return grid

    field_name = str(color_by)
    gA = build_grid(snapA, field_name)
    gB = build_grid(snapB, field_name)

    # ----------------------------
    # Plotter integration
    # ----------------------------
    own_plotter = False
    if plotter is None:
        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)
        row = 0
        col0 = 0
        own_plotter = True

    titleA = _title_for_snap(ia, snapA)
    titleB = _title_for_snap(ib, snapB)

    for j, (grid, title) in enumerate(((gA, titleA), (gB, titleB))):
        plotter.subplot(row, col0 + j)
        plotter.add_text(f"{title}\ncolor_by={color_by}", font_size=10)

        if color_by == "grain_id":
            # Use direct RGB -> identical look to plot_snapshots_voxel(..., color_by="grain_id")
            plotter.add_mesh(
                grid,
                scalars="RGB",
                rgb=True,
                lighting=lighting,
                show_edges=show_edges,
            )
        else:
            # Scalar rendering for continuous or huge discrete ranges
            vals = grid.cell_data[field_name]
            if color_by == "voxel_id":
                # avoid meaningless discrete colormap explosion
                clim = (float(vals.min()), float(vals.max()))
                cmap = "viridis"
            else:
                clim = (float(vals.min()), float(vals.max()))
                cmap = "viridis"

            plotter.add_mesh(
                grid,
                scalars=field_name,
                cmap=cmap,
                clim=clim,
                lighting=lighting,
                show_edges=show_edges,
            )

        plotter.show_axes()

    if own_plotter:
        plotter.link_views()

    if show:
        plotter.show()

    return plotter, (ia, ib)
