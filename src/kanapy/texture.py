"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""

import copy
import logging
import warnings
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from kanapy.core import get_grain_geom
from kanapy.ebsd_io import EBSDDataLoader, LoadedEBSDData
from kanapy.ebsd_hex_grid import resample_ebsd_to_rect_grid
from scipy.stats import lognorm, vonmises
from scipy.spatial import ConvexHull, KDTree
from scipy.special import legendre, beta
from scipy.optimize import fminbound, linear_sum_assignment
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import connected_components as connected_components_csgraph
from skimage.segmentation import mark_boundaries
from orix import io
from orix import plot as ox_plot
from orix.quaternion import Orientation, symmetry
from orix.quaternion.symmetry import Symmetry
from orix.sampling import get_sample_fundamental
from orix.vector import Miller
from abc import ABC
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Any, Sequence, Union, Dict, Tuple, List, Optional


MIN_GRAIN_SIZE_PIXELS = 10.0
"""
Minimum number of pixels for a grain to be considered valid.
"""

DEFAULT_MAX_MISORIENTATION_DEGREES = 5.0
"""
Default maximum misorientation angle in degrees for grain segmentation.
"""

DEFAULT_SIMILAR_ORIENTATION_MERGE_DEGREES = DEFAULT_MAX_MISORIENTATION_DEGREES
"""
Default maximum orientation difference for merging neighboring graph nodes.
"""

DEFAULT_MIN_VOLUME_FRACTION = 0.03
"""
Default minimum phase volume fraction for EBSD analysis.
"""

DEFAULT_CONNECTIVITY = 4
"""
Default pixel connectivity for graph construction.
"""

DEFAULT_BOUNDARY_ARTIFACT_SIZE_FACTOR = 2.0
"""
Default size factor for boundary artifact candidates.
"""

DEFAULT_BOUNDARY_ARTIFACT_FRACTION_MIN = 0.85
"""
Default minimum boundary pixel fraction for artifact candidates.
"""

CONFIDENCE_INDEX_THRESHOLD = 0.1
BAD_PIXEL_MAX_ITERATIONS_MULTIPLIER = 2

CONVEX_HULL_MIN_POINTS = 3
"""
Minimum number of points required to compute a convex hull.
"""


def get_distinct_colormap(N: int, cmap: str = 'prism') -> Any:
    """
    Generate N visually distinct colors as an RGB colormap.

    Parameters:
    - N: int, number of colors
    - seed: optional int, random seed for reproducibility

    Returns:
    - cmap: list of N RGB tuples in [0, 1]
    """
    colors = plt.get_cmap(cmap, N)
    col_ = [colors(i)[:3] for i in range(N)]
    return col_


def neighbors(r: int, c: int, connectivity: int = 4) -> list[tuple[int, int]]:
    """
    Return the neighboring coordinates of a cell in a 2D grid.

    Parameters
    ----------
    r : int
        Row index of the cell.
    c : int
        Column index of the cell.
    connectivity : int, optional
        Type of connectivity. Options are:
        - 1 or 4 : 4-connected neighbors (up, down, left, right)
        - 8      : 8-connected neighbors (includes diagonals)
        Default is 4.

    Returns
    -------
    neighbors : list of tuple
        List of (row, column) tuples representing neighboring cells.

    Examples
    --------
    >>> neighbors(2, 3, connectivity=4)
    [(3, 3), (1, 3), (2, 4), (2, 2)]

    >>> neighbors(2, 3)
    [(3, 2), (3, 3), (3, 4), (2, 2), (2, 4), (1, 2), (1, 3), (1, 4)]
    """
    if connectivity == 1 or connectivity == 4:
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    else:
        return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                if not (i == 0 and j == 0)]


def mean_orientation_data(pixel_orientations: Any, sym: Any) -> Any:
    """
    Return a normalized quaternion mean after symmetry equivalent alignment.

    Parameters
    ----------
    pixel_orientations : array_like
        Quaternion orientation data with quaternion components on the last axis.
    sym : Symmetry
        Crystal symmetry used to align symmetry equivalent orientations.

    Returns
    -------
    ndarray
        Normalized mean quaternion.
    """
    quat = np.asarray(pixel_orientations, dtype=float)
    quat = quat.reshape((-1, quat.shape[-1]))

    # normalize each pixel quaternion independently
    # [a, b, c, d] becomes [a/norm, b/norm, c/norm, d/norm]
    quat = quat / np.linalg.norm(quat, axis=1)[:, None]

    # use the first pixel as the reference orientation
    ref = quat[0]

    if getattr(sym, 'contains_inversion', False):
        # use only proper symmetry operators for averaging
        proper = sym.proper_subgroup
    else:
        proper = sym
    # the length of sym_ops is according to the symmetry type
    # for example, cubic symmetry has 24 equivalent proper rotations
    sym_ops = np.asarray(proper.data, dtype=float)

    best_quat = quat.copy()
    best_score = np.abs(best_quat @ ref)

    # choose the symmetry equivalent orientation closest to the reference
    for sym_op in sym_ops:
        w1, x1, y1, z1 = sym_op
        w2 = quat[:, 0]
        x2 = quat[:, 1]
        y2 = quat[:, 2]
        z2 = quat[:, 3]

        candidate = np.column_stack((
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ))
        candidate = candidate / np.linalg.norm(candidate, axis=1)[:, None]
        score = np.abs(candidate @ ref)

        # keep the closest symmetry equivalent for each pixel
        update = score > best_score
        best_quat[update] = candidate[update]
        best_score[update] = score[update]

    # put q and -q on the same branch before forming the mean
    signs = np.sign(best_quat @ ref)
    signs[signs == 0.0] = 1.0
    best_quat = best_quat * signs[:, None]

    # compute the quaternion mean from the dominant eigenvector
    accumulator = best_quat.T @ best_quat
    eigvals, eigvecs = np.linalg.eigh(accumulator)
    mean_quat = eigvecs[:, int(np.argmax(eigvals))]

    # keep the scalar component positive for a stable representation
    if mean_quat[0] < 0.0:
        mean_quat = -mean_quat

    return mean_quat / np.linalg.norm(mean_quat)


def get_proper_symmetry_quaternions(sym: Any) -> Any:
    """
    Return proper symmetry operators as scalar first quaternions.

    Graph misorientation helpers use this compact array for vectorized
    symmetry reduction.

    Parameters
    ----------
    sym : object, numpy.ndarray or dict
        ORIX symmetry, an existing quaternion array, or symmetries keyed by phase ID.

    Returns
    -------
    numpy.ndarray or dict
        Proper symmetry operators, optionally keyed by phase ID.
    """
    if isinstance(sym, dict):
        return {phase: get_proper_symmetry_quaternions(value) for phase, value in sym.items()}
    if isinstance(sym, np.ndarray):
        return np.asarray(sym, dtype=float)

    if getattr(sym, "contains_inversion", False):
        sym = sym.proper_subgroup

    return np.asarray(sym.data, dtype=float)


def _quat_left_mul_batch(q, Q):
    """
    Left multiply one quaternion with a batch of quaternions.

    Parameters
    ----------
    q : numpy.ndarray
        Single scalar first quaternion with shape ``(4,)``.
    Q : numpy.ndarray
        Batch of scalar first quaternions with shape ``(n, 4)``.

    Returns
    -------
    numpy.ndarray
        Quaternion products with shape ``(n, 4)``.
    """
    w1, x1, y1, z1 = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _batch_misorientation_angle(q_a, q_b, sym_ops):
    """
    Return symmetry reduced misorientation angles for quaternion pairs.

    The implementation keeps memory bounded by looping over the first symmetry
    side and vectorizing over the second.

    Parameters
    ----------
    q_a : numpy.ndarray
        First quaternion batch with shape ``(n, 4)``.
    q_b : numpy.ndarray
        Second quaternion batch with shape ``(n, 4)``.
    sym_ops : numpy.ndarray
        Proper symmetry operators with shape ``(n_ops, 4)``.

    Returns
    -------
    numpy.ndarray
        Misorientation angles in radians.
    """
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    sym_ops = np.asarray(sym_ops, dtype=float)

    norms_a = np.linalg.norm(q_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(q_b, axis=1, keepdims=True)
    q_a = q_a / np.where(norms_a > 0, norms_a, 1.0)
    q_b = q_b / np.where(norms_b > 0, norms_b, 1.0)

    w1, x1, y1, z1 = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    w2, x2, y2, z2 = q_a[:, 0], -q_a[:, 1], -q_a[:, 2], -q_a[:, 3]
    raw = np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

    sign = np.array([1.0, -1.0, -1.0, -1.0])
    max_abs_w = np.zeros(len(q_a), dtype=np.float64)
    for sym_left in sym_ops:
        left = _quat_left_mul_batch(sym_left, raw)
        candidate_w = (left * sign) @ sym_ops.T
        np.maximum(max_abs_w, np.abs(candidate_w).max(axis=1), out=max_abs_w)

    np.clip(max_abs_w, 0.0, 1.0, out=max_abs_w)
    return 2.0 * np.arccos(max_abs_w)


def _same_phase(G, node1, node2):
    """
    Check whether two grain nodes belong to the same phase.

    Legacy graphs without phase attributes remain usable when both nodes
    lack the attribute. A tagged node cannot merge with an untagged node.

    Parameters
    ----------
    G : networkx.Graph
        Graph containing the grain nodes.
    node1, node2 : int
        Grain identifiers to compare.

    Returns
    -------
    bool
        Whether both phase identifiers match.
    """
    return G.nodes[node1].get("phase_id") == G.nodes[node2].get("phase_id")


def _compute_all_edge_misorientations(G, sym_ops):
    """
    Compute and cache misorientation values for edges within the same phase.

    Cross phase edges have no comparable angle under the graph's symmetry.
    Any old angle on those edges is removed before merge selection.

    Parameters
    ----------
    G : networkx.Graph
        Microstructure graph with average orientations on nodes.
    sym_ops : numpy.ndarray or dict
        Proper symmetry operators, optionally keyed by phase ID.
    """
    edges = []
    for node_a, node_b in G.edges():
        if _same_phase(G, node_a, node_b):
            edges.append((node_a, node_b))
        else:
            G[node_a][node_b].pop("misorientation", None)
    if not edges:
        return

    if isinstance(sym_ops, dict):
        phase_edges = {}
        for node_a, node_b in edges:
            phase = G.nodes[node_a]["phase_id"]
            phase_edges.setdefault(phase, []).append((node_a, node_b))
        groups = [(pairs, sym_ops[phase]) for phase, pairs in phase_edges.items()]
    else:
        groups = [(edges, sym_ops)]
    for pairs, operators in groups:
        q_a = np.array([G.nodes[node_a]["ori_av"] for node_a, _node_b in pairs])
        q_b = np.array([G.nodes[node_b]["ori_av"] for _node_a, node_b in pairs])
        angles = _batch_misorientation_angle(q_a, q_b, operators)
        for (node_a, node_b), angle in zip(pairs, angles):
            G[node_a][node_b]["misorientation"] = float(angle)


def _refresh_node_misorientation_cache(G, node, sym_ops=None):
    """
    Refresh cached misorientation values for one node's incident edges.

    Merging changes the surviving grain orientation, so cached values around
    that node must be recomputed before later neighbor searches reuse them.

    Parameters
    ----------
    G : networkx.Graph
        Microstructure graph with average orientations on nodes.
    node : int
        Grain node whose incident edge cache values should be refreshed.
    sym_ops : numpy.ndarray or dict, optional
        Proper symmetry operators, optionally keyed by phase ID. If omitted,
        they are built from graph symmetry metadata.
    """
    neighbors_sorted = []
    for neighbor in sorted(G.adj[node]):
        if _same_phase(G, node, neighbor):
            neighbors_sorted.append(neighbor)
        else:
            G[node][neighbor].pop("misorientation", None)
    if not neighbors_sorted:
        return

    if sym_ops is None:
        sym_ops = get_proper_symmetry_quaternions(G.graph["symmetry"])
    if isinstance(sym_ops, dict):
        sym_ops = sym_ops[G.nodes[node]["phase_id"]]

    q_a = np.tile(G.nodes[node]["ori_av"], (len(neighbors_sorted), 1))
    q_b = np.array([G.nodes[neigh]["ori_av"] for neigh in neighbors_sorted])
    angles = _batch_misorientation_angle(q_a, q_b, sym_ops)

    for neigh, angle in zip(neighbors_sorted, angles):
        G[node][neigh]["misorientation"] = float(angle)


def merge_nodes(
    G: nx.Graph,
    node1: int,
    node2: int,
    preserve_target_orientation: bool = False,
) -> None:
    """
    Merge two grain nodes in the microstructure graph.

    The graph cleanup pipeline uses this operation whenever one grain should be
    absorbed into a neighboring grain while keeping pixels, orientations, edges,
    and the label map consistent.

    Combines grain `node1` into grain `node2`, then removes `node1` from
    the graph. This operation updates pixel assignments, recomputes the
    average orientation, recalculates the convex hull, transfers all adjacency
    edges, and updates the label map.

    Parameters
    ----------
    G : nx.Graph
        Microstructure graph with grain nodes. Must contain:
        - Node attributes: 'pixels', 'npix', 'ori_av', optionally 'hull'
        - Graph attributes: 'label_map', 'dx', 'dy'. If 'rotations' and
          'symmetry' are available, the merged average orientation is
          recomputed from the original pixel orientations.
    node1 : int
        Grain ID to be merged (will be removed)
    node2 : int
        Grain ID to merge into (will be updated and retained)
    preserve_target_orientation : bool, optional
        Keep the retained node orientation unchanged. Cleanup passes use this
        when an absorbed small region should not affect the crystallographic
        representative orientation of the larger node.

    Raises
    ------
    ValueError
        If the nodes have different phases or both have zero pixels.

    Algorithm
    ---------
    1. Concatenate pixel lists from both grains
    2. Recompute the average orientation unless target orientation is preserved
    3. Recompute convex hull for merged grain (if sufficient points)
    4. Transfer all edges from node1's neighbors to node2
    5. Remove node1 from graph
    6. Update label_map to remap node1 pixels to node2

    Notes
    -----
    This function modifies the graph in place.

    The convex hull is only recomputed if the merged grain has at least
    CONVEX_HULL_MIN_POINTS (typically 3) pixels. Otherwise, the hull
    attribute is removed. The explicit Kanapy convention uses scaled rows and
    columns. Existing handoffs with origins and no such marker retain their
    physical x and y coordinates.
    """
    if not _same_phase(G, node1, node2):
        raise ValueError("Cannot merge grains from different phases.")

    has_misorientation_cache = any(
        "misorientation" in data for _u, _v, data in G.edges([node1, node2], data=True)
    )

    p1 = G.nodes[node1]["pixels"]
    p2 = G.nodes[node2]["pixels"]
    merged_pixels = np.concatenate((p1, p2))
    G.nodes[node2]["pixels"] = merged_pixels

    n1 = int(G.nodes[node1]["npix"])
    n2 = int(G.nodes[node2]["npix"])
    ntot = n1 + n2
    if ntot == 0:
        raise ValueError("Both nodes have zero pixels")

    if not preserve_target_orientation:
        if "rotations" in G.graph and "symmetry" in G.graph:
            sym = G.graph["symmetry"]
            if isinstance(sym, dict):
                sym = sym[G.nodes[node2]["phase_id"]]
            G.nodes[node2]["ori_av"] = mean_orientation_data(
                G.graph["rotations"][merged_pixels],
                sym,
            )
        else:
            # keep old graph objects usable when original pixel orientations are missing
            ori1 = np.asarray(G.nodes[node1]["ori_av"], dtype=np.float64)
            ori2 = np.asarray(G.nodes[node2]["ori_av"], dtype=np.float64)
            ori_av = (ori1 * n1 + ori2 * n2) / ntot
            G.nodes[node2]["ori_av"] = ori_av / np.linalg.norm(ori_av)

    G.nodes[node2]["npix"] = ntot

    if "hull" in G.nodes[node2]:
        sh = G.graph["label_map"].shape
        rows, columns = np.unravel_index(G.nodes[node2]["pixels"], sh)
        if G.graph.get("geometry_convention") != "kanapy_row_column" and (
                "origin_x" in G.graph or "origin_y" in G.graph):
            origin_x = float(G.graph.get("origin_x", 0.0))
            origin_y = float(G.graph.get("origin_y", 0.0))
            x = origin_x + columns * float(G.graph["dx"])
            y = origin_y + rows * float(G.graph["dy"])
            pts = np.column_stack((x, y))
        else:
            pts = np.column_stack((rows * G.graph["dx"], columns * G.graph["dy"]))
        if pts.shape[0] >= CONVEX_HULL_MIN_POINTS:
            G.nodes[node2]["hull"] = ConvexHull(pts)
        else:
            G.nodes[node2].pop("hull", None)

    for neigh in sorted(G.adj[node1]):
        if neigh != node2:
            phase_boundary = G[node1][neigh].get("is_phase_boundary")
            G.add_edge(node2, neigh)
            if phase_boundary is not None:
                G[node2][neigh]["is_phase_boundary"] = phase_boundary

    # remove node1 from graph and update label_map
    G.remove_node(node1)
    ix, iy = np.nonzero(G.graph["label_map"] == node1)
    G.graph["label_map"][ix, iy] = node2

    if has_misorientation_cache:
        _refresh_node_misorientation_cache(G, node2)


def find_largest_neighbor(G: nx.Graph, node: int) -> int:
    """
    Find the neighboring grain with the most pixels.

    Small or geometrically invalid grains are merged into this neighbor during
    graph cleanup, using the lowest node id as a deterministic tie breaker.

    Searches grains of the same phase adjacent to the given node and returns the ID
    of the neighbor with the highest pixel count. This is commonly used
    when merging small grains into a larger neighboring grain.

    Parameters
    ----------
    G : nx.Graph
        Microstructure graph with grain nodes
    node : int
        Grain ID whose largest neighbor should be found

    Returns
    -------
    int
        Grain ID of the largest neighboring grain

    Raises
    ------
    ValueError
        If the grain has no neighbors (isolated grain)
    ValueError
        If the graph has a self loop (corrupted graph structure)
    """
    neighbor_rows = []
    for neigh in sorted(G.adj[node]):
        if not _same_phase(G, node, neigh):
            continue
        neighbor_rows.append((int(G.nodes[neigh]["npix"]), int(neigh)))

    if not neighbor_rows:
        raise ValueError(f"Grain {node} has no neighbors in the same phase.")

    _size_ln, num_ln = max(neighbor_rows, key=lambda row: (row[0], -row[1]))
    if num_ln == node:
        raise ValueError(f"Corrupted graph with circular edges: {node, num_ln}.")

    return num_ln


def find_sim_neighbor(G: nx.Graph, nn: Any, sym_ops: Any = None) -> Any:
    """
    Find the neighboring node most similar in orientation to the given node.

    This function computes the misorientation angles between the given node and
    its neighbors in the same phase, or reads cached edge values when available.

    Parameters
    ----------
    G : networkx.Graph
        Graph object where nodes have an 'ori_av' attribute representing average orientation.
        The graph also contains a 'symmetry' key in G.graph used for orientation calculations.
    nn : int
        Node ID for which the most similar neighbor is to be found.
    sym_ops : numpy.ndarray or dict, optional
        Proper symmetry operators, optionally keyed by phase ID. Pass this
        when calling in a loop to avoid rebuilding symmetry data.

    Returns
    -------
    tuple
        A tuple (neighbor_id, angle) where:
        - neighbor_id : int, ID of the neighbor with smallest misorientation
        - angle : float, misorientation angle with the given node

    Raises
    ------
    ValueError
        If the node has no neighbors.
    """
    neighbors_sorted = [neigh for neigh in sorted(G.adj[nn]) if _same_phase(G, nn, neigh)]
    if not neighbors_sorted:
        raise ValueError(f"Grain {nn} has no neighbors in the same phase.")

    if all("misorientation" in G[nn][neigh] for neigh in neighbors_sorted):
        angles = [G[nn][neigh]["misorientation"] for neigh in neighbors_sorted]
    else:
        if sym_ops is None:
            sym_ops = get_proper_symmetry_quaternions(G.graph["symmetry"])
        if isinstance(sym_ops, dict):
            sym_ops = sym_ops[G.nodes[nn]["phase_id"]]

        q_a = np.tile(G.nodes[nn]["ori_av"], (len(neighbors_sorted), 1))
        q_b = np.array([G.nodes[neigh]["ori_av"] for neigh in neighbors_sorted])
        angles = _batch_misorientation_angle(q_a, q_b, sym_ops)

    rows = []
    for neigh, angle in zip(neighbors_sorted, angles):
        rows.append((float(angle), int(neigh)))

    angle, neighbor = min(rows, key=lambda row: (row[0], row[1]))
    return neighbor, angle


def get_node_boundary_stats(G: nx.Graph, node_id: Any, connectivity: int = 4) -> dict[str, Any]:
    """
    Compute boundary diagnostics for one graph node.

    Parameters
    ----------
    G : networkx.Graph
        Graph built from labeled pixels. The graph must store ``label_map``.
    node_id : int
        Node ID to inspect.
    connectivity : int, optional
        Connectivity criterion for neighbors (4 or 8). Default is 4.

    Returns
    -------
    dict
        Boundary statistics for the selected node.
    """
    label_map = G.graph['label_map']
    rows, cols = label_map.shape
    pix = G.nodes[node_id]['pixels']
    rr, cc = np.unravel_index(pix, label_map.shape)

    # use coordinate pairs to check local neighborhood relations
    pixel_set = set(zip(rr.tolist(), cc.tolist()))
    boundary_pixels = 0
    touches_map_edge = False
    neighbor_labels = set()

    # a pixel is on the boundary if any checked neighbor has another label
    for r0, c0 in pixel_set:
        is_boundary = False
        for r1, c1 in neighbors(r0, c0, connectivity=connectivity):
            if r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols:
                touches_map_edge = True
                is_boundary = True
                continue

            other_label = int(label_map[r1, c1])
            if other_label != node_id:
                is_boundary = True
                if other_label > 0:
                    neighbor_labels.add(other_label)

        if is_boundary:
            boundary_pixels += 1

    # bounding box fill helps identify thin or fragmented regions
    bbox_height = int(rr.max() - rr.min() + 1)
    bbox_width = int(cc.max() - cc.min() + 1)

    return {
        'boundary_pixels': int(boundary_pixels),
        'boundary_fraction': float(boundary_pixels / len(pixel_set)),
        'touches_map_edge': bool(touches_map_edge),
        'neighbor_count': int(len(neighbor_labels)),
        'neighbor_labels': sorted(int(lbl) for lbl in neighbor_labels),
        'bbox_height': bbox_height,
        'bbox_width': bbox_width,
        'bbox_fill_fraction': float(len(pixel_set) / (bbox_height * bbox_width)),
    }


def merge_boundary_artifact_nodes(
    G: nx.Graph,
    gs_min: float,
    connectivity: int = 4,
    npix_factor: float = 2.0,
    boundary_fraction_min: float = 0.85,
    store_merge_pixels: bool = False,
) -> list[dict]:
    """
    Merge small nodes with a high boundary pixel fraction.

    Each selected node is absorbed into its crystallographically closest
    neighbor before the remaining graph cleanup steps. A record is retained
    for every performed merge.

    Parameters
    ----------
    G : nx.Graph
        Initial microstructure graph.
    gs_min : float
        Minimum grain size threshold from the EBSD workflow.
    connectivity : int, optional
        Connectivity criterion for neighbors. Default is 4.
    npix_factor : float, optional
        Candidate size threshold as ``max(npix_factor * gs_min, 40)``.
        Default is 2.0.
    boundary_fraction_min : float, optional
        Minimum boundary pixel fraction for a node to be considered.
        Default is 0.85.
    store_merge_pixels : bool, optional
        If True, copy source and target pixel arrays before each merge.
        Default is False.

    Returns
    -------
    list of dict
        Records for all merges performed in this pass.
    """
    debug_records = []
    npix_limit = max(int(npix_factor * gs_min), 40)
    sym_ops = get_proper_symmetry_quaternions(G.graph["symmetry"])

    changed = True
    while changed:
        changed = False
        _compute_all_edge_misorientations(G, sym_ops)
        candidate_rows = []

        for node_id in sorted(G.nodes):
            npix = int(G.nodes[node_id]["npix"])
            if npix > npix_limit:
                continue
            if G.degree[node_id] <= 0:
                continue

            stats = get_node_boundary_stats(G, node_id, connectivity=connectivity)
            if stats["boundary_fraction"] < boundary_fraction_min:
                continue

            try:
                similar_neighbor, ori_angle = find_sim_neighbor(
                    G, node_id, sym_ops=sym_ops
                )
            except ValueError:
                continue

            candidate_rows.append(
                (
                    -stats["boundary_fraction"],
                    npix,
                    float(ori_angle),
                    int(node_id),
                    stats,
                    int(similar_neighbor),
                )
            )

        candidate_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        for _, _, ori_angle, node_id, stats, similar_neighbor in candidate_rows:
            if node_id not in G.nodes or similar_neighbor not in G.nodes:
                continue

            record = {
                "node": int(node_id),
                "reason": "boundary_artifact_premerge",
                "target": int(similar_neighbor),
                "angle_rad": float(ori_angle),
                "angle_deg": float(np.degrees(ori_angle)),
                "node_npix": int(G.nodes[node_id]["npix"]),
                "target_npix": int(G.nodes[similar_neighbor]["npix"]),
                "boundary_pixels": int(stats["boundary_pixels"]),
                "boundary_fraction": float(stats["boundary_fraction"]),
                "neighbor_count": int(stats["neighbor_count"]),
                "bbox_fill_fraction": float(stats["bbox_fill_fraction"]),
                "npix_limit": int(npix_limit),
                "boundary_fraction_min": float(boundary_fraction_min),
                "will_merge": True,
            }
            source_pixels = np.asarray(G.nodes[node_id]["pixels"], dtype=int)
            if source_pixels.size > 0:
                rows, columns = np.unravel_index(
                    source_pixels,
                    G.graph["label_map"].shape,
                )
                record["source_center"] = [
                    float(np.mean(columns)),
                    float(np.mean(rows)),
                ]

            if store_merge_pixels:
                record["node_pixels"] = source_pixels.copy()
                record["target_pixels_before"] = np.asarray(
                    G.nodes[similar_neighbor]["pixels"],
                    dtype=int,
                ).copy()

            debug_records.append(record)
            merge_nodes(
                G,
                node_id,
                similar_neighbor,
                preserve_target_orientation=True,
            )
            changed = True
            break

    return debug_records


def summarize_labels(
    label_array: np.ndarray,
    rotations: Any,
    sym: Any,
    wanted_labels: Any = None,
    phase_map: Any = None) -> list[tuple[int, dict[str, Any]]]:
    """
    Summarize labeled pixels with average orientation and statistics.

    This function computes, for each label in `label_array` (or a subset of `wanted_labels`):
    - Number of pixels
    - Pixel indices
    - Average orientation vector
    - Standard deviation of orientation

    Parameters
    ----------
    label_array : ndarray of shape (H, W)
        2D array of integer labels.
    rotations : ndarray of shape (H*W, D)
        Orientation data corresponding to each pixel (e.g., emap.rotations.data).
    sym : Symmetry or dict
        Crystal symmetry, or symmetries keyed by phase ID for a shared map.
    wanted_labels : sequence of int, optional
        List of label IDs to summarize. If None, all labels in `label_array` are used.
    phase_map : ndarray, optional
        Phase IDs on the same grid as the labels. Each grain must contain one phase.

    Returns
    -------
    list of tuples
        Each tuple is `(label, info_dict)` where `info_dict` contains:
        - 'npix' : int, number of pixels
        - 'pixels' : array of 1D indices in `label_array.ravel()`
        - 'ori_av' : ndarray of shape (D,), average orientation vector
        - 'ori_std' : ndarray of shape (D,), standard deviation of orientation

    Notes
    -----
    - Computes average orientations using crystal symmetry.
    - The returned list preserves the order of `wanted_labels` if provided,
      otherwise follows sorted unique labels from `label_array`
    - Does not modify input arrays
    """
    lab = label_array.ravel()
    N = lab.size
    rot = rotations  # shape (N, D)

    # Gruppierung: alle Pixel nach Label sortieren
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]

    # Eindeutige Labels + Segmentstarts + Counts
    uniq, starts, counts = np.unique(lab_sorted, return_index=True, return_counts=True)
    ends = starts + counts

    # Rotations in derselben Reihenfolge sortieren
    rot_sorted = rot[order]

    # component wise standard deviation is kept as a simple diagnostic
    sq = rot_sorted * rot_sorted
    sums_sq = np.add.reduceat(sq, starts, axis=0)
    sums = np.add.reduceat(rot_sorted, starts, axis=0)
    means_component = sums / counts[:, None]
    vars_ = sums_sq / counts[:, None] - means_component**2
    stds = np.sqrt(np.maximum(vars_, 0.0))

    # Pixelindizes pro Label (als Listen von 1D-Indizes)
    pixels_list = np.split(order, starts[1:])  # Liste der Segmente in uniq-Reihenfolge

    # Auswahl / Re-Ordering auf gewünschte Labels
    if wanted_labels is None:
        valid = uniq > 0
        labels_out = uniq[valid]
        idx = np.arange(len(uniq))[valid]
    else:
        labels_out = np.asarray(wanted_labels)
        pos = {int(l): i for i, l in enumerate(uniq)}
        idx = np.array([pos[int(l)] for l in labels_out], dtype=int)

    nodes = []
    phases = None if phase_map is None else np.asarray(phase_map).ravel()
    for lbl, i in zip(labels_out, idx):
        pix = pixels_list[i]
        node_symmetry = sym
        if phases is not None:
            phase = int(phases[pix[0]])
            if np.any(phases[pix] != phase):
                raise ValueError("A grain cannot contain pixels from different phases.")
            node_symmetry = sym[phase]
        info = {
            "npix": int(counts[i]),
            "pixels": pix,
            "ori_av": mean_orientation_data(rot[pix], node_symmetry),
            "ori_std": stds[i],
        }
        if phases is not None:
            info["phase_id"] = phase
        nodes.append((int(lbl), info))
    return nodes


def build_graph_from_labeled_pixels(
    label_array: np.ndarray,
    rot: Any,
    sym: Any,
    dx: float,
    dy: float,
    connectivity: int = 4,
    phase_map: Any = None) -> nx.Graph:
    """
    Build a graph representation of a microstructure from labeled pixels.

    Each node in the graph represents a grain (label) and stores pixel indices,
    average orientation, and orientation statistics. Edges connect neighboring grains
    based on pixel adjacency.

    Parameters
    ----------
    label_array : ndarray of shape (H, W)
        2D array of integer labels representing grains.
    rot : ndarray
        Quaternion orientation data for all pixels in ``label_array``.
    sym : Symmetry or dict
        Crystal symmetry, or symmetries keyed by phase ID for a shared map.
    dx : float
        Physical pixel spacing along x, the image column direction.
    dy : float
        Physical pixel spacing along y, the image row direction.
    connectivity : int, optional
        Connectivity criterion for neighbors (4 or 8). Default is 4.
    phase_map : ndarray, optional
        Phase IDs on the label grid. Enables node phase IDs and interface edge flags.

    Returns
    -------
    networkx.Graph
        Graph object with:
        - Nodes representing grains, storing 'pixels', 'ori_av', 'ori_std', 'npix'
        - Edges representing neighboring grains (shared boundaries)
        - Graph attributes: 'label_map', 'symmetry', 'rotations', 'dx', 'dy'

    Notes
    -----
    - Uses `summarize_labels` to extract node properties from labeled pixels
    - Loops over all pixels to add edges between neighboring grains
    - The function preserves the shape and indices of `label_array`
    """
    nodes = summarize_labels(label_array, rot, sym, phase_map=phase_map)
    G = nx.Graph(label_map=label_array.copy(), symmetry=sym, rotations=rot,
                 dx=dx, dy=dy)
    G.add_nodes_from(nodes)

    edges = set()

    left = label_array[:, :-1].ravel()
    right = label_array[:, 1:].ravel()
    mask = left != right
    if mask.any():
        pairs = np.stack((left[mask], right[mask]), axis=1)
        for label_a, label_b in pairs:
            if label_a > 0 and label_b > 0 and label_a != label_b:
                edge = tuple(sorted((int(label_a), int(label_b))))
                edges.add(edge)

    top = label_array[:-1, :].ravel()
    bottom = label_array[1:, :].ravel()
    mask = top != bottom
    if mask.any():
        pairs = np.stack((top[mask], bottom[mask]), axis=1)
        for label_a, label_b in pairs:
            if label_a > 0 and label_b > 0 and label_a != label_b:
                edge = tuple(sorted((int(label_a), int(label_b))))
                edges.add(edge)

    if connectivity == 8:
        top_left = label_array[:-1, :-1].ravel()
        bottom_right = label_array[1:, 1:].ravel()
        mask = top_left != bottom_right
        if mask.any():
            pairs = np.stack((top_left[mask], bottom_right[mask]), axis=1)
            for label_a, label_b in pairs:
                if label_a > 0 and label_b > 0 and label_a != label_b:
                    edge = tuple(sorted((int(label_a), int(label_b))))
                    edges.add(edge)

        top_right = label_array[:-1, 1:].ravel()
        bottom_left = label_array[1:, :-1].ravel()
        mask = top_right != bottom_left
        if mask.any():
            pairs = np.stack((top_right[mask], bottom_left[mask]), axis=1)
            for label_a, label_b in pairs:
                if label_a > 0 and label_b > 0 and label_a != label_b:
                    edge = tuple(sorted((int(label_a), int(label_b))))
                    edges.add(edge)

    G.add_edges_from(sorted(edges))
    if phase_map is not None:
        for node_a, node_b in G.edges:
            G[node_a][node_b]["is_phase_boundary"] = not _same_phase(G, node_a, node_b)
    return G


def _graph_points_to_image(graph, points):
    """
    Convert grain geometry to image column and row coordinates.

    The explicit Kanapy convention stores scaled rows and columns. Existing
    handoffs with origins and no such marker retain physical x and y coordinates.

    Parameters
    ----------
    graph : networkx.Graph
        Graph containing pixel spacing and optional coordinate origins.
    points : array_like
        One point or an array of points in the graph coordinate convention.

    Returns
    -------
    numpy.ndarray
        Image column and row coordinates with the input shape.
    """
    values = np.asarray(points, dtype=float).copy()
    dx = float(graph.graph.get('dx', 1.0))
    dy = float(graph.graph.get('dy', 1.0))
    if graph.graph.get('geometry_convention') != 'kanapy_row_column' and (
            'origin_x' in graph.graph or 'origin_y' in graph.graph):
        values[..., 0] = (values[..., 0] - graph.graph.get('origin_x', 0.0)) / dx
        values[..., 1] = (values[..., 1] - graph.graph.get('origin_y', 0.0)) / dy
    else:
        values = values[..., ::-1] / np.array([dy, dx])
    return values


def visualize_graph(G: nx.Graph, node_size: int = 100, fs: int = 12) -> Any:
    """
    Visualize a microstructure graph using a spring layout

    This function draws the nodes and edges of the graph with labels,
    using a spring layout for positioning. Node color, edge color,
    node size, and font size can be adjusted.

    Parameters
    ----------
    G : networkx.Graph
        Graph object representing the microstructure. Nodes should have meaningful labels.
    node_size : int, optional
        Size of the nodes in the plot. Default is 100.
    fs : int, optional
        Font size for node labels. Default is 12.

    Notes
    -----
    - Uses NetworkX's `spring_layout` for node positioning
    - Uses Matplotlib to render the plot
    - Does not modify the graph
    """
    pos = nx.spring_layout(G, seed=42)  # positioning
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=node_size, font_size=fs)
    plt.show()


def export_graph(G: nx.Graph, filename: Union[str, Path], format: str = "graphml") -> None:
    """
    Export a microstructure graph to a file in the specified format

    This function writes the NetworkX graph to disk in either GraphML or GEXF format.

    Parameters
    ----------
    G : networkx.Graph
        Graph object representing the microstructure.
    filename : str
        Path to the output file.
    format : str, optional
        File format for export. Supported options are 'graphml' and 'gexf'. Default is 'graphml'.

    Notes
    -----
    - Raises a ValueError if an unsupported format is provided
    - Does not modify the graph
    """
    if format == "graphml":
        nx.write_graphml(G, filename)
    elif format == "gexf":
        nx.write_gexf(G, filename)
    else:
        raise ValueError("Only 'graphml' or 'gexf' formats are supported.")


def find_similar_regions(array: np.ndarray, tolerance: float = 0.087, connectivity: int = 1) -> Any:
    """
    Identify connected regions of similar values in a 2D array

    This function labels all connected pixels in the input array whose values
    differ by at most `tolerance`. Connectivity can be 4- or 8-connected.

    Parameters
    ----------
    array : ndarray of shape (H, W)
        2D input array with values to segment.
    tolerance : float, optional
        Maximum allowed difference between connected values. Default is 0.087.
    connectivity : int, optional
        Connectivity criterion for neighbors: 1 for 4-connectivity, 2 for 8-connectivity. Default is 1.

    Returns
    -------
    labeled_array : ndarray of shape (H, W)
        Array where each connected region of similar values has a unique label.
    num_features : int
        Total number of connected regions found.

    Notes
    -----
    - Uses a depth-first search to find connected regions.
    - Does not modify the input array.
    - Assumes `neighbors(i, j, connectivity)` returns valid neighboring indices.
    """

    array = np.asarray(array)
    visited = np.full(array.shape, False, dtype=bool)
    labeled_array = np.zeros(array.shape, dtype=int)
    current_label = 1
    rows, cols = array.shape
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                ref_val = array[r, c]
                stack = [(r, c)]
                while stack:
                    i, j = stack.pop()
                    if (0 <= i < rows) and (0 <= j < cols) and not visited[i, j]:
                        if abs(array[i, j] - ref_val) <= tolerance:
                            visited[i, j] = True
                            labeled_array[i, j] = current_label
                            stack.extend(neighbors(i, j, connectivity))
                current_label += 1
    return labeled_array, current_label - 1



def find_similar_regions_by_misorientation(
        ori_map, phase_mask, sym, tolerance=np.deg2rad(5.0), connectivity=4, phase_map=None):
    """
    Identify connected regions using local pixel to pixel misorientation.

    Parameters
    ----------
    ori_map : ndarray of shape (H, W, 4)
        Quaternion map for the full 2D image.
    phase_mask : ndarray of shape (H, W)
        Boolean mask selecting pixels of the retained phase or phases.
    sym : Symmetry or dict
        Crystal symmetry, or symmetries keyed by phase ID for a shared map.
    tolerance : float, optional
        Maximum allowed local misorientation in radians. Default is five degrees.
    connectivity : int, optional
        Connectivity criterion for neighbors (4 or 8). Default is 4.
    phase_map : ndarray of shape (H, W), optional
        Phase IDs for unified segmentation. Only pixels of the same phase can
        connect, using that phase's symmetry before one global labeling pass.

    Returns
    -------
    labeled_array : ndarray of shape (H, W)
        Label image of connected regions. Pixels outside the phase mask stay 0.
    num_features : int
        Number of connected regions inside the phase mask.
    """
    ori_map = np.asarray(ori_map)
    phase_mask = np.asarray(phase_mask, dtype=bool)

    if ori_map.ndim != 3 or ori_map.shape[2] != 4:
        raise ValueError("ori_map must have shape (rows, cols, 4).")
    if ori_map.shape[:2] != phase_mask.shape:
        raise ValueError("ori_map and phase_mask must have matching spatial dimensions.")
    if phase_map is not None:
        phase_map = np.asarray(phase_map)
        if phase_map.shape != phase_mask.shape:
            raise ValueError("phase_map and phase_mask must have matching spatial dimensions.")

    rows, cols = phase_mask.shape
    sym_ops = get_proper_symmetry_quaternions(sym)
    flat_index = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)

    directions = [
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :]),
    ]
    if connectivity == 8:
        directions += [
            (np.s_[:-1, :-1], np.s_[1:, 1:]),
            (np.s_[:-1, 1:], np.s_[1:, :-1]),
        ]

    source_rows = []
    target_rows = []
    for slice_a, slice_b in directions:
        valid = phase_mask[slice_a] & phase_mask[slice_b]
        if phase_map is not None:
            valid &= phase_map[slice_a] == phase_map[slice_b]
        if not np.any(valid):
            continue

        q_a = ori_map[slice_a][valid]
        q_b = ori_map[slice_b][valid]
        if phase_map is None:
            same_region = _batch_misorientation_angle(q_a, q_b, sym_ops) <= tolerance
        else:
            pair_phases = phase_map[slice_a][valid]
            same_region = np.zeros(len(q_a), dtype=bool)
            for phase in np.unique(pair_phases):
                selected = pair_phases == phase
                same_region[selected] = _batch_misorientation_angle(
                    q_a[selected], q_b[selected], sym_ops[int(phase)],
                ) <= tolerance
        if not np.any(same_region):
            continue

        source_rows.append(flat_index[slice_a][valid][same_region])
        target_rows.append(flat_index[slice_b][valid][same_region])

    pixel_count = rows * cols
    if source_rows:
        source = np.concatenate(source_rows)
        target = np.concatenate(target_rows)
        adjacency = coo_matrix(
            (np.ones(len(source), dtype=np.float32), (source, target)),
            shape=(pixel_count, pixel_count),
        )
        adjacency = adjacency + adjacency.T
    else:
        adjacency = coo_matrix((pixel_count, pixel_count), dtype=np.float32)

    _, labels_flat = connected_components_csgraph(
        adjacency.tocsr(),
        directed=False,
    )

    labeled_array = np.zeros(pixel_count, dtype=int)
    phase_flat = phase_mask.ravel()
    phase_labels = labels_flat[phase_flat]
    unique_components = np.unique(phase_labels)
    remap = np.zeros(int(labels_flat.max()) + 1, dtype=int)
    remap[unique_components] = np.arange(1, len(unique_components) + 1)
    labeled_array[phase_flat] = remap[phase_labels]

    return labeled_array.reshape(rows, cols), len(unique_components)


def calc_error(odf_ref: Any, odf_test: Any, res: float = 10.) -> float:
    """
    Compute the normalized difference between two orientation distribution functions (ODFs)

    This function evaluates the ODFs on a sample grid in fundamental space and
    computes the L1 norm of the difference after normalization.

    Parameters
    ----------
    odf_ref : object
        Reference ODF object. Must have `.orientations.symmetry` and `.evaluate()` method.
    odf_test : object
        Test ODF object to compare. Must have the same symmetry as `odf_ref`.
    res : float, optional
        Resolution for sampling the fundamental space grid. Default is 10.

    Returns
    -------
    float
        L1 error between the normalized ODFs.

    Notes
    -----
    - Raises a RuntimeError if the symmetries of the two ODFs do not match.
    - Uses `get_sample_fundamental` to sample the fundamental zone and `Orientation` for evaluation.
    - Does not modify the input ODFs.
    """
    cs = odf_ref.orientations.symmetry
    if cs != odf_test.orientations.symmetry:
        raise RuntimeError("Symmetries of ODF's do not match.")
    so3g = get_sample_fundamental(resolution=res, point_group=cs)
    so3g = Orientation(so3g.data, symmetry=cs)
    p1 = odf_ref.evaluate(so3g)
    p2 = odf_test.evaluate(so3g)
    err = np.sum(np.abs(p1/np.sum(p1) - p2/np.sum(p2)))
    return err


def calc_orientations(odf: Any, nori: int, res: Any = None) -> Any:
    """
    Generate a set of orientations sampled from an orientation distribution function (ODF)

    This function uses Monte Carlo sampling to generate `nori` orientations according to
    the probabilities defined by the input ODF. Sampling is performed in the fundamental
    zone of the crystal symmetry.

    Parameters
    ----------
    odf : object
        Orientation distribution function object. Must have `.orientations.symmetry` and `.evaluate()` method.
    nori : int
        Number of orientations to generate.
    res : float, optional
        Angular resolution in degrees for sampling the fundamental zone grid.
        If None, a resolution between 2° and 5° is selected based on ODF half-width.

    Returns
    -------
    Orientation
        Orientation object containing `nori` orientations sampled from the ODF.

    Notes
    -----
    - Raises a RuntimeError if the Monte Carlo sampling does not converge after 200 iterations.
    - Logs a warning if the SO3 grid resolution is too coarse for the requested number of orientations.
    - Uses `get_sample_fundamental` to generate the sampling grid and `Orientation` to store results.
    - Does not modify the input ODF.
    """
    oq = np.zeros((nori, 4))
    indstart = 0
    rem = nori
    cs = odf.orientations.symmetry
    hw = odf.halfwidth
    if res is None:
        res = np.clip(np.rad2deg(hw), 2, 5)
    so3g = get_sample_fundamental(resolution=res, point_group=cs)  # generate fine mesh in fund. region; res in degree!
    so3g = Orientation(so3g.data, symmetry=cs)
    if so3g.size < 5*nori:
        logging.warning(f'Resolution of SO3 grid in "calc_orientations" is too coarse: {res}°.\n'
                        f'Only {so3g.size} grid points available for {nori} desired orientations.')
    val = odf.evaluate(so3g)  # value of ODF at each gridpoint
    # do MC sampling of ODF to generate nori orientations
    ctr = 0
    while rem > 0 and ctr < 200:
        rn = np.random.rand(so3g.size)
        vn = val * np.linalg.norm(rn) * nori / np.linalg.norm(val)
        ori = so3g[vn >= rn]
        no = min(ori.size, rem)
        oq[indstart:indstart+no, :] = ori.data[0:no, :]
        indstart += no
        rem -= no
        ctr += 1
    if ctr >= 200:
        raise RuntimeError(f'Monte Carlo algorithm in "calc_orientations" did not converge for resolution {res}.\n'
                           f'Orientations found {indstart} orientations. Target was {nori}.')
    return Orientation(oq, symmetry=cs)


def odf_est(
    ori: Any,
    odf: Any,
    nstep: int = 50,
    step: float = 0.5,
    halfwidth: Any = None,
    verbose: bool = False) -> Any:
    """
    Estimate an ODF from a set of orientations using iterative half-width adjustment

    This function generates a new orientation distribution function (ODF) from
    input orientations by iteratively adjusting the half-width to minimize the
    error relative to a reference ODF.

    Parameters
    ----------
    ori : Orientation
        Input orientations used to estimate the ODF.
    odf : ODF
        Reference orientation distribution function for error comparison.
    nstep : int, optional
        Maximum number of iterations for adjusting the half-width. Default is 50.
    step : float, optional
        Increment step in degrees for half-width adjustment. Default is 0.5.
    halfwidth : float, optional
        Initial half-width in radians. If None, a value based on `odf.halfwidth` is used.
    verbose : bool, optional
        If True, prints iteration details. Default is False.

    Returns
    -------
    ODF
        Estimated orientation distribution function corresponding to input orientations.

    Notes
    -----
    - Uses `calc_error` to compare the estimated ODF with the reference ODF.
    - Iteratively increases half-width until the error starts to increase.
    - The returned ODF corresponds to the last half-width before error increased.
    - Does not modify the input orientations or reference ODF.
    """
    e0 = 1e8
    st_rad = np.radians(step)
    hwmin = np.radians(2)
    hwmax = np.radians(30) - 0.5 * nstep * st_rad
    if halfwidth is None:
        hw = np.clip(odf.halfwidth - 0.5 * nstep * st_rad, hwmin, hwmax)
        if verbose:
            print(f'Initial halfwidth set to {np.degrees(hw)}')
    else:
        hw = np.clip(halfwidth - 2 * st_rad, hwmin, hwmax)
        if verbose:
            print(f'Initial halfwidth is {np.degrees(hw)}')
    for c in range(nstep):
        todf = ODF(ori, halfwidth=hw)
        e = calc_error(odf, todf)
        if verbose:
            print(f'Iteration {c}: error={e}, halfwidth={np.degrees(hw)}°')
        if e > e0:
            break
        else:
            e0 = e
        hw += st_rad
    # return last value before error increased
    hw -= st_rad
    todf = ODF(ori, halfwidth=hw)
    return todf


def plot_pole_figure(orientations, phase, vector=None,
                     size=None, alpha=None):
    """
    Plot an inverse pole figure and pole density function for given orientations

    This function visualizes the orientation distribution of a crystal phase by
    plotting the stereographic projection of specified poles and the corresponding
    pole density function (PDF).

    Parameters
    ----------
    orientations : Orientation
        Orientation object containing crystal orientations to plot.
    phase : object
        Crystal phase object containing symmetry information and name.
    vector : array-like of shape (3,), optional
        Miller indices of the pole vector to plot. Default is [0, 0, 1].
    size : float, optional
        Marker size for scatter plot. Automatically scaled if None.
    alpha : float, optional
        Marker transparency for scatter plot. Automatically scaled if None.

    Notes
    -----
    - Uses Matplotlib for plotting with stereographic projection.
    - The first subplot shows the inverse pole figure (scatter of poles).
    - The second subplot shows the pole density function.
    - Marker size and transparency are scaled inversely with the number of orientations.
    - Does not modify the input `orientations` or `phase`.
    """
    # plot inverse pole figure
    # <111> poles in the sample reference frame
    if vector is None:
        vector = [0, 0, 1]
    assert isinstance(orientations, Orientation)
    if orientations.size < 1:
        logging.warning(f'No orientations provided: {orientations.size}.')
        scf = 1.0
    else:
        scf = 1.0 / np.sqrt(orientations.size)
    if size is None:
        size = np.clip(250*scf, 0.25, 25)
    if alpha is None:
        alpha = np.clip(4*scf, 0.05, 0.5)
    t_ = Miller(uvw=vector, phase=phase).symmetrise(unique=True)
    t_all = orientations.inv().outer(t_)
    fig = plt.figure(figsize=(9, 12))
    ax1 = fig.add_subplot(211, projection="stereographic")
    ax1.scatter(t_all, s=size, alpha=alpha)
    ax1.set_labels("X", "Y", None)
    ax1.set_title(phase.name + r" $\left<" +
                  f"{vector[0]}{vector[1]}{vector[2]}" +
                  r"\right>$ PF")
    ax2 = fig.add_subplot(212, projection="stereographic")
    ax2.pole_density_function(t_all)
    ax2.set_labels("X", "Y", None)
    ax2.set_title(phase.name + r" $\left<" +
                  f"{vector[0]}{vector[1]}{vector[2]}" +
                  r"\right>$ PDF")
    plt.show()


def plot_pole_figure_proj(orientations, phase, vector=None,
                          title=None, alpha=None, size=None):
    """
    Plot an inverse pole figure (IPF) projection and pole density function for EBSD data.

    This function visualizes crystal orientations along a specified sample direction
    using an IPF projection. The first subplot shows the scatter of rotated poles,
    and the second subplot shows the corresponding pole density function (PDF).

    Parameters
    ----------
    orientations : Orientation
        Orientation object containing crystal orientations to plot.
    phase : object
        Crystal phase object with symmetry information and name.
    vector : array-like of shape (3,), optional
        Sample direction to project (Miller indices). Default is [0, 0, 1].
    title : str, optional
        Title for the plot. If None, automatically uses the principal axis of `vector`.
    alpha : float, optional
        Marker transparency for scatter plot. Automatically scaled if None.
    size : float, optional
        Marker size for scatter plot. Automatically scaled if None.

    Notes
    -----
    - Uses Matplotlib with stereographic projection (IPF) for plotting.
    - Marker size and transparency are scaled inversely with the number of orientations.
    - The first subplot shows the inverse pole figure (scatter of rotated poles).
    - The second subplot shows the pole density function (PDF).
    - Does not modify the input `orientations` or `phase`.
    """
    # Some sample direction, v
    if vector is None:
        vector = [0, 0, 1]
    v = Miller(vector)
    if title is None:
        v_title = ["X", "Y", "Z"][np.argmax(vector)]
    else:
        v_title = title
    if orientations.size < 1:
        logging.warning(f'No orientations provided: {orientations.size}.')
        scf = 1.0
    else:
        scf = 1.0 / np.sqrt(orientations.size)
    if size is None:
        size = np.clip(250*scf, 0.25, 25)
    if alpha is None:
        alpha = np.clip(4*scf, 0.05, 0.5)
    assert isinstance(orientations, Orientation)
    # Rotate sample direction v into every crystal orientation O
    t_ = orientations * v

    # Set IPDF range
    vmin, vmax = (0, 3)

    subplot_kw = {"projection": "ipf", "symmetry": phase.point_group.laue, "direction": v}
    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.5)
    # leave room for the crystal direction labels above the IPF sector
    title_pad = plt.rcParams["font.size"] + 12

    ax0 = fig.add_subplot(211, **subplot_kw)
    ax0.scatter(t_, s=size, alpha=alpha)
    _ = ax0.set_title(f"EBSD data, {phase.name}, {v_title}", pad=title_pad)

    ax2 = fig.add_subplot(212, **subplot_kw)
    ax2.pole_density_function(t_, vmin=vmin, vmax=vmax)
    _ = ax2.set_title(f"EBSD data, {phase.name}, {v_title}", pad=title_pad)

    plt.show()

def find_orientations_fast(ori1: Orientation, ori2: Orientation, tol: float = 1e-3) -> np.ndarray:
    """
    Find closest matches in ori1 for each orientation in ori2 using KDTree

    Parameters
    ----------
    ori1 : Orientation
        Orientation database (e.g., EBSD orientations).
    ori2 : Orientation
        Orientations to match (e.g., grain mean orientations).
    tol : float
        Angular tolerance in radians.

    Returns
    -------
    matches : np.ndarray
        Array of indices in ori1 matching each entry in ori2; -1 if no match found.
    """
    # Get quaternions
    q1 = ori1.data
    q2 = ori2.data

    tree = KDTree(q2)  # KDTree in 4D quaternion space for ori2 (typically, regular SO3 grid)
    dists, indices = tree.query(q1)  # For every orientation in ori1, look for nearest neighbor in ori2 (grid)

    # create list of length or ori2, and store counts and indices of nearest neighbors in ori1
    matches = np.zeros(ori2.size, dtype=int)
    neigh_dict = dict()
    for i, idx in enumerate(indices):
        matches[idx] += 1
        if idx in neigh_dict.keys():
            neigh_dict[idx].append(i)
        else:
            neigh_dict[idx] = [i]

    return matches, neigh_dict


def texture_reconstruction(ns, ebsd=None, ebsdfile=None, orientations=None,
                          grainsfile=None, grains=None, kernel=None, kernel_halfwidth=5,
                          res_low=5, res_high=25, res_step=2, lim=5, hw_init=None, verbose=False):
    """
    Reconstruct a reduced ODF from EBSD or orientation data

    This function systematically reconstructs an orientation distribution function (ODF)
    using a given number of orientations or grains in a representative volume element (RVE).
    The reconstructed ODF approximates the misorientation distribution of the input data.
    Follows the method described in Biswas et al. (https://doi.org/10.1107/S1600576719017138).

    Parameters
    ----------
    ns : int
        Number of reduced orientations/grains in the RVE.
    ebsd : EBSDmap, optional
        EBSD map containing orientations for a single phase.
    ebsdfile : str, optional
        Path to a *.mat file with EBSD data (not yet supported).
    orientations : Orientation, optional
        Predefined orientations to use instead of EBSD data.
    grainsfile : str, optional
        Path to estimated grains file (*.mat). Not used for kernel estimation.
    grains : object, optional
        Grain data. Not used for kernel estimation.
    kernel : DeLaValleePoussinKernel, optional
        Kernel for ODF estimation. Default: halfwidth=kernel_halfwidth.
    kernel_halfwidth : float, optional
        Halfwidth in degrees for default DeLaValleePoussin kernel. Default is 5°.
    res_low : float, optional
        Minimum resolution (in degrees) of orientation grid for reconstruction. Default 5°.
    res_high : float, optional
        Maximum resolution (in degrees) of orientation grid. Default 25°.
    res_step : float, optional
        Step size (in degrees) for grid resolution. Default 2°.
    lim : int, optional
        Maximum number of consecutive worsening errors before stopping. Default 5.
    verbose : bool, optional
        If True, print progress and error information. Default False.

    Returns
    -------
    orired_f : Orientation
        Reduced set of orientations in the fundamental region.
    odfred_f : ODF
        Reconstructed ODF using the reduced orientations.
    ero : float
        L1 error between reference ODF and reconstructed ODF.
    res : float
        Grid resolution (degrees) at which the minimum error was achieved.

    Notes
    -----
    - Uses Monte Carlo sampling and kernel smoothing to generate reduced orientations.
    - Only supports single-phase EBSD data.
    - Kernel estimation from grains or files is not yet supported; default kernel is used.
    - The reconstructed ODF approximates the misorientation distribution of the input data.
    """
    ori = None
    psi = None
    if ebsdfile is not None:
        raise ModuleNotFoundError('Option "ebsdMatFile" is not yet supported.')
        # ind = args.index('ebsdMatFile') + 1
        # ebsd = loadmat(args[ind])
        # ebsd_var = list(ebsd.keys())[0]
        # ebsd = ebsd[ebsd_var]
        # assert len(np.unique(ebsd.phaseId)) == 1, 'EBSD has multiple phases'
        # ori = ebsd.orientations
    if ebsd is not None:
        if not isinstance(ebsd, EBSDmap):
            raise TypeError('Argument "ebsd" must be of type EBSDmap.')
        assert len(ebsd.emap.phases.ids) == 1, 'EBSD has multiple phases'
        if ori is not None:
            logging.warning('Both arguments "ebsd" and "ori" are given, using EBSD map orientations.')
        else:
            ori = ebsd.emap.orientations
    if orientations is not None:
        if not isinstance(orientations, Orientation):
            raise TypeError('Argument "orientations" must be of type Orientation,')
        if ori is not None:
            logging.warning('Both EBSD map and orientations are given, using EBSD.')
        else:
            ori = orientations

    if grainsfile is not None:
        logging.warning('Estimation of optimal kernel from grain files not supported. '
                        'DeLaValleePoussinKernel with halfwidth 5° will be used.\n'
                        'Please use kanapy-mtex for support of optimal kernels.')
        # ind = args.index('grainsMatFile') + 1
        # grains = loadmat(args[ind])
        # grains_var = list(grains.keys())[0]
        # grains = grains[grains_var]
        # assert len(np.unique(grains.phaseId)) == 1, 'Grains has multiple phases'
        # print('Optimum kernel estimated from mean orientations of grains')
        # psi = calcKernel(grains.meanOrientation)
    if grains is not None:
        logging.warning('Estimation of optimal kernel from grains not supported. '
                        'DeLaValleePoussinKernel with halfwidth 5° will be used.\n'
                        'Please use kanapy-mtex for support of optimal kernels.')
        #assert len(np.unique(grains.phaseId)) == 1, 'Grains has multiple phases'
        #print('Optimum kernel estimated from mean orientations of grains')
        #psi = calcKernel(grains.meanOrientation)
    if kernel is not None:
        if not isinstance(kernel, DeLaValleePoussinKernel):
            raise TypeError('Only kernels of type "DeLaValeePoussinKernel" are supported.')
        psi = kernel
    if psi is None:
        psi = DeLaValleePoussinKernel(halfwidth=np.radians(kernel_halfwidth))

    # Step 1: Create reference odf from given orientations with proper kernel
    odf = ODF(ori, kernel=psi)
    cs = ori.symmetry

    ero = 10.
    e_mod = []
    hw_stored = hw_init
    for hw in np.arange(res_low, res_high + res_step, res_step):
        # Step 2: create equispaced grid of orientations
        S3G = get_sample_fundamental(resolution=hw, point_group=cs)  # resolution in degrees! ori.SS not considered
        S3G = Orientation(S3G.data, symmetry=cs)
        # Step 3: calculate number of orientations close to each grid point (0 if no orientation close to GP)
        # count close orientations from EBSD map for each grid point, and get list of neighbor indices
        M, neighs = find_orientations_fast(ori, S3G, tol=np.radians(0.5))
        ictr = np.nonzero(M > 0)[0]  # indices of gridpoints with non-zero counts
        weights = M[ictr]  # create weights from non-zero counts of EBSD orientations at gridpoints
        weights = weights / np.sum(weights)  # calculate weights
        # Step 4: Calculate scaling factor hval such that sum of all int(weights*hval) = ns
        lval = 0
        hval = float(ns)
        ifc = 1.0
        ihval = np.sum(np.round(hval * weights))
        while (hval - lval > hval * 1e-15 or ihval < ns) and ihval != ns:
            if ihval < ns:
                hval_old = hval
                hval = hval + ifc * (hval - lval) / 2.0
                lval = hval_old
                ifc = ifc * 2.0
            else:
                hval = (lval + hval) / 2.0
                ifc = 1.0
            ihval = np.sum(np.round(hval * weights))
        screen = np.round(weights * hval)  # number of orientations associated to each grid point
        diff = np.sum(screen) - ns  # difference to desired number of orientations
        weights_loc = np.argsort(weights)
        co = 1
        while diff > 0:
            if screen[weights_loc[co]] > 0:
                screen[weights_loc[co]] = screen[weights_loc[co]] - 1
                diff = np.sum(screen) - ns
            co = co + 1

        # Step 5: Subdivide orientations around grid points into screen orientations
        #         and estimate mean orientation for each group or take orientation of grid point
        #         if only one orientation needs to be generated
        ori_red = np.zeros((ns, 4))
        octr = 0
        for i, no in enumerate(screen):
            nl = neighs[ictr[i]]
            if len(nl) < no:
                raise RuntimeError(f'{len(nl)} < {no} @ {i}, ictr: {ictr[i]}, weight: {weights[i]}')
            if 0.5 < no < 1.5:
                ori_red[octr, :] = S3G[ictr[i]].data
                octr += 1
                assert np.isclose(no, 1), f'no: {no}, should be 1'
            elif no >= 1.5:
                # split orientations in EBSD map matching to one grid point according to required number of
                # orientations at this point
                ind = np.linspace(0, len(nl), int(no)+1, dtype=int)
                idx = np.split(np.array(nl), ind[1:-1])
                ho = octr
                for j in idx:
                    ori_red[octr, :] = np.mean(ori[j].data, axis=0)
                    octr += 1
                if not np.isclose(no, octr-ho):
                    print(f'len_nl: {len(nl)}, len_idx: {len(idx)}')
                    raise RuntimeError(f'no: {no}, but increment is only #{octr-ho}')
        # create Orientations from quaternion array
        ori_f = Orientation(ori_red, symmetry=cs)
        assert not np.isnan(ori_f.data).any()
        assert not np.isinf(ori_f.data).any()

        # Step 6: Compute reduced ODF
        odfred = odf_est(ori_f, odf, halfwidth=hw_stored, verbose=verbose)
        hw_stored = odfred.halfwidth

        # Step 7: Compute error for kernel optimization
        er = calc_error(odf, odfred)
        if verbose:
            print(f'Resolution: {hw}, Error: {er}, Reduced HW: {np.degrees(odfred.halfwidth)}°')
        # store best results and evaluate stopping criterion
        if er < ero:
            orired_f = ori_f
            odfred_f = odfred
            ero = er
            res = hw
        e_mod.append(er)
        if len(e_mod) - np.argmin(e_mod) > lim:
            break
    orired_f = orired_f.in_euler_fundamental_region()
    if verbose:
        print(f'Final resolution: {res}, reduced HW: {np.degrees(odfred_f.halfwidth)}°')
    return orired_f, odfred_f, ero, res


class Kernel(ABC):
    """
    Abstract base class for kernels used in orientation distribution estimation.

    Attributes
    ----------
    A : ndarray
        Flattened array of kernel parameters. Initialized to empty array if not provided.
    """
    def __init__(self, A=None):
        self.A = np.array(A).flatten() if A is not None else np.array([])

    @property
    def bandwidth(self):
        """
        Returns the bandwidth of the kernel.

        The bandwidth is defined as the number of elements in `A` minus one.

        Returns
        -------
        int
            Bandwidth of the kernel.
        """
        return len(self.A) - 1

    @bandwidth.setter
    def bandwidth(self, L):
        """
        Set the bandwidth of the kernel.

        Truncates the kernel parameter array `A` to have at most `L + 1` elements.

        Parameters
        ----------
        L : int
            Desired bandwidth.
        """
        self.A = self.A[:min(L + 1, len(self.A))]

    def __str__(self):
        """
        Return a human-readable string representation of the kernel.

        Displays the kernel type and its halfwidth in degrees.

        Returns
        -------
        str
            String describing the kernel.
        """
        return f"custom, halfwidth {np.degrees(self.halfwidth()):.2f}°"

    def __eq__(self, other):
        """
        Check if two kernels are equal.

        Comparison is based on the truncated arrays of kernel parameters `A` up to
        the minimum bandwidth of the two kernels. Returns True if the relative
        difference is less than 1e-6.

        Parameters
        ----------
        other : Kernel
            Another kernel to compare against.

        Returns
        -------
        bool
            True if kernels are considered equal, False otherwise.
        """
        L = min(self.bandwidth, other.bandwidth)
        return np.linalg.norm(self.A[:L + 1] - other.A[:L + 1]) / np.linalg.norm(self.A) < 1e-6

    def __mul__(self, other):
        """
        Multiply two kernels element-wise with scaling by (2l + 1).

        Only the first `L + 1` elements are used, where L is the minimum bandwidth
        of the two kernels. Returns a new Kernel instance with the resulting array.

        Parameters
        ----------
        other : Kernel
            Another kernel to multiply with.

        Returns
        -------
        Kernel
            New kernel resulting from element-wise multiplication and scaling.
        """
        L = min(self.bandwidth, other.bandwidth)
        l = np.arange(L + 1)
        return Kernel(self.A[:L + 1] * other.A[:L + 1] / (2 * l + 1))

    def __pow__(self, p):
        """
        Raise kernel elements to a given power with scaling.

        Each element of the kernel parameter array `A` is scaled by (2l + 1),
        raised to the power `p`, and then rescaled by (2l + 1).

        Parameters
        ----------
        p : float
            Power to raise the kernel elements to.

        Returns
        -------
        Kernel
            New kernel resulting from the power operation with scaling.
        """
        l = np.arange(self.bandwidth + 1)
        return Kernel(((self.A / (2 * l + 1)) ** p) * (2 * l + 1))

    def norm(self):
        """
        Compute the L2 norm of the squared kernel elements.

        Returns
        -------
        float
            L2 norm of the squared elements of the kernel array `A`.
        """
        return np.linalg.norm(self.A ** 2)

    def cutA(self, fft_accuracy=1e-2):
        """
        Truncate the kernel array `A` based on the specified FFT accuracy.

        Elements of `A` are scaled by 1/(l^2), and elements smaller than
        a threshold determined from `fft_accuracy` are removed.

        Parameters
        ----------
        fft_accuracy : float, optional
            Desired FFT accuracy (default is 1e-2).

        Notes
        -----
        The first element is never truncated.
        """
        epsilon = fft_accuracy / 150
        A_mod = self.A / (np.arange(1, len(self.A) + 1) ** 2)
        idx = np.where(A_mod[1:] <= max(min([np.min(A_mod[1:]), 10 * epsilon]), epsilon))[0]
        if idx.size > 0:
            self.A = self.A[:idx[0] + 2]

    def halfwidth(self):
        """
        Compute the halfwidth of the kernel.

        The halfwidth is determined by finding the angle `omega` that minimizes
        the squared difference between K(1) and 2*K(cos(omega/2)).

        Returns
        -------
        float
            Halfwidth angle in radians.
        """
        def error_fn(omega):
            """
            Compute the squared difference used to determine kernel halfwidth.

            Parameters
            ----------
            omega : float
                Angle in radians to evaluate the kernel function.

            Returns
            -------
            float
                Squared difference: (K(1) - 2*K(cos(omega/2)))**2
            """
            return (self.K(1) - 2 * self.K(np.cos(omega / 2))) ** 2

        return fminbound(error_fn, 0, 3 * np.pi / 4)

    def K(self, co2):
        """
        Evaluate the kernel function at a given squared cosine value.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine squared value(s), will be clipped to [-1, 1].

        Returns
        -------
        float or ndarray
            Value(s) of the kernel function.
        """
        co2 = np.clip(co2, -1, 1)
        omega = 2 * np.arccos(co2)
        return self._clenshawU(self.A, omega)

    def K_orientations(self, orientations_ref, orientations):
        """
        Evaluate the kernel function for the misorientation angles between two sets of orientations.

        Parameters
        ----------
        orientations_ref : Orientation
            Reference set of orientations.
        orientations : Orientation
            Target set of orientations to compare.

        Returns
        -------
        ndarray
            Kernel values corresponding to misorientation angles.
        """
        misangles = orientations.angle_with(orientations_ref)
        co2 = np.cos(misangles / 2)
        return self.K(co2)

    def RK(self, d):
        """
        Evaluate the kernel using the Clenshaw-L method for given cosine distances.

        Parameters
        ----------
        d : array_like
            Cosine of angles (distance metric) in the range [-1, 1].

        Returns
        -------
        ndarray
            Kernel values for the given distances.
        """
        d = np.clip(d, -1, 1)
        return self._clenshawL(self.A, d)

    def RRK(self, dh, dr):
        """
        Evaluate the kernel on two sets of cosine distances using a reduced rotational kernel.

        Parameters
        ----------
        dh : array_like
            Cosines of angles for the first set, clipped to [-1, 1].
        dr : array_like
            Cosines of angles for the second set, clipped to [-1, 1].

        Returns
        -------
        ndarray
            2D array of kernel values for each combination of dh and dr.
        """
        dh = np.clip(dh, -1, 1)
        dr = np.clip(dr, -1, 1)
        L = self.bandwidth
        result = np.zeros((len(dh), len(dr)))

        if len(dh) < len(dr):
            for i, dh_i in enumerate(dh):
                Plh = [legendre(l)(dh_i) for l in range(L + 1)]
                result[i, :] = self._clenshawL(np.array(Plh) * self.A, dr)
        else:
            for j, dr_j in enumerate(dr):
                Plr = [legendre(l)(dr_j) for l in range(L + 1)]
                result[:, j] = self._clenshawL(np.array(Plr) * self.A, dh)
        result[result < 0] = 0
        return result

    def _clenshawU(self, A, omega):
        """
        Evaluate the kernel using the Clenshaw algorithm in the Chebyshev U basis.

        Parameters
        ----------
        A : array_like
            Coefficients of the kernel.
        omega : array_like
            Angles in radians.

        Returns
        -------
        ndarray
            Kernel values evaluated at the given angles.
        """
        omega = omega / 2
        res = np.ones_like(omega) * A[0]
        for l in range(1, len(A)):
            term = np.cos(2 * l * omega) + np.cos(omega) * np.cos((2 * l - 1) * omega) + \
                   (np.cos(omega) ** 2)
            res += A[l] * term
        return res

    def _clenshawL(self, A, x):
        """
        Evaluate the kernel using the Clenshaw algorithm in the Legendre basis.

        Parameters
        ----------
        A : array_like
            Coefficients of the kernel.
        x : array_like
            Input values where the kernel is evaluated (should be within [-1, 1]).

        Returns
        -------
        ndarray or float
            Kernel values evaluated at the given input(s).
        """
        b_next, b_curr = 0.0, 0.0
        x2 = 2 * x
        for a in reversed(A[1:]):
            b_next, b_curr = b_curr, a + x2 * b_curr - b_next
        return A[0] + x * b_curr - b_next

    def calc_fourier(self, L, max_angle=np.pi, fft_accuracy=1e-2):
        """
        Compute the Fourier coefficients of the kernel up to order L.

        Parameters
        ----------
        L : int
            Maximum order of the Fourier coefficients.
        max_angle : float, optional
            Upper limit of integration in radians (default is pi).
        fft_accuracy : float, optional
            Threshold below which coefficients are considered negligible and
            computation stops early (default is 1e-2).

        Returns
        -------
        ndarray
            Array of Fourier coefficients of length <= L+1.
        """
        A = []
        small = 0
        for l in range(L + 1):
            def integrand(omega):
                return self.K(np.cos(omega / 2)) * np.sin((2 * l + 1) * omega / 2) * np.sin(omega / 2)

            coeff, _ = quad(integrand, 0, max_angle, limit=2000)
            coeff *= 2 / np.pi
            A.append(coeff)
            if abs(coeff) < fft_accuracy:
                small += 1
            else:
                small = 0
            if small == 10:
                break
        return np.array(A)

    def plot_K(self, n_points=200):
        """
        Plot the kernel function K as a function of misorientation angle.

        Parameters
        ----------
        n_points : int, optional
            Number of points used for plotting the kernel function (default is 200).
        """
        omega = np.linspace(0, np.pi, n_points)
        co2 = np.cos(omega / 2)
        values = self.K(co2)
        plt.figure()
        plt.plot(np.degrees(omega), values)
        plt.xlabel("Misorientation angle (degrees)")
        plt.ylabel("K(cos(omega/2))")
        plt.title("Kernel Function")
        plt.grid(True)
        plt.show()


class DeLaValleePoussinKernel(Kernel):
    """
    De La Vallee Poussin kernel class for orientation distribution functions

    Parameters
    ----------
    kappa : float, optional
        Shape parameter of the kernel
    halfwidth : float, optional
        Halfwidth in radians; overrides kappa if provided
    bandwidth : int, optional
        Maximum degree of the series expansion

    Attributes
    ----------
    A : ndarray
        Series coefficients of the kernel
    kappa : float
        Shape parameter of the kernel
    C : float
        Normalization constant
    """
    def __init__(self, kappa=None, halfwidth=None, bandwidth=None):
        if halfwidth is not None:
            kappa = 0.5 * np.log(0.5) / np.log(np.cos(0.5*halfwidth))
        elif kappa is None:
            kappa = 90

        self.kappa = kappa
        L = bandwidth if bandwidth is not None else round(kappa)
        C = beta(1.5, 0.5) / beta(1.5, kappa + 0.5)
        self.C = C

        A = np.ones(L + 1)
        A[1] = kappa / (kappa + 2)

        for l in range(1, L - 1):
            A[l + 1] = ((kappa - l + 1) * A[l - 1] - (2 * l + 1) * A[l]) / (kappa + l + 2)

        for l in range(0, L + 1):
            A[l] *= (2 * l + 1)

        super().__init__(A)
        self.cutA()

    def K(self, co2):
        """
        Evaluate the De La Vallee Poussin kernel function.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine of half the misorientation angle. Values will be clipped to [-1, 1].

        Returns
        -------
        float or ndarray
            Kernel value(s) evaluated at the input `co2`.
        """
        co2 = np.clip(co2, -1, 1)
        return self.C * co2 ** (2 * self.kappa)

    def DK(self, co2):
        """
        Evaluate the derivative of the De La Vallee Poussin kernel with respect to misorientation.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine of half the misorientation angle. Values should be in [-1, 1].

        Returns
        -------
        float or ndarray
            Derivative of the kernel function evaluated at the input `co2`.
        """
        return -self.C * self.kappa * np.sqrt(1 - co2 ** 2) * co2 ** (2 * self.kappa - 1)

    def RK(self, t):
        """
        Evaluate the reduced kernel function R_K at a given input.

        Parameters
        ----------
        t : float or ndarray
            Input value, typically representing a normalized distance or cosine value.

        Returns
        -------
        float or ndarray
            Value of the reduced kernel function R_K at `t`.
        """
        return (1 + self.kappa) * ((1 + t) / 2) ** self.kappa

    def DRK(self, t):
        """
        Evaluate the derivative of the reduced kernel function R_K at a given input.

        Parameters
        ----------
        t : float or ndarray
            Input value, typically representing a normalized distance or cosine value.

        Returns
        -------
        float or ndarray
            Value of the derivative of R_K at `t`.
        """
        return self.kappa * (1 + self.kappa) * ((1 + t) / 2) ** (self.kappa - 1) / 2

    def halfwidth(self):
        """
        Compute the halfwidth of the DeLaValleePoussin kernel.

        The halfwidth is the misorientation angle ω where the kernel value
        drops to half its maximum.

        Returns
        -------
        float
            Halfwidth angle in radians.
        """
        return 2 * np.arccos(0.5 ** (1 / (2 * self.kappa)))


class ODF(object):
    """
    Estimate an Orientation Distribution Function (ODF) from a set of orientations
    using kernel density estimation.

    Parameters
    ----------
    orientations : orix.quaternion.Orientation
        Input orientation set.
    halfwidth : float, optional
        Halfwidth of the kernel in radians (default: 10 degrees).
    weights : array-like, optional
        Weights for each orientation. If None, uniform weights are used.
    kernel : Kernel instance, optional
        Kernel function to use. If None, DeLaValleePoussinKernel is used.
    exact : bool, optional
        If False and orientation count > 1000, approximation using grid is applied.

    Attributes
    ----------
    orientations : orix.quaternion.Orientation
        Orientation set stored in the ODF.
    weights : ndarray
        Normalized weights of the orientations.
    kernel : Kernel
        Kernel used for density estimation.
    halfwidth : float
        Halfwidth of the kernel in radians.
    """
    def __init__(self, orientations, halfwidth=np.radians(10), weights=None, kernel=None, exact=False):

        if orientations.size == 0:
            raise ValueError("Orientation set is empty.")

        if weights is None:
            weights = np.ones(orientations.size) / orientations.size

        # Set up kernel
        if kernel is None:
            kernel = DeLaValleePoussinKernel(halfwidth=halfwidth)
        hw = kernel.halfwidth()

        # Gridify if too many orientations and not exact
        if orientations.size > 1000 and not exact:
            # Placeholder: replace with proper gridify function if needed
            # Currently using simple thinning and weighting
            step = max(1, orientations.size // 1000)
            orientations = orientations[::step]
            weights = weights[::step]
            weights = weights / np.sum(weights)

        self.orientations = orientations
        self.weights = weights
        self.kernel = kernel
        self.halfwidth = hw

    def evaluate(self, ori):
        """
        Evaluate the ODF at given orientations.

        Parameters
        ----------
        ori : Orientation
            Orientation(s) at which to evaluate the ODF.

        Returns
        -------
        values : ndarray
            ODF values at the specified orientations.
        """
        values = np.zeros(ori.size)
        for o in self.orientations:
            values += self.kernel.K_orientations(o, ori)
        return values


class EBSDmap:
    """
    Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate
    synthetic RVEs

    Parameters
    ----------
    fname : str or pathlib.Path, optional
        EBSD file path, required unless loaded data is supplied.
    gs_min : float, optional
        Minimum grain size in pixels. Grains smaller than this are disregarded
        for statistical analysis. Default is 10.0.
    vf_min : float, optional
        Minimum volume fraction. Phases with smaller values are disregarded.
        Default is 0.03.
    max_angle : float, optional
        Maximum misorientation angle (degrees) used for grain merging.
        Default is 5.0.
    connectivity : int, optional
        Connectivity for grain identification. Default is 4.
    boundary_artifact_size_factor : float, optional
        Size factor used to identify small boundary artifact nodes.
    boundary_artifact_fraction_min : float, optional
        Minimum boundary pixel fraction for boundary artifact nodes.
    convex_hull_min_points : int, optional
        Minimum number of node points required for convex hull calculation.
    similar_orientation_merge_degrees : float, optional
        Maximum orientation difference in degrees for merging neighboring
        graph nodes after invalid regions have been absorbed.
    show_plot : bool, optional
        If True, plots microstructure maps. Default is True.
    show_hist : bool, optional
        If True, plots histograms of grain statistics. Default follows `show_plot`.
    felzenszwalb : bool, optional
        If True, applies Felzenszwalb segmentation. Default is False.
    show_grains : bool, optional
        If True, plots grain labeling. Default is False.
    show_graph : bool, optional
        If True, plots graph nodes and edges on the EBSD IPF map. Default is False.
    graph_only : bool, optional
        If True, build the EBSD graph and skip grain statistics extraction.
        Default is False.
    hist : bool, optional
        Deprecated. Use `show_hist` instead.
    plot : bool, optional
        Deprecated. Use `show_plot` instead.
    hex_grid_target_spacing : float, optional
        Isotropic physical spacing for hexagonal measurements. The default
        keeps the rectangular map shape reported by orix.
    ebsd_data : LoadedEBSDData, optional
        Measurements and grid metadata already loaded by ``EBSDDataLoader``.

    Attributes
    ----------
    emap : object
        Loaded EBSD map object.
    sh_x, sh_y : int
        Shape of EBSD map in pixels.
    dx, dy : float
        Pixel size in microns.
    npx : int
        Total number of pixels in the map.
    ms_data : list of dict
        Microstructure information for each phase, including:
        - name : str — phase name
        - vf : float — volume fraction
        - index : int — phase index
        - ori : Orientation — grain orientations
        - cs : crystal symmetry / Laue group
        - mo_map : ndarray — misorientation field
        - rgb_im : ndarray — IPF color image
        - ngrains : int — number of grains
        - gs_param, ar_param, om_param : ndarray — grain size, aspect ratio, main axis angle parameters
        - gs_data, ar_data, om_data : ndarray — raw distributions
        - gs_moments, ar_moments, om_moments : list — distribution moments
        - graph : object — microstructure graph
    graph : networkx.Graph or None
        All retained grains with globally unique IDs and edges across phase
        boundaries. For one retained phase this is its existing graph object.
    ci_map : ndarray
        Optional confidence index map if available.
    ngrains : int
        Total number of grains after merging/pruning.
    """

    def __init__(self, fname=None, gs_min=MIN_GRAIN_SIZE_PIXELS,
                 vf_min=DEFAULT_MIN_VOLUME_FRACTION,
                 max_angle=DEFAULT_MAX_MISORIENTATION_DEGREES,
                 connectivity=DEFAULT_CONNECTIVITY,
                 boundary_artifact_size_factor=DEFAULT_BOUNDARY_ARTIFACT_SIZE_FACTOR,
                 boundary_artifact_fraction_min=DEFAULT_BOUNDARY_ARTIFACT_FRACTION_MIN,
                 convex_hull_min_points=CONVEX_HULL_MIN_POINTS,
                 similar_orientation_merge_degrees=DEFAULT_SIMILAR_ORIENTATION_MERGE_DEGREES,
                 show_plot=True, show_hist=None, felzenszwalb=False, show_grains=False,
                 show_graph=False, graph_only=False, hist=None, plot=None, *,
                 hex_grid_target_spacing=None, ebsd_data=None):
        """
        Analyze measured EBSD phases and build their grain graphs.

        Existing plotting flags retain their order and defaults. Optional loaded
        data supplies the same workflow without reading another file.

        Parameters
        ----------
        fname : str or pathlib.Path, optional
            EBSD file to load when ``ebsd_data`` is not provided.
        gs_min, vf_min : float, optional
            Minimum grain pixel count and retained phase fraction.
        max_angle, similar_orientation_merge_degrees : float, optional
            Segmentation and graph merge misorientation thresholds in degrees.
        connectivity : int, optional
            Four or eight pixel connectivity.
        boundary_artifact_size_factor, boundary_artifact_fraction_min : float, optional
            Boundary cleanup size factor and minimum boundary pixel fraction.
        convex_hull_min_points : int, optional
            Minimum point count required for a valid grain hull.
        show_plot, show_hist, show_grains, show_graph : bool, optional
            Display the corresponding analysis figures.
        felzenszwalb : bool, optional
            Display the alternative image segmentation.
        graph_only : bool, optional
            Build cleaned graphs without fitting statistics or creating figures.
        hist, plot : bool, optional
            Deprecated aliases for ``show_hist`` and ``show_plot``.
        hex_grid_target_spacing : float, optional
            Isotropic physical spacing for hexagonal measurements. By default,
            preserve the rectangular shape reported by orix.
        ebsd_data : LoadedEBSDData, optional
            Already loaded measurements and grid metadata.

        Raises
        ------
        ValueError
            If input data is missing or a grid or hull parameter is invalid.
        """
        # interpret parameters
        if plot is not None:
            show_plot = plot
            logging.warning('Use of "plot" is depracted, use argument "show_plot" instead.')
        if hist is not None:
            show_hist = hist
            logging.warning('Use of "hist" is depracted, use argument "show_plot" instead.')
        if show_hist is None:
            show_hist = show_plot
        if graph_only:
            show_plot = False
            show_hist = False
            show_grains = False
            show_graph = False
        if convex_hull_min_points < CONVEX_HULL_MIN_POINTS:
            raise ValueError(
                f"convex_hull_min_points must be at least {CONVEX_HULL_MIN_POINTS}"
            )
        segmentation_angle = max_angle * np.pi / 180
        similar_orientation_merge_angle = (
            similar_orientation_merge_degrees * np.pi / 180
        )
        self.gs_min = gs_min
        self.vf_min = vf_min
        self.max_angle = segmentation_angle
        self.connectivity = connectivity
        self.boundary_artifact_size_factor = boundary_artifact_size_factor
        self.boundary_artifact_fraction_min = boundary_artifact_fraction_min
        self.convex_hull_min_points = int(convex_hull_min_points)
        self.similar_orientation_merge_angle = similar_orientation_merge_angle
        self.ms_data = []
        self.graph = None
        self.logger = logging.getLogger(__name__)
        if hex_grid_target_spacing is not None and hex_grid_target_spacing <= 0.0:
            raise ValueError("hex_grid_target_spacing must be positive")
        self.hex_grid_target_spacing = hex_grid_target_spacing
        if ebsd_data is None:
            if fname is None:
                raise ValueError("Either 'fname' or 'ebsd_data' must be provided")
            ebsd_data = EBSDDataLoader().load(fname)
        self._load_from_data(ebsd_data)
        retained_phases = []
        for n_ph, ind in enumerate(self.emap.phases.ids):
            if ind == -1:
                continue
            vf = np.count_nonzero(self.emap.phase_id == ind) / self.original_npx
            if vf >= vf_min:
                retained_phases.append((n_ph, ind, vf))

        if self.hex_grid and (len(retained_phases) > 1 or self.hex_grid_target_spacing is not None):
            self._prepare_hex_grid()

        print(f'Imported EBSD map with {len(self.emap.phases.ids)} phases.')
        phase_rows = []
        for n_ph, ind, vf in retained_phases:
            data = {
                "vf": vf,
                "name": self.emap.phases.names[n_ph],
                "index": ind,
                "len_x": self.sh_x,
                "len_y": self.sh_y,
                "delta_x": self.dx,
                "delta_y": self.dy,
                "cs": self.emap.phases[ind].point_group.laue,
            }
            if self.hex_grid:
                self._compute_hex_phase_data(data, ind)
            else:
                self._compute_square_phase_data(data, ind)
            self._handle_bad_pixels(data)

            data['mo_map'] = np.zeros(self.npx)
            data['mo_map'][data['idx']] = data['ori'].angle
            rgb_values = np.zeros((self.npx, 3))
            ipf_key = ox_plot.IPFColorKeyTSL(data['cs'])
            rgb_values[data['idx']] = ipf_key.orientation2color(data['ori'])
            data['rgb_im'] = rgb_values.reshape(self.sh_x, self.sh_y, 3)
            phase_rows.append(data)

        if len(phase_rows) > 1:
            self._build_multiphase_graph(phase_rows)

        for data in phase_rows:
            ind = data["index"]
            if len(phase_rows) > 1:
                ms_graph = data["graph"]
                self.ngrains = len(ms_graph)
            else:
                graph_rotations = np.full((self.npx, 4), np.nan, dtype=float)
                graph_rotations[data['idx']] = np.asarray(data['ori'].data)
                orientation_map = graph_rotations.reshape(self.sh_x, self.sh_y, 4)
                phase_mask = np.zeros(self.npx, dtype=bool)
                phase_mask[data['idx']] = True
                phase_mask = phase_mask.reshape(self.sh_x, self.sh_y)
                labels, n_regions = find_similar_regions_by_misorientation(
                    orientation_map, phase_mask, data['cs'],
                    tolerance=self.max_angle, connectivity=self.connectivity,
                )
                ms_graph = build_graph_from_labeled_pixels(
                    labels, graph_rotations, data['cs'], self.dx, self.dy,
                    connectivity=self.connectivity,
                )
                print(f"Phase #{ind} ({data['name']}): Identified Grains: {n_regions}")
                self._record_graph_grid_geometry(ms_graph)
                ms_graph.graph.update(phase_id=int(ind), phase_name=data["name"])
                nx.set_node_attributes(ms_graph, int(ind), "phase_id")
                nx.set_node_attributes(ms_graph, data["name"], "phase_name")
                ms_graph.name = 'Graph of microstructure'
                data['initial_labels'] = labels.copy()
                data['ori_map_debug'] = orientation_map.copy()
                data['phase_mask_debug'] = phase_mask.copy()
                data['graph_initial'] = copy.deepcopy(ms_graph)
                data['merge_debug'] = []
                self._prune_graph(ms_graph, data)
                self.graph = ms_graph
            data['ngrains'] = self.ngrains

            if graph_only:
                data['graph'] = ms_graph
                self.ms_data.append(data)
                print(
                    f"Built graph of phase #{data['index']} ({data['name']}) "
                    f"with {self.ngrains} grains."
                )
                continue

            # Extract grain statistics and axes
            data["grain_euler_angles"] = {
                f"grain {index}": Orientation(node["ori_av"], data["cs"])
                .to_euler(degrees=True)[0].tolist()
                for index, node in enumerate(ms_graph.nodes.values(), start=1)
            }
            arr_a = []
            arr_b = []
            arr_eqd = []
            arr_om = []
            for num, node in ms_graph.nodes.items():
                hull = node['hull']
                eqd = 2.0 * (hull.volume / np.pi) ** 0.5  # area of convex hull approximates grain better than pixels
                pts = hull.points[hull.vertices]  # outer nodes of grain
                # analyze geometry of point cloud
                ea, eb, va, vb = get_grain_geom(pts, method='ellipsis', two_dim=True)  # std: 'ellipsis', failsafe: 'raw'
                # assert ea >= eb
                if eb < 0.01 * ea:
                    logging.warning(f'Grain {num} has too high aspect ratio: main axes: {ea}, {eb}')
                    eb = 0.01 * ea
                sc_fct = eqd / np.sqrt(ea**2 + eb**2)  # rescale axes of ellipsis to ensure consistency with eqd
                ea *= sc_fct
                eb *= sc_fct
                # assert np.dot(va, vb) < 1.e-9
                # assert np.isclose(np.linalg.norm(va), 1.0)
                omega = np.arccos(va[0])  # Kanapy angle against the scaled row axis in [0, pi]
                node['max_dia'] = ea
                node['min_dia'] = eb
                node['equ_dia'] = eqd
                node['maj_ax'] = va
                node['min_ax'] = vb
                node['omega'] = omega
                node['center'] = hull.points.mean(axis=0)
                arr_a.append(ea)
                arr_b.append(eb)
                arr_eqd.append(eqd)
                arr_om.append(omega)
            arr_om = np.array(arr_om)
            arr_a = np.array(arr_a)
            arr_b = np.array(arr_b)
            arr_eqd = np.array(arr_eqd)
            print('\n--------------------------------------------------------')
            print('Statistical microstructure parameters in pixel map ')
            print('--------------------------------------------------------')
            print(np.median(arr_a), np.std(arr_a))
            print(np.median(arr_b), np.std(arr_b))
            print(np.median(arr_eqd), np.std(arr_eqd))

            # calculate equivalent diameters
            doffs = 0.
            deq_log = np.log(arr_eqd)
            dscale = np.exp(np.median(deq_log))
            dsig = np.std(deq_log)
            data['gs_param'] = np.array([dsig, doffs, dscale])
            data['gs_data'] = arr_eqd
            data['gs_moments'] = [dscale, dsig]

            # grain aspect ratio
            asp = arr_a / arr_b
            aoffs = 0.
            asp_log = np.log(asp)
            ascale = np.exp(np.median(asp_log))  # lognorm.median(asig, loc=aoffs, scale=ascale)
            asig = np.std(asp_log)  # lognorm.std(asig, loc=aoffs, scale=ascale)
            data['ar_param'] = np.array([asig, aoffs, ascale])
            data['ar_data'] = asp
            data['ar_moments'] = [ascale, asig]

            # angles of main axis
            # fit von Mises distribution (circular normal distribution) to data
            kappa, oloc, oscale = vonmises.fit(2*arr_om - np.pi)
            med_om = vonmises.median(kappa, loc=oloc)  # scale parameter has no effect on vonmises distribution
            std_om = vonmises.std(kappa, loc=oloc)
            data['om_param'] = np.array([kappa, oloc])
            data['om_data'] = arr_om
            data['om_moments'] = [med_om, std_om]

            print(f"Analyzed microstructure of phase #{data['index']} ({data['name']}) with {self.ngrains} grains.")
            print(f'Median values: equiv. diameter: {dscale.round(3)} micron, ' +
                  f'aspect ratio: {ascale.round(3)}, ' +
                  f'tilt angle: {(med_om * 180 / np.pi).round(3)}°')
            print(f'Std. dev: equivalent diameter: {dsig.round(3)} micron, ' +
                  f'aspect ratio: {asig.round(3)}, ' +
                  f'tilt angle: {(std_om * 180 / np.pi).round(3)}°')
            data['graph'] = ms_graph
            self.ms_data.append(data)
            if show_hist:
                self._plot_phase_histograms(data)

        if len(phase_rows) > 1:
            for data in self.ms_data:
                for node_id, node in data["graph"].nodes.items():
                    self.graph.nodes[node_id].update(node)
            self.ngrains = len(self.graph)

        if show_plot:
            self.plot_mo_map()
            self.plot_ipf_map()
            self.plot_segmentation()
            self.plot_pf()
        if show_graph:
            self.plot_graph_overlay(iphase=None)

        if show_grains:
            self.plot_grains()

        if felzenszwalb:
            self.plot_felsenszwalb()
        return

    def _build_multiphase_graph(self, phase_rows):
        """
        Segment and clean all retained phases in one grain graph.

        Pixel connectivity uses the symmetry of each phase before a single
        component labeling pass. Grain adjacency includes phase boundaries,
        while the existing cleanup passes reject merges across those boundaries.

        Parameters
        ----------
        phase_rows : list of dict
            Prepared phases receiving their graphs and cleanup snapshots.
        """
        rotations = np.full((self.npx, 4), np.nan)
        phase_map = np.full(self.npx, -1, dtype=int)
        symmetries = {}
        names = {}
        for data in phase_rows:
            phase = int(data["index"])
            rotations[data["idx"]] = data["ori"].data
            phase_map[data["idx"]] = phase
            symmetries[phase] = data["cs"]
            names[phase] = data["name"]
        phase_map = phase_map.reshape(self.sh_x, self.sh_y)
        orientations = rotations.reshape(self.sh_x, self.sh_y, 4)
        labels, count = find_similar_regions_by_misorientation(
            orientations, phase_map != -1, symmetries, tolerance=self.max_angle,
            connectivity=self.connectivity, phase_map=phase_map,
        )
        graph = build_graph_from_labeled_pixels(
            labels, rotations, symmetries, self.dx, self.dy,
            connectivity=self.connectivity, phase_map=phase_map,
        )
        for node in graph.nodes.values():
            node["phase_name"] = names[node["phase_id"]]
        self._record_graph_grid_geometry(graph)
        graph.name = "Graph of microstructure"
        print(f"All retained phases: Identified Grains: {count}")
        initial = copy.deepcopy(graph)
        cleanup = {"merge_debug": []}
        self._prune_graph(graph, cleanup)
        self.graph = graph

        for data in phase_rows:
            mask = phase_map == data["index"]
            initial_nodes = set(np.unique(labels[mask]))
            data["graph"] = self._extract_phase_graph(graph, data, initial_nodes)
            data["graph_initial"] = self._extract_phase_graph(initial, data, initial_nodes)
            data["graph_boundary_merged"] = self._extract_phase_graph(
                cleanup["graph_boundary_merged"], data, initial_nodes,
            )
            data["initial_labels"] = np.where(mask, labels, 0)
            data["phase_mask_debug"] = mask
            data["ori_map_debug"] = np.where(mask[..., None], orientations, np.nan)
            data["merge_debug"] = [
                record for record in cleanup["merge_debug"] if record["node"] in initial_nodes
            ]

    def _extract_phase_graph(self, graph, data, initial_nodes):
        """
        Extract a phase graph for existing statistics and RVE consumers.

        Grain IDs remain global. Labels and cleanup records outside this phase
        are excluded from its analysis view; the full graph retains all edges.

        Parameters
        ----------
        graph : networkx.Graph
            Shared graph at one stage of reconstruction.
        data : dict
            Phase metadata and pixel indices.
        initial_nodes : set of int
            Grain IDs belonging to the phase before cleanup.

        Returns
        -------
        networkx.Graph
            Graph containing this phase with its own symmetry and label map.
        """
        nodes = [node for node in graph if graph.nodes[node]["phase_id"] == data["index"]]
        phase_graph = graph.subgraph(nodes).copy()
        labels = np.zeros(self.npx, dtype=int)
        labels[data["idx"]] = graph.graph["label_map"].ravel()[data["idx"]]
        phase_graph.graph.update(
            label_map=labels.reshape(self.sh_x, self.sh_y), symmetry=data["cs"],
            phase_id=int(data["index"]), phase_name=data["name"],
        )
        for key in ("boundary_artifact_merge_records", "invalid_hull_merge_records",
                    "similar_orientation_merge_records", "small_node_merge_records"):
            if key in graph.graph:
                phase_graph.graph[key] = [
                    record for record in graph.graph[key] if record["node"] in initial_nodes
                ]
        return phase_graph

    def _load_from_data(self, ebsd_data: LoadedEBSDData) -> None:
        """
        Initialize EBSDmap attributes from loaded EBSD data.

        This method enables testing without file system by accepting
        data that is already in memory.

        Parameters
        ----------
        ebsd_data : LoadedEBSDData
            Measurements with their original and rectangular grid metadata.
        """
        self.emap = ebsd_data.emap
        self.sh_x = ebsd_data.sh_x
        self.sh_y = ebsd_data.sh_y
        self.dx = ebsd_data.dx
        self.dy = ebsd_data.dy
        self.npx = ebsd_data.npx
        self.hex_grid = getattr(ebsd_data, "hex_grid", False)
        self.original_npx = getattr(ebsd_data, "original_npx", None) or self.npx
        self.source_grid = getattr(ebsd_data, "source_grid", None)
        self.source_xstep = getattr(ebsd_data, "source_xstep", None)
        self.source_ystep = getattr(ebsd_data, "source_ystep", None)

    def _prepare_hex_grid(self):
        """
        Resample all staggered hexagonal fields onto one rectangular grid.

        The operation is performed once for the complete map so phase labels,
        confidence values, and orientations share the same grid. By default,
        the output keeps the map shape reported by orix. Callers may request
        an isotropic physical spacing explicitly.
        """
        target_spacing = self.hex_grid_target_spacing
        if target_spacing is None:
            grid_options = {
                "nx": int(self.sh_y),
                "ny": int(self.sh_x),
            }
            resampling_basis = "orix map shape"
        else:
            grid_options = {
                "dx_out": target_spacing,
                "dy_out": target_spacing,
            }
            resampling_basis = "requested physical spacing"

        xy = np.column_stack([self.emap.x, self.emap.y])
        phase_values = np.asarray(self.emap.phase_id)
        quaternion_values = np.asarray(self.emap.rotations.data)

        if "ci" in self.emap.prop.keys():
            confidence_values = np.asarray(self.emap.prop["ci"])
        else:
            confidence_values = np.zeros(self.original_npx)

        (
            x_grid,
            y_grid,
            phase_grid,
            confidence_grid,
            quaternion_grid,
        ) = resample_ebsd_to_rect_grid(
            xy,
            phase_values,
            quaternion_values,
            confidence_values,
            **grid_options,
        )

        self._hex_x_grid = x_grid
        self._hex_y_grid = y_grid
        self._hex_phase_grid = phase_grid
        self._hex_ci_grid = confidence_grid
        self._hex_quat_grid = quaternion_grid

        self.sh_x, self.sh_y = phase_grid.shape
        self.npx = int(phase_grid.size)
        self.dx = (
            float(x_grid[0, 1] - x_grid[0, 0])
            if self.sh_y > 1
            else float(target_spacing or self.dx)
        )
        self.dy = (
            float(y_grid[1, 0] - y_grid[0, 0])
            if self.sh_x > 1
            else float(target_spacing or self.dy)
        )
        self.grid_origin_x = float(x_grid[0, 0])
        self.grid_origin_y = float(y_grid[0, 0])

        self.logger.info(
            "Resampled staggered hexagonal EBSD map using "
            f"{resampling_basis}: x spacing {self.dx:g}, y spacing "
            f"{self.dy:g}, shape ({self.sh_x}, {self.sh_y})."
        )

    def _compute_square_phase_data(self, data, phase_id):
        """
        Prepare one phase on the measured square grid.

        Orientations are kept in the full map coordinate frame while low
        confidence points are repaired. The selected phase is also retained
        as a compact array for the later graph operations.

        Parameters
        ----------
        data : dict
            Phase data dictionary being filled.
        phase_id : int
            Phase ID to extract from the measured map.

        Raises
        ------
        ValueError
            If confidence index values cannot be aligned with either the full
            map or the selected phase.
        """
        phase_flat = np.asarray(self.emap.phase_id).reshape(-1)
        selected = phase_flat == phase_id
        selected_indices = np.flatnonzero(selected)

        euler_angles = self.emap[
            selected
        ].orientations.in_euler_fundamental_region()
        orientations = Orientation.from_euler(euler_angles, symmetry=data["cs"])

        quaternion_full = np.full((self.npx, 4), np.nan)
        quaternion_full[selected_indices] = orientations.data

        data["idx"] = selected_indices
        data["ori"] = orientations
        data["phase_id_flat"] = phase_flat
        data["quat_full"] = quaternion_full

        if "ci" not in self.emap.prop.keys():
            return

        confidence_values = np.asarray(self.emap.prop["ci"]).reshape(-1)
        if confidence_values.size == self.npx:
            data["ci_values"] = confidence_values
            return

        if confidence_values.size == selected_indices.size:
            confidence_full = np.full(self.npx, np.nan)
            confidence_full[selected_indices] = confidence_values
            data["ci_values"] = confidence_full
            return

        raise ValueError(
            "Confidence index values do not match the full map or selected phase."
        )

    def _compute_hex_phase_data(self, data, phase_id):
        """
        Resample one phase onto the rectangular EBSD grid.

        One retained phase keeps the existing default interpolation from its
        own measurements. Multiple retained phases and explicit physical
        spacing use a shared grid so their positions remain aligned.

        Parameters
        ----------
        data : dict
            Phase data dictionary being filled.
        phase_id : int
            Phase ID to extract from the resampled map.
        """
        if not hasattr(self, "_hex_phase_grid"):
            if "ci" in self.emap.prop.keys():
                confidence_values = np.asarray(self.emap.prop["ci"])
            else:
                confidence_values = np.zeros(self.original_npx)

            phase_mask = self.emap.phase_id == phase_id
            xy = np.column_stack([self.emap.x, self.emap.y])

            euler_angles = self.emap[
                phase_mask
            ].orientations.in_euler_fundamental_region()
            orientations = Orientation.from_euler(euler_angles)
            quaternions = np.column_stack(
                [
                    orientations.a,
                    orientations.b,
                    orientations.c,
                    orientations.d,
                ]
            )

            _, _, phase_grid, confidence_grid, quaternion_grid = (
                resample_ebsd_to_rect_grid(
                    xy[phase_mask],
                    self.emap.phase_id[phase_mask],
                    quaternions,
                    confidence_values[phase_mask],
                    nx=self.sh_y,
                    ny=self.sh_x,
                )
            )
            phase_flat = phase_grid.ravel()
            ci_flat = confidence_grid.ravel()
            quat_flat = quaternion_grid.reshape(-1, 4)
            valid = np.isfinite(quat_flat).all(axis=1)
        else:
            phase_flat = self._hex_phase_grid.ravel()
            ci_flat = self._hex_ci_grid.ravel()
            quat_flat = self._hex_quat_grid.reshape(-1, 4)
            valid = (phase_flat == phase_id) & np.isfinite(quat_flat).all(axis=1)

        data["idx"] = np.flatnonzero(valid)
        data["ori"] = Orientation(quat_flat[valid], data["cs"])
        data["phase_id_flat"] = phase_flat
        data["ci_values"] = ci_flat
        data["quat_full"] = quat_flat.copy()

    def _handle_bad_pixels(self, data):
        """
        Detect and correct bad pixels by reassigning neighboring orientations.

        Pixels with low confidence index values or no valid indexing are copied
        from nearby valid pixels in the same phase.

        Parameters
        ----------
        data : dict
            Phase orientations and full map confidence and phase arrays.
        """

        def _reassign(pixel, ori, phase_id, bads):
            """
            Reassign orientation of a pixel using its valid neighbor in the same phase.

            Full map indices prevent copying across row ends or phase boundaries.

            Parameters
            ----------
            pixel : int
                Flat index of the measurement being repaired.
            ori : numpy.ndarray
                Full map orientations, modified in place.
            phase_id : numpy.ndarray
                Full map phase identifiers.
            bads : set of int
                Measurements still awaiting repair.
            """

            phase = phase_id[pixel]
            row, col = divmod(pixel, self.sh_y)
            ix = -1
            if col > 0 and phase_id[pixel - 1] == phase and (pixel - 1) not in bads:
                ix = pixel - 1
            elif (
                col < self.sh_y - 1
                and phase_id[pixel + 1] == phase
                and (pixel + 1) not in bads
            ):
                ix = pixel + 1
            elif (
                row > 0
                and phase_id[pixel - self.sh_y] == phase
                and (pixel - self.sh_y) not in bads
            ):
                ix = pixel - self.sh_y
            elif (
                row < self.sh_x - 1
                and phase_id[pixel + self.sh_y] == phase
                and (pixel + self.sh_y) not in bads
            ):
                ix = pixel + self.sh_y

            if ix >= 0:
                ori[pixel] = ori[ix]  # pixel orientation is reassigned
            else:
                bads.add(pixel)  # no valid neighbor found add pixel again to list

        phase_id_values = data.get("phase_id_flat", self.emap.phase_id)
        orientation_values = data.get("quat_full", data["ori"])

        # Identify non-indexed pixels and pixels with low confidence index (CI)
        if "ci_values" in data:
            ci_values = data["ci_values"]
            self.ci_map = ci_values
            bad_mask = (ci_values < CONFIDENCE_INDEX_THRESHOLD) & (
                phase_id_values == data["index"]
            )
            bad_pixels = set(np.nonzero(bad_mask)[0])
        elif "ci" in self.emap.prop.keys():
            ci_values = self.emap.prop["ci"]
            if len(ci_values) == self.npx:
                self.ci_map = ci_values
            else:
                self.ci_map = np.zeros(self.npx)
                self.ci_map[self.emap.phase_id == 0] = ci_values

            bad_pixels = set(np.nonzero(ci_values < CONFIDENCE_INDEX_THRESHOLD)[0])
        else:
            bad_pixels = set()

        if len(bad_pixels) > 0:
            niter = 0
            self.logger.info(f"Initial number of bad pixels: {len(bad_pixels)}")
            nbad = len(bad_pixels)
            while (
                len(bad_pixels) > 0
                and niter < BAD_PIXEL_MAX_ITERATIONS_MULTIPLIER * nbad
            ):
                bp = bad_pixels.pop()
                _reassign(bp, orientation_values, phase_id_values, bad_pixels)
                niter += 1
            self.logger.info(
                f"After {niter} loops: number of bad pixels: {len(bad_pixels)}"
            )

        if "quat_full" in data:
            data["ori"] = Orientation(orientation_values[data["idx"]], data["cs"])
            del data["quat_full"]
            data.pop("phase_id_flat", None)
            data.pop("ci_values", None)

    def _record_graph_grid_geometry(self, graph):
        """
        Store the rectangular grid coordinate system on a graph.

        Source origins are retained as metadata. Grain geometry uses the
        explicit Kanapy row and column convention for grain statistics.

        Parameters
        ----------
        graph : networkx.Graph
            Microstructure graph prepared from the rectangular label map.
        """
        graph.graph.update(
            {
                "dx": float(self.dx),
                "dy": float(self.dy),
                "origin_x": float(getattr(self, "grid_origin_x", 0.0)),
                "origin_y": float(getattr(self, "grid_origin_y", 0.0)),
                "source_grid": getattr(self, "source_grid", None),
                "resampled_from_hex": bool(self.hex_grid),
                "geometry_convention": "kanapy_row_column",
                "grain_method": "graph",
            }
        )

    def _cleanup_merge_record(
        self,
        ms_graph,
        node,
        target,
        reason,
        angle_rad=None,
    ):
        """
        Build one graph cleanup merge record.

        The record keeps compact metadata for every cleanup merge. Pixel arrays
        are copied only when local zoom figures are requested.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph before the merge is applied.
        node : int
            Source node that will be merged away.
        target : int
            Target node that will remain after the merge.
        reason : str
            Cleanup stage that triggered the merge.
        angle_rad : float, optional
            Misorientation angle in radians when the stage uses orientation.

        Returns
        -------
        dict
            Merge record suitable for graph metadata and optional plotting.
        """
        record = {
            "node": int(node),
            "reason": reason,
            "target": int(target),
            "node_npix": int(ms_graph.nodes[node]["npix"]),
            "target_npix": int(ms_graph.nodes[target]["npix"]),
        }
        source_pixels = np.asarray(ms_graph.nodes[node]["pixels"], dtype=int)
        if source_pixels.size > 0:
            rows, columns = np.unravel_index(
                source_pixels,
                ms_graph.graph["label_map"].shape,
            )
            record["source_center"] = [
                float(np.mean(columns)),
                float(np.mean(rows)),
            ]

        if angle_rad is not None:
            record["angle_rad"] = float(angle_rad)
            record["angle_deg"] = float(np.degrees(angle_rad))

        if getattr(self, "save_graph_step_zoom_figures", False):
            record["node_pixels"] = source_pixels.copy()
            record["target_pixels_before"] = np.asarray(
                ms_graph.nodes[target]["pixels"],
                dtype=int,
            ).copy()

        return record

    def _node_points(self, ms_graph, node):
        """
        Return scaled row and column coordinates for one graph node.

        Convex hull checks use these points to decide whether a graph node can
        represent a valid grain region.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph containing node pixel lists.
        node : int
            Graph node ID.

        Returns
        -------
        numpy.ndarray
            Coordinate array with shape ``(n_points, 2)``.
        """
        rows, columns = np.unravel_index(
            ms_graph.nodes[node]["pixels"],
            (self.sh_x, self.sh_y),
        )
        return np.column_stack((rows * self.dx, columns * self.dy))

    def _assign_convex_hull(self, ms_graph, node):
        """
        Assign a convex hull to one node if possible.

        Nodes that cannot form a hull are handled by the invalid region merge
        stage.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph.
        node : int
            Graph node ID.

        Returns
        -------
        bool
            True when the node has a valid convex hull.
        """
        pts = self._node_points(ms_graph, node)
        if pts.shape[0] < self.convex_hull_min_points:
            return False

        try:
            ms_graph.nodes[node]["hull"] = ConvexHull(pts)
        except Exception:
            return False

        return True

    def _merge_invalid_hull_nodes(self, ms_graph):
        """
        Merge nodes that cannot form a valid convex hull.

        Invalid geometric regions are absorbed into their largest neighbor,
        preserving the neighbor orientation during this cleanup stage.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph to modify in place.
        """
        records = []
        ms_graph.graph["invalid_hull_merge_records"] = records
        changed = True
        while changed:
            changed = False
            for grain_id in sorted(list(ms_graph.nodes)):
                if grain_id not in ms_graph.nodes:
                    continue

                if self._assign_convex_hull(ms_graph, grain_id):
                    continue

                try:
                    largest_neighbor = find_largest_neighbor(ms_graph, grain_id)
                except ValueError:
                    continue

                records.append(
                    self._cleanup_merge_record(
                        ms_graph,
                        grain_id,
                        largest_neighbor,
                        "invalid_hull_merge",
                    )
                )
                merge_nodes(
                    ms_graph,
                    grain_id,
                    largest_neighbor,
                    preserve_target_orientation=True,
                )
                changed = True
                break

    def _merge_similar_orientation_nodes(self, ms_graph):
        """
        Merge neighboring nodes whose average orientations are similar.

        This cleanup stage reduces over segmentation after invalid geometric
        nodes have already been absorbed.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph to modify in place.
        """
        records = []
        ms_graph.graph["similar_orientation_merge_records"] = records
        sym_ops = get_proper_symmetry_quaternions(ms_graph.graph["symmetry"])
        changed = True
        while changed:
            changed = False
            _compute_all_edge_misorientations(ms_graph, sym_ops)

            for grain_id in sorted(list(ms_graph.nodes)):
                if grain_id not in ms_graph.nodes:
                    continue

                try:
                    similar_neighbor, ori_angle = find_sim_neighbor(
                        ms_graph,
                        grain_id,
                        sym_ops=sym_ops,
                    )
                except ValueError:
                    continue

                if ori_angle > self.similar_orientation_merge_angle:
                    continue

                record = self._cleanup_merge_record(
                    ms_graph,
                    grain_id,
                    similar_neighbor,
                    "similar_orientation_merge",
                    angle_rad=ori_angle,
                )
                record["threshold_deg"] = float(
                    np.degrees(self.similar_orientation_merge_angle)
                )
                records.append(record)
                merge_nodes(ms_graph, grain_id, similar_neighbor)
                changed = True
                break

        self.ngrains = len(ms_graph.nodes)
        self.logger.info(
            f"After merging of similar regions, {self.ngrains} grains left."
        )

    def _merge_small_nodes(self, ms_graph):
        """
        Merge remaining nodes below the minimum grain size.

        Small nodes are absorbed into their largest neighbor while keeping the
        neighbor orientation unchanged. Queued nodes are checked again because
        earlier merges may have grown them to the minimum grain size.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Microstructure graph to modify in place.
        """
        records = []
        ms_graph.graph["small_node_merge_records"] = records
        tiny_grains = sorted(
            nid for nid in ms_graph.nodes if ms_graph.nodes[nid]["npix"] < self.gs_min
        )
        tiny_grains_set = set(tiny_grains)
        while tiny_grains:
            grain_id = tiny_grains.pop(0)
            tiny_grains_set.discard(grain_id)
            if grain_id not in ms_graph.nodes:
                continue
            if ms_graph.nodes[grain_id]["npix"] >= self.gs_min:
                continue
            try:
                largest_neighbor = find_largest_neighbor(ms_graph, grain_id)
            except ValueError:
                continue

            record = self._cleanup_merge_record(
                ms_graph,
                grain_id,
                largest_neighbor,
                "small_node_merge",
            )
            record["threshold_npix"] = float(self.gs_min)
            records.append(record)
            merge_nodes(
                ms_graph,
                grain_id,
                largest_neighbor,
                preserve_target_orientation=True,
            )

            if (
                ms_graph.nodes[largest_neighbor]["npix"] < self.gs_min
                and largest_neighbor not in tiny_grains_set
            ):
                tiny_grains.append(largest_neighbor)
                tiny_grains.sort()
                tiny_grains_set.add(largest_neighbor)

        self.ngrains = len(ms_graph.nodes)
        self.logger.info(
            f"After elimination of small grains, {self.ngrains} grains left."
        )

    def _ensure_final_convex_hulls(self, ms_graph):
        """
        Ensure every remaining graph node has a valid convex hull.

        This validation keeps the downstream grain statistics step from using
        geometrically invalid graph nodes.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Cleaned microstructure graph.

        Raises
        ------
        ValueError
            If a remaining node cannot form a valid convex hull.
        """
        for grain_id, node in list(ms_graph.nodes(data=True)):
            pts = self._node_points(ms_graph, grain_id)
            if pts.shape[0] < self.convex_hull_min_points:
                raise ValueError(
                    f"Grain {grain_id} has too few pixels to compute a convex hull."
                )

            if "hull" in node:
                continue

            try:
                node["hull"] = ConvexHull(pts)
            except Exception as exc:
                raise ValueError(
                    f"Grain {grain_id} does not form a valid convex hull."
                ) from exc

    def _prune_graph(self, ms_graph, data=None):
        """
        Apply boundary, hull, orientation, and size cleanup in order.

        Each pass consumes the completed graph from the preceding stage, so
        orientation changes are revisited before the size threshold is applied.

        Parameters
        ----------
        ms_graph : networkx.Graph
            Graph to simplify in place.
        data : dict, optional
            Phase metadata receiving compatibility snapshots and merge records.
        """
        boundary_records = merge_boundary_artifact_nodes(
            ms_graph, self.gs_min, connectivity=self.connectivity,
            npix_factor=self.boundary_artifact_size_factor,
            boundary_fraction_min=self.boundary_artifact_fraction_min,
        )
        ms_graph.graph['boundary_artifact_merge_records'] = boundary_records
        if data is not None:
            data['merge_debug'].extend(boundary_records)
            data['graph_boundary_merged'] = copy.deepcopy(ms_graph)
        self._merge_invalid_hull_nodes(ms_graph)
        self._merge_similar_orientation_nodes(ms_graph)
        self._merge_small_nodes(ms_graph)
        self._ensure_final_convex_hulls(ms_graph)
        if data is not None:
            for key in ('invalid_hull_merge_records',
                        'similar_orientation_merge_records', 'small_node_merge_records'):
                data['merge_debug'].extend(ms_graph.graph[key])

    @staticmethod
    def _plot_phase_histograms(data):
        """
        Plot the measured distributions and fitted densities for one phase.

        Axis limits, bin counts and density transforms follow the original
        Kanapy histograms.

        Parameters
        ----------
        data : dict
            Retained phase row containing raw distributions and fitted parameters.
        """
        for prefix, title, xlabel in [
            ("gs", "Histogram of grain equivalent diameters", "Equivalent diameter (micron)"),
            ("ar", "Histogram of grain aspect ratios", "aspect ratio"),
            ("om", "Histogram of tilt angles of major axes", "angle (rad)"),
        ]:
            values = data[f"{prefix}_data"]
            params = data[f"{prefix}_param"]
            figure, axis = plt.subplots()
            if prefix == "om":
                x = np.linspace(-np.pi, np.pi, 200)
                y = vonmises.pdf(x, params[0], loc=params[1])
                axis.plot(0.5 * (x + np.pi), 2 * y, "r-", label="von Mises fit")
                axis.hist(values, bins=40, density=True, label="data")
            else:
                counts, edges = np.histogram(values, bins=20)
                singletons = np.flatnonzero(counts == 1)
                lower = edges[0]
                upper = edges[singletons[0]] if len(singletons) else np.max(values)
                x = np.linspace(lower, upper, 150)
                y = lognorm.pdf(x, params[0], loc=params[1], scale=params[2])
                axis.plot(x, y, "r-", label="lognorm fit")
                axis.hist(values, bins=20, range=(lower, upper), density=True, label="data")
            ylabel = "Normalized frequency" if prefix == "gs" else "normalized frequency"
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            axis.legend()
            plt.show()

    def plot_histograms(self, iphase=None):
        """
        Plot grain size, aspect ratio and tilt angle histograms with their fits.

        Each selected phase produces three figures from the stored statistics.
        A map loaded with ``graph_only=True`` has no fitted distributions.

        Parameters
        ----------
        iphase : int, optional
            Position in ``ms_data`` to plot. If None, plot all retained phases.

        Raises
        ------
        IndexError
            If the requested position is outside the retained phase list.
        KeyError
            If the selected phase has no extracted grain statistics.
        """
        phases = self.ms_data if iphase is None else [self.ms_data[iphase]]
        for data in phases:
            self._plot_phase_histograms(data)

    def calcORI(self, Ng, iphase=0, shared_area=None, nbins=12,
                res_low=5, res_high=25, res_step=2, lim=5, hw_init=None,
                verbose=False, full_output=False):
        """
        Estimate optimum kernel half-width and produce reduced set of
        orientations for given number of grains

        Parameters
        ----------
        Ng : int
            Numbr of grains for which orientation is requested.
        iphase : int, optional
            Phase for which data is requested. The default is 0.
        shared_area : array, optional
            Grain boundary data. The default is None.
        nbins : int, optional
            number of bins for GB texture. The default is 12.

        Returns
        -------
        ori : (Ng, 3)-array
            Array with Ng Euler angles.

        """
        ms = self.ms_data[iphase]
        orired, odfred, ero, res = texture_reconstruction(Ng, orientations=ms['ori'],
                                                          res_low=res_low, res_high=res_high,
                                                          res_step=res_step, lim=lim, hw_init=hw_init,
                                                          verbose=verbose)

        if shared_area is None:
            if full_output:
                return orired, odfred, ero, res
            else:
                return orired
        else:
            raise ModuleNotFoundError('Shared area is not implemented yet in pure Python version.\n'
                                      'Use kanapy-mtex for this option.')
            #orilist, ein, eout, mbin = \
            #    self.eng.gb_textureReconstruction(ms['grains'], orired,
            #                                      matlab.double(shared_area), nbins, nargout=4)
            #return np.array(self.eng.Euler(orilist))

    def showIPF(self):
        """
        Plot the inverse pole figure (IPF) color key for all phases in the EBSD map
        """
        for i in self.emap.phases.ids:
            pg = self.emap.phases[i].point_group.laue
            fig = plt.figure(figsize=(8, 8))
            ax0 = fig.add_subplot(111, projection="ipf", symmetry=pg, zorder=2)
            ax0.plot_ipf_color_key(show_title=False)
            ax0.patch.set_facecolor("None")
            plt.show()

    def plot_ci_map(self):
        """
        Plot the confidence index (CI) map of the EBSD data if available
        """
        if 'ci' in self.emap.prop.keys():
            plt.imshow(self.ci_map.reshape((self.sh_x, self.sh_y)))
            plt.title('CI values in EBSD map')
            plt.colorbar(label="CI")
            plt.show()

    def plot_pf(self, vector=None):
        """
        Plot pole figures for retained phases using the specified sample direction.

        Phase metadata is selected using each retained row's original identifier.

        Parameters
        ----------
        vector : array-like, optional
            Sample reference vector for the pole figure. Default is [0, 0, 1].

        Notes
        -----
        Plots <111> poles in the sample reference frame for each phase.
        """
        # plot inverse pole figure
        # <111> poles in the sample reference frame
        if vector is None:
            vector = [0, 0, 1]
        for data in self.ms_data:
            orientations = data['ori']
            plot_pole_figure(orientations, self.emap.phases[data['index']], vector=vector)
            #t_ = Miller(uvw=vector, phase=self.emap.phases[ind]).symmetrise(unique=True)
            #t_all = orientations.inv().outer(t_)
            #fig = plt.figure(figsize=(8, 8))
            #ax = fig.add_subplot(111, projection="stereographic")
            #ax.scatter(t_all)
            #ax.set_labels("X", "Y", None)
            #ax.set_title(data['name'] + r" $\left<001\right>$ PF")
            #plt.show()

    def plot_pf_proj(self, vector=None):
        """
        Plot projected pole figures for retained phases using a sample direction.

        Phase metadata is selected using each retained row's original identifier.

        Parameters
        ----------
        vector : array-like, optional
            Sample reference vector for the projected pole figure. Default is [0, 0, 1].

        Notes
        -----
        Uses a projection method to visualize <111> poles in the sample reference frame.
        """
        # plot inverse pole figure
        # <111> poles in the sample reference frame
        if vector is None:
            vector = [0, 0, 1]
        for data in self.ms_data:
            orientations = data['ori']
            plot_pole_figure_proj(orientations, self.emap.phases[data['index']], vector=vector)

    def plot_grains_marked(self):
        """
        Plot grains with labels and major and minor axes for all phases.

        Multiphase maps use compact display indices for colors while retaining
        the original global grain IDs in annotations and the colorbar.

        Notes
        -----
        Each grain is annotated with its number. Major axes are plotted in black,
        minor axes in red. The plot uses a distinct colormap for different grains.
        """
        # plot grain with numbers and axes
        for data in self.ms_data:
            ngr = data['ngrains']
            cols = get_distinct_colormap(ngr)
            if len(self.ms_data) > 1:
                labels = data['graph'].graph['label_map']
                grain_ids, display_indices = np.unique(labels, return_inverse=True)
                cmap = ListedColormap(['white'] + cols)
                plt.imshow(display_indices.reshape(labels.shape), cmap=cmap, vmin=0, vmax=ngr)
            else:
                cmap = LinearSegmentedColormap.from_list('segs', cols, N=ngr)
                plt.imshow(data['graph'].graph['label_map'] / ngr, cmap=cmap)
            for num, node in data['graph'].nodes.items():
                ctr = data['graph'].nodes[num]['center']
                plt.annotate(str(num), xy=_graph_points_to_image(data['graph'], ctr))
                pts = np.zeros((4, 2))
                pts[0, :] = node['center'] - node['max_dia'] * node['maj_ax']
                pts[1, :] = node['center'] + node['max_dia'] * node['maj_ax']
                pts[2, :] = node['center'] - node['min_dia'] * node['min_ax']
                pts[3, :] = node['center'] + node['min_dia'] * node['min_ax']
                pts = _graph_points_to_image(data['graph'], pts)
                plt.plot(pts[0:2, 0], pts[0:2, 1], color='k')
                plt.plot(pts[2:4, 0], pts[2:4, 1], color='red')
            plt.title(f"Phase #{data['index']} ({data['name']}): Grain labels and axes: {ngr}")
            colorbar = plt.colorbar(label='Grain Number')
            if len(self.ms_data) > 1:
                ticks = np.linspace(1, ngr, min(ngr, 10), dtype=int)
                colorbar.set_ticks(ticks, labels=grain_ids[ticks])
            plt.show()

    def get_ipf_image(self, iphase=None):
        """
        Return the IPF image for one or all retained phases.

        Multiple phases use their own pixels on the common EBSD grid. A
        single phase returns its existing image without allocating a composite.

        Parameters
        ----------
        iphase : int, optional
            Position in ``ms_data``. None includes all retained phases.

        Returns
        -------
        numpy.ndarray
            RGB image with shape ``(sh_x, sh_y, 3)``.
        """
        phases = self.ms_data if iphase is None else [self.ms_data[iphase]]
        if len(phases) == 1:
            return phases[0]["rgb_im"].reshape(self.sh_x, self.sh_y, 3)

        image = np.zeros((self.sh_x * self.sh_y, 3))
        for data in phases:
            pixels = data["idx"]
            image[pixels] = data["rgb_im"].reshape(-1, 3)[pixels]
        return image.reshape(self.sh_x, self.sh_y, 3)

    def plot_graph_overlay(
            self, iphase=None, show_ids=False, marker_size=3.0, line_width=1.0,
            save_path=None, show=True, phase_colors=None):
        """
        Plot retained grain graphs together on the EBSD IPF map.

        Grain nodes and edges use one color per phase. Single phase maps keep
        black markers and lines, and individual phases can still be selected.
        Gray edges record adjacency between grains from different phases.

        Parameters
        ----------
        iphase : int, optional
            Position in ``self.ms_data``. None includes all retained phases.
        show_ids : bool, optional
            If True, draw node IDs next to the node centers. Default is False.
        marker_size : float, optional
            Marker size for node centers. Default is 3.0.
        line_width : float, optional
            Line width for graph edges. Default is 1.0.
        save_path : str or pathlib.Path, optional
            Path where the figure is saved. If None, the figure is not saved.
            Default is None.
        show : bool, optional
            If True, show the figure with Matplotlib. Default is True.
        phase_colors : dict, optional
            Matplotlib colors keyed by original phase ID. Unspecified phases
            use black followed by colors from the tab10 palette.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        phases = self.ms_data if iphase is None else [self.ms_data[iphase]]
        colors = {}
        for index, data in enumerate(self.ms_data):
            colors[data["index"]] = "black" if index == 0 else plt.get_cmap("tab10")((index - 1) % 10)
        if phase_colors is not None:
            colors.update(phase_colors)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.get_ipf_image(iphase))
        legend = []
        all_positions = {}
        for data in phases:
            graph = data["graph"]
            color = colors[data["index"]]
            positions = {}
            for node_id, node in graph.nodes.items():
                center = node.get("center")
                if center is not None:
                    positions[node_id] = _graph_points_to_image(graph, center)
                elif len(node.get("pixels", [])):
                    rows, columns = np.unravel_index(node["pixels"], (self.sh_x, self.sh_y))
                    positions[node_id] = (columns.mean(), rows.mean())

            for u, v in graph.edges():
                if u not in positions or v not in positions:
                    continue
                xu, yu = positions[u]
                xv, yv = positions[v]
                ax.plot([xu, xv], [yu, yv], color=color, linewidth=line_width)

            for node_id, (x, y) in positions.items():
                ax.plot(x, y, "o", color=color, markersize=marker_size)
                if show_ids:
                    label = str(node_id) if len(phases) == 1 else f"{data['index']}:{node_id}"
                    ax.text(x, y, label, color="white", fontsize=8, ha="center", va="center")
            legend.append(Line2D([], [], color=color, marker="o",
                                 label=f"{data['name']} (phase {data['index']})"))
            all_positions.update(positions)

        if len(phases) > 1:
            has_phase_boundary = False
            for node_a, node_b in self.graph.edges:
                if _same_phase(self.graph, node_a, node_b):
                    continue
                if node_a not in all_positions or node_b not in all_positions:
                    continue
                xa, ya = all_positions[node_a]
                xb, yb = all_positions[node_b]
                ax.plot([xa, xb], [ya, yb], color="gray", linewidth=line_width, zorder=1.5)
                has_phase_boundary = True
            if has_phase_boundary:
                legend.append(Line2D([], [], color="gray", label="Phase boundary"))

        if len(phases) == 1:
            data = phases[0]
            ax.set_title(f"Phase #{data['index']} ({data['name']}): Microstructure graph")
        else:
            ax.set_title("Microstructure graph: all retained phases")
            ax.legend(handles=legend)
        ax.set_axis_off()
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_graph_with_black_marker(self, iphase=0, show_ids=False, marker_size=12, line_width=1.0):
        """
        Plot the microstructure graph on top of the EBSD IPF map.

        Parameters
        ----------
        iphase : int, optional
            Phase index in ``self.ms_data``. Default is 0.
        show_ids : bool, optional
            If True, draw node IDs next to the node centers. Default is False.
        marker_size : float, optional
            Legacy marker size. Default is 12.
        line_width : float, optional
            Line width for graph edges. Default is 1.0.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        return self.plot_graph_overlay(
            iphase=iphase,
            show_ids=show_ids,
            marker_size=marker_size / 4,
            line_width=line_width,
        )

    def plot_mo_map(self):
        """
        Plot the misorientation map for all phases

        Notes
        -----
        Shows the misorientation angle of each pixel with respect to a reference orientation.
        """
        for data in self.ms_data:
            plt.imshow(data['mo_map'].reshape((self.sh_x, self.sh_y)))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation angle wrt reference")
            plt.colorbar(label="Misorientation (rad)")
            plt.show()

    def plot_segmentation(self, show_mo=True, show_ipf=True):
        """
        Plot segmentation results for all phases

        Parameters
        ----------
        show_mo : bool, optional
            If True, overlay grain boundaries on the misorientation map. Default is True.
        show_ipf : bool, optional
            If True, overlay grain boundaries on the IPF map. Default is True.

        Notes
        -----
        Uses similarity-based segmentation to highlight grain boundaries on the maps.
        """
        # plot segmentation results
        for data in self.ms_data:
            if show_mo:
                gscale_map = data['mo_map'].reshape((self.sh_x, self.sh_y)) / np.pi
                plt.imshow(mark_boundaries(gscale_map, data['graph'].graph['label_map']))
                plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation map with similarity segmentation")
                plt.show()
            if show_ipf:
                plt.imshow(mark_boundaries(data['rgb_im'], data['graph'].graph['label_map']))
                plt.title(f"Phase #{data['index']} ({data['name']}): IPF map with similarity segmentation")
                plt.show()

    def plot_felsenszwalb(self):
        """
        Apply Felzenszwalb segmentation to the misorientation map and plot the results

        Notes
        -----
        Segments the misorientation map using the Felzenszwalb algorithm and overlays
        the segment boundaries for visualization.
        """
        from skimage.segmentation import felzenszwalb
        for data in self.ms_data:
            gscale_map = data['mo_map'].reshape((self.sh_x, self.sh_y)) / np.pi
            labels_fz = felzenszwalb(gscale_map, scale=300, sigma=0.6, min_size=25)  # sc=300, sig=0.8, min_s=25
            plt.imshow(mark_boundaries(gscale_map, labels_fz))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation map with Felzenszwalb segmentation")
            plt.show()

    def plot_graph(self):
        """
        Visualize the microstructure graph for all phases

        Notes
        -----
        Uses the `visualize_graph` function to display nodes and connections of each phase's graph.
        """
        # visualize graph
        for data in self.ms_data:
            visualize_graph(data['graph'])

    def plot_ipf_map(self, iphase=None):
        """
        Plot the IPF map for one or all retained phases on the same image.

        Each retained phase keeps its crystallographic colors and grid positions.
        Pixels outside the selected phases remain black.

        Parameters
        ----------
        iphase : int, optional
            Position in ``ms_data``. None includes all retained phases.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.get_ipf_image(iphase))
        plt.show()
        return fig, ax

    def plot_grains(self, N=5):
        """
        Plot the N largest grains with one-hot maps, convex hulls, and principal axes

        Parameters
        ----------
        N : int, optional
            Number of largest grains to plot for each phase. Default is 5.

        Notes
        -----
        Each grain is visualized with its pixels, convex hull, and major/minor axes.
        Major axes are plotted in cyan, minor axes in green.
        """
        # select N largest gains and create a one-hot-plot for each grain
        # showing its pixels, the vertices of the convex hull and the convex hull itself
        # together with the principal axes of the grain
        for data in self.ms_data:
            glist = [0]
            slist = [0]
            for num, node in data['graph'].nodes.items():
                hs = node['equ_dia']
                if hs > min(slist):
                    glist.append(num)
                    slist.append(hs)
                    if len(glist) > N:
                        i = np.argmin(slist)
                        glist.pop(i)
                        slist.pop(i)
            for num in glist:
                pix_c = np.unravel_index(data['graph'].nodes[num]['pixels'], (self.sh_x, self.sh_y))
                oh_grain = np.zeros((self.sh_x, self.sh_y))
                oh_grain[pix_c] = 0.3
                node = data['graph'].nodes[num]
                ind = node['hull'].vertices
                oh_grain[pix_c[0][ind], pix_c[1][ind]] = 1
                plt.imshow(oh_grain, cmap='gray')
                # add convex hull
                for i in range(len(ind) - 1):
                    ix = [ind[i], ind[i + 1]]
                    plt.plot(pix_c[1][ix], pix_c[0][ix], color='y')
                ix = [ind[i + 1], ind[0]]
                plt.plot(pix_c[1][ix], pix_c[0][ix], color='y')
                # add axes
                ctr = node['center']
                pts = np.zeros((4, 2))
                pts[0, :] = ctr - node['max_dia'] * node['maj_ax']
                pts[1, :] = ctr + node['max_dia'] * node['maj_ax']
                pts[2, :] = ctr - node['min_dia'] * node['min_ax']
                pts[3, :] = ctr + node['min_dia'] * node['min_ax']
                pts = _graph_points_to_image(data['graph'], pts)
                plt.plot(pts[0:2, 0], pts[0:2, 1], color='cyan')
                plt.plot(pts[2:4, 0], pts[2:4, 1], color='green')
                plt.title(f"Phase #{data['index']} ({data['name']}): Grain #{num}")
                plt.show()


def get_ipf_colors(ori_list: Any, color_key: int = 0) -> np.ndarray:
    """
    Get RGB colors for a list of orientations in radians

    Parameters
    ----------
    ori_list : (N, 3) ndarray
        List of N Euler angles in radians

    color_key : int, optional
        Index of the IPF color key to use. Default is 0

    Returns
    -------
    colors : (N, 3) ndarray
        RGB values corresponding to each orientation

    Notes
    -----
    Assumes cubic crystal symmetry and cubic specimen symmetry
    """

    # get colors
    if not isinstance(ori_list, Orientation):
        ori_list = Orientation.from_euler(ori_list)
    ckey = ox_plot.IPFColorKeyTSL(ori_list.symmetry)
    ocol = ckey.orientation2color(ori_list)
    return ocol


def createOriset(
    num: int,
    ang: Any,
    omega: float,
    hist: Any = None,
    shared_area: Any = None,
    cs: Any = None,
    degree: bool = True,
    Nbase: int = 10000,
    resolution: Optional[float] = None,
    res_low: int = 5,
    res_high: int = 25,
    res_step: int = 2,
    lim: int = 5,
    hw_init: Any = None,
    verbose: bool = False,
    full_output: bool = False) -> Any:
    """
    Create a set of Euler angles according to an ODF defined by input orientations and kernel half-width

    Parameters
    ----------
    num : int
        Number of Euler angles to generate
    ang : (3,) or (M, 3) array or Orientation
        Input set of Euler angles (in degrees or radians) defining the ODF
    omega : float
        Kernel half-width (in degrees or radians)
    hist : array or None, optional, default=None
        Histogram of MDF.
    shared_area : array or None, optional, default=None
        Shared area between pairs of grains.
    cs : Symmetry, optional
        Crystal symmetry group. Default is 'm3m'
    degree : bool, optional, default=True
        If True, input angles and omega are in degrees.
    Nbase : int, optional, default=10000
        Base number of orientations for artificial ODF.
    resolution : float or None, optional, default=None
        Resolution for orientation generation. If None, derived from omega
    res_low, res_high, res_step, lim : int, optional
        Parameters for texture reconstruction
    verbose : bool, optional, default=False
        If True, prints progress messages.
    full_output : bool, optional, default=False
        If True, returns additional reconstruction outputs.

    Returns
    -------
    ori : (num, 3) ndarray
        Reduced set of Euler angles

    Notes
    -----
    Can generate artificial ODFs if input set is small. Grain-boundary-texture
    reconstruction with hist or shared_area requires the kanapy-mtex module.
    """
    # prepare parameters
    if hist is not None or shared_area is not None:
        raise ModuleNotFoundError('The grain-boundary-texture module is currently only available in kanapy-mtex.')
    if cs is None or not isinstance(cs, Symmetry):
        logging.warning('Crystal Symmetry "cs" must be provided as Symmetry object.')
        print(type(cs))
    if resolution is None:
        if degree:
            resolution = omega
        else:
            resolution = np.rad2deg(omega)  # resolution must be given in degrees
    if degree:
        omega = np.deg2rad(omega)
    if isinstance(ang, list):
        if degree:
            ang = np.deg2rad(ang)
        else:
            ang = np.array(ang)
        ang = Orientation.from_euler(ang, cs)
    else:
        assert isinstance(ang, Orientation)    
    # psi = DeLaValleePoussinKernel(halfwidth=omega)
    if ang.size < Nbase/100:
        # create artificial ODF for monomodal texture or small orientation set
        if verbose:
            logging.info(f'Creating artificial ODF centered around orientation {ang} with '
                         f'kernel halfwidth: {omega}')
        odf = ODF(ang, halfwidth=omega)
        if verbose:
            logging.info(f'Creating {Nbase} orientation from artificial ODF.')
        ori = calc_orientations(odf, Nbase, res=resolution)
        assert ori.size == Nbase
    else:
        ori = ang
    logging.info(f'Texture reconstruction generating {num} orientations from {ori.size} inputs.'
                 f'Kernel halfwidth: {omega}')
    ori_red, odfred, ero, res = texture_reconstruction(num, orientations=ori,
                                                   kernel_halfwidth=omega,
                                                   res_low=res_low, res_high=res_high,
                                                   res_step=res_step, lim=lim, hw_init=hw_init,
                                                   verbose=verbose)
    if hist is None:
        if full_output:
            return ori_red, odfred, ero, res
        else:
            return ori_red
    else:
        pass
        #if shared_area is None:
        #    raise ValueError('Microstructure.shared_area must be provided if hist is given.')
        #orilist, ein, eout, mbin = \
        #    eng.gb_textureReconstruction(matlab.double(hist), ori,
        #                                 matlab.double(shared_area), len(hist), nargout=4)
        #return np.array(eng.Euler(orilist))


def createOrisetRandom(
    num: int,
    omega: Optional[float] = None,
    hist: Any = None,
    shared_area: Any = None,
    cs: Any = None,
    Nbase: Optional[int] = None) -> Any:
    """
    Create a set of Euler angles for a random texture

    Parameters
    ----------
    num : int
        Number of Euler angles to generate
    omega : float, optional
        Kernel half-width in degrees. Default is 7.5
    hist : array, optional
        Histogram of MDF. Default is None
    shared_area : array, optional
        Shared area between pairs of grains. Default is None
    cs : Symmetry, optional
        Crystal symmetry group. Default is 'm3m'
    Nbase : int, optional
        Base number of orientations for random texture. Default is 5000

    Returns
    -------
    ori : (num, 3) ndarray
        Set of randomly distributed Euler angles

    Notes
    -----
    Unlike `createOriset`, this function directly generates `num` random orientations
    without reducing a larger artificial EBSD set. Grain-boundary-texture reconstruction
    using `hist` or `shared_area` requires the kanapy-mtex module.
    """
    if hist is not None or shared_area is not None:
        raise ModuleNotFoundError('The grain-boundary-texture module is currently only available in kanapy-mtex.')
    if cs is None or not isinstance(cs, Symmetry):
        logging.warning('Crystal Symmetry "cs" must be provided as Symmetry object.')
        print(type(cs))

    ori = Orientation.random(num, cs).in_euler_fundamental_region()
    if hist is None:
        return ori
    else:
        pass
    # ori = eng.project2FundamentalRegion(ori)
    #    if shared_area is None:
    #        raise ValueError('Shared grain boundary area (geometry["GBarea"]) must be provided if hist is given.')
    #    orilist, ein, eout, mbin = \
    #        eng.gb_textureReconstruction(matlab.double(hist), ori,
    #                                     matlab.double(shared_area), len(hist), nargout=4)
    #    return np.array(eng.Euler(orilist))

def segment_microstructure(
    json_path: str,
    *,
    th_cut_deg: float = 5.0,
    th_merge_deg: float = 7.5,
    min_voxels: int = 4,
    print_topk_rag: int = 10,
    tiny_cutoff: int = 5,
    verbose: bool = True,
    log_path: str | None = None,
    progress: bool = False,
) -> None:
    """
    Segment all snapshots in a  microstructure JSON in-place.

    - Snapshot 0 is assumed already segmented and is kept as reference for phase_id mapping.
    - Snapshots 1..last are re-segmented from voxel orientations using:
        Step 1: 6-neighbor voxel graph + ORIX misorientation on edges
        Step 2: threshold cut (<= th_cut_deg) + connected components
        Step 3: segment-RAG boundary stats + ORIX mean orientation per segment
        Step 4: merge/prune via union-find:
            Rule A: merge adjacent segments if mean-orientation miso <= th_merge_deg
            Rule B: absorb tiny components (<= min_voxels) into best neighbor using root-mean miso
        Step 5: diagnostics (post-merge boundary stats)
        Step 6/7: write voxel grain_id + rebuild grains[] with ORIX mean orientation

    This function overwrites `json_path` (no new file).

    Parameters
    ----------
    json_path : str
        Path to the JSON file to update in-place.
    th_cut_deg : float
        Edge cut threshold in degrees (keep edges <= th_cut_deg).
    th_merge_deg : float
        Merge threshold in degrees for Rule A (mean-orientation miso).
    min_voxels : int
        Tiny component cutoff (<= min_voxels) for Rule B.
    print_topk_rag : int
        Number of highest-contact boundaries printed for diagnostics.
    tiny_cutoff : int
        Diagnostics: count segments with <= tiny_cutoff voxels after Step 2.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    None
        Updates the JSON file in-place.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=np.exceptions.ComplexWarning,
        module=r"quaternion(\.|$)",
    )
    # optional tqdm import only when needed (avoids hard dependency)
    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
        except Exception as e:
            raise ImportError(
                "progress=True requires tqdm. Install it via `pip install tqdm` or disable progress."
            ) from e

    # open Log file if requested (append mode) and route _p() to it
    log_fh = None
    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8")

    def _p(*args):
        if not verbose:
            return
        if log_fh is not None:
            print(*args, file=log_fh, flush=True)
        else:
            print(*args)

    try:
        # -----------------------------
        # helpers: union-find
        # -----------------------------
        def _uf_find(parent: np.ndarray, x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> bool:
            ra, rb = _uf_find(parent, a), _uf_find(parent, b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        # -----------------------------
        # Load JSON
        # -----------------------------
        _p("=== LOAD JSON ===")
        with open(json_path, "r") as f:
            data = json.load(f)

        if "microstructure" not in data:
            raise KeyError("Top-level key 'microstructure' not found.")

        micro = data["microstructure"]
        n_snaps = len(micro)
        if n_snaps < 2:
            _p("Nothing to do: only one snapshot found.")
            return

        # Snapshot 0 reference phase mapping
        snap0 = micro[0]
        if "grains" not in snap0:
            raise KeyError("Snapshot 0 must contain 'grains' for phase mapping.")
        grainid_to_phase = {int(g["grain_id"]): int(g.get("phase_id", 0)) for g in snap0["grains"]}

        _p("File:", json_path)
        _p("Number of snapshots:", n_snaps)
        _p("Will update snapshots:", list(range(1, n_snaps)))

        # -----------------------------
        # Process each snapshot INC = 1..last
        # -----------------------------
        # optionally wrap iteration with tqdm progress bar
        it = range(1, n_snaps)
        if progress:
            it = tqdm(it, total=max(n_snaps - 1, 0), desc="Segmenting snapshots", leave=True)

        for inc in it:
            _p("\n" + "=" * 90)
            _p(f"=== PROCESS SNAPSHOT {inc}  time={micro[inc].get('time', None)} ===")
            _p("=" * 90)

            snap = micro[inc]
            voxels = snap.get("voxels", None)
            if voxels is None:
                raise KeyError(f"Snapshot {inc} missing 'voxels'.")

            N = len(voxels)
            if N == 0:
                _p(f"Snapshot {inc}: empty voxels; skipping.")
                continue

            # -----------------------------------------------------------------------------
            # STEP 1 — BUILD VOXEL GRAPH (6-neigh) + MISORIENTATION EDGE WEIGHTS (ORIX)
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 1: VOXEL GRAPH + MISORIENTATION (ORIX) ===")

            idx_to_pos = {}
            eulers = np.empty((N, 3), dtype=float)
            voxel_vol = np.empty(N, dtype=float)
            old_gid = np.empty(N, dtype=int)

            for pos, v in enumerate(voxels):
                ijk = tuple(int(x) for x in v["voxel_index"])  # 1-based indices
                idx_to_pos[ijk] = pos
                eulers[pos, :] = np.asarray(v["orientation"], dtype=float)
                voxel_vol[pos] = float(v["voxel_volume"])
                old_gid[pos] = int(v["grain_id"])

            edges_pos_u = []
            edges_pos_v = []

            for (i, j, k), pos in idx_to_pos.items():
                nbs = [
                    (i - 1, j, k), (i + 1, j, k),
                    (i, j - 1, k), (i, j + 1, k),
                    (i, j, k - 1), (i, j, k + 1),
                ]
                for nb in nbs:
                    nb_pos = idx_to_pos.get(nb, None)
                    if nb_pos is None:
                        continue
                    if nb_pos > pos:  # store undirected edge once
                        edges_pos_u.append(pos)
                        edges_pos_v.append(nb_pos)

            edges_pos_u = np.asarray(edges_pos_u, dtype=int)
            edges_pos_v = np.asarray(edges_pos_v, dtype=int)
            E = len(edges_pos_u)

            _p("N voxels:", N)
            _p("E edges (6-neigh undirected):", E)

            cs = symmetry.Oh  # cubic m-3m
            O = Orientation.from_euler(eulers, symmetry=cs, direction="lab2crystal", degrees=False)
            w_deg = np.asarray(O[edges_pos_u].angle_with(O[edges_pos_v], degrees=True), dtype=float)

            _p("Misorientation (deg): min/mean/max =",
               float(w_deg.min()), float(w_deg.mean()), float(w_deg.max()))
            _p("=== STEP 1 COMPLETE ===")

            # -----------------------------------------------------------------------------
            # STEP 2 — INITIAL CUT + CONNECTED COMPONENTS => INITIAL SEGMENTS
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 2: INITIAL CUT SEGMENTATION ===")
            _p("TH_CUT (deg):", th_cut_deg)

            keep = (w_deg <= th_cut_deg)
            u = edges_pos_u[keep]
            v = edges_pos_v[keep]
            _p("edges kept:", int(keep.sum()), "edges cut:", int((~keep).sum()))

            rows = np.concatenate([u, v])
            cols = np.concatenate([v, u])
            vals = np.ones(rows.shape[0], dtype=np.int8)
            A = coo_matrix((vals, (rows, cols)), shape=(N, N))

            n_seg, labels = connected_components(A, directed=False, connection="weak")
            sizes = np.bincount(labels, minlength=n_seg)
            sizes_sorted = np.sort(sizes)[::-1]

            _p("n_initial_segments:", int(n_seg))
            _p("largest 10 sizes:", sizes_sorted[:10].tolist())
            _p("smallest 10 sizes:", sizes_sorted[-10:].tolist())
            _p(f"segments with <= {tiny_cutoff} voxels:", int((sizes <= tiny_cutoff).sum()))
            _p("=== STEP 2 COMPLETE ===")

            # -----------------------------------------------------------------------------
            # STEP 3 — BUILD SEGMENT-RAG + MEAN ORIENTATIONS (ORIX)
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 3: BUILD SEGMENT-RAG ===")
            _p("segments:", int(n_seg), "labels min/max:", int(labels.min()), int(labels.max()))

            seg_size = sizes.astype(int)

            seg_mean_O = [None] * n_seg
            for s in range(n_seg):
                mask = (labels == s)
                seg_mean_O[s] = O[mask].mean()

            # Boundary aggregation using voxel edges
            rag_contact = {}   # (sa,sb)->int
            rag_miso_sum = {}  # (sa,sb)->float (voxel-edge miso sum)

            for i in range(E):
                a = int(labels[int(edges_pos_u[i])])
                b = int(labels[int(edges_pos_v[i])])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                key = (a, b)
                rag_contact[key] = rag_contact.get(key, 0) + 1
                rag_miso_sum[key] = rag_miso_sum.get(key, 0.0) + float(w_deg[i])

            rag_miso_mean = {k: rag_miso_sum[k] / rag_contact[k] for k in rag_contact}
            _p("segment-RAG edges:", len(rag_contact))

            if print_topk_rag > 0:
                top_edges = sorted(rag_contact.items(), key=lambda kv: kv[1], reverse=True)[:print_topk_rag]
                _p(f"\nTop {print_topk_rag} neighbors by boundary contact count:")
                for (sa, sb), cnt in top_edges:
                    _p(f"  seg {sa} -- seg {sb} : contacts={cnt}, boundary_mean_miso_deg={rag_miso_mean[(sa, sb)]:.3f}")

            _p("=== STEP 3 COMPLETE ===")

            # -----------------------------------------------------------------------------
            # STEP 4 — MERGE / PRUNE USING UNION-FIND
            #   Rule A: merge neighbors if mean-orientation miso <= th_merge_deg
            #   Rule B: absorb tiny components (<= min_voxels) into best neighbor
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 4: MERGE / PRUNE SEGMENTS ===")
            _p("TH_MERGE (deg):", th_merge_deg, "MIN_VOXELS:", min_voxels)

            parent = np.arange(n_seg, dtype=int)
            rank = np.zeros(n_seg, dtype=int)

            # Rule A (UPDATED: require BOTH mean-orientation and boundary mean miso)
            merge_A = 0
            for (sa, sb) in rag_contact.keys():

                # 1) mean-orientation misorientation (deg)
                miso_meanO = float(
                    np.asarray(seg_mean_O[sa].angle_with(seg_mean_O[sb], degrees=True)).squeeze()
                )

                # 2) boundary mean misorientation (deg)
                boundary_mean_miso = float(rag_miso_mean[(sa, sb)])

                # Merge only if BOTH are small
                if (miso_meanO <= th_merge_deg) and (boundary_mean_miso <= th_merge_deg):
                    if _uf_union(parent, rank, sa, sb):
                        merge_A += 1

            _p("Rule A merges performed:", merge_A)

            # Build root members after Rule A
            root_members = {}
            for s in range(n_seg):
                r = _uf_find(parent, s)
                root_members.setdefault(r, []).append(s)

            # Root sizes
            root_size = {r: int(sum(seg_size[s] for s in members)) for r, members in root_members.items()}
            tiny_roots = [r for r, sz in root_size.items() if sz <= min_voxels]
            _p("tiny components after Rule A:", len(tiny_roots), "roots:", tiny_roots)

            # Root mean orientation (ORIX) from voxel sets
            root_mean_O = {}
            for r, members in root_members.items():
                mask = np.isin(labels, members)
                root_mean_O[r] = O[mask].mean()

            # Rule B: absorb each tiny root
            merge_B = 0
            for r0 in tiny_roots:
                r = _uf_find(parent, r0)  # update in case previous unions changed it
                if r not in root_size:
                    continue
                if root_size[r] > min_voxels:
                    continue

                best = None  # (miso_root_deg, -contact, nbr_root)

                for (sa, sb), contact in rag_contact.items():
                    ra = _uf_find(parent, sa)
                    rb = _uf_find(parent, sb)
                    if ra == rb:
                        continue
                    if ra != r and rb != r:
                        continue

                    nbr = rb if ra == r else ra

                    miso_root = float(np.asarray(root_mean_O[r].angle_with(root_mean_O[nbr], degrees=True)).squeeze())
                    score = (miso_root, -int(contact), nbr)
                    if (best is None) or (score < best):
                        best = score

                if best is None:
                    continue

                _, _, nbr_root = best
                if _uf_union(parent, rank, r, nbr_root):
                    merge_B += 1

                    # Update bookkeeping (sizes + mean orientations) for stability
                    newr = _uf_find(parent, r)
                    oldr = nbr_root if newr == r else r  # whichever got absorbed

                    root_size[newr] = root_size.get(r, 0) + root_size.get(nbr_root, 0)
                    root_size.pop(oldr, None)

                    # Recompute mean orientation for new root (safe and correct)
                    # (voxel mask uses ORIGINAL segment labels; rebuild members by scanning is OK at this scale)
                    members_new = [s for s in range(n_seg) if _uf_find(parent, s) == newr]
                    mask_new = np.isin(labels, members_new)
                    root_mean_O[newr] = O[mask_new].mean()

            _p("Rule B merges performed:", merge_B)

            # Finalize merged voxel labels (compact 0..n_final-1)
            root_to_new = {}
            labels_merged = np.empty_like(labels)
            next_id = 0
            for pos in range(N):
                s = int(labels[pos])
                r = _uf_find(parent, s)
                if r not in root_to_new:
                    root_to_new[r] = next_id
                    next_id += 1
                labels_merged[pos] = root_to_new[r]

            n_final = next_id
            final_sizes = np.bincount(labels_merged, minlength=n_final)

            _p("\nAfter merging:")
            _p("n_final_segments:", n_final)
            _p("sum(final_sizes) == N:", int(final_sizes.sum()), "==", N)
            _p("sorted final sizes:", np.sort(final_sizes)[::-1].tolist())
            _p("segments with <= 5 voxels:", int((final_sizes <= 5).sum()))
            _p("=== STEP 4 COMPLETE ===")

            # -----------------------------------------------------------------------------
            # STEP 5 — VALIDATE (post-merge boundary diagnostic)
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 5: POST-MERGE RAG VALIDATION ===")
            rag2_contact = {}
            rag2_sum = {}

            for i in range(E):
                a = int(labels_merged[int(edges_pos_u[i])])
                b = int(labels_merged[int(edges_pos_v[i])])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                key = (a, b)
                rag2_contact[key] = rag2_contact.get(key, 0) + 1
                rag2_sum[key] = rag2_sum.get(key, 0.0) + float(w_deg[i])

            rag2_mean = {k: rag2_sum[k] / rag2_contact[k] for k in rag2_contact}
            _p("post-merge RAG edges:", len(rag2_contact))

            if print_topk_rag > 0:
                top2 = sorted(rag2_contact.items(), key=lambda kv: kv[1], reverse=True)[:print_topk_rag]
                _p(f"\nTop {print_topk_rag} post-merge neighbors by contact:")
                for (a, b), cnt in top2:
                    _p(f"  seg {a} -- seg {b} : contacts={cnt}, boundary_mean_miso_deg={rag2_mean[(a,b)]:.3f}")

            _p("=== STEP 5 COMPLETE ===")

            # -----------------------------------------------------------------------------
            # STEP 6/7 — UPDATE SNAPSHOT: voxels[*].grain_id + rebuild grains[]
            # -----------------------------------------------------------------------------
            _p("\n=== STEP 6/7: UPDATE JSON (voxels + grains) ===")

            # Write new grain_id (1-based) into voxels
            new_gid = labels_merged.astype(int) + 1
            for pos in range(N):
                voxels[pos]["grain_id"] = int(new_gid[pos])

            n_grains_out = int(new_gid.max())
            _p("n_final grains:", n_grains_out)
            _p("first 10 new grain_id:", new_gid[:10].tolist())

            if progress:
                try:
                    it.set_postfix(grains=n_grains_out, refresh=True)
                except Exception:
                    pass

            # Rebuild grains[]
            new_grains = []
            for k in range(1, n_grains_out + 1):
                mask = (new_gid == k)
                if not np.any(mask):
                    continue

                gv = float(voxel_vol[mask].sum())

                Omean = O[mask].mean()
                e_mean = np.asarray(Omean.to_euler(), dtype=float).reshape(3)

                phases = [grainid_to_phase[int(og)] for og in old_gid[mask] if int(og) in grainid_to_phase]
                phase_id = Counter(phases).most_common(1)[0][0] if phases else 0

                new_grains.append({
                    "grain_id": int(k),
                    "phase_id": int(phase_id),
                    "grain_volume": gv,
                    "orientation": [float(e_mean[0]), float(e_mean[1]), float(e_mean[2])],
                })

            new_grains.sort(key=lambda g: g["grain_id"])
            snap["grains"] = new_grains

            _p("rebuilt grains[] -> n_grains:", len(new_grains))
            _p("total grain_volume:", float(sum(g["grain_volume"] for g in new_grains)))
            _p("=== STEP 6/7 COMPLETE ===")

        # -----------------------------
        # Overwrite the same JSON file
        # -----------------------------
        _p("\n=== WRITE JSON (IN-PLACE) ===")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        _p("Updated in-place:", json_path)
    finally:
        if log_fh is not None:
            log_fh.close()


def step1_extract_gid_arrays(
        json_path: str,
        t: int,
        verbose: bool = True,
        log_fh=None,) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """
    Step 1: Build aligned grain-id arrays for snapshots t and t+1 using voxel_index.

    Parameters
    ----------
    json_path : str
        Path to the microstructure JSON.
    t : int
        Snapshot index (tracks from t -> t+1).
    verbose : bool
        If True, prints alignment summary.

    Returns
    -------
    gid_t : np.ndarray, shape (N,)
        Grain IDs for snapshot t aligned by voxel positions.
    gid_tp1 : np.ndarray, shape (N,)
        Grain IDs for snapshot t+1 aligned by voxel positions (same ordering as gid_t).
    idx_list : list[tuple[int,int,int]]
        The ordered voxel_index list that defines the alignment.
    """
    def _p(msg: str):
        if not verbose:
            return
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
        else:
            print(msg)

    with open(json_path, "r") as f:
        data = json.load(f)

    micro = data["microstructure"]
    snap_t = micro[t]
    snap_tp1 = micro[t + 1]

    vox_t = snap_t["voxels"]
    vox_tp1 = snap_tp1["voxels"]

    # Build index -> grain_id dict for each snapshot
    map_t = {tuple(v["voxel_index"]): int(v["grain_id"]) for v in vox_t}
    map_tp1 = {tuple(v["voxel_index"]): int(v["grain_id"]) for v in vox_tp1}

    # Ensure same voxel set (critical assumption)
    idx_set_t = set(map_t.keys())
    idx_set_tp1 = set(map_tp1.keys())
    if idx_set_t != idx_set_tp1:
        missing_in_tp1 = idx_set_t - idx_set_tp1
        missing_in_t = idx_set_tp1 - idx_set_t
        raise ValueError(
            f"Voxel index sets differ between snapshots {t} and {t + 1}.\n"
            f"Missing in t+1: {len(missing_in_tp1)}\n"
            f"Missing in t  : {len(missing_in_t)}"
        )
    # Define a stable order (sort by k, then j, then i OR your preferred)
    idx_list = sorted(idx_set_t)
    gid_t = np.array([map_t[idx] for idx in idx_list], dtype=np.int32)
    gid_tp1 = np.array([map_tp1[idx] for idx in idx_list], dtype=np.int32)

    _p(f"\n=== STEP 1: GID ALIGNMENT CHECK @ snapshot {t} ===")
    _p(f"Aligned voxel count        : {len(idx_list):,}")
    _p(f"Unique grains @ t          : {len(set(gid_t))}")
    _p(f"Unique grains @ t+1        : {len(set(gid_tp1))}")
    _p("Voxel index alignment      : ✅ PASSED\n")

    return gid_t, gid_tp1, idx_list


def step2_overlap_counts(gid_t: np.ndarray, gid_tp1: np.ndarray, verbose: bool = True, log_fh=None,):
    """
    Step 2: Compute overlap counts C(g_t, g_tp1).

    Parameters
    ----------
    gid_t, gid_tp1 : np.ndarray, shape (N,)
        Aligned grain-id arrays for snapshots t and t+1.
    verbose : bool
        If True, prints overlap statistics and match summaries.

    Returns
    -------
    C : np.ndarray, shape (Gt+1, Gtp1+1)
        Overlap count matrix where C[i,j] = number of voxels shared.
    row_sum : np.ndarray
        Total voxels per grain in t (row sums).
    col_sum : np.ndarray
        Total voxels per grain in t+1 (col sums).
    top_pairs : list[tuple]
        Sorted list of (count, i, j) for the largest overlaps.
    """

    def _p(msg: str):
        if not verbose:
            return
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
        else:
            print(msg)

    if gid_t.shape != gid_tp1.shape:
        raise ValueError("gid arrays must have same shape")

    Gt = int(gid_t.max())
    Gp = int(gid_tp1.max())

    # Build overlap matrix (IDs assumed positive ints)
    C = np.zeros((Gt + 1, Gp + 1), dtype=np.int32)

    # Fast accumulation (vectorized via bincount on combined index)
    # combined_index = i*(Gp+1) + j
    combined = gid_t.astype(np.int64) * (Gp + 1) + gid_tp1.astype(np.int64)
    bc = np.bincount(combined, minlength=(Gt + 1) * (Gp + 1))
    C[:, :] = bc.reshape((Gt + 1, Gp + 1))

    row_sum = C.sum(axis=1)  # voxels per grain at t
    col_sum = C.sum(axis=0)  # voxels per grain at t+1

    # Extract top overlaps ignoring 0 row/col
    pairs = []
    for i in range(1, Gt + 1):
        for j in range(1, Gp + 1):
            c = int(C[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    _p("\n=== STEP 2: OVERLAP MATRIX SUMMARY ===")
    _p(f"Overlap matrix shape         : {C.shape} (row 0 / col 0 unused)")
    _p(f"Total overlapping voxels     : {int(C.sum()):,}")
    _p(f"Grains @ t (nonzero rows)    : {(row_sum[1:] > 0).sum()}  | Max grain ID: {Gt}")
    _p(f"Grains @ t+1 (nonzero cols)  : {(col_sum[1:] > 0).sum()}  | Max grain ID: {Gp}")

    _p(f"\n Overlaps (count, g_t -> g_t+1, frac_of_t, frac_of_t+1):")

    for c, i, j in pairs:
        frac_t = c / row_sum[i] if row_sum[i] else 0.0
        frac_p = c / col_sum[j] if col_sum[j] else 0.0
        _p(f"  {c:4d}   {i:2d} → {j:2d}    frac_t={frac_t:6.3f}    frac_tp1={frac_p:6.3f}")

    _p("\nBest match per grain in t:")
    for i in range(1, len(row_sum)):
        if row_sum[i] == 0:
            continue
        j_best = int(np.argmax(C[i, :]))
        c_best = int(C[i, j_best])
        frac_t = c_best / row_sum[i]
        frac_p = c_best / col_sum[j_best] if col_sum[j_best] else 0.0
        _p(f"  g_t={i:2d}: best g_t+1={j_best:2d}  count={c_best:4d}  frac_t={frac_t:6.3f}  frac_p={frac_p:6.3f}")

    return C, row_sum, col_sum, pairs


def step3_track_from_overlap(C, row_sum, col_sum, *,
                             tau_parent=0.30,  # child must come >=30% from a parent to count as parent
                             tau_child=0.30,  # parent must send >=30% of its mass to child to count as child
                             allow_multi_parent=True,
                             verbose: bool = True,
                             log_fh=None,):
    """
    Build tracking relations from overlap matrix.

    Returns
    -------
    parents_of_child : dict[int, list[tuple[int, float, float, int]]]
        child j -> list of (parent i, frac_of_parent, frac_of_child, count)
    best_parent : dict[int, int]
        child j -> parent i with max overlap
    best_child  : dict[int, int]
        parent i -> child j with max overlap
    events : dict[str, list]
        merge/split/birth/death/continue records
    """
    def _p(msg: str):
        if not verbose:
            return
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
        else:
            print(msg)

    Gt = C.shape[0] - 1
    Gp1 = C.shape[1] - 1

    # Candidate parent links per child
    parents_of_child = defaultdict(list)
    children_of_parent = defaultdict(list)

    for i in range(1, Gt + 1):
        if row_sum[i] == 0:
            continue
        for j in range(1, Gp1 + 1):
            c = int(C[i, j])
            if c == 0:
                continue
            f_parent = c / row_sum[i]
            f_child = c / col_sum[j] if col_sum[j] else 0.0

            # Keep links that are meaningful from either side
            if (f_parent >= tau_child) or (f_child >= tau_parent):
                parents_of_child[j].append((i, f_parent, f_child, c))
                children_of_parent[i].append((j, f_parent, f_child, c))

    # Best matches by raw overlap
    best_parent = {}
    for j in range(1, Gp1 + 1):
        i_best = int(np.argmax(C[:, j]))
        best_parent[j] = i_best if i_best != 0 else None

    best_child = {}
    for i in range(1, Gt + 1):
        j_best = int(np.argmax(C[i, :]))
        best_child[i] = j_best if j_best != 0 else None

    # Event classification
    events = {"continue": [], "merge": [], "split": [], "birth": [], "death": []}

    # Births: child has no meaningful parent links
    for j in range(1, Gp1 + 1):
        if col_sum[j] == 0:
            continue
        if j not in parents_of_child:
            events["birth"].append(j)

    # Deaths: parent has no meaningful child links
    for i in range(1, Gt + 1):
        if row_sum[i] == 0:
            continue
        if i not in children_of_parent:
            events["death"].append(i)

    # Continuations + merges (child perspective)
    for j, plist in parents_of_child.items():
        # sort by overlap count descending
        plist = sorted(plist, key=lambda x: x[3], reverse=True)
        if len(plist) == 1:
            i = plist[0][0]
            # mutual best → strong continuation candidate
            if best_child.get(i, None) == j and best_parent.get(j, None) == i:
                events["continue"].append((i, j, plist[0]))
        else:
            # multiple parents → merge
            if allow_multi_parent:
                parents = [p[0] for p in plist]
                events["merge"].append((parents, j, plist))

    # Splits (parent perspective): one parent has multiple strong children
    for i, clist in children_of_parent.items():
        # keep only strong-from-parent links
        strong = [x for x in clist if x[1] >= tau_child]
        if len(strong) >= 2:
            strong = sorted(strong, key=lambda x: x[3], reverse=True)
            children = [x[0] for x in strong]
            events["split"].append((i, children, strong))

    _p("\n=== STEP 3: TRACKING EVENT SUMMARY ===")
    _p(f"Continues : {len(events['continue'])}")
    _p(f"Merges    : {len(events['merge'])}")
    _p(f"Splits    : {len(events['split'])}")
    _p(f"Births    : {len(events['birth'])}")
    _p(f"Deaths    : {len(events['death'])}")

    return parents_of_child, best_parent, best_child, events

def step4_assign_tracked_ids(
        pairs, C, row_sum, col_sum,
        events,
        parents_of_child,
        verbose: bool = True,
        log_fh=None,):
    """
    Step 4: Assign tracked grain IDs at t+1 using Hungarian matching and event logic.

    Parameters
    ----------
    pairs : list of tuples
        Each tuple is (count, grain_t, grain_t+1), sorted by overlap.
    C : np.ndarray
        Overlap matrix where C[i, j] is the count of voxels shared between grain i at t and grain j at t+1.
    row_sum : np.ndarray
        Total voxel count per grain at t.
    col_sum : np.ndarray
        Total voxel count per grain at t+1.
    events : dict[str, list]
        Event records from tracking: contains 'split', 'merge', 'birth', etc.
    parents_of_child : dict[int, list[tuple]]
        Mapping from child grain ID at t+1 to list of parent grain contributions from t.
    verbose : bool, optional
        Whether to print detailed summary and logs (default: True).

    Returns
    -------
    final_ids : dict[int, int]
        Mapping of grain_id at t+1 → tracked global grain ID.
    """
    def _p(msg: str):
        if not verbose:
            return
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
        else:
            print(msg)

    # Step 1: extract unique grain IDs
    grains_t = sorted(set(i for _, i, _ in pairs))
    grains_tp1 = sorted(set(j for _, _, j in pairs))

    id_to_row = {gid: idx for idx, gid in enumerate(grains_t)}
    id_to_col = {gid: idx for idx, gid in enumerate(grains_tp1)}

    # Step 2: build cost matrix for Hungarian matching
    M = np.full((len(grains_t), len(grains_tp1)), fill_value=10.0)
    for c, i, j in pairs:
        M[id_to_row[i], id_to_col[j]] = -c  # maximize overlap

    row_idx, col_idx = linear_sum_assignment(M)

    # Step 3: extract assigned pairs
    assigned_pairs = []
    for r, c in zip(row_idx, col_idx):
        i = grains_t[r]
        j = grains_tp1[c]
        cost = M[r, c]
        if cost < 1.0:
            assigned_pairs.append((i, j, cost))

    _p("\n=== STEP 4: Hungarian Matches ===")
    for i, j, cost in assigned_pairs:
        _p(f"  {i}@t → {j}@t+1 | cost={cost:.3f}")

    final_ids = {}
    assigned_ids = set()
    max_gid_t = max(grains_t)
    next_id = max_gid_t + 1

    # 1. Assign IDs from Hungarian matches
    for i, j, _ in assigned_pairs:
        if j not in final_ids and i not in assigned_ids:
            final_ids[j] = i
            assigned_ids.add(i)

    # 2. Splits
    for i, children, info in events.get("split", []):
        children = sorted(children, key=lambda j: C[i, j], reverse=True)
        for idx, j in enumerate(children):
            if j in final_ids:
                continue
            if i not in assigned_ids:
                final_ids[j] = i
                assigned_ids.add(i)
            else:
                final_ids[j] = next_id
                assigned_ids.add(next_id)
                next_id += 1

    # 3. Merges
    for parents, j, info in events.get("merge", []):
        if j in final_ids:
            continue
        dominant_i = max(info, key=lambda x: x[3])[0]
        if dominant_i not in assigned_ids:
            final_ids[j] = dominant_i
            assigned_ids.add(dominant_i)
        else:
            final_ids[j] = next_id
            assigned_ids.add(next_id)
            next_id += 1

    # 4. Births
    for j in events.get("birth", []):
        if j not in final_ids:
            final_ids[j] = next_id
            assigned_ids.add(next_id)
            next_id += 1

    # 5. Fallback
    for j in grains_tp1:
        if j not in final_ids:
            final_ids[j] = next_id
            assigned_ids.add(next_id)
            next_id += 1

    # === Summary Output ===
    _p("\n=== Final Tracked IDs at t+1 ===")
    for j in sorted(final_ids):
        _p(f"  Grain {j}@t+1 → Tracked ID {final_ids[j]}")

    _p("\n=== Parent Summary ===")
    for j in sorted(grains_tp1):
        parents = parents_of_child.get(j, [])
        tracked_id = final_ids[j]
        if not parents:
            status = "birth"
            parent_str = "-"
        elif len(parents) == 1:
            status = "1 parent"
            parent_str = f"{parents[0][0]}"
        else:
            status = f"{len(parents)} parents (merge)"
            parent_str = ", ".join(str(p[0]) for p in parents)
        _p(f"  Grain {j}@t+1 → ID {tracked_id}, {status}, from parent(s): {parent_str}")

    # Duplicates
    duplicates = [k for k, v in Counter(final_ids.values()).items() if v > 1]
    if duplicates:
        _p(f"\n⚠️  Duplicated tracked IDs at t+1: {duplicates}")
    else:
        _p("\n✅ No duplicated tracked IDs at t+1")

    return final_ids

def step5_update_tracked_ids_in_json(
        json_path, t,
        final_ids,
        parents_of_child,
        verbose: bool = True,
        log_fh=None,):
    """
    Step 5: Update JSON file with new tracked grain IDs and parent relationships.

    Parameters
    ----------
    json_path : str
        Path to the microstructure JSON file.
    t : int
        Snapshot index to update (this modifies snapshot at t+1).
    final_ids : dict[int, int]
        Mapping of grain_id at t+1 to globally tracked IDs.
    parents_of_child : dict[int, list[tuple]]
        Mapping from child grain ID to parent grain info (from t).
    """
    def _p(msg: str):
        if not verbose:
            return
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
        else:
            print(msg)

    with open(json_path, 'r') as f:
        data = json.load(f)

    micro = data["microstructure"]
    snapshot = micro[t + 1]  # we are updating t+1

    # --- Update grains ---
    updated_grains = []
    for grain in snapshot["grains"]:
        old_gid = grain["grain_id"]
        if old_gid not in final_ids:
            continue
        new_gid = final_ids[old_gid]
        new_grain = grain.copy()
        new_grain["grain_id"] = new_gid

        parents = parents_of_child.get(old_gid, [])
        if len(parents) == 1:
            new_grain["parent_grain_id"] = parents[0][0]
        elif len(parents) > 1:
            sorted_parents = sorted(parents, key=lambda x: x[3], reverse=True)
            new_grain["parent_grain_id"] = [p[0] for p in sorted_parents]

        updated_grains.append(new_grain)

    snapshot["grains"] = updated_grains

    # --- Update voxels ---
    for voxel in snapshot["voxels"]:
        old_gid = voxel["grain_id"]
        if old_gid in final_ids:
            voxel["grain_id"] = final_ids[old_gid]

    # --- Save back to file ---
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    _p(f"✅ Snapshot {t+1} successfully updated and saved to: {json_path}")


def track_all_grains_across_snapshots(
        json_path: str,
        *,
        log_path: str | None = None,
        progress: bool = False,
        verbose: bool = True,) -> None:
    """
    Run the full grain tracking pipeline across all snapshot pairs in the microstructure JSON.

    Parameters
    ----------
    json_path : str
        Path to the input JSON file containing microstructure data.
    """
    if progress:
        try:
            from tqdm.auto import tqdm
        except Exception as e:
            raise ImportError(
                "progress=True requires tqdm. Install it via `pip install tqdm` or disable progress."
            ) from e

    log_fh = None
    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8")

    def _p(*args):
        if not verbose:
            return
        if log_fh is not None:
            print(*args, file=log_fh, flush=True)
        else:
            print(*args)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        num_snapshots = len(data["microstructure"])
        _p(f"📦 Starting grain tracking on {num_snapshots} snapshots...")

        # progress bar over snapshot-pairs (t -> t+1)
        it = range(num_snapshots - 1)
        if progress:
            it = tqdm(it, total=max(num_snapshots - 1, 0), desc="Tracking grain IDs", leave=True)

        for t in it:
            _p(f"\n\n🚩 Processing snapshot pair: {t} → {t+1}")

            # Step 1: Extract aligned GID arrays
            gid_t, gid_tp1, idx_list = step1_extract_gid_arrays(json_path, t, verbose=verbose, log_fh=log_fh)

            # Step 2: Compute overlap matrix
            C, row_sum, col_sum, pairs = step2_overlap_counts(gid_t, gid_tp1, verbose=verbose, log_fh=log_fh)

            # Step 3: Track from overlap
            parents_of_child, best_parent, best_child, events = step3_track_from_overlap(C, row_sum, col_sum,
                                                                                         verbose=verbose, log_fh=log_fh)

            # Step 4: Assign tracked IDs
            final_ids = step4_assign_tracked_ids(pairs, C, row_sum, col_sum, events, parents_of_child,
                                                 verbose=verbose, log_fh=log_fh)

            # Step 5: Update JSON file in-place for snapshot t+1
            step5_update_tracked_ids_in_json(json_path, t, final_ids, parents_of_child, verbose=verbose, log_fh=log_fh)

            # optional live postfix info on progress bar (no stdout spam)
            if progress:
                try:
                    n_grains_t = int(gid_t.max()) if gid_t.size else 0
                    n_grains_tp1 = int(gid_tp1.max()) if gid_tp1.size else 0
                    it.set_postfix(
                        grains_t=n_grains_t,
                        grains_tp1=n_grains_tp1,
                        merges=len(events.get("merge", [])),
                        splits=len(events.get("split", [])),
                        refresh=True,
                    )
                except Exception:
                    pass

        _p("\n✅ Grain tracking and ID propagation completed across all snapshots.")

    finally:
        if log_fh is not None:
            log_fh.close()

def add_ipf_color(
    json_path: str | Path,
    l: Sequence[float] = (1, 0, 0),
    *,
    out_path: str | Path | None = None,
) -> Path:
    """
    Add per-voxel IPF RGB colors to every snapshot in the JSON.

    - Reads symmetry from: data["phases"][0]["orientation"]["crystal_symmetry_group"]
    - Computes IPF color along lab direction `l` (default (1,0,0))
    - Writes into each voxel:
        "IPFcolor_(l0 l1 l2)": [R, G, B]   (8-bit)

    Assumptions:
    - Single phase (data["phases"] has exactly one element).
    - Voxel orientation stored as Euler angles [phi1, Phi, phi2] in radians.
    """
    from orix import plot
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d

    json_path = Path(json_path)
    out_path = Path(out_path) if out_path is not None else json_path

    # --- Load JSON ---
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    # --- Extract symmetry (your JSON stores phases as a list) ---
    phases = data["phase"]
    if not isinstance(phases, list) or len(phases) != 1:
        raise ValueError(f'Expected one phase in data["phase"], got {type(phases)} with len={len(phases) if isinstance(phases, list) else "?"}.')

    sym_str = phases[0]["orientation"]["crystal_symmetry_group"]

    # Minimal mapping (extend later if needed)
    if sym_str != "m-3m":
        raise ValueError(f'Unsupported symmetry "{sym_str}" in this simplified function. Expected "m-3m".')
    laue_sym = symmetry.Oh  # cubic m-3m

    # --- Build IPF color key for direction `l` ---
    l = np.asarray(l, dtype=float)
    if l.shape != (3,) or np.allclose(l, 0.0):
        raise ValueError("`l` must be a non-zero length-3 vector, e.g. (1,0,0).")

    ckey = plot.IPFColorKeyTSL(symmetry=laue_sym, direction=Vector3d(l))

    # Store using your requested label style
    ipf_key = f"IPFcolor_({int(l[0])} {int(l[1])} {int(l[2])})"

    # --- Process snapshots (vectorized per snapshot) ---
    for snap in data["microstructure"]:
        voxels = snap.get("voxels", [])
        if not voxels:
            continue

        # Collect voxel Euler angles into (N,3)
        eulers = np.asarray([v["orientation"] for v in voxels], dtype=float)
        if eulers.ndim != 2 or eulers.shape[1] != 3:
            raise ValueError("Each voxel['orientation'] must be a length-3 Euler list.")

        # Euler -> Orientation (radians)
        O = Orientation.from_euler(eulers, symmetry=laue_sym, degrees=False)

        # Orientation -> RGB floats in [0,1] -> uint8 [0..255]
        rgb = ckey.orientation2color(O)
        rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

        # Write back to each voxel
        for v, c in zip(voxels, rgb_u8):
            v[ipf_key] = c.tolist()

    # --- Save JSON ---
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path

def extract_initial_texture_dict(
    json_path: Union[str, Path],
    *,
    ebsd: object = None,
    snap_idx_initial: int = 0,
    pole_vector: Tuple[int, int, int] = (0, 0, 1),
    initial_orientation_source: str = "ebsd",  # "ebsd" or "json"
) -> Dict[str, Any]:
    """
    Build the texture payload for the initial microstructure snapshot.

    Returns
    -------
    dict
        {'O': O0, 't_all': t0_all, 'n': n0, 'phase': phase, 'pole_vector': tuple}
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    snaps = data.get("microstructure", [])
    if not isinstance(snaps, list) or len(snaps) == 0:
        raise ValueError("No snapshots found under key 'microstructure'.")
    if not (-len(snaps) <= snap_idx_initial < len(snaps)):
        raise ValueError(f"snap_idx_initial={snap_idx_initial} out of range for {len(snaps)} snapshots.")

    snap0 = snaps[snap_idx_initial]
    grains0 = snap0.get("grains", [])
    if not grains0:
        raise ValueError("No grains found in the initial snapshot.")

    # --- phase mapping (EBSD) ---
    # phase_id in your data object should correspond to ebsd.emap.phases.ids[*]
    try:
        phase = ebsd.emap.phases[ebsd.emap.phases.ids[0]]
    except Exception as exc:
        raise ValueError("Could not resolve `phase` from EBSD (`ebsd.emap.phases[...]`).") from exc

    # --- initial orientations ---
    src = str(initial_orientation_source).strip().lower()

    if src == "ebsd":
        if ebsd is None:
            raise ValueError("`ebsd` must be provided when initial_orientation_source='ebsd'.")
        ori_obj = ebsd.ms_data[0]["ori"]
        if not isinstance(ori_obj, Orientation):
            q = np.asarray(ori_obj, float).reshape(-1, 4)
            ori_obj = Orientation.from_quaternion(q)
        O0 = Orientation.from_euler(ori_obj.to_euler(degrees=False), degrees=False)
        n0 = int(O0.size) if hasattr(O0, "size") else len(ori_obj)

    elif src == "json":
        E0 = np.asarray([g["orientation"] for g in grains0], dtype=float)
        if E0.ndim != 2 or E0.shape[1] != 3:
            raise ValueError("JSON grain orientations must be (N,3) Euler [phi1,Phi,phi2] in radians.")
        O0 = Orientation.from_euler(E0, degrees=False)
        n0 = int(O0.size) if hasattr(O0, "size") else len(E0)

    else:
        raise ValueError("`initial_orientation_source` must be 'ebsd' or 'json'.")

    # --- pole projection set ---
    t0 = Miller(uvw=list(pole_vector), phase=phase).symmetrise(unique=True)
    t0_all = O0.inv().outer(t0)

    return {
        "O": O0,
        "t_all": t0_all,
        "n": n0,
        "phase": phase,
        "pole_vector": tuple(pole_vector),
    }

def extract_regridded_texture_dict(
    json_path: Union[str, Path],
    *,
    ebsd: Any,
    pole_vector: Tuple[int, int, int] = (0, 0, 1),
    # selection policy:
    # - if snap_idx_regridded is given: use it, but enforce grid.status == "undeformed"
    # - otherwise: find the LAST snapshot with time==target_time AND status=="undeformed"
    snap_idx_regridded: int | None = -1,
    target_time: float | None = None,
) -> Dict[str, Any]:
    """
    Build the texture payload for a regridded (undeformed grid) snapshot.

    Returns
    -------
    dict
        {'O': OR, 't_all': tR_all, 'n': nR, 'phase': phase, 'pole_vector': tuple}
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    snaps = data.get("microstructure", [])
    if not isinstance(snaps, list) or len(snaps) == 0:
        raise ValueError("No snapshots found under key 'microstructure'.")

    # --- phase mapping (EBSD) ---
    try:
        phase = ebsd.emap.phases[ebsd.emap.phases.ids[0]]
    except Exception as exc:
        raise ValueError("Could not resolve `phase` from EBSD (`ebsd.emap.phases[...]`).") from exc

    def _is_undeformed(snap: dict) -> bool:
        grid = snap.get("grid", {})
        status = str(grid.get("status", "")).strip().lower()
        return status == "undeformed"

    # --- choose snapshot ---
    snapR = None

    if snap_idx_regridded is not None:
        # explicit index selection
        idx = snap_idx_regridded
        if not (-len(snaps) <= idx < len(snaps)):
            raise ValueError(f"snap_idx_regridded={idx} out of range for {len(snaps)} snapshots.")
        candidate = snaps[idx]
        if not _is_undeformed(candidate):
            raise RuntimeError(
                f"Requested snapshot_index={idx} but grid.status='{candidate.get('grid', {}).get('status')}', "
                "expected 'undeformed' for regridded texture extraction."
            )
        snapR = candidate

    else:
        # search policy: last snapshot matching time (if provided) and undeformed
        if target_time is None:
            # last undeformed snapshot
            for s in reversed(snaps):
                if _is_undeformed(s):
                    snapR = s
                    break
        else:
            # last undeformed snapshot at target_time (exact match with tolerance)
            tol = 1e-12
            for s in reversed(snaps):
                t = s.get("time", None)
                if t is None:
                    continue
                if abs(float(t) - float(target_time)) <= tol and _is_undeformed(s):
                    snapR = s
                    break

        if snapR is None:
            raise RuntimeError(
                "Could not find an undeformed regridded snapshot with the requested selection policy "
                f"(target_time={target_time})."
            )

    voxelsR = snapR.get("voxels", [])
    if not voxelsR:
        raise ValueError("No voxels found in the selected regridded snapshot.")

    # --- orientations from voxel Euler angles ---
    ER = np.asarray([v["orientation"] for v in voxelsR], dtype=float)
    if ER.ndim != 2 or ER.shape[1] != 3:
        raise ValueError("Regridded voxel orientations must be (N,3) Euler [phi1,Phi,phi2] in radians.")

    OR = Orientation.from_euler(ER, degrees=False)
    nR = int(OR.size) if hasattr(OR, "size") else len(ER)

    # --- pole projection set ---
    tR = Miller(uvw=list(pole_vector), phase=phase).symmetrise(unique=True)
    tR_all = OR.inv().outer(tR)

    return {
        "O": OR,
        "t_all": tR_all,
        "n": nR,
        "phase": phase,
        "pole_vector": tuple(pole_vector),
    }
