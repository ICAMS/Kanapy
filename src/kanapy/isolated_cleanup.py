"""
Resolve graph regions without same phase neighbors after normal phase cleanup.
"""

import heapq

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _prepare_shared_boundaries(graph):
    """
    Measure shared pixel sides while preserving existing same phase edges.

    Parameters
    ----------
    graph : networkx.Graph
        Grain graph containing a rectangular label map and physical spacing.
    """
    for first, second, data in graph.edges(data=True):
        same_phase = graph.nodes[first]["phase_id"] == graph.nodes[second]["phase_id"]
        data["boundary_type"] = "same_phase" if same_phase else "interphase"
        data["is_phase_boundary"] = not same_phase
        data["shared_boundary_length"] = 0.0

    labels = graph.graph["label_map"]
    dx = float(graph.graph.get("dx", 1.0))
    dy = float(graph.graph.get("dy", 1.0))
    for first, second, side_length in (
        (labels[:, :-1], labels[:, 1:], dy),
        (labels[:-1, :], labels[1:, :], dx),
    ):
        contact = (first > 0) & (second > 0) & (first != second)
        pairs = np.column_stack((first[contact], second[contact]))
        pairs.sort(axis=1)
        pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (first_id, second_id), count in zip(pairs, counts):
            first_id, second_id = int(first_id), int(second_id)
            if not graph.has_edge(first_id, second_id):
                same_phase = (
                    graph.nodes[first_id]["phase_id"]
                    == graph.nodes[second_id]["phase_id"]
                )
                graph.add_edge(
                    first_id,
                    second_id,
                    boundary_type="same_phase" if same_phase else "interphase",
                    is_phase_boundary=not same_phase,
                    shared_boundary_length=0.0,
                )
            graph[first_id][second_id]["shared_boundary_length"] += float(count * side_length)

    for first, second, data in list(graph.edges(data=True)):
        if data["boundary_type"] == "interphase" and data["shared_boundary_length"] == 0:
            graph.remove_edge(first, second)


def _boundary_pixel_count(graph, node_id, connectivity):
    """
    Count region pixels exposed to another label or the edge of the map.

    Parameters
    ----------
    graph : networkx.Graph
        Grain graph with pixel membership and labels.
    node_id : int
        Region whose boundary fraction is needed.
    connectivity : int
        Four or eight neighbor convention used for the boundary fraction.

    Returns
    -------
    int
        Number of boundary pixels, counting each pixel once.
    """
    labels = graph.graph["label_map"]
    rows, columns = np.unravel_index(graph.nodes[node_id]["pixels"], labels.shape)
    boundary = np.zeros(rows.size, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity not in (1, 4):
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for row_offset, column_offset in offsets:
        next_rows = rows + row_offset
        next_columns = columns + column_offset
        inside = (
            (next_rows >= 0) & (next_rows < labels.shape[0])
            & (next_columns >= 0) & (next_columns < labels.shape[1])
        )
        boundary |= ~inside
        boundary[inside] |= labels[next_rows[inside], next_columns[inside]] != node_id
    return int(np.count_nonzero(boundary))


def _invalid_hull_reason(graph, node_id, minimum_points):
    """
    Return a reason when grain pixel centers cannot form a valid hull.

    Preserve the graph's geometry convention when storing a valid hull.

    Parameters
    ----------
    graph : networkx.Graph
        Grain graph with spatial metadata.
    node_id : int
        Region to validate.
    minimum_points : int
        Required number of pixel centers.

    Returns
    -------
    str or None
        Geometry failure reason, or None after a successful hull assignment.
    """
    node = graph.nodes[node_id]
    if int(node["npix"]) < minimum_points:
        return "too_few_hull_points"
    if "hull" in node:
        return None
    rows, columns = np.unravel_index(node["pixels"], graph.graph["label_map"].shape)
    dx = float(graph.graph.get("dx", 1.0))
    dy = float(graph.graph.get("dy", 1.0))
    if graph.graph.get("geometry_convention") == "kanapy_row_column":
        points = np.column_stack((rows * dx, columns * dy))
    else:
        x = float(graph.graph.get("origin_x", 0.0)) + columns * dx
        y = float(graph.graph.get("origin_y", 0.0)) + rows * dy
        points = np.column_stack((x, y))
    try:
        node["hull"] = ConvexHull(points)
    except QhullError:
        return "degenerate_hull"
    return None


def _candidate_details(
    graph, node_id, step, gs_min, npix_limit, fraction_min, minimum_points, connectivity,
):
    """
    Apply one original cleanup condition after checking same phase priority.

    Parameters
    ----------
    graph : networkx.Graph
        Current combined grain graph.
    node_id : int
        Candidate region identifier.
    step : int
        Cleanup step, either 3, 4, or 6.
    gs_min : float
        Minimum size for Step 6.
    npix_limit : int
        Maximum size for Step 3.
    fraction_min : float
        Minimum boundary fraction for Step 3.
    minimum_points : int
        Required number of points for Step 4.
    connectivity : int
        Boundary pixel neighborhood convention.

    Returns
    -------
    dict or None
        Reason and threshold diagnostics, or None when the node is ineligible.
    """
    node = graph.nodes[node_id]
    phase_id = node["phase_id"]
    if any(graph.nodes[neighbor]["phase_id"] == phase_id for neighbor in graph[node_id]):
        return None
    npix = int(node["npix"])
    if step == 3:
        if npix > npix_limit:
            return None
        boundary_pixels = _boundary_pixel_count(graph, node_id, connectivity)
        fraction = boundary_pixels / npix
        if fraction < fraction_min:
            return None
        return {
            "reason": "boundary_artifact",
            "boundary_pixels": boundary_pixels,
            "boundary_fraction": fraction,
            "npix_limit": npix_limit,
            "boundary_fraction_min": fraction_min,
        }
    if step == 4:
        reason = _invalid_hull_reason(graph, node_id, minimum_points)
        if reason is None:
            return None
        return {"reason": "invalid_hull", "geometry_unavailable_reason": reason}
    if npix < gs_min:
        return {"reason": "small_region", "threshold_npix": float(gs_min)}
    return None


def _select_target(graph, node_id, step):
    """
    Select a contacting neighbor by shared boundary or pixel count.

    Parameters
    ----------
    graph : networkx.Graph
        Current graph with physical shared boundary lengths.
    node_id : int
        Source node with no same phase neighbors.
    step : int
        Cleanup step determining the target selection rule.

    Returns
    -------
    int or None
        Target node identifier, or None when no shared side exists.
    """
    candidates = [
        neighbor for neighbor, data in graph[node_id].items()
        if data["shared_boundary_length"] > 0
    ]
    if not candidates:
        return None
    if step == 3:
        longest = max(graph[node_id][neighbor]["shared_boundary_length"] for neighbor in candidates)
        tolerance = 64 * np.finfo(float).eps * longest
        candidates = [
            neighbor for neighbor in candidates
            if abs(graph[node_id][neighbor]["shared_boundary_length"] - longest) <= tolerance
        ]
    return min(candidates, key=lambda neighbor: (-graph.nodes[neighbor]["npix"], neighbor))


def _absorb_region(graph, source_id, target_id, connectivity):
    """
    Transfer a source region while retaining the target phase and orientation.

    Parameters
    ----------
    graph : networkx.Graph
        Current graph to mutate.
    source_id : int
        Region absorbed and removed.
    target_id : int
        Region receiving the pixels and boundary contacts.
    connectivity : int
        Phase specific connectivity used to restore same phase corner edges.
    """
    source = graph.nodes[source_id]
    target = graph.nodes[target_id]
    source_pixels = np.asarray(source["pixels"], dtype=int)
    target["pixels"] = np.concatenate((target["pixels"], source_pixels))
    target["npix"] = int(target["npix"]) + int(source["npix"])
    graph.graph["label_map"].ravel()[source_pixels] = target_id
    for neighbor, data in list(graph[source_id].items()):
        if neighbor == target_id:
            continue
        shared_length = float(data["shared_boundary_length"])
        if graph.has_edge(target_id, neighbor):
            shared_length += graph[target_id][neighbor]["shared_boundary_length"]
        same_phase = target["phase_id"] == graph.nodes[neighbor]["phase_id"]
        graph.add_edge(
            target_id,
            neighbor,
            boundary_type="same_phase" if same_phase else "interphase",
            is_phase_boundary=not same_phase,
            shared_boundary_length=shared_length,
        )
    graph.remove_node(source_id)
    if connectivity not in (1, 4):
        labels = graph.graph["label_map"]
        rows, columns = np.unravel_index(source_pixels, labels.shape)
        for row_offset, column_offset in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            next_rows = rows + row_offset
            next_columns = columns + column_offset
            inside = (
                (next_rows >= 0) & (next_rows < labels.shape[0])
                & (next_columns >= 0) & (next_columns < labels.shape[1])
            )
            adjacent_labels = labels[next_rows[inside], next_columns[inside]]
            for neighbor in np.unique(adjacent_labels):
                if neighbor <= 0 or neighbor == target_id:
                    continue
                if graph.nodes[neighbor]["phase_id"] != target["phase_id"]:
                    continue
                if not graph.has_edge(target_id, neighbor):
                    graph.add_edge(
                        target_id,
                        int(neighbor),
                        boundary_type="same_phase",
                        is_phase_boundary=False,
                        shared_boundary_length=0.0,
                    )
    for key in (
        "hull", "max_dia", "min_dia", "equ_dia", "maj_ax", "min_ax", "omega",
        "center", "geometry_available", "geometry_unavailable_reason",
    ):
        target.pop(key, None)


def cleanup_isolated_nodes(
    graph,
    *,
    gs_min,
    boundary_size_factor=2.0,
    boundary_fraction_min=0.85,
    convex_hull_min_points=3,
    connectivity=4,
):
    """
    Clean regions that have no remaining same phase neighbor.

    Steps 3, 4, and 6 run in that order. Each pass visits nodes in ascending
    ID order, rechecking the receiving node after an absorption. Nodes with a
    same phase neighbor are left for phase specific cleanup. Qualifying nodes
    without any shared side are excluded, with their source pixels recorded.

    Parameters
    ----------
    graph : networkx.Graph
        Combined grain adjacency graph to update in place.
    gs_min : float
        Minimum grain size in pixels used in Step 6.
    boundary_size_factor : float, optional
        Multiplier used in the Step 3 size cutoff.
    boundary_fraction_min : float, optional
        Minimum boundary pixel fraction for Step 3.
    convex_hull_min_points : int, optional
        Minimum number of points needed for a valid hull.
    connectivity : int, optional
        Existing boundary pixel convention, four or eight neighbors.

    Returns
    -------
    list of dict
        Provenance records for the merges and exclusions.
    """
    records = []
    graph.graph["isolated_cleanup_records"] = records
    _prepare_shared_boundaries(graph)
    npix_limit = max(int(boundary_size_factor * gs_min), 40)
    for step in (3, 4, 6):
        pending = list(graph)
        heapq.heapify(pending)
        while pending:
            node_id = heapq.heappop(pending)
            if node_id not in graph:
                continue
            details = _candidate_details(
                graph, node_id, step, gs_min, npix_limit, boundary_fraction_min,
                convex_hull_min_points, connectivity,
            )
            if details is None:
                continue
            node = graph.nodes[node_id]
            pixels = np.asarray(node["pixels"], dtype=int)
            record = {
                "step": step,
                "node": int(node_id),
                "source_grain_id": int(node["source_grain_id"]),
                "source_phase_id": int(node["phase_id"]),
                "npix": int(node["npix"]),
                "pixels": pixels.tolist(),
                **details,
            }
            target_id = _select_target(graph, node_id, step)
            if target_id is None:
                record["action"] = "exclude"
                graph.graph["label_map"].ravel()[pixels] = 0
                graph.remove_node(node_id)
            else:
                target = graph.nodes[target_id]
                record.update(
                    action="merge",
                    target=int(target_id),
                    target_grain_id=int(target["source_grain_id"]),
                    target_phase_id=int(target["phase_id"]),
                    target_npix=int(target["npix"]),
                    shared_boundary_length=float(
                        graph[node_id][target_id]["shared_boundary_length"]
                    ),
                )
                _absorb_region(graph, node_id, target_id, connectivity)
                heapq.heappush(pending, target_id)
            records.append(record)
    return records
