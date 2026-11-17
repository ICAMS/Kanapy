"""
Combine phase graphs for shared cleanup and spatial inspection.
"""

from copy import deepcopy

import networkx as nx
import numpy as np


def build_multiphase_graph(ms_data, image_shape):
    """
    Combine cleaned phase graphs and add contacts across shared pixel sides.

    This builder copies the phase graphs without mutating them. The shared
    graph is used for supplementary isolated region cleanup and, after final
    statistics, for plotting. Each node retains its phase and source grain ID.

    Parameters
    ----------
    ms_data : list of dict
        Phase records containing graph, index, name, and original vf.
    image_shape : tuple of int
        Common rectangular grid shape as rows and columns.

    Returns
    -------
    networkx.Graph
        Combined adjacency with phase metadata and a global label map.

    Raises
    ------
    ValueError
        If phases repeat, pixels overlap, or pixel indices leave the grid.
    """
    label_map = np.zeros(image_shape, dtype=int)
    labels = label_map.ravel()
    graph = nx.Graph(
        name="Multiphase grain adjacency",
        label_map=label_map,
        phases=[],
        interphase_connectivity=4,
    )
    phases = sorted(ms_data, key=lambda data: (-data["vf"], data["index"]))
    phase_ids = [int(data["index"]) for data in phases]
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("Phase IDs must be unique in the overview graph.")

    spatial_keys = (
        "dx", "dy", "origin_x", "origin_y", "source_grid", "resampled_from_hex",
        "geometry_convention", "grain_method",
    )
    if phases:
        first_metadata = phases[0]["graph"].graph
        for key in spatial_keys:
            if key in first_metadata:
                graph.graph[key] = deepcopy(first_metadata[key])

    node_phases = [-1]
    for data in phases:
        phase_id = int(data["index"])
        phase_name = str(data["name"])
        source_graph = data["graph"]
        graph.graph["phases"].append(
            {"id": phase_id, "name": phase_name, "vf": float(data["vf"])}
        )
        node_ids = {}
        for source_id in sorted(source_graph.nodes):
            attributes = deepcopy(source_graph.nodes[source_id])
            pixels = np.asarray(attributes["pixels"], dtype=int).reshape(-1)
            if np.any(pixels < 0) or np.any(pixels >= labels.size):
                raise ValueError("Grain pixels must lie within the overview grid.")
            if np.any(labels[pixels] != 0):
                raise ValueError("Grain pixels overlap on the overview grid.")

            node_id = len(node_phases)
            node_ids[source_id] = node_id
            node_phases.append(phase_id)
            attributes.update(
                phase_id=phase_id,
                phase_name=phase_name,
                source_grain_id=int(source_id),
            )
            graph.add_node(node_id, **attributes)
            labels[pixels] = node_id

        for first, second, source_attributes in source_graph.edges(data=True):
            attributes = deepcopy(source_attributes)
            attributes["boundary_type"] = "same_phase"
            attributes["is_phase_boundary"] = False
            graph.add_edge(node_ids[first], node_ids[second], **attributes)

    node_phases = np.asarray(node_phases, dtype=int)
    for first, second in (
        (label_map[:, :-1], label_map[:, 1:]),
        (label_map[:-1, :], label_map[1:, :]),
    ):
        contact = (first > 0) & (second > 0)
        contact &= node_phases[first] != node_phases[second]
        if not np.any(contact):
            continue
        pairs = np.column_stack((first[contact], second[contact]))
        pairs.sort(axis=1)
        for first_id, second_id in np.unique(pairs, axis=0):
            graph.add_edge(
                int(first_id), int(second_id), boundary_type="interphase",
                is_phase_boundary=True,
            )

    return graph


def build_multiphase_rgb_image(ms_data, image_shape):
    """
    Assemble the retained phases' IPF colors on their common analysis grid.

    Phase graph colors are drawn separately from these orientation colors.
    Positions without retained measurements remain black.

    Parameters
    ----------
    ms_data : list of dict
        Phase records with flat pixel indices and RGB values from 0 to 255.
    image_shape : tuple of int
        Common rectangular grid shape as rows and columns.

    Returns
    -------
    numpy.ndarray
        Float RGB image with values from 0 to 1.
    """
    image = np.zeros((*image_shape, 3), dtype=np.uint8)
    flat_image = image.reshape(-1, 3)
    for data in ms_data:
        flat_image[data["idx"]] = data["rgb_vals"]
    return image.astype(float) / 255.0
