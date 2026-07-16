"""
Tests for EBSD graph helper functions.

These tests cover graph misorientation caches and merge behavior used by the
EBSD cleanup workflow.
"""

import networkx as nx
import numpy as np
from orix.crystal_map import Phase

from kanapy.texture import (
    _batch_misorientation_angle,
    _compute_all_edge_misorientations,
    find_sim_neighbor,
    get_proper_symmetry_quaternions,
    merge_nodes,
)


def _make_orientation_graph():
    """
    Return a graph with four nodes at known misorientations.

    The graph is small enough to make direct and cached misorientation results
    easy to compare.

    Returns
    -------
    networkx.Graph
        Test graph with average orientations and map metadata.
    """
    sym = Phase(point_group="m-3m").point_group
    graph = nx.Graph()
    graph.graph["symmetry"] = sym
    graph.graph["label_map"] = np.ones((8, 8), dtype=int)
    graph.graph["dx"] = 1.0
    graph.graph["dy"] = 1.0

    angles_deg = {1: 0.0, 2: 3.0, 3: 10.0, 4: 20.0}
    for node_id, angle_deg in angles_deg.items():
        half_angle = np.deg2rad(angle_deg) / 2.0
        graph.add_node(
            node_id,
            npix=10,
            pixels=np.arange((node_id - 1) * 10, node_id * 10),
            ori_av=np.array(
                [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
            ),
        )

    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 4)
    return graph


def test_edge_misorientation_cache_matches_direct_batch_result():
    """
    Cached edge misorientation values match direct batch computations.
    """
    graph = _make_orientation_graph()
    sym_ops = get_proper_symmetry_quaternions(graph.graph["symmetry"])

    expected = {}
    for node_a, node_b in graph.edges():
        q_a = np.array([graph.nodes[node_a]["ori_av"]])
        q_b = np.array([graph.nodes[node_b]["ori_av"]])
        expected[(node_a, node_b)] = float(
            _batch_misorientation_angle(q_a, q_b, sym_ops)[0]
        )

    _compute_all_edge_misorientations(graph, sym_ops)

    for node_a, node_b in graph.edges():
        assert "misorientation" in graph[node_a][node_b]
        assert np.isclose(
            graph[node_a][node_b]["misorientation"],
            expected[(node_a, node_b)],
            atol=1e-12,
        )


def test_find_sim_neighbor_uses_cached_values_without_changing_result():
    """
    Similar neighbor search returns the same result with or without cache.
    """
    graph = _make_orientation_graph()
    sym_ops = get_proper_symmetry_quaternions(graph.graph["symmetry"])

    direct = {
        node_id: find_sim_neighbor(graph, node_id, sym_ops=sym_ops)
        for node_id in graph.nodes
    }
    _compute_all_edge_misorientations(graph, sym_ops)
    cached = {node_id: find_sim_neighbor(graph, node_id) for node_id in graph.nodes}

    for node_id in graph.nodes:
        assert direct[node_id][0] == cached[node_id][0]
        assert np.isclose(direct[node_id][1], cached[node_id][1], atol=1e-12)


def test_merge_nodes_refreshes_surviving_node_misorientation_cache():
    """
    Merge updates cached edge misorientation values around the surviving node.
    """
    graph = nx.Graph()
    graph.graph["symmetry"] = Phase(point_group="m-3m").point_group
    graph.graph["label_map"] = np.ones((6, 5), dtype=int)
    graph.graph["dx"] = 1.0
    graph.graph["dy"] = 1.0

    for node_id, angle_deg in [(1, 20.0), (2, 0.0), (3, 30.0)]:
        half_angle = np.deg2rad(angle_deg) / 2.0
        graph.add_node(
            node_id,
            npix=10,
            pixels=np.arange((node_id - 1) * 10, node_id * 10),
            ori_av=np.array(
                [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
            ),
        )

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    sym_ops = get_proper_symmetry_quaternions(graph.graph["symmetry"])
    _compute_all_edge_misorientations(graph, sym_ops)

    merge_nodes(graph, 1, 2)

    expected_angle = _batch_misorientation_angle(
        np.array([graph.nodes[2]["ori_av"]]),
        np.array([graph.nodes[3]["ori_av"]]),
        sym_ops,
    )[0]

    assert "misorientation" in graph[2][3]
    assert np.isclose(graph[2][3]["misorientation"], expected_angle, atol=1e-12)
