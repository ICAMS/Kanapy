"""
Check that final phase graphs can share contacts without sharing cleanup state.
"""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from kanapy.multiphase_graph import (
    build_multiphase_graph,
    build_multiphase_rgb_image,
)


def make_phase(phase_id, fraction, grains, edges=()):
    """
    Make a small phase graph from explicit final grain pixels.

    Parameters
    ----------
    phase_id : int
        Indexed phase identifier.
    fraction : float
        Original measured fraction.
    grains : dict
        Original grain IDs mapped to final flat pixel indices.
    edges : iterable, optional
        Existing edges within this phase.

    Returns
    -------
    dict
        Phase data in the analyzer format.
    """
    graph = nx.Graph(dx=0.5, dy=1.0, origin_x=2.0, origin_y=3.0,
                     source_grid="HexGrid", resampled_from_hex=True,
                     geometry_convention="kanapy_row_column",
                     symmetry="phase specific")
    for grain_id, pixels in grains.items():
        graph.add_node(grain_id, pixels=np.array(pixels), npix=len(pixels),
                       geometry_available=True)
    graph.add_edges_from(edges)
    return {"index": phase_id, "name": f"Phase {phase_id}", "vf": fraction,
            "graph": graph}


def phase_node_ids(graph):
    """
    Find overview IDs by their phase and original grain ID.

    Parameters
    ----------
    graph : networkx.Graph
        Combined final graph.

    Returns
    -------
    dict
        Original phase and grain pairs mapped to overview node IDs.
    """
    return {(node["phase_id"], node["source_grain_id"]): node_id
            for node_id, node in graph.nodes(data=True)}


def test_three_phase_contacts_keep_local_grain_ids_distinct():
    """
    Preserve phase edges and add each shared interface only once.
    """
    phase_a = make_phase(7, 0.6, {1: [0, 1, 4, 5], 5: [8, 9, 12, 13]}, [(1, 5)])
    phase_b = make_phase(2, 0.3, {1: [2, 3, 6, 7]})
    phase_c = make_phase(9, 0.1, {1: [10, 11, 14, 15]})

    graph = build_multiphase_graph([phase_b, phase_c, phase_a], (4, 4))
    ids = phase_node_ids(graph)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert graph[ids[7, 1]][ids[7, 5]]["boundary_type"] == "same_phase"
    assert graph[ids[7, 1]][ids[7, 5]]["is_phase_boundary"] is False
    for first, second in [((7, 1), (2, 1)), ((7, 5), (9, 1)), ((2, 1), (9, 1))]:
        assert graph[ids[first]][ids[second]]["boundary_type"] == "interphase"
        assert graph[ids[first]][ids[second]]["is_phase_boundary"] is True
    assert not graph.has_edge(ids[7, 1], ids[9, 1])
    assert not graph.has_edge(ids[7, 5], ids[2, 1])
    assert graph.graph["phases"] == [
        {"id": 7, "name": "Phase 7", "vf": 0.6},
        {"id": 2, "name": "Phase 2", "vf": 0.3},
        {"id": 9, "name": "Phase 9", "vf": 0.1},
    ]
    assert graph.graph["interphase_connectivity"] == 4
    assert graph.graph["dx"] == 0.5
    assert graph.graph["origin_y"] == 3.0
    assert graph.graph["resampled_from_hex"] is True
    assert graph.graph["geometry_convention"] == "kanapy_row_column"
    assert "symmetry" not in graph.graph
    for node_id, node in graph.nodes(data=True):
        assert node["phase_name"] == f"Phase {node['phase_id']}"
        np.testing.assert_array_equal(graph.graph["label_map"].ravel()[node["pixels"]], node_id)


@pytest.mark.parametrize("other_pixel", [2, 3])
def test_gaps_and_corner_contacts_do_not_create_interface_edges(other_pixel):
    """
    Keep grains separated by a blank pixel or a point contact disconnected.

    Parameters
    ----------
    other_pixel : int
        Second grain location on the first row or the opposite corner.
    """
    shape = (2, 2) if other_pixel == 3 else (1, 3)
    phases = [make_phase(1, 0.6, {1: [0]}), make_phase(2, 0.4, {1: [other_pixel]})]
    graph = build_multiphase_graph(phases, shape)
    assert graph.number_of_edges() == 0


def test_composition_uses_final_pixels_and_keeps_invalid_isolate_status():
    """
    Use merged grain membership and retain geometry flags despite new contacts.
    """
    phase_a = make_phase(1, 0.75, {4: [0, 1, 2]})
    phase_b = make_phase(2, 0.25, {4: [3]})
    phase_b["graph"].nodes[4].update(
        geometry_available=False, geometry_unavailable_reason="too_few_pixels")
    phase_a["graph"].graph["label_map"] = np.array([[4, 8, 8, 0]])

    graph = build_multiphase_graph([phase_a, phase_b], (1, 4))
    ids = phase_node_ids(graph)

    assert graph.has_edge(ids[1, 4], ids[2, 4])
    assert graph.nodes[ids[2, 4]]["geometry_available"] is False
    assert graph.nodes[ids[2, 4]]["geometry_unavailable_reason"] == "too_few_pixels"
    assert phase_b["graph"].degree(4) == 0
    np.testing.assert_array_equal(graph.graph["label_map"],
                                  [[ids[1, 4], ids[1, 4], ids[1, 4], ids[2, 4]]])


def test_overview_has_no_mutable_aliases_to_phase_graphs():
    """
    Keep later edits to overview nodes and edges away from phase analysis.
    """
    phase_a = make_phase(1, 0.75, {1: [0], 2: [1, 2]}, [(1, 2)])
    phase_a["graph"][1][2]["measurements"] = [3.0]
    phase_a["graph"].nodes[1]["nested"] = {"flags": [False]}
    phase_b = make_phase(2, 0.25, {1: [3]})
    before = deepcopy(phase_a["graph"])

    graph = build_multiphase_graph([phase_a, phase_b], (1, 4))
    ids = phase_node_ids(graph)
    graph.nodes[ids[1, 1]]["pixels"][0] = 99
    graph.nodes[ids[1, 1]]["nested"]["flags"][0] = True
    graph[ids[1, 1]][ids[1, 2]]["measurements"].append(4.0)

    np.testing.assert_array_equal(phase_a["graph"].nodes[1]["pixels"],
                                  before.nodes[1]["pixels"])
    assert phase_a["graph"].nodes[1]["nested"] == before.nodes[1]["nested"]
    assert phase_a["graph"][1][2] == before[1][2]
    assert "phase_id" not in phase_a["graph"].nodes[1]
    assert "boundary_type" not in phase_a["graph"][1][2]


def test_equal_phase_fractions_use_phase_id_order():
    """
    Keep the color order deterministic when original phase fractions tie.
    """
    graph = build_multiphase_graph(
        [make_phase(9, 0.5, {1: [0]}), make_phase(2, 0.5, {1: [1]})], (1, 2))
    assert [phase["id"] for phase in graph.graph["phases"]] == [2, 9]


def test_overlapping_phase_pixels_are_rejected():
    """
    Reject contradictory ownership rather than silently overwrite a grain.
    """
    with pytest.raises(ValueError, match="overlap"):
        build_multiphase_graph(
            [make_phase(1, 0.5, {1: [0]}), make_phase(2, 0.5, {1: [0]})], (1, 2))


def test_empty_phase_list_produces_an_empty_overview():
    """
    Keep empty filtered maps representable without an invalid source lookup.
    """
    graph = build_multiphase_graph([], (2, 3))
    assert graph.number_of_nodes() == 0
    assert graph.graph["phases"] == []
    np.testing.assert_array_equal(graph.graph["label_map"], np.zeros((2, 3)))


def test_combined_rgb_image_keeps_all_phases_and_blank_positions():
    """
    Combine IPF colors without recoloring orientations or filling missing data.
    """
    phases = [
        {"idx": np.array([0, 2]), "rgb_vals": np.array([[255, 0, 0], [0, 128, 0]])},
        {"idx": np.array([3]), "rgb_vals": np.array([[0, 0, 255]])},
    ]
    image = build_multiphase_rgb_image(phases, (2, 2))
    assert image.shape == (2, 2, 3)
    assert image.dtype.kind == "f"
    np.testing.assert_allclose(image, [[[1, 0, 0], [0, 0, 0]],
                                      [[0, 128 / 255, 0], [0, 0, 1]]])
