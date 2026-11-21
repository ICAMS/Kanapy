"""
Exercise supplementary cleanup with physical contacts and explicit exclusions.
"""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from kanapy.isolated_cleanup import cleanup_isolated_nodes


def make_graph(labels, phases, dx=1.0, dy=1.0):
    """
    Build a labeled graph with hand specified phase membership.

    Parameters
    ----------
    labels : numpy.ndarray
        Positive grain labels with zero for unknown positions.
    phases : dict
        Grain labels mapped to indexed phase identifiers.
    dx, dy : float, optional
        Physical grid spacing along columns and rows.

    Returns
    -------
    networkx.Graph
        Graph with pixel side adjacency and distinct grain orientations.
    """
    graph = nx.Graph(label_map=labels.copy(), dx=dx, dy=dy)
    for node_id, phase_id in phases.items():
        pixels = np.flatnonzero(labels.ravel() == node_id)
        angle = 0.1 * node_id
        graph.add_node(
            node_id,
            pixels=pixels,
            npix=len(pixels),
            ori_av=np.array([np.cos(angle), np.sin(angle), 0.0, 0.0]),
            phase_id=phase_id,
            phase_name=f"Phase {phase_id}",
            source_grain_id=node_id + 100,
        )
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        contact = (first > 0) & (second > 0) & (first != second)
        for one, two in zip(first[contact], second[contact]):
            boundary_type = "same_phase" if phases[one] == phases[two] else "interphase"
            graph.add_edge(int(one), int(two), boundary_type=boundary_type)
    return graph


def test_step3_uses_physical_boundary_before_neighbor_size_and_preserves_orientation():
    """
    Prefer one long shared side to two shorter sides of a larger neighbor.
    """
    labels = np.full((20, 20), 3)
    labels[:, :5] = 2
    labels[7, 5] = 1
    labels[7, 6] = 0
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3}, dx=1.0, dy=5.0)
    graph.nodes[2]["ori_av"] = np.array([0.0, 1.0, 0.0, 0.0])
    graph.nodes[3]["ori_av"] = graph.nodes[1]["ori_av"].copy()
    target_orientation = graph.nodes[2]["ori_av"].copy()
    source_pixel = 7 * 20 + 5

    records = cleanup_isolated_nodes(graph, gs_min=10)

    assert set(graph) == {2, 3}
    assert graph.nodes[2]["npix"] == 101
    assert graph.nodes[2]["phase_id"] == 2
    assert graph.nodes[2]["source_grain_id"] == 102
    np.testing.assert_array_equal(graph.nodes[2]["ori_av"], target_orientation)
    assert graph.graph["label_map"].flat[source_pixel] == 2
    assert len(records) == 1
    assert records[0]["step"] == 3
    assert records[0]["action"] == "merge"
    assert records[0]["target"] == 2
    assert records[0]["shared_boundary_length"] == pytest.approx(5.0)
    assert records[0]["source_phase_id"] == 1
    assert records[0]["target_phase_id"] == 2
    assert records[0]["pixels"] == [source_pixel]
    assert records[0]["target_npix"] == 100
    assert graph.graph["isolated_cleanup_records"] == records
    assert sum(node["npix"] for _, node in graph.nodes(data=True)) == 399


def test_equal_boundary_length_uses_largest_neighbor_then_stable_id():
    """
    Resolve a physical boundary tie by size before using a stable node ID.
    """
    labels = np.full((15, 15), 3)
    labels[:, :5] = 2
    labels[7, 5] = 1
    labels[6, 5] = labels[8, 5] = 0
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3})
    records = cleanup_isolated_nodes(graph, gs_min=10)
    assert records[0]["target"] == 3

    labels = np.zeros((11, 11), dtype=int)
    labels[:, :5] = 2
    labels[:, 6:] = 3
    labels[5, 5] = 1
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3})
    records = cleanup_isolated_nodes(graph, gs_min=10)
    assert records[0]["target"] == 2


def test_existing_same_phase_neighbor_blocks_cross_phase_cleanup():
    """
    Keep a same phase candidate even when its other phase neighbor is larger.
    """
    labels = np.full((15, 15), 3)
    labels[:, :5] = 2
    labels[7, 5] = 1
    graph = make_graph(labels, {1: 1, 2: 1, 3: 2})
    original_labels = graph.graph["label_map"].copy()
    original_orientation = graph.nodes[1]["ori_av"].copy()

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert records == []
    assert set(graph) == {1, 2, 3}
    np.testing.assert_array_equal(graph.graph["label_map"], original_labels)
    np.testing.assert_array_equal(graph.nodes[1]["ori_av"], original_orientation)


def test_step4_merges_large_collinear_region_into_largest_neighbor():
    """
    Apply invalid hull cleanup even above the small fragment size cutoff.
    """
    labels = np.full((6, 50), 3)
    labels[:2] = 2
    labels[2] = 1
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3})
    target_orientation = graph.nodes[3]["ori_av"].copy()

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert len(records) == 1
    assert records[0]["step"] == 4
    assert records[0]["target"] == 3
    assert records[0]["npix"] == 50
    assert graph.nodes[3]["npix"] == 200
    np.testing.assert_array_equal(graph.nodes[3]["ori_av"], target_orientation)
    np.testing.assert_array_equal(graph.graph["label_map"][2], 3)


def test_step6_merges_compact_small_region_into_largest_neighbor():
    """
    Merge a valid compact hull that passes Step 3 but falls below gs_min.
    """
    labels = np.full((12, 16), 3)
    labels[:, :6] = 2
    labels[4:8, 4:8] = 1
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3})
    target_orientation = graph.nodes[3]["ori_av"].copy()

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert len(records) == 1
    assert records[0]["step"] == 6
    assert records[0]["target"] == 3
    assert graph.nodes[3]["npix"] == 128
    np.testing.assert_array_equal(graph.nodes[3]["ori_av"], target_orientation)


@pytest.mark.parametrize("kind, expected_step, expected_pixels", [
    ("thin", 3, 30),
    ("line", 4, 52),
    ("compact", 6, 16),
])
def test_regions_without_any_neighbor_are_excluded_by_each_cleanup_step(
    kind, expected_step, expected_pixels,
):
    """
    Exclude unsupported fragments without inventing contact across blank data.

    Parameters
    ----------
    kind : str
        Geometry of the disconnected indexed region.
    expected_step : int
        First cleanup condition satisfied by that geometry.
    expected_pixels : int
        Hand counted number of pixels excluded from analysis.
    """
    labels = np.zeros((10, 60), dtype=int)
    if kind == "thin":
        labels[3:5, 3:18] = 1
    elif kind == "line":
        labels[3, 3:55] = 1
    else:
        labels[3:7, 3:7] = 1
    graph = make_graph(labels, {1: 1})
    before = labels.copy()

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert not graph
    assert not graph.graph["label_map"].any()
    assert len(records) == 1
    assert records[0]["action"] == "exclude"
    assert records[0]["step"] == expected_step
    assert records[0]["npix"] == expected_pixels
    assert records[0]["pixels"] == np.flatnonzero(before).tolist()
    np.testing.assert_array_equal(labels, before)


def test_large_valid_isolated_region_stays_even_when_its_outline_is_concave():
    """
    Retain a sufficiently large isolated L shape with a valid convex hull.
    """
    labels = np.zeros((12, 12), dtype=int)
    labels[2:10, 2:10] = 1
    labels[6:10, 6:10] = 0
    graph = make_graph(labels, {1: 1})
    original_node = deepcopy(graph.nodes[1])

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert records == []
    assert graph.nodes[1]["npix"] == 48
    np.testing.assert_array_equal(graph.nodes[1]["pixels"], original_node["pixels"])
    np.testing.assert_array_equal(graph.nodes[1]["ori_av"], original_node["ori_av"])
    assert graph.nodes[1]["hull"].volume > 0


def test_hull_preserves_kanapy_row_column_geometry_on_anisotropic_grid():
    """
    Keep Kanapy's grain geometry axes separate from physical contact lengths.
    """
    labels = np.zeros((10, 12), dtype=int)
    labels[2:8, 3:11] = 1
    graph = make_graph(labels, {1: 1}, dx=2.0, dy=5.0)
    graph.graph.update(
        geometry_convention="kanapy_row_column", origin_x=100.0, origin_y=200.0,
    )

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert records == []
    rows, columns = np.where(labels == 1)
    expected = np.column_stack((rows * 2.0, columns * 5.0))
    np.testing.assert_array_equal(graph.nodes[1]["hull"].points, expected)


def test_cleanup_edges_keep_kanapy_phase_boundary_alias_after_absorption():
    """
    Expose both boundary attributes on retained and newly created contacts.
    """
    labels = np.zeros((10, 12), dtype=int)
    labels[:, :6] = 2
    labels[4, 6] = 1
    labels[4, 7] = 3
    graph = make_graph(labels, {1: 1, 2: 2, 3: 2})

    cleanup_isolated_nodes(graph, gs_min=20)

    assert graph[2][3]["boundary_type"] == "same_phase"
    assert graph[2][3]["is_phase_boundary"] is False


@pytest.mark.parametrize("offset", [0, 1])
def test_gap_and_corner_contacts_do_not_allow_cross_phase_merge(offset):
    """
    Treat a corner contact and a wider gap as no shared boundary.

    Parameters
    ----------
    offset : int
        Additional gap beyond the corner contact.
    """
    labels = np.zeros((10, 10), dtype=int)
    labels[0, 0] = 1
    labels[1 + offset:8 + offset, 1 + offset:8 + offset] = 2
    graph = make_graph(labels, {1: 1, 2: 2})

    records = cleanup_isolated_nodes(graph, gs_min=20, connectivity=8)

    assert set(graph) == {2}
    assert graph.nodes[2]["npix"] == 49
    assert records[0]["action"] == "exclude"
    assert graph.graph["label_map"][0, 0] == 0


def test_new_same_phase_contact_prevents_later_cross_phase_selection():
    """
    Recheck phase isolation after a bridge fragment changes phase ownership.
    """
    labels = np.zeros((10, 12), dtype=int)
    labels[:, :6] = 2
    labels[4, 6] = 1
    labels[4, 7] = 3
    graph = make_graph(labels, {1: 1, 2: 2, 3: 2})
    third_orientation = graph.nodes[3]["ori_av"].copy()

    records = cleanup_isolated_nodes(graph, gs_min=20)

    assert len(records) == 1
    assert records[0]["node"] == 1
    assert records[0]["target"] == 2
    assert set(graph) == {2, 3}
    assert graph[2][3]["boundary_type"] == "same_phase"
    assert graph[2][3]["shared_boundary_length"] == pytest.approx(1.0)
    np.testing.assert_array_equal(graph.nodes[3]["ori_av"], third_orientation)


def test_existing_same_phase_corner_edge_is_preserved():
    """
    Respect same phase diagonal adjacency from an existing eight neighbor graph.
    """
    labels = np.zeros((10, 10), dtype=int)
    labels[0, 0] = 1
    labels[1:8, 1:8] = 2
    graph = make_graph(labels, {1: 1, 2: 1})
    graph.add_edge(1, 2, boundary_type="same_phase")

    records = cleanup_isolated_nodes(graph, gs_min=20, connectivity=8)

    assert records == []
    assert graph.has_edge(1, 2)
    assert graph[1][2]["shared_boundary_length"] == 0.0


@pytest.mark.parametrize("connectivity, expected_nodes", [(4, {2}), (8, {2, 3})])
def test_absorption_adds_new_same_phase_corner_contacts(connectivity, expected_nodes):
    """
    Protect a new same phase corner neighbor under eight neighbor connectivity.

    Parameters
    ----------
    connectivity : int
        Connectivity selected for phase specific graph construction.
    expected_nodes : set of int
        Final nodes after the bridge is absorbed and isolation is reevaluated.
    """
    labels = np.zeros((10, 10), dtype=int)
    labels[:7, :5] = 2
    labels[2, 5] = 1
    labels[3, 6] = 3
    graph = make_graph(labels, {1: 1, 2: 2, 3: 2})

    records = cleanup_isolated_nodes(graph, gs_min=20, connectivity=connectivity)

    assert records[0]["node"] == 1
    assert records[0]["target"] == 2
    assert set(graph) == expected_nodes
    if connectivity == 8:
        assert len(records) == 1
        assert graph[2][3]["boundary_type"] == "same_phase"
        assert graph[2][3]["shared_boundary_length"] == 0.0
        assert graph.graph["label_map"][3, 6] == 3
    else:
        assert records[1]["node"] == 3
        assert records[1]["action"] == "exclude"


def test_boundary_length_is_updated_when_contacts_are_joined():
    """
    Transfer interface length after absorption without losing existing contact.
    """
    labels = np.full((12, 12), 3)
    labels[:, :6] = 2
    labels[5, 5] = 1
    graph = make_graph(labels, {1: 1, 2: 2, 3: 3}, dx=2.0, dy=3.0)

    records = cleanup_isolated_nodes(graph, gs_min=10)

    assert records[0]["target"] == 2
    assert graph[2][3]["shared_boundary_length"] == pytest.approx(36.0)
    assert graph[2][3]["boundary_type"] == "interphase"


def test_empty_graph_has_an_empty_audit_record():
    """
    Allow all phases to have no remaining grain nodes.
    """
    graph = make_graph(np.zeros((2, 3), dtype=int), {})
    assert cleanup_isolated_nodes(graph, gs_min=20) == []
    assert graph.graph["isolated_cleanup_records"] == []
