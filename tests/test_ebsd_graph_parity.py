"""
Regression checks for the synchronized EBSD graph workflow.
"""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Orientation, symmetry
from scipy.spatial import ConvexHull

from kanapy.texture import (
    EBSDmap,
    _graph_points_to_image,
    build_graph_from_labeled_pixels,
    find_largest_neighbor,
    find_similar_regions_by_misorientation,
    merge_nodes,
)


def test_largest_neighbor_ties_use_lowest_id():
    """
    Equal size neighbors are chosen independently of edge insertion order.
    """
    graph = nx.Graph()
    graph.add_nodes_from([(1, {"npix": 1}), (8, {"npix": 5}), (2, {"npix": 5})])
    graph.add_edges_from([(1, 8), (1, 2)])
    assert find_largest_neighbor(graph, 1) == 2


def test_standalone_segmentation_uses_five_degree_threshold():
    """
    Adjacent measurements seven degrees apart remain separate by default.
    """
    ori = Orientation.from_euler([[0, 0, 0], [np.deg2rad(7), 0, 0]])
    labels, count = find_similar_regions_by_misorientation(
        ori.data.reshape(1, 2, 4), np.ones((1, 2), dtype=bool), symmetry.Oh,
    )
    assert count == 2
    assert labels[0, 0] != labels[0, 1]


def test_grain_segmentation_distinguishes_rotation_axes():
    """
    Separate regions with equal rotation angles but different rotation axes.

    Comparing scalar angles alone would incorrectly join these two regions.
    """
    euler = np.zeros((6, 12, 3))
    euler[:, :6, 1] = np.pi / 6
    euler[:, 6:, 0] = np.pi / 6
    emap = CrystalMap(
        rotations=Orientation.from_euler(euler.reshape(-1, 3), symmetry=symmetry.Oh),
        phase_id=np.zeros(72, dtype=int),
        x=np.tile(np.arange(12), 6), y=np.repeat(np.arange(6), 12),
        phase_list=PhaseList(names=["phase"], point_groups=["m-3m"], ids=[0]),
    )
    loaded = SimpleNamespace(emap=emap, sh_x=6, sh_y=12, dx=1., dy=1., npx=72)

    ebsd = EBSDmap(ebsd_data=loaded, graph_only=True, gs_min=4, max_angle=5)

    phase = ebsd.ms_data[0]
    assert len(phase["graph"]) == 2
    labels = phase["graph"].graph["label_map"]
    assert np.unique(labels[:, :6]).size == 1
    assert np.unique(labels[:, 6:]).size == 1
    assert labels[0, 0] != labels[0, 6]


@pytest.mark.parametrize("physical", [True, False])
def test_merge_hull_preserves_coordinate_convention(physical):
    """
    Merged hulls retain physical coordinates or the legacy row convention.

    Parameters
    ----------
    physical : bool
        Whether graph metadata marks physical x and y coordinates.
    """
    labels = np.array([[1, 1, 2], [1, 2, 2]])
    rotations = np.tile([1., 0., 0., 0.], (6, 1))
    graph = build_graph_from_labeled_pixels(labels, rotations, symmetry.Oh, 2., 3.)
    if physical:
        graph.graph.update(origin_x=10., origin_y=20.)
        points = [[12., 23.], [14., 20.], [14., 23.]]
        expected_min, expected_max = [10., 20.], [14., 23.]
    else:
        points = [[2., 3.], [0., 6.], [2., 6.]]
        expected_min, expected_max = [0., 0.], [2., 6.]
    graph.nodes[2]["hull"] = ConvexHull(points)
    merge_nodes(graph, 1, 2)
    np.testing.assert_allclose(graph.nodes[2]["hull"].points.min(axis=0), expected_min)
    np.testing.assert_allclose(graph.nodes[2]["hull"].points.max(axis=0), expected_max)
    rows, columns = np.unravel_index(graph.nodes[2]["pixels"], labels.shape)
    np.testing.assert_allclose(
        _graph_points_to_image(graph, graph.nodes[2]["hull"].points),
        np.column_stack((columns, rows)),
    )


@pytest.mark.parametrize("minimum, nodes, edges", [(10, 122, 309), (20, 109, 276)])
def test_316l_reference_graph_counts(minimum, nodes, edges):
    """
    Real fixture cleanup reproduces the reference graph at both thresholds.

    Parameters
    ----------
    minimum : int
        Minimum retained grain size.
    nodes, edges : int
        Independently measured reference graph counts.
    """
    source = Path(__file__).resolve().parents[1] / "examples/fixtures/ebsd_316L_500x500.ang"
    result = EBSDmap(str(source), gs_min=minimum, graph_only=True)
    graph = result.ms_data[0]["graph"]
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (nodes, edges)
    expected = {
        10: ("3c3f9b2992c7eb2b4579315792b1b958272de9562c565328b536ecf082e5039c",
             "e906d7a64ff649a9250b26766286195fb25c6de1bdfdfabf6e74af213f7eeb03"),
        20: ("7d46a2e3418fb76186274a5ca32d7ac0bb2f589dd70ae29105e317273be9f226",
             "9249bd1bbc337cb58aa7391f4c566b56ce7085db3d92297666aab9d40eb6f141"),
    }
    labels = graph.graph["label_map"]
    ordered_ids = sorted(graph.nodes, key=lambda node: int(np.flatnonzero(labels.ravel() == node)[0]))
    mapping = {node: index + 1 for index, node in enumerate(ordered_ids)}
    canonical = np.zeros(labels.shape, dtype="<i4")
    for node, canonical_id in mapping.items():
        canonical[labels == node] = canonical_id
    canonical_edges = sorted(tuple(sorted((mapping[a], mapping[b]))) for a, b in graph.edges())
    partition_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
    adjacency_hash = hashlib.sha256(json.dumps(canonical_edges, separators=(",", ":")).encode()).hexdigest()
    assert (partition_hash, adjacency_hash) == expected[minimum]


@pytest.mark.parametrize("pixel, phases, donor", [
    (4, [7] * 12, 5),
    (6, [2, 7, 7, 2, 2, 2, 7, 2, 2, 2, 7, 2], 2),
    (7, [2, 2, 7, 7, 7, 2, 2, 7, 2, 2, 2, 7], 3),
])
def test_ci_repair_uses_same_phase_grid_neighbors(pixel, phases, donor):
    """
    Repair uses true row neighbors and global indices on rectangular maps.

    Parameters
    ----------
    pixel : int
        Low confidence measurement index.
    phases : list of int
        Phase identifiers in a three by four grid.
    donor : int
        Hand checked valid neighbor in the same phase.
    """
    analyzer = EBSDmap.__new__(EBSDmap)
    analyzer.sh_x, analyzer.sh_y, analyzer.npx = 3, 4, 12
    analyzer.logger = logging.getLogger(__name__)
    phases = np.asarray(phases)
    analyzer.emap = SimpleNamespace(phase_id=phases, prop={})
    euler = np.zeros((12, 3))
    euler[:, 0] = np.arange(12) * 0.05
    rotations = Orientation.from_euler(euler, symmetry=symmetry.Oh)
    ci = np.ones(12)
    ci[pixel] = 0.01
    indices = np.flatnonzero(phases == 7)
    data = {"index": 7, "idx": indices, "cs": symmetry.Oh,
            "ori": rotations[indices], "quat_full": rotations.data.copy(),
            "phase_id_flat": phases, "ci_values": ci}
    analyzer._handle_bad_pixels(data)
    repaired = data["ori"].data[np.flatnonzero(indices == pixel)[0]]
    np.testing.assert_allclose(repaired, rotations.data[donor])


def test_multiphase_counts_and_repaired_graph_orientations():
    """
    Nonconsecutive phase IDs retain direct fractions and corrected node pixels.
    """
    phases = np.tile([2, 2, 7, 7, 9, 9, 9, 9, 9, 9], 6)
    euler = np.tile([0.1, 0.2, 0.3], (60, 1))
    euler[52] = [1.2, 0.9, 0.7]
    rotations = Orientation.from_euler(euler, symmetry=symmetry.Oh)
    ci = np.ones(60)
    ci[52] = 0.01
    emap = CrystalMap(
        rotations=rotations, phase_id=phases,
        x=np.tile(np.arange(10), 6) * 2., y=np.repeat(np.arange(6), 10) * 3.,
        phase_list=PhaseList(names=["two", "seven", "nine"],
                             point_groups=["m-3m"] * 3, ids=[2, 7, 9]),
        prop={"ci": ci},
    )
    loaded = SimpleNamespace(emap=emap, sh_x=6, sh_y=10, dx=2., dy=3., npx=60)
    analyzer = EBSDmap(ebsd_data=loaded, graph_only=True, vf_min=0.1)
    assert [data["index"] for data in analyzer.ms_data] == [2, 7, 9]
    np.testing.assert_allclose([data["vf"] for data in analyzer.ms_data], [0.2, 0.2, 0.6])
    data = analyzer.ms_data[1]
    graph = data["graph"]
    donor = data["ori"].data[np.flatnonzero(data["idx"] == 53)[0]]
    np.testing.assert_allclose(graph.graph["rotations"][52], donor)
    assert graph.number_of_nodes() == 1


def test_similarity_cleanup_revisits_previously_checked_node():
    """
    A changed mean can enable a merge with a node visited before that change.
    """
    graph = nx.Graph(label_map=np.array([[1, 2, 3]]), dx=1., dy=1., symmetry=symmetry.Oh)
    angles = np.deg2rad([0., 6., 2.])
    for node, angle in enumerate(angles, start=1):
        graph.add_node(node, npix=1, pixels=np.array([node - 1]),
                       ori_av=np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)]))
    graph.add_edges_from([(1, 2), (2, 3)])
    analyzer = EBSDmap.__new__(EBSDmap)
    analyzer.logger = logging.getLogger(__name__)
    analyzer.similar_orientation_merge_angle = np.deg2rad(5.)
    analyzer._merge_similar_orientation_nodes(graph)
    assert graph.number_of_nodes() == 1
    assert len(graph.graph["similar_orientation_merge_records"]) == 2


@pytest.mark.parametrize("sizes, expected_counts", [
    ((6, 6, 100), {2: 12, 3: 100}),
    ((4, 6, 100), {2: 10, 3: 100}),
    ((3, 6, 100), {3: 109}),
])
def test_small_grain_cleanup_rechecks_size_after_absorbing_neighbor(sizes, expected_counts):
    """
    Retain a queued grain once an earlier merge brings it to the size threshold.

    A grain that remains below the threshold must still be absorbed into its
    largest neighbor.

    Parameters
    ----------
    sizes : tuple of int
        Pixel counts of three grains connected in a path.
    expected_counts : dict of int to int
        Retained grain IDs and their expected pixel counts.
    """
    labels = np.repeat([1, 2, 3], sizes).reshape(1, -1)
    graph = nx.Graph(label_map=labels, dx=1., dy=1.)
    for grain_id, size in enumerate(sizes, start=1):
        graph.add_node(
            grain_id, npix=size,
            pixels=np.flatnonzero(labels.ravel() == grain_id),
            ori_av=np.array([1., 0., 0., 0.]),
        )
    graph.add_edges_from([(1, 2), (2, 3)])
    analyzer = EBSDmap.__new__(EBSDmap)
    analyzer.logger = logging.getLogger(__name__)
    analyzer.gs_min = 10

    analyzer._merge_small_nodes(graph)

    assert dict(graph.nodes(data="npix")) == expected_counts
    assert all(record["node_npix"] < 10 for record in graph.graph["small_node_merge_records"])


@pytest.mark.parametrize("physical", [True, False])
def test_overlay_converts_physical_centers_to_image_pixels(physical):
    """
    Overlay edges align with image pixels for new and older graph handoffs.

    Parameters
    ----------
    physical : bool
        Whether graph centers use current physical coordinates or legacy rows.
    """
    graph = nx.Graph(dx=2., dy=3.)
    if physical:
        graph.graph.update(origin_x=10., origin_y=20.)
        centers = ([12., 23.], [16., 26.])
    else:
        centers = ([2., 3.], [4., 9.])
    graph.add_node(1, center=np.asarray(centers[0]))
    graph.add_node(2, center=np.asarray(centers[1]))
    graph.add_edge(1, 2)
    analyzer = EBSDmap.__new__(EBSDmap)
    analyzer.sh_x, analyzer.sh_y, analyzer.dx, analyzer.dy = 4, 5, 2., 3.
    analyzer.ms_data = [{"graph": graph, "index": 7, "name": "phase",
                         "rgb_im": np.ones((4, 5, 3))}]
    figure, axes = analyzer.plot_graph_overlay(show=False)
    np.testing.assert_allclose(axes.lines[0].get_xdata(), [1., 3.])
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1., 2.])
    plt.close(figure)


@pytest.mark.parametrize("method", ["plot_grains_marked", "plot_grains"])
def test_grain_axes_convert_physical_geometry_to_image_pixels(method, monkeypatch):
    """
    Both grain annotation views align ellipse axes with anisotropic images.

    Parameters
    ----------
    method : str
        Grain visualization method to exercise.
    monkeypatch : pytest.MonkeyPatch
        Suppress the interactive display while retaining plotted artists.
    """
    graph = nx.Graph(dx=2., dy=3., origin_x=10., origin_y=20.,
                     label_map=np.ones((2, 2), dtype=int))
    hull = ConvexHull([[10., 20.], [12., 20.], [10., 23.], [12., 23.]])
    graph.add_node(1, pixels=np.arange(4), hull=hull, equ_dia=2.,
                   center=np.array([11., 21.5]), max_dia=2., min_dia=3.,
                   maj_ax=np.array([1., 0.]), min_ax=np.array([0., 1.]))
    analyzer = EBSDmap.__new__(EBSDmap)
    analyzer.sh_x = analyzer.sh_y = 2
    analyzer.dx, analyzer.dy = 2., 3.
    analyzer.ms_data = [{"graph": graph, "index": 7, "name": "phase", "ngrains": 1}]
    monkeypatch.setattr(plt, "show", lambda: None)
    figure = plt.figure()
    if method == "plot_grains":
        analyzer.plot_grains(N=1)
    else:
        analyzer.plot_grains_marked()
    axes = figure.axes[0]
    np.testing.assert_allclose(axes.lines[-2].get_xdata(), [-0.5, 1.5])
    np.testing.assert_allclose(axes.lines[-2].get_ydata(), [0.5, 0.5])
    np.testing.assert_allclose(axes.lines[-1].get_xdata(), [0.5, 0.5])
    np.testing.assert_allclose(axes.lines[-1].get_ydata(), [-0.5, 1.5])
    plt.close(figure)
