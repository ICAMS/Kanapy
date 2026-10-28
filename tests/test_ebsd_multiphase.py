"""
Exercise phase boundaries and shared EBSD figures while preserving single phase analysis.
"""

import copy
import json
import pickle
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Orientation, symmetry

from kanapy.graph_workflow import (
    EBSDGraphConfig, EBSDGraphOutputOptions, EBSDGraphResult, write_graph_result_outputs,
)
from kanapy.texture import (
    EBSDmap, _compute_all_edge_misorientations, _refresh_node_misorientation_cache,
    build_graph_from_labeled_pixels, find_similar_regions_by_misorientation,
    find_largest_neighbor, find_sim_neighbor, get_proper_symmetry_quaternions,
    merge_nodes,
)


def make_mixed_graph():
    """
    Create a misleading cross phase neighbor and a valid same phase candidate.

    The other phase is both larger and closer in orientation, so unconstrained
    neighbor searches select the wrong grain.

    Returns
    -------
    networkx.Graph
        Three grain graph with a shared boundary between different phases.
    """
    graph = nx.Graph(label_map=np.array([[1, 2, 3]]), dx=1., dy=1., symmetry=symmetry.Oh)
    for node, phase, size, angle in [(1, 2, 1, 0.), (2, 7, 100, 0.), (3, 2, 5, 0.2)]:
        graph.add_node(node, phase_id=phase, phase_name=str(phase),
                       pixels=np.array([node - 1]), npix=size,
                       ori_av=np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)]))
    graph.add_edges_from([(1, 2), (1, 3)])
    return graph


def test_neighbor_searches_exclude_other_phases():
    """
    Prevent another phase from winning size or orientation based merge selection.
    """
    graph = make_mixed_graph()
    assert find_largest_neighbor(graph, 1) == 3
    assert find_sim_neighbor(graph, 1)[0] == 3
    graph.remove_edge(1, 3)
    with pytest.raises(ValueError):
        find_largest_neighbor(graph, 1)
    with pytest.raises(ValueError):
        find_sim_neighbor(graph, 1)


def test_cross_phase_merge_rejected_before_mutation():
    """
    Leave every graph field unchanged when a direct cross phase merge is requested.
    """
    graph = make_mixed_graph()
    before = copy.deepcopy(graph)
    with pytest.raises(ValueError, match="phase"):
        merge_nodes(graph, 1, 2)
    assert set(graph.nodes) == set(before.nodes)
    assert set(graph.edges) == set(before.edges)
    np.testing.assert_array_equal(graph.graph["label_map"], before.graph["label_map"])
    for node in before:
        for key, value in before.nodes[node].items():
            np.testing.assert_equal(graph.nodes[node][key], value)
    merge_nodes(graph, 1, 3)
    assert graph.nodes[3]["phase_id"] == 2
    assert graph.nodes[3]["phase_name"] == "2"


def test_tagged_grain_does_not_merge_with_unknown_phase():
    """
    Reject mixed known and unknown phase metadata while supporting legacy graphs.
    """
    graph = make_mixed_graph()
    del graph.nodes[3]["phase_id"]
    with pytest.raises(ValueError, match="phase"):
        merge_nodes(graph, 1, 3)
    del graph.nodes[1]["phase_id"]
    merge_nodes(graph, 1, 3)
    assert 1 not in graph


@pytest.mark.parametrize("refresh", [False, True])
def test_misorientation_cache_excludes_cross_phase_edges(refresh):
    """
    Clear stale cross phase angles instead of treating them as merge candidates.

    Parameters
    ----------
    refresh : bool
        Recompute all edges or only the selected node's incident edges.
    """
    graph = make_mixed_graph()
    graph[1][2]["misorientation"] = 0.
    if refresh:
        _refresh_node_misorientation_cache(graph, 1)
    else:
        _compute_all_edge_misorientations(graph, get_proper_symmetry_quaternions(symmetry.Oh))
    assert "misorientation" not in graph[1][2]
    assert graph[1][3]["misorientation"] == pytest.approx(0.2)


@pytest.fixture
def square_map():
    """
    Build two phases with two adjacent grains each at known image positions.

    Returns
    -------
    kanapy.texture.EBSDmap
        Map analyzed without grain statistics or interactive figures.
    """
    rows, columns = np.indices((8, 20))
    phases = np.where(columns.ravel() < 10, 2, 7)
    euler = np.zeros((160, 3))
    euler[:, 0] = np.deg2rad((columns.ravel() // 5) * 20)
    emap = CrystalMap(
        rotations=Orientation.from_euler(euler), phase_id=phases,
        x=columns.ravel().astype(float), y=rows.ravel().astype(float),
        phase_list=PhaseList(names=["A", "B"], point_groups=["m-3m", "6/mmm"], ids=[2, 7]),
    )
    data = SimpleNamespace(emap=emap, sh_x=8, sh_y=20, dx=1., dy=1., npx=160)
    return EBSDmap(ebsd_data=data, graph_only=True)


def make_hex_data():
    """
    Build two phases occupying opposite sides of a translated staggered scan.

    Returns
    -------
    types.SimpleNamespace
        Measurements and original grid metadata, before EBSD analysis.
    """
    xy = []
    for row in range(6):
        for column in range(8 - row % 2):
            xy.append([10. + column + 0.5 * (row % 2), 20. + row * np.sqrt(3.) / 2.])
    xy = np.asarray(xy)
    phases = np.where(xy[:, 0] < 13.75, 2, 7)
    euler = np.zeros((45, 3))
    euler[phases == 7] = [0.4, 0.3, 0.2]
    emap = CrystalMap(
        rotations=Orientation.from_euler(euler), phase_id=phases, x=xy[:, 0], y=xy[:, 1],
        phase_list=PhaseList(names=["A", "B"], point_groups=["m-3m", "6/mmm"], ids=[2, 7]),
        prop={"ci": np.where(phases == 2, 0.8, 0.4)},
    )
    return SimpleNamespace(emap=emap, sh_x=6, sh_y=8, dx=1., dy=np.sqrt(3.) / 2.,
                           npx=48, original_npx=45, hex_grid=True)


def test_nodes_keep_phase_identity_and_regions_do_not_cross_phases(square_map):
    """
    Retain phase labels through cleanup and exclude pixels of the other phase.

    Parameters
    ----------
    square_map : kanapy.texture.EBSDmap
        Two phase map with independently known grain counts.
    """
    for data in square_map.ms_data:
        graph = data["graph"]
        assert len(graph) == 2
        for node in graph.nodes.values():
            assert node["phase_id"] == data["index"]
            assert node["phase_name"] == data["name"]
            np.testing.assert_array_equal(square_map.emap.phase_id[node["pixels"]], data["index"])


def test_multiphase_hex_uses_disjoint_masks_on_one_full_grid():
    """
    Preserve the left and right phases instead of expanding both across the image.
    """
    ebsd = EBSDmap(ebsd_data=make_hex_data(), graph_only=True, gs_min=1)
    first, second = ebsd.ms_data
    assert np.intersect1d(first["idx"], second["idx"]).size == 0
    np.testing.assert_array_equal(np.union1d(first["idx"], second["idx"]), np.arange(48))
    labels = ebsd._hex_phase_grid
    np.testing.assert_array_equal(labels[:, 0], 2)
    np.testing.assert_array_equal(labels[:, -1], 7)
    np.testing.assert_allclose(ebsd._hex_x_grid[0], np.arange(10., 18.))
    for data in ebsd.ms_data:
        assert len(data["graph"]) == 1
        expected_ci = 0.8 if data["index"] == 2 else 0.4
        np.testing.assert_allclose(ebsd._hex_ci_grid.ravel()[data["idx"]], expected_ci)
        angles = [0., 0., 0.] if data["index"] == 2 else [0.4, 0.3, 0.2]
        quaternion = Orientation.from_euler(angles).data[0]
        np.testing.assert_allclose(np.abs(data["ori"].data @ quaternion), 1.)


def test_hex_filter_leaving_one_phase_preserves_existing_resampling():
    """
    Keep the old one phase grid and filling behavior after the other phase is excluded.
    """
    ebsd = EBSDmap(ebsd_data=make_hex_data(), graph_only=True, gs_min=1, vf_min=0.49)
    assert len(ebsd.ms_data) == 1
    assert ebsd.ms_data[0]["index"] == 2
    assert not hasattr(ebsd, "_hex_phase_grid")
    np.testing.assert_array_equal(ebsd.ms_data[0]["idx"], np.arange(48))
    assert (ebsd.dx, ebsd.dy) == (1., np.sqrt(3.) / 2.)


@pytest.mark.parametrize("minor_pixels, expected_phases", [(2, [2]), (3, [2, 7])])
def test_phase_fraction_cutoff_and_equal_orientation_boundaries(minor_pixels, expected_phases):
    """
    Retain exactly three percent and separate touching phases with identical orientations.

    Parameters
    ----------
    minor_pixels : int
        Number of minor phase pixels among one hundred measurements.
    expected_phases : list of int
        Independently expected retained phase IDs at the default cutoff.
    """
    rows, columns = np.indices((10, 10))
    phases = np.full(100, 2)
    phases[np.array([0, 1, 10])[:minor_pixels]] = 7
    emap = CrystalMap(
        rotations=Orientation.from_euler(np.zeros((100, 3))), phase_id=phases,
        x=columns.ravel().astype(float), y=rows.ravel().astype(float),
        phase_list=PhaseList(names=["A", "B"], point_groups=["m-3m"] * 2, ids=[2, 7]),
    )
    loaded = SimpleNamespace(emap=emap, sh_x=10, sh_y=10, dx=1., dy=1., npx=100)
    ebsd = EBSDmap(ebsd_data=loaded, graph_only=True, gs_min=1)
    assert [data["index"] for data in ebsd.ms_data] == expected_phases
    for data in ebsd.ms_data:
        assert len(data["graph"]) == 1
        for node in data["graph"].nodes.values():
            np.testing.assert_array_equal(phases[node["pixels"]], data["index"])


def test_combined_overlay_preserves_background_and_phase_colors(square_map):
    """
    Plot both phase graphs on one IPF image, including graph only nodes without centers.

    Parameters
    ----------
    square_map : kanapy.texture.EBSDmap
        Two phases with two graph nodes and one edge each.
    """
    fig, ax = square_map.plot_graph_overlay(iphase=None, show=False)
    try:
        background = np.asarray(ax.images[0].get_array())
        np.testing.assert_array_equal(background[:, :10], square_map.ms_data[0]["rgb_im"][:, :10])
        np.testing.assert_array_equal(background[:, 10:], square_map.ms_data[1]["rgb_im"][:, 10:])
        markers = [line for line in ax.lines if line.get_marker() == "o"]
        edges = [line for line in ax.lines if len(line.get_xdata()) == 2]
        assert len(markers) == 4
        assert len(edges) == 3
        assert markers[0].get_color() == markers[1].get_color() == edges[0].get_color()
        assert markers[2].get_color() == markers[3].get_color() == edges[1].get_color()
        assert markers[0].get_color() != markers[2].get_color()
        np.testing.assert_allclose([line.get_xdata()[0] for line in markers], [2., 7., 12., 17.])
        np.testing.assert_allclose(edges[2].get_xdata(), [7., 12.])
        assert edges[2].get_color() == "gray"
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "A (phase 2)", "B (phase 7)", "Phase boundary",
        ]
    finally:
        plt.close(fig)


def test_multiphase_handoff_includes_all_retained_graphs(square_map, tmp_path):
    """
    Keep both phase graphs in optional files while preserving the selected RVE phase.

    Parameters
    ----------
    square_map : kanapy.texture.EBSDmap
        Analyzed map with two retained phases.
    tmp_path : pathlib.Path
        Temporary export directory.
    """
    result = EBSDGraphResult.from_phase_data(
        square_map.ms_data[0], 160, 8, 20, None, EBSDGraphConfig(), ebsd=square_map,
    )
    written = write_graph_result_outputs(result, EBSDGraphOutputOptions(
        output_dir=tmp_path, write_pickle=True, write_summary_json=True,
        write_graph_overlay=True, write_ipf_map=True,
    ))
    with written["pickle"].open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["phase"]["index"] == 2
    assert [row["index"] for row in payload["phases"]] == [2, 7]
    assert [len(row["graph"]) for row in payload["phases"]] == [2, 2]
    full_graph = payload["full_graph"]
    assert len(full_graph) == 4
    assert full_graph.number_of_edges() == 3
    assert result.full_graph is square_map.graph
    assert sum(edge["is_phase_boundary"] for *_, edge in full_graph.edges(data=True)) == 1
    summary = json.loads(written["summary_json"].read_text())
    assert [row["index"] for row in summary["phases"]] == [2, 7]
    assert summary["full_graph_node_count"] == 4
    assert summary["full_graph_edge_count"] == 3
    assert summary["phase_boundary_edge_count"] == 1
    assert all(path.stat().st_size > 0 for path in written.values())


def test_global_graph_preserves_contact_and_phase_identity(square_map):
    """
    Preserve the shared phase boundary without joining its two grains.

    Parameters
    ----------
    square_map : kanapy.texture.EBSDmap
        Four rectangular grains across two phases.
    """
    graph = square_map.graph
    assert len(graph) == 4
    assert set(graph.edges) == {(1, 2), (2, 3), (3, 4)}
    assert [graph.nodes[node]["phase_id"] for node in graph] == [2, 2, 7, 7]
    assert graph[2][3]["is_phase_boundary"] is True
    assert "misorientation" not in graph[2][3]
    assert graph[1][2]["is_phase_boundary"] is False
    np.testing.assert_array_equal(graph.graph["label_map"][0], np.repeat([1, 2, 3, 4], 5))
    with pytest.raises(ValueError, match="phase"):
        merge_nodes(graph, 2, 3)
    assert len(graph) == 4


def test_multiphase_segmenter_uses_each_crystal_symmetry():
    """
    Treat a quarter turn as cubic equivalence but not hexagonal equivalence.

    Equal orientations across the phase boundary must still remain separate.
    """
    phase_map = np.array([[2] * 4, [7] * 4])
    angles = np.zeros((2, 4, 3))
    angles[:, 2:, 0] = np.pi / 2
    rotations = Orientation.from_euler(angles.reshape(-1, 3)).data
    symmetries = {2: symmetry.Oh, 7: symmetry.D6h}
    labels, count = find_similar_regions_by_misorientation(
        rotations.reshape(2, 4, 4), np.ones((2, 4), dtype=bool), symmetries,
        phase_map=phase_map,
    )
    assert count == 3
    np.testing.assert_array_equal(labels, [[1, 1, 1, 1], [2, 2, 3, 3]])
    graph = build_graph_from_labeled_pixels(
        labels, rotations, symmetries, 1., 1., phase_map=phase_map,
    )
    assert graph[1][2]["is_phase_boundary"] is True
    assert graph[1][3]["is_phase_boundary"] is True
    assert graph[2][3]["is_phase_boundary"] is False
    np.testing.assert_allclose(np.abs(graph.nodes[1]["ori_av"][0]), 1.)


def test_multiphase_uses_one_component_labeling_pass(monkeypatch):
    """
    Label all retained phases together on the common rectangular grid.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Record the actual global connectivity calls without replacing their work.
    """
    import kanapy.texture as texture

    calls = []
    label_components = texture.connected_components_csgraph

    def record_components(adjacency, **kwargs):
        """
        Record the grid size and execute the real connectivity algorithm.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Pixel adjacency for the shared grid.
        **kwargs : object
            Connectivity options.

        Returns
        -------
        tuple
            Connected component count and pixel labels.
        """
        calls.append(adjacency.shape)
        return label_components(adjacency, **kwargs)

    monkeypatch.setattr(texture, "connected_components_csgraph", record_components)
    ebsd = EBSDmap(ebsd_data=make_hex_data(), graph_only=True, gs_min=1)
    assert calls == [(48, 48)]
    assert len(ebsd.graph) == 2
    assert ebsd.graph.number_of_edges() == 1


def test_same_phase_merge_preserves_cross_phase_contact():
    """
    Transfer an interface edge when a grain merges into its own phase neighbor.

    The global graph uses distinct symmetries, while interface edges never
    receive a same phase misorientation value.
    """
    graph = make_mixed_graph()
    graph.graph["symmetry"] = {2: symmetry.Oh, 7: symmetry.D6h}
    graph.graph["rotations"] = np.array([graph.nodes[node]["ori_av"] for node in graph])
    graph[1][2]["is_phase_boundary"] = True
    graph[1][3]["is_phase_boundary"] = False
    operators = get_proper_symmetry_quaternions(graph.graph["symmetry"])
    _compute_all_edge_misorientations(graph, operators)
    assert find_sim_neighbor(graph, 1)[0] == 3
    merge_nodes(graph, 1, 3)
    assert set(graph) == {2, 3}
    assert graph[2][3]["is_phase_boundary"] is True
    assert "misorientation" not in graph[2][3]
    np.testing.assert_array_equal(graph.graph["label_map"], [[3, 2, 3]])


def test_filtered_pixels_do_not_create_interface_edges():
    """
    Leave grains disconnected when excluded pixels separate their regions.
    """
    labels = np.array([[1, 0, 2], [1, 0, 2]])
    phases = np.array([[2, -1, 7], [2, -1, 7]])
    rotations = np.tile([1., 0., 0., 0.], (6, 1))
    graph = build_graph_from_labeled_pixels(
        labels, rotations, {2: symmetry.Oh, 7: symmetry.D6h}, 1., 1.,
        phase_map=phases,
    )
    assert len(graph) == 2
    assert graph.number_of_edges() == 0


def test_marked_grains_keep_distinct_colors_with_global_ids(monkeypatch):
    """
    Show each grain distinctly even when its phase has larger global IDs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Capture the real rendered colors instead of opening interactive windows.
    """
    rows, columns = np.indices((8, 30))
    angles = np.zeros((240, 3))
    angles[:, 0] = np.deg2rad((columns.ravel() // 5) * 20)
    emap = CrystalMap(
        rotations=Orientation.from_euler(angles),
        phase_id=np.where(columns.ravel() < 15, 2, 7),
        x=columns.ravel().astype(float), y=rows.ravel().astype(float),
        phase_list=PhaseList(names=["A", "B"], point_groups=["m-3m", "6/mmm"], ids=[2, 7]),
    )
    loaded = SimpleNamespace(emap=emap, sh_x=8, sh_y=30, dx=1., dy=1., npx=240)
    ebsd = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)
    colors = []
    annotations = []

    def capture_grains():
        """
        Read the actual grain colors and labels before closing each preview.
        """
        fig = plt.gcf()
        axes = fig.axes[0]
        image = axes.images[-1]
        rgba = image.to_rgba(image.get_array())
        mask = ebsd.ms_data[len(colors)]["graph"].graph["label_map"] > 0
        colors.append(len(np.unique(rgba[mask], axis=0)))
        annotations.append([text.get_text() for text in axes.texts])
        plt.close(fig)

    monkeypatch.setattr(plt, "show", capture_grains)
    ebsd.plot_grains_marked()
    assert colors == [3, 3]
    assert annotations == [["1", "2", "3"], ["4", "5", "6"]]
