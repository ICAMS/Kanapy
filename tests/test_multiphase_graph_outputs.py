"""
Verify phase colors, isolated cleanup provenance, and reusable graph exports.
"""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np

from kanapy.graph_workflow import (
    EBSDGraphConfig, EBSDGraphResult, EBSDGraphOutputOptions,
    load_graph_result, write_graph_result_outputs,
)
from kanapy.multiphase_plot import phase_graph_colors, plot_multiphase_graph_overlay


def make_result():
    """
    Build two touching phases and an exclusion record from the smaller phase.

    Returns
    -------
    EBSDGraphResult
        Complete overview with a separate selected phase graph.
    """
    image = np.ones((4, 6, 3))
    phases = []
    overview = nx.Graph(phases=[
        {"id": 7, "name": "Matrix", "vf": 0.6},
        {"id": 2, "name": "Minor", "vf": 0.4},
    ])
    for phase_id, name, fraction, pixels in [
        (7, "Matrix", 0.6, np.array([0, 1, 6, 7])),
        (2, "Minor", 0.4, np.array([2, 3, 8, 9])),
    ]:
        graph = nx.Graph()
        graph.add_node(1, phase_id=phase_id, pixels=pixels, npix=len(pixels))
        phases.append({"index": phase_id, "name": name, "vf": fraction,
                       "graph": graph, "idx": pixels, "rgb_im": image})
        overview.add_node(len(phases), **graph.nodes[1])
    overview.add_edge(1, 2, boundary_type="interphase")
    overview.graph["isolated_cleanup_records"] = [
        {"action": "exclude", "source_phase_id": 2, "npix": 3, "pixels": [20, 21, 22]},
        {"action": "merge", "source_phase_id": 2, "npix": 2, "pixels": [18, 19]},
    ]
    ebsd = SimpleNamespace(ms_data=phases, graph=overview)
    return EBSDGraphResult.from_phase_data(
        phases[0], 24, 4, 6, None, EBSDGraphConfig(), ebsd=ebsd,
    )


def test_saved_overview_keeps_secondary_cleanup_and_can_be_replotted(tmp_path):
    """
    Preserve all phases and exclusions after the originating analyzer is gone.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    """
    result = make_result()
    files = write_graph_result_outputs(result, EBSDGraphOutputOptions(
        output_dir=tmp_path, write_pickle=True,
    ))
    restored = load_graph_result(files["pickle"])
    assert restored.ebsd is None
    assert len(restored.graph) == 1
    assert len(restored.full_graph) == 2
    assert restored.full_graph.nodes[2]["phase_id"] == 2
    assert restored.to_summary()["isolated_cleanup"] == {
        "merged_region_count": 1, "excluded_region_count": 1, "excluded_pixel_count": 3,
    }
    assert restored.to_summary()["phase_boundary_edge_count"] == 1
    assert restored.to_summary()["phases"][0]["vf"] == 0.6
    output = write_graph_result_outputs(restored, EBSDGraphOutputOptions(
        output_dir=tmp_path / "restored", write_graph_overlay=True, write_ipf_map=True,
    ))
    assert all(path.stat().st_size > 1000 for path in output.values())
    payload = restored.to_payload()
    assert [phase["index"] for phase in payload["phases"]] == [7, 2]
    assert payload["full_graph"] is restored.full_graph


def test_direct_result_construction_keeps_attached_overview():
    """
    Preserve the prior full graph contract for direct dataclass construction.
    """
    result = make_result()
    direct = replace(result, overview_graph=None, overview_rgb_image=None, phase_records=[])
    assert direct.full_graph is result.ebsd.graph
    assert len(direct.to_payload()["phases"]) == 2
    np.testing.assert_array_equal(direct.overview_rgb_image, result.overview_rgb_image)


def test_palette_does_not_repeat_for_many_retained_phases():
    """
    Distinguish up to thirty three phases above the default fraction cutoff.
    """
    phases = [{"id": phase_id} for phase_id in range(33)]
    colors = phase_graph_colors(phases)
    assert len(set(colors.values())) == 33
    assert colors[0] == (0.0, 0.0, 0.0)


def test_old_single_phase_pickle_still_loads(tmp_path):
    """
    Accept existing handoffs without optional overview fields.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for a trusted legacy payload.
    """
    result = make_result()
    result.ebsd = None
    result.overview_graph = None
    result.overview_rgb_image = None
    payload = result.to_payload()
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(payload))
    restored = load_graph_result(path)
    assert restored.full_graph is restored.graph
    assert restored.overview_rgb_image is None


def test_old_multiphase_pickle_keeps_graph_when_overview_image_was_not_saved(tmp_path):
    """
    Use the available phase image while restoring the previous full graph format.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for a trusted legacy payload and restored plot.
    """
    payload = make_result().to_payload()
    del payload["overview_graph"]
    del payload["overview_rgb_image"]
    del payload["full_graph"].graph["phases"]
    path = tmp_path / "legacy-multiphase.pkl"
    path.write_bytes(pickle.dumps(payload))
    restored = load_graph_result(path)
    assert len(restored.full_graph) == 2
    assert restored.full_graph.graph["phases"][0]["id"] == 7
    np.testing.assert_array_equal(restored.overview_rgb_image, restored.rgb_image)
    paths = write_graph_result_outputs(restored, EBSDGraphOutputOptions(
        output_dir=tmp_path, write_graph_overlay=True,
    ))
    assert paths["graph_overlay"].stat().st_size > 1000


def test_phase_markers_and_cross_contacts_use_distinct_styles():
    """
    Use black for the largest original phase and dashed gray for its interface.
    """
    result = make_result()
    graph = result.full_graph
    graph.graph["phases"].reverse()
    fig, axis = plot_multiphase_graph_overlay(graph, np.ones((4, 6, 3)))
    try:
        markers = [line for line in axis.lines if line.get_marker() == "o"]
        edge = [line for line in axis.lines if len(line.get_xdata()) == 2][0]
        assert to_rgba(markers[0].get_color()) == to_rgba("black")
        assert to_rgba(markers[1].get_color()) == to_rgba("#0072B2")
        assert to_rgba(edge.get_color()) == to_rgba("#555555")
        assert edge.get_linestyle() == "--"
        assert len(edge.get_path_effects()) == 2
        assert [label.get_text() for label in axis.get_legend().get_texts()] == [
            "Phase #7 (Matrix): 60.0%", "Phase #2 (Minor): 40.0%", "Cross phase boundary",
        ]
    finally:
        plt.close(fig)


def test_phase_legend_fits_inside_portrait_canvas():
    """
    Keep the phase legend visible in previews and PNGs without tight cropping.
    """
    result = make_result()
    fig, axis = plot_multiphase_graph_overlay(result.full_graph, np.ones((60, 40, 3)))
    try:
        fig.canvas.draw()
        legend = axis.get_legend().get_window_extent()
        assert legend.x0 >= fig.bbox.x0
        assert legend.y0 >= fig.bbox.y0
        assert legend.x1 <= fig.bbox.x1
        assert legend.y1 <= fig.bbox.y1
    finally:
        plt.close(fig)


def test_unavailable_statistics_remain_explicit_in_export():
    """
    Preserve empty phase provenance while marking its statistics unavailable.
    """
    result = make_result()
    result.phase_data["statistics_available"] = False
    result.phase_data["ngrains"] = 0
    for prefix in ("gs", "ar", "om"):
        result.phase_data[f"{prefix}_param"] = None
        result.phase_data[f"{prefix}_moments"] = None
        result.phase_data[f"{prefix}_data"] = np.array([])
    statistics = result.to_statistics()["experimental"]
    assert statistics["statistics_available"] is False
    assert statistics["gs_param"] is None
    assert statistics["ngrains"] == 0
