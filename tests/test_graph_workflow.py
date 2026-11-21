"""
Tests for the EBSD graph workflow objects.

These tests keep the graph handoff in memory and make file outputs explicit.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from kanapy.graph_workflow import (
    EBSD_GRAPH_FILENAME,
    GRAPH_IPF_MAP_FILENAME,
    GRAPH_OVERLAY_FILENAME,
    GRAPH_SUMMARY_FILENAME,
    EBSDGraphConfig,
    EBSDGraphOutputOptions,
    EBSDGraphResult,
    build_ebsd_graph,
    write_graph_result_outputs,
)
import kanapy.graph_workflow as graph_workflow


def make_phase_data():
    """
    Build a minimal phase row for graph workflow tests.

    The row mirrors the fields produced by ``EBSDmap.ms_data`` that are needed
    for plotting and serialization.

    Returns
    -------
    dict
        Minimal phase data containing a graph and an RGB image.
    """
    graph = nx.Graph()
    graph.graph["label_map"] = np.array([[1, 1], [2, 2]], dtype=int)
    graph.graph["dx"] = 1.0
    graph.graph["dy"] = 1.0
    graph.add_node(1, pixels=np.array([0, 1]), npix=2)
    graph.add_node(2, pixels=np.array([2, 3]), npix=2)
    graph.add_edge(1, 2)

    return {
        "index": 0,
        "name": "Ni",
        "vf": 1.0,
        "graph": graph,
        "rgb_im": np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ],
            dtype=float,
        ),
    }


def test_graph_config_exposes_user_thresholds():
    """
    Graph configuration exposes defaults and user overrides.
    """
    config = EBSDGraphConfig()

    assert config.min_phase_pixel_fraction == 0.03
    assert config.max_misorientation_degrees == 5.0
    assert config.min_grain_size_pixels == 10.0
    assert config.connectivity == 4
    assert config.boundary_artifact_size_factor == 2.0
    assert config.boundary_artifact_fraction_min == 0.85
    assert config.convex_hull_min_points == 3
    assert config.similar_orientation_merge_degrees == 5.0

    custom = EBSDGraphConfig(
        min_phase_pixel_fraction=0.05,
        max_misorientation_degrees=4.0,
        min_grain_size_pixels=25.0,
        connectivity=8,
        boundary_artifact_size_factor=3.0,
        boundary_artifact_fraction_min=0.75,
        convex_hull_min_points=4,
        similar_orientation_merge_degrees=3.5,
    )

    assert custom.to_dict()["connectivity"] == 8
    assert custom.to_dict()["similar_orientation_merge_degrees"] == 3.5


def test_graph_workflow_is_exported_from_kanapy():
    """
    Top level kanapy exports the graph workflow entry points.
    """
    import kanapy as knpy

    assert knpy.EBSDGraphConfig is EBSDGraphConfig
    assert knpy.EBSDGraphOutputOptions is EBSDGraphOutputOptions
    assert knpy.EBSDGraphResult is EBSDGraphResult
    assert callable(knpy.build_ebsd_graph)


def test_graph_outputs_are_explicit_and_exclude_step_figures(tmp_path):
    """
    Selected graph outputs match the paper figure handoff files.
    """
    result = EBSDGraphResult.from_phase_data(
        phase_data=make_phase_data(),
        npx=4,
        sh_x=2,
        sh_y=2,
        source_input_file=tmp_path / "source.ang",
        config=EBSDGraphConfig(),
    )
    options = EBSDGraphOutputOptions(
        output_dir=tmp_path,
        write_pickle=True,
        write_summary_json=True,
        write_graph_overlay=True,
        write_ipf_map=True,
        write_step_figures=False,
    )

    written_files = write_graph_result_outputs(result, options)

    assert set(written_files) == {
        "pickle",
        "summary_json",
        "graph_overlay",
        "ipf_map",
    }
    assert written_files["pickle"].name == EBSD_GRAPH_FILENAME
    assert written_files["summary_json"].name == GRAPH_SUMMARY_FILENAME
    assert written_files["graph_overlay"].name == GRAPH_OVERLAY_FILENAME
    assert written_files["ipf_map"].name == GRAPH_IPF_MAP_FILENAME
    assert "step0" not in GRAPH_IPF_MAP_FILENAME
    assert not list(tmp_path.glob("graph_step1*.png"))

    with written_files["summary_json"].open(encoding="utf-8") as file_handle:
        summary = json.load(file_handle)

    assert summary["graph_node_count"] == 2
    assert summary["graph_edge_count"] == 1
    assert summary["graph_overlay"] == GRAPH_OVERLAY_FILENAME
    assert summary["ipf_map"] == GRAPH_IPF_MAP_FILENAME


def test_build_ebsd_graph_uses_graph_only_map_analysis(tmp_path, monkeypatch):
    """
    Graph workflow requests graph only EBSD analysis.

    This keeps the example focused on the graph handoff instead of running the
    full statistical EBSD analysis path.
    """
    calls = []

    class FakeEBSDMap:
        """
        Minimal EBSD map replacement for workflow tests.
        """

        def __init__(self, *_args, **kwargs):
            """
            Record constructor options and expose one phase row.

            Parameters
            ----------
            *_args : tuple
                Positional arguments passed by the workflow.
            **kwargs : dict
                Keyword arguments passed by the workflow.
            """
            calls.append(kwargs)
            self.npx = 4
            self.sh_x = 2
            self.sh_y = 2
            self.ms_data = [make_phase_data()]

    input_file = tmp_path / "source.ang"
    input_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(graph_workflow, "EBSDmap", FakeEBSDMap)

    result = build_ebsd_graph(
        input_file=input_file,
        config=EBSDGraphConfig(),
        output_options=EBSDGraphOutputOptions(),
    )

    assert calls[0]["graph_only"] is True
    assert result.graph.number_of_nodes() == 2


def test_graph_statistics_match_full_ebsd_analysis():
    """
    Return complete statistics for the same grains as the full EBSD analysis.

    A real map catches graph results that omit statistics or calculate them
    from a different segmentation than the graph returned to the caller.
    """
    from kanapy import EBSDmap

    input_file = (Path(__file__).resolve().parents[1] / "examples" / "fixtures"
                  / "ebsd_316L_500x500.ang")
    config = EBSDGraphConfig(min_grain_size_pixels=20.0)
    result = build_ebsd_graph(input_file, config=config, include_statistics=True)
    reference = EBSDmap(
        str(input_file), gs_min=20.0, show_plot=False, show_hist=False,
        show_grains=False, show_graph=False,
    )
    expected = reference.ms_data[0]
    actual = result.phase_data

    assert result.ebsd is not None
    assert actual is result.ebsd.ms_data[0]
    assert actual["name"] == "Iron fcc"
    assert actual["ngrains"] == result.graph.number_of_nodes()
    assert actual["ngrains"] > 0
    for quantity, parameter_count in (("gs", 3), ("ar", 3), ("om", 2)):
        values = actual[f"{quantity}_data"]
        assert len(values) == actual["ngrains"]
        assert np.all(np.isfinite(values))
        assert actual[f"{quantity}_param"].shape == (parameter_count,)
        for suffix in ("data", "param", "moments"):
            np.testing.assert_allclose(
                actual[f"{quantity}_{suffix}"], expected[f"{quantity}_{suffix}"])

    np.testing.assert_allclose(
        np.sort(actual["gs_data"]),
        np.sort([node["equ_dia"] for _, node in result.graph.nodes(data=True)]),
    )
    assert result.written_files == {}


def test_selected_phase_position_resolves_filtered_original_id():
    """
    Keep an original phase ID distinct from its filtered list position.
    """
    phase = make_phase_data()
    phase["index"] = 7
    result = EBSDGraphResult.from_phase_data(
        phase, 4, 2, 2, None, EBSDGraphConfig(),
        ebsd=SimpleNamespace(ms_data=[{"index": 2}, phase]),
    )
    assert result.phase_index == 7
    assert result.selected_phase_position == 1


@pytest.mark.parametrize("phase_rows", [None, [{"index": 2}]])
def test_selected_phase_position_rejects_unavailable_phase(phase_rows):
    """
    Reject an absent source map or a phase missing from its retained results.

    Parameters
    ----------
    phase_rows : list of dict or None
        Retained phase rows, or no originating map.
    """
    ebsd = None if phase_rows is None else SimpleNamespace(ms_data=phase_rows)
    result = EBSDGraphResult.from_phase_data(
        make_phase_data(), 4, 2, 2, None, EBSDGraphConfig(), ebsd=ebsd,
    )
    with pytest.raises(ValueError):
        _ = result.selected_phase_position


@pytest.mark.parametrize("include_euler_angles", [False, True])
def test_statistics_payload_preserves_arrays_units_and_fit_conventions(include_euler_angles):
    """
    Expose reusable EBSD statistics without requiring a particular RVE report.

    Parameters
    ----------
    include_euler_angles : bool
        Whether the phase contains per-grain Euler angle metadata.
    """
    phase = make_phase_data()
    phase["ngrains"] = 2
    phase["gs_data"] = np.array([1.0, 2.0])
    phase["ar_data"] = np.array([1.0, 3.0])
    phase["om_data"] = np.array([0.2, 0.4])
    for prefix in ("gs", "ar", "om"):
        phase[f"{prefix}_param"] = np.array([0.2, 0.0] if prefix == "om" else [0.2, 0.0, 1.4])
        phase[f"{prefix}_moments"] = [1.4, 0.2]
    if include_euler_angles:
        phase["grain_euler_angles"] = {"grain 1": [10., 20., 30.], "grain 2": [40., 50., 60.]}
    result = EBSDGraphResult.from_phase_data(
        phase, 4, 2, 2, None, EBSDGraphConfig(),
    )
    payload = result.to_statistics()
    units = {"length": "um", "tilt_angle": "rad", "aspect_ratio": "1"}
    if include_euler_angles:
        units["grain_euler_angles"] = "deg"
        assert payload["experimental"]["grain_euler_angles"] == phase["grain_euler_angles"]
    assert payload["units"] == units
    assert payload["experimental"]["ngrains"] == 2
    assert payload["experimental"]["index"] == 0
    np.testing.assert_array_equal(payload["experimental"]["gs_data"], [1.0, 2.0])
    np.testing.assert_array_equal(payload["experimental"]["om_data"], [0.2, 0.4])
    for prefix in ("gs", "ar", "om"):
        for suffix in ("data", "param", "moments"):
            np.testing.assert_array_equal(payload["experimental"][f"{prefix}_{suffix}"], phase[f"{prefix}_{suffix}"])
    assert payload["fit_conventions"]["om_param"] == "von Mises [kappa, location] fitted to 2 * angle - pi"
    assert "rve_descriptor" not in payload


def test_statistics_payload_rejects_results_without_statistics():
    """
    Explain how callers can request statistics when using graph analysis only.
    """
    result = EBSDGraphResult.from_phase_data(
        make_phase_data(), 4, 2, 2, None, EBSDGraphConfig(),
    )
    with pytest.raises(ValueError, match="include_statistics=True"):
        result.to_statistics()


def test_graph_overlay_uses_physical_xy_centers_with_scan_origin():
    """
    Project physical x and y coordinates onto the same pixels as the image.
    """
    graph = nx.Graph(dx=2.0, dy=3.0, origin_x=10.0, origin_y=20.0)
    graph.add_node(1, center=np.array([14.0, 23.0]))
    assert graph_workflow._node_center(graph, 1, (3, 5)) == (2.0, 1.0)


def test_graph_overlay_keeps_legacy_row_column_handoffs_usable():
    """
    Preserve old graph centers whose metadata predates physical x and y.
    """
    graph = nx.Graph(dx=2.0, dy=3.0)
    graph.add_node(1, center=np.array([2.0, 6.0]))
    assert graph_workflow._node_center(graph, 1, (3, 5)) == (2.0, 1.0)


@pytest.mark.parametrize("spacing", [0.0, -1.0])
def test_graph_config_rejects_nonpositive_hex_spacing(spacing):
    """
    Reject physical grid spacings that cannot define a hex resampling grid.

    Parameters
    ----------
    spacing : float
        Invalid physical target spacing.
    """
    with pytest.raises(ValueError, match="hex_grid_target_spacing"):
        EBSDGraphConfig(hex_grid_target_spacing=spacing)


def test_graph_workflow_forwards_physical_hex_spacing(tmp_path, monkeypatch):
    """
    Carry a caller's physical resampling request into EBSD graph analysis.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary input location.
    monkeypatch : pytest.MonkeyPatch
        Replacement for the expensive map construction boundary.
    """
    options = {}

    class RecordingMap:
        """
        Record graph construction options and return a minimal retained phase.
        """

        def __init__(self, *args, **kwargs):
            """
            Record the map boundary settings used by the public workflow.

            Parameters
            ----------
            *args, **kwargs : object
                EBSD map construction options.
            """
            options.update(kwargs)
            self.ms_data = [make_phase_data()]
            self.npx, self.sh_x, self.sh_y = 4, 2, 2

    source = tmp_path / "source.ang"
    source.touch()
    monkeypatch.setattr(graph_workflow, "EBSDmap", RecordingMap)
    config = EBSDGraphConfig(hex_grid_target_spacing=0.75)
    result = build_ebsd_graph(source, config=config)
    assert options["hex_grid_target_spacing"] == 0.75
    assert result.config.to_dict()["hex_grid_target_spacing"] == 0.75
