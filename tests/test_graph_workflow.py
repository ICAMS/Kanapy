"""
Tests for the EBSD graph workflow objects.

These tests keep the graph handoff in memory and make file outputs explicit.
"""

import json

import networkx as nx
import numpy as np

from kanapy.graph_workflow import (
    EBSD_GRAPH_FILENAME,
    GRAPH_IPF_MAP_FILENAME,
    GRAPH_OVERLAY_FILENAME,
    GRAPH_SUMMARY_FILENAME,
    EBSDGraphConfig,
    EBSDGraphOutputOptions,
    EBSDGraphResult,
    write_graph_result_outputs,
)


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
    assert not list(tmp_path.glob("graph_step1*.png"))

    with written_files["summary_json"].open(encoding="utf-8") as file_handle:
        summary = json.load(file_handle)

    assert summary["graph_node_count"] == 2
    assert summary["graph_edge_count"] == 1
    assert summary["graph_overlay"] == GRAPH_OVERLAY_FILENAME
    assert summary["ipf_map"] == GRAPH_IPF_MAP_FILENAME
