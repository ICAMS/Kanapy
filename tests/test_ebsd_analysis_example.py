"""
Verify complete two dimensional analysis without starting RVE generation.
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest


def load_analysis_example():
    """
    Load the standalone example without executing its workflow.

    Returns
    -------
    module
        Example settings and analysis functions.
    """
    path = Path(__file__).parents[1] / "examples/EBSD_analysis/graph_based_analysis.py"
    spec = importlib.util.spec_from_file_location("graph_based_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_example(monkeypatch, output_dir):
    """
    Keep all example output in the test directory, including before the change.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Temporary replacements for example settings.
    output_dir : pathlib.Path
        Destination isolated from user results.

    Returns
    -------
    module
        Example using real EBSD analysis and a temporary output directory.
    """
    example = load_analysis_example()
    monkeypatch.setattr(example, "output_dir", output_dir, raising=False)
    build_graph = example.knpy.build_ebsd_graph

    def build_analysis(*args, **kwargs):
        """
        Run the real analysis while preventing writes to existing user results.

        Parameters
        ----------
        *args, **kwargs : object
            Arguments supplied by the example.

        Returns
        -------
        kanapy.EBSDGraphResult
            Complete analysis of the actual ANG fixture.
        """
        assert kwargs.get("include_statistics") is True
        kwargs["output_options"] = replace(kwargs["output_options"], output_dir=output_dir)
        return build_graph(*args, **kwargs)

    monkeypatch.setattr(example.knpy, "build_ebsd_graph", build_analysis)
    return example


@pytest.mark.parametrize("preview", [False, True])
def test_standalone_exports_statistics_and_all_ebsd_plots(tmp_path, monkeypatch, preview):
    """
    Produce numerical distributions and real plots without constructing an RVE.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Replacements for output location and the forbidden RVE stage.
    preview : bool
        Whether saved figures are also previewed interactively.
    """
    example = configure_example(monkeypatch, tmp_path)
    monkeypatch.setattr(example, "show_figures", preview)
    monkeypatch.setitem(plt.rcParams, "interactive", preview)
    shown_titles = []
    preview_blocks = []

    def record_preview(*args, **kwargs):
        """
        Record the actual figure offered for interactive display.

        Parameters
        ----------
        *args, **kwargs : object
            Display options from the example or plotting context.
        """
        shown_titles.append(plt.gcf().axes[0].get_title())
        preview_blocks.append(kwargs.get("block", not plt.isinteractive()))

    monkeypatch.setattr(plt, "show", record_preview)

    def reject_rve(*args, **kwargs):
        """
        Reject an accidental transition from analysis to RVE construction.

        Parameters
        ----------
        *args, **kwargs : object
            Unexpected microstructure constructor arguments.
        """
        raise AssertionError("Standalone EBSD analysis must not construct an RVE")

    monkeypatch.setattr(example.knpy, "Microstructure", reject_rve)
    result = example.main()
    statistics = json.loads((tmp_path / "statistics.json").read_text())
    assert statistics["units"]["length"] == "um"
    assert statistics["units"]["tilt_angle"] == "rad"
    for prefix in ("gs", "ar", "om"):
        values = statistics["experimental"][f"{prefix}_data"]
        assert len(values) == statistics["experimental"]["ngrains"]
        assert len(values) > 1
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, result.phase_data[f"{prefix}_data"])
    assert len(list(tmp_path.glob("ebsd_histograms_*.png"))) == 3
    for stage in ("ebsd_misorientation", "ebsd_ipf", "ebsd_segmentation",
                  "ebsd_pole_figures", "ebsd_ipf_projection"):
        assert list(tmp_path.glob(f"{stage}_*.png")), stage
    for filename in ("ebsd_graph.pkl", "ebsd_graph_summary.json",
                     "00_graph_overlay.png", "graph_ipf_map.png"):
        assert (tmp_path / filename).is_file()
    assert sorted(path.name for path in tmp_path.glob("*.png"))[0] == "00_graph_overlay.png"
    if preview:
        assert "graph" in shown_titles[0].lower()
        assert all(preview_blocks)
    else:
        assert shown_titles == []
    assert plt.isinteractive() is preview
    for path in tmp_path.glob("*.png"):
        assert plt.imread(path).size > 0
    summary = json.loads((tmp_path / "ebsd_graph_summary.json").read_text())
    assert summary["statistics_json"] == "statistics.json"
    assert not list(tmp_path.glob("*voxels*"))
    assert plt.get_fignums() == []


def test_standalone_can_return_statistics_without_writing_files(tmp_path, monkeypatch):
    """
    Keep analysis in memory when every example output switch is disabled.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary destination that must remain absent.
    monkeypatch : pytest.MonkeyPatch
        Temporary output settings.
    """
    destination = tmp_path / "disabled"
    example = configure_example(monkeypatch, destination)
    for setting in ("write_graph_pickle", "write_graph_summary", "write_graph_overlay",
                    "write_ipf_map", "write_statistics_json", "write_analysis_figures",
                    "clear_output_dir", "show_figures"):
        monkeypatch.setattr(example, setting, False, raising=False)
    result = example.main()
    assert len(result.phase_data["gs_data"]) > 1
    assert result.written_files == {}
    assert not destination.exists()
    assert plt.get_fignums() == []


def test_statistics_export_keeps_every_phase_and_its_identity(tmp_path):
    """
    Preserve distinct distributions even when the selected phase is not first.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary statistics destination.
    """
    example = load_analysis_example()
    phases = []
    for phase_id, fraction in ((7, 0.3), (2, 0.7)):
        phase = {"index": phase_id, "name": f"phase {phase_id}", "vf": fraction,
                 "ngrains": 2, "graph": nx.path_graph(2)}
        for prefix in ("gs", "ar", "om"):
            phase[f"{prefix}_data"] = np.array([phase_id, phase_id + 0.5])
            phase[f"{prefix}_param"] = np.array([0.2, 0.0, 1.4])
            phase[f"{prefix}_moments"] = np.array([1.4, 0.2])
        phase["om_param"] = np.array([0.2, 0.0])
        phases.append(phase)
    selected = phases[1]
    result = example.knpy.EBSDGraphResult(
        graph=selected["graph"], phase_data=selected, phase_index=2,
        phase_name=selected["name"], rgb_image=np.zeros((1, 2, 3)),
        npx=2, sh_x=1, sh_y=2, ebsd=SimpleNamespace(ms_data=phases),
    )
    example.write_statistics(tmp_path / "statistics.json", result)
    payload = json.loads((tmp_path / "statistics.json").read_text())
    assert payload["experimental"]["index"] == 2
    exported = {phase["experimental"]["index"]: phase for phase in payload["phases"]}
    assert set(exported) == {2, 7}
    assert exported[7]["experimental"]["gs_data"] == [7.0, 7.5]
    assert exported[2]["experimental"]["gs_data"] == [2.0, 2.5]
    assert exported[7]["experimental"]["vf"] == 0.3
    assert exported[2]["units"]["tilt_angle"] == "rad"
