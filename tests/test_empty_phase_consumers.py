"""
Keep retained empty phase records usable for reporting and safe for RVE input.
"""

import importlib.util
from pathlib import Path
import runpy
from types import SimpleNamespace
import warnings

import networkx as nx
import numpy as np
import pytest

import kanapy
from kanapy.graph_workflow import EBSDGraphResult


def _load_example(relative_path):
    """
    Import an example without invoking its main workflow.

    Parameters
    ----------
    relative_path : str
        Path relative to the repository examples folder.

    Returns
    -------
    module
        Example functions and editable settings.
    """
    path = Path(__file__).parents[1] / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def empty_result():
    """
    Provide a complete handoff for a phase removed from the analysis graph.

    Returns
    -------
    EBSDGraphResult
        Result with provenance metadata but unavailable shape statistics.
    """
    phase = {
        "index": 2, "name": "Removed phase", "vf": 0.6, "ngrains": 0,
        "graph": nx.Graph(), "statistics_available": False,
        "grain_euler_angles": {},
    }
    for prefix in ("gs", "ar", "om"):
        phase[f"{prefix}_data"] = np.array([])
        phase[f"{prefix}_param"] = None
        phase[f"{prefix}_moments"] = None
    return EBSDGraphResult(
        graph=phase["graph"], phase_data=phase, phase_index=2,
        phase_name=phase["name"], rgb_image=np.zeros((1, 1, 3)),
        npx=1, sh_x=1, sh_y=1, ebsd=SimpleNamespace(ms_data=[phase]),
    )


def test_analysis_summary_reports_unavailable_statistics_without_nan(empty_result, capsys):
    """
    Explain an empty phase instead of evaluating medians on empty arrays.

    Parameters
    ----------
    empty_result : EBSDGraphResult
        Handoff retaining an absorbed or excluded phase.
    capsys : pytest.CaptureFixture
        Captured user facing summary.
    """
    example = _load_example("EBSD_analysis/graph_based_analysis.py")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        example.print_graph_handoff_summary(empty_result)
    output = capsys.readouterr().out
    assert "Removed phase" in output
    assert "unavailable" in output.lower()
    assert "nan" not in output.lower()


def test_complete_rve_example_rejects_empty_selected_phase(empty_result, tmp_path, monkeypatch):
    """
    Stop before plotting or fitting when the selected original major phase is empty.

    Parameters
    ----------
    empty_result : EBSDGraphResult
        Handoff with no geometry available for the selected phase.
    tmp_path : pathlib.Path
        Temporary output folder that protects existing example results.
    monkeypatch : pytest.MonkeyPatch
        Substitute only the already tested EBSD analysis result.
    """
    example = _load_example("EBSD_to_RVE/graph_to_rve.py")

    def build_graph(*args, **kwargs):
        """
        Return the empty selected phase at the EBSD to RVE handoff.

        Parameters
        ----------
        *args, **kwargs : object
            Input and output settings passed by the example.

        Returns
        -------
        EBSDGraphResult
            Existing analysis result.
        """
        return empty_result

    monkeypatch.setattr(example.knpy, "build_ebsd_graph", build_graph)
    args = example.parse_args(["--output-dir", str(tmp_path), "--no-show"])
    with pytest.raises(ValueError, match="no grain shape statistics.*RVE"):
        example.run_workflow(args)


def test_original_rve_example_rejects_empty_first_phase(empty_result, monkeypatch):
    """
    Keep the original first phase selection and reject unavailable geometry early.

    Parameters
    ----------
    empty_result : EBSDGraphResult
        Analysis whose first retained record has become empty.
    monkeypatch : pytest.MonkeyPatch
        Substitute the analyzer before the standalone script can plot or write.
    """
    def analyze(*args, **kwargs):
        """
        Supply the completed analysis at the example's normal loading boundary.

        Parameters
        ----------
        *args, **kwargs : object
            EBSD settings supplied by the original example.

        Returns
        -------
        object
            Analysis containing the retained empty phase.
        """
        return empty_result.ebsd

    monkeypatch.setattr(kanapy, "EBSDmap", analyze)
    path = Path(__file__).parents[1] / "examples/EBSD_to_RVE/convert_ang2rve.py"
    with pytest.raises(ValueError, match="no grain shape statistics.*RVE"):
        runpy.run_path(str(path), run_name="__main__")
