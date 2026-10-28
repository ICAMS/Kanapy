"""
Exercise the file and plotting behavior of the complete EBSD example.
"""

import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import pytest
from orix.quaternion import symmetry


def load_example():
    """
    Load the example without launching the expensive RVE pipeline.

    Returns
    -------
    module
        Example entry point and its focused output helpers.
    """
    path = Path(__file__).parents[1] / "examples/EBSD_to_RVE/graph_to_rve.py"
    spec = importlib.util.spec_from_file_location("graph_to_rve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_parameters_control_defaults_and_cli_overrides(tmp_path, monkeypatch):
    """
    Use the editable example settings unless the command line overrides them.

    This prevents the visible parameter block from becoming disconnected from
    the actual run configuration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Alternate input and output paths.
    monkeypatch : pytest.MonkeyPatch
        Temporary edits to the example settings.
    """
    example = load_example()
    assert example.parse_args([]).connectivity == 4
    settings = {
        "fname": tmp_path / "measurement.ang", "output_dir": tmp_path / "results",
        "nvox": 12, "box_length": 30, "periodic": True, "seed": 42,
        "min_size": 20., "max_angle": 7.5, "vf_min": 0.05,
        "connectivity": 4, "show_figures": True, "extra_outputs": True,
    }
    for name, value in settings.items():
        monkeypatch.setattr(example, name, value)

    args = example.parse_args([])

    assert args.input == settings["fname"]
    assert args.output_dir == settings["output_dir"]
    assert (args.voxels, args.size, args.periodic, args.seed) == (12, 30, True, 42)
    assert (args.min_size, args.max_angle, args.vf_min) == (20., 7.5, 0.05)
    assert args.show and args.extra_outputs
    overridden = example.parse_args(["--connectivity", "8", "--min-size", "15", "--no-show"])
    assert (overridden.connectivity, overridden.min_size, overridden.show) == (8, 15., False)
    assert overridden.voxels == 12
    assert overridden.max_angle == 7.5


def test_manifest_tracks_only_current_artifacts(tmp_path):
    """
    Exclude stale files while retaining numeric metadata in JSON.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    example = load_example()
    (tmp_path / "stale.png").write_bytes(b"old")
    current = tmp_path / "current.png"
    current.write_bytes(b"new")
    target = tmp_path / "manifest.json"
    example.write_manifest(target, {"dimensions": np.array([30, 30, 30])}, [current])
    manifest = json.loads(target.read_text())
    assert manifest["dimensions"] == [30, 30, 30]
    assert manifest["files"] == ["current.png"]
    assert (tmp_path / "stale.png").read_bytes() == b"old"


def test_statistics_preserve_raw_distributions_and_units(tmp_path):
    """
    Export experimental arrays as numbers with their statistical meaning.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    example = load_example()
    data = {"index": 7, "name": "austenite", "vf": np.float64(0.9), "ngrains": 2}
    for prefix in ["gs", "ar", "om"]:
        data[f"{prefix}_data"] = np.array([1.0, 2.0])
        data[f"{prefix}_param"] = np.array([0.2, 0.0, 1.4])
        data[f"{prefix}_moments"] = [np.float64(1.4), np.float64(0.2)]
    result = example.knpy.EBSDGraphResult(
        graph=nx.path_graph(2), phase_data=data, phase_index=7,
        phase_name="austenite", rgb_image=np.zeros((1, 2, 3)),
        npx=2, sh_x=1, sh_y=2,
    )
    target = tmp_path / "statistics.json"
    example.write_statistics(target, result, {"Phase": {"Number": 0}})
    statistics = json.loads(target.read_text())
    assert statistics["experimental"]["gs_data"] == [1.0, 2.0]
    assert statistics["experimental"]["index"] == 7
    assert statistics["units"] == {"length": "um", "tilt_angle": "rad", "aspect_ratio": "1"}
    assert statistics["rve_descriptor"]["Phase"]["Number"] == 0


def test_mesh_orientations_preserve_experimental_symmetry():
    """
    Preserve crystal symmetry and Euler angles for the RVE pole figures.
    """
    example = load_example()
    mesh = SimpleNamespace(grain_ori_dict={3: [0, 0, 0], 8: [0.2, 0.5, 0.8]})
    orientations = example.mesh_orientations(mesh, symmetry.Oh)
    assert orientations.symmetry == symmetry.Oh
    np.testing.assert_allclose(orientations.to_euler(), [[0, 0, 0], [0.2, 0.5, 0.8]], atol=1e-7)


@pytest.mark.parametrize("preview", [False, True])
def test_complete_example_saves_and_previews_graph_before_histograms(tmp_path, monkeypatch, preview):
    """
    Put the real graph first in both saved output and interactive previews.

    The workflow runs its real EBSD stages and stops before RVE construction.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Record figure operations and stop before three dimensional computation.
    preview : bool
        Whether each saved figure is previewed.
    """
    example = load_example()
    saved_names = []
    shown_titles = []
    save_figure = Figure.savefig

    def record_save(figure, filename, *args, **kwargs):
        """
        Record image order while retaining the actual file write.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure being saved.
        filename : str or pathlib.Path
            Output image path.
        *args, **kwargs : object
            Remaining save options.
        """
        saved_names.append(Path(filename).name)
        save_figure(figure, filename, *args, **kwargs)

    def record_preview(*args, **kwargs):
        """
        Inspect the figure passed to an interactive preview.

        Parameters
        ----------
        *args, **kwargs : object
            Display arguments.
        """
        shown_titles.append(plt.gcf().axes[0].get_title())

    monkeypatch.setattr(Figure, "savefig", record_save)
    monkeypatch.setattr(plt, "show", record_preview)
    monkeypatch.setattr(example.knpy, "set_stats", Mock(side_effect=RuntimeError("stop before RVE")))
    args = example.parse_args(["--output-dir", str(tmp_path), "--show" if preview else "--no-show"])
    with pytest.raises(RuntimeError, match="stop before RVE"):
        example.run_workflow(args)
    assert saved_names[0] == "ebsd_graph_overlay_01.png"
    if preview:
        assert "graph" in shown_titles[0].lower()
    else:
        assert shown_titles == []
    assert plt.get_fignums() == []


def test_original_example_uses_separate_output_from_any_working_directory(tmp_path, monkeypatch):
    """
    Route the relocated original example's export away from graph RVE results.

    RVE computation is substituted here so the test checks only file routing
    and runs the real example's configuration and export call.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary example tree and unrelated working directory.
    monkeypatch : pytest.MonkeyPatch
        Substitutes for expensive EBSD and RVE calculations.
    """
    import kanapy as knpy

    source = Path(__file__).parents[1] / "examples/EBSD_to_RVE/convert_ang2rve.py"
    assert source.is_file()
    example_dir = tmp_path / "examples/EBSD_to_RVE"
    example_dir.mkdir(parents=True)
    copied_script = example_dir / source.name
    copied_script.write_text(source.read_text())
    fixtures = tmp_path / "examples/fixtures"
    fixtures.mkdir()
    input_file = fixtures / "ebsd_316L_500x500.ang"
    input_file.touch()
    graph_output = example_dir / "outputs"
    graph_output.mkdir()
    existing = graph_output / "Iron fcc_voxels.json"
    existing.write_text("existing graph result")
    working_directory = tmp_path / "elsewhere"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    phase = {"name": "Iron fcc", "gs_param": np.array([0.2, 0.0, 12.0]),
             "ar_param": np.array([0.2, 0.0, 1.5]), "om_param": np.array([1.2, 0.0]),
             "gs_data": np.array([10.0, 14.0]), "ar_data": np.array([1.2, 1.8])}
    ebsd = Mock(ms_data=[phase], emap=SimpleNamespace(phases=[object()]))
    load_map = Mock(return_value=ebsd)
    monkeypatch.setattr(knpy, "EBSDmap", load_map)
    microstructure = Mock(mesh=SimpleNamespace(grain_ori_dict={1: [0.0, 0.0, 0.0]}))

    def export_voxels(*, file, path="./", **kwargs):
        """
        Materialize the export at the exact destination chosen by the example.

        Parameters
        ----------
        file : str
            Export filename supplied by the script.
        path : str or pathlib.Path, optional
            Export directory, defaulting to the caller's working directory.
        **kwargs : object
            Remaining metadata and mesh options.
        """
        destination = Path(path) / file
        destination.write_text('{"origin": "original example"}')

    microstructure.write_voxels.side_effect = export_voxels
    monkeypatch.setattr(knpy, "Microstructure", Mock(return_value=microstructure))
    monkeypatch.setattr(knpy, "plot_pole_figure", Mock())
    monkeypatch.setattr(knpy, "plot_pole_figure_proj", Mock())
    runpy.run_path(str(copied_script), run_name="__main__")

    assert Path(load_map.call_args.args[0]).resolve() == input_file
    assert (example_dir / "Iron fcc_voxels.json").is_file()
    assert not (example_dir / "ang_to_rve_outputs").exists()
    assert load_map.call_args.kwargs["show_plot"] is False
    assert load_map.call_args.kwargs["show_hist"] is False
    plot_calls = [call[0] for call in ebsd.mock_calls if call[0].startswith("plot_")]
    assert plot_calls[:2] == ["plot_graph_overlay", "plot_histograms"]
    assert existing.read_text() == "existing graph result"
    assert list(working_directory.iterdir()) == []
