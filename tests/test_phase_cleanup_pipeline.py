"""
Exercise phase transfer and isolated region cleanup through real EBSD maps.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from orix import plot as ox_plot
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh

from kanapy.texture import EBSDmap, merge_nodes
from kanapy.ebsd_io import LoadedEBSDData

def _loaded_map(phase_map, source_pixels=()):
    """
    Build measured square data with distinct phase and fragment orientations.

    Parameters
    ----------
    phase_map : numpy.ndarray
        Phase identifiers, with minus one representing unindexed positions.
    source_pixels : sequence of int, optional
        Indexed positions with an orientation distinct from their own phase.

    Returns
    -------
    LoadedEBSDData
        Real orix map and its rectangular measurement geometry.
    """
    rows, columns = phase_map.shape
    ids = sorted(set(phase_map.ravel()) - {-1})
    eulers = np.tile([0.3, 0.4, 0.5], (phase_map.size, 1))
    eulers[phase_map.ravel() == 2] = [0.8, 0.6, 0.2]
    eulers[np.asarray(source_pixels, dtype=int)] = [1.0, 0.3, 0.2]
    phases = []
    for phase in ids:
        point_group = "6/mmm" if phase == 2 else "m-3m"
        phases.append(Phase(name=f"Phase {phase}", point_group=point_group))
    phase_list = PhaseList(phases=phases, ids=ids)
    emap = CrystalMap(
        rotations=Rotation.from_euler(eulers),
        x=np.tile(np.arange(columns, dtype=float), rows),
        y=np.repeat(np.arange(rows, dtype=float), columns),
        phase_id=phase_map.ravel(),
        phase_list=phase_list,
    )
    return LoadedEBSDData(emap, rows, columns, 1.0, 1.0, phase_map.size)


def _phase_data(analyzer, phase_id):
    """
    Select the final analysis record of a measured phase.

    Parameters
    ----------
    analyzer : EBSDmap
        Completed analyzer.
    phase_id : int
        Original phase identifier.

    Returns
    -------
    dict
        Phase graph, pixel data, and derived statistics.
    """
    return next(data for data in analyzer.ms_data if data["index"] == phase_id)


def test_cross_phase_absorption_updates_pixels_colors_and_statistics():
    """
    Absorb a retained thin phase island and finalize all phase representations.

    Its twelve pixels exceed the three percent phase cutoff. Supplementary
    cleanup must therefore transfer them only after ordinary phase analysis.
    """
    phase_map = np.ones((15, 15), dtype=int)
    phase_map[6:8, 5:11] = 2
    loaded = _loaded_map(phase_map)
    original_ids = loaded.emap.phase_id.copy()
    original_rotations = loaded.emap.rotations.data.copy()
    absorbed = np.flatnonzero(phase_map.ravel() == 2)

    analyzer = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)

    phase_a = _phase_data(analyzer, 1)
    phase_b = _phase_data(analyzer, 2)
    np.testing.assert_array_equal(phase_a["idx"], np.arange(225))
    assert phase_b["idx"].size == 0
    assert len(phase_a["graph"]) == 1
    assert len(phase_b["graph"]) == 0
    assert phase_a["statistics_available"]
    assert not phase_b["statistics_available"]
    assert phase_b["gs_param"] is None
    assert phase_b["grain_euler_angles"] == {}
    assert phase_a["processed_pixel_fraction"] == pytest.approx(1.0)
    assert phase_b["processed_pixel_fraction"] == pytest.approx(0.0)
    assert phase_a["vf"] == pytest.approx(213 / 225)
    assert phase_b["vf"] == pytest.approx(12 / 225)

    node = next(iter(phase_a["graph"].nodes.values()))
    assert node["npix"] == 225
    np.testing.assert_array_equal(np.sort(node["pixels"]), np.arange(225))
    assert node["hull"].volume == pytest.approx(196.0)
    np.testing.assert_allclose(phase_a["gs_data"], [28 / np.sqrt(np.pi)])
    pixel_rotations = phase_a["graph"].graph["rotations"][absorbed]
    expected_rotations = np.tile(node["ori_av"], (absorbed.size, 1))
    np.testing.assert_array_equal(pixel_rotations, expected_rotations)
    np.testing.assert_allclose(
        phase_a["ori"].data[absorbed], expected_rotations, rtol=0, atol=1e-14,
    )
    angle = Orientation(node["ori_av"], Oh).angle_with(
        Orientation(original_rotations[0], Oh), degrees=True,
    )
    np.testing.assert_allclose(angle, 0.0, atol=1e-5)
    expected_rgb = ox_plot.IPFColorKeyTSL(Oh).orientation2color(
        Orientation(node["ori_av"], Oh),
    )[0]
    np.testing.assert_allclose(
        phase_a["rgb_im"].reshape(-1, 3)[absorbed],
        np.tile(expected_rgb, (absorbed.size, 1)),
    )

    assert analyzer.graph.number_of_nodes() == 1
    assert analyzer.graph.number_of_edges() == 0
    assert analyzer.graph.graph["isolated_cleanup_records"] == analyzer.isolated_cleanup_records
    overview_labels = analyzer.graph.graph["label_map"]
    assert np.all(overview_labels > 0)
    assert np.unique(overview_labels).size == 1
    np.testing.assert_array_equal(loaded.emap.phase_id, original_ids)
    np.testing.assert_array_equal(loaded.emap.rotations.data, original_rotations)


def test_smaller_same_phase_neighbor_wins_before_cross_phase_cleanup():
    """
    Merge a boundary fragment into its 197 pixel same phase neighbor.

    The other phase has 200 pixels and touches the fragment. Its larger size
    must not overrule the preceding phase specific cleanup pass.
    """
    phase_map = np.ones((20, 20), dtype=int)
    phase_map[:, 10:] = 2
    fragment = np.array([149, 169, 189])
    loaded = _loaded_map(phase_map, source_pixels=fragment)
    expected_orientation = loaded.emap.rotations.data[0].copy()

    analyzer = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)

    phase_a = _phase_data(analyzer, 1)
    phase_b = _phase_data(analyzer, 2)
    assert len(phase_a["graph"]) == len(phase_b["graph"]) == 1
    np.testing.assert_array_equal(phase_a["idx"], np.flatnonzero(phase_map.ravel() == 1))
    np.testing.assert_array_equal(phase_b["idx"], np.flatnonzero(phase_map.ravel() == 2))
    node = next(iter(phase_a["graph"].nodes.values()))
    assert node["npix"] == 200
    assert set(fragment).issubset(node["pixels"])
    np.testing.assert_allclose(
        Orientation(node["ori_av"], Oh).angle_with(
            Orientation(expected_orientation, Oh), degrees=True,
        ),
        0.0,
        atol=1e-5,
    )


def test_unindexed_gaps_leave_only_large_valid_isolated_region():
    """
    Exclude tiny and thin isolated regions without crossing unindexed gaps.

    A compact 36 pixel component survives. A valid three pixel triangle and
    a valid 30 pixel strip are removed by the approved boundary cleanup rule.
    """
    phase_map = np.full((20, 30), -1, dtype=int)
    phase_map[11:17, 2:8] = 1
    phase_map[1:3, 10:25] = 1
    phase_map.flat[[31, 32, 61]] = 1
    phase_map[18, 28] = 1
    retained = np.flatnonzero(phase_map.ravel() == 1)
    retained = retained[(retained // 30 >= 11) & (retained // 30 < 17)]
    loaded = _loaded_map(phase_map)
    original_ids = loaded.emap.phase_id.copy()

    analyzer = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)

    phase = _phase_data(analyzer, 1)
    np.testing.assert_array_equal(phase["idx"], retained)
    assert len(phase["graph"]) == 1
    assert phase["statistics_available"]
    node = next(iter(phase["graph"].nodes.values()))
    assert node["npix"] == 36
    assert node["hull"].volume == pytest.approx(25.0)
    assert phase["processed_pixel_fraction"] == pytest.approx(36 / 600)
    assert phase["vf"] == pytest.approx(70 / 600)
    labels = phase["graph"].graph["label_map"].ravel()
    np.testing.assert_array_equal(np.flatnonzero(labels), retained)
    np.testing.assert_array_equal(loaded.emap.phase_id, original_ids)


def test_cross_phase_transfer_restores_same_phase_cleanup_of_collinear_region():
    """
    Clean an invalid region after a transferred bridge joins its own phase.

    An eight pixel A strip separates a compact B grain from a 41 pixel B
    line. The line exceeds the Step 3 size cap, so the A strip merges first.
    The resulting same phase contact must let Step 4 absorb the B line.
    """
    phase_map = np.full((20, 60), -1, dtype=int)
    phase_map[2:10, 2:8] = 2
    phase_map[2:10, 8] = 1
    phase_map[5, 9:50] = 2
    phase_map[12:18, 52:58] = 1
    line_pixels = np.arange(309, 350)
    loaded = _loaded_map(phase_map, source_pixels=line_pixels)
    original_ids = loaded.emap.phase_id.copy()
    original_rotations = loaded.emap.rotations.data.copy()
    expected_orientation = original_rotations[122].copy()
    retained_a = np.flatnonzero(phase_map.ravel() == 1)
    retained_a = retained_a[retained_a // 60 >= 12]
    expected_b_mask = phase_map == 2
    expected_b_mask[2:10, 8] = True
    expected_b = np.flatnonzero(expected_b_mask)

    analyzer = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)

    phase_a = _phase_data(analyzer, 1)
    phase_b = _phase_data(analyzer, 2)
    np.testing.assert_array_equal(phase_a["idx"], retained_a)
    np.testing.assert_array_equal(phase_b["idx"], expected_b)
    assert len(phase_a["graph"]) == len(phase_b["graph"]) == 1
    node = next(iter(phase_b["graph"].nodes.values()))
    assert node["npix"] == 97
    np.testing.assert_array_equal(np.sort(node["pixels"]), expected_b)
    assert node["hull"].volume > 0
    np.testing.assert_allclose(
        Orientation(node["ori_av"], phase_b["cs"]).angle_with(
            Orientation(expected_orientation, phase_b["cs"]), degrees=True,
        ),
        0.0,
        atol=1e-5,
    )
    assert phase_a["vf"] == pytest.approx(44 / 1200)
    assert phase_b["vf"] == pytest.approx(89 / 1200)
    assert phase_a["processed_pixel_fraction"] == pytest.approx(36 / 1200)
    assert phase_b["processed_pixel_fraction"] == pytest.approx(97 / 1200)
    np.testing.assert_array_equal(loaded.emap.phase_id, original_ids)
    np.testing.assert_array_equal(loaded.emap.rotations.data, original_rotations)


def test_default_cleanup_returns_empty_statistics_when_every_node_is_excluded():
    """
    Return an empty analysis instead of fitting nonexistent grain data.

    Both indexed components satisfy the isolated boundary cleanup condition,
    so the original measurements remain while all analysis pixels disappear.
    """
    phase_map = np.full((10, 10), -1, dtype=int)
    phase_map.flat[[0, 1, 10, 99]] = 1
    loaded = _loaded_map(phase_map)
    original_ids = loaded.emap.phase_id.copy()

    analyzer = EBSDmap(ebsd_data=loaded, show_plot=False, show_hist=False)

    phase = _phase_data(analyzer, 1)
    assert phase["idx"].size == 0
    assert phase["ori"].size == 0
    assert phase["graph"].number_of_nodes() == 0
    assert phase["graph"].number_of_edges() == 0
    assert not np.any(phase["graph"].graph["label_map"])
    assert not phase["statistics_available"]
    assert phase["grain_euler_angles"] == {}
    for prefix in ("gs", "ar", "om"):
        assert phase[f"{prefix}_param"] is None
        assert phase[f"{prefix}_data"].size == 0
    assert phase["vf"] == pytest.approx(0.04)
    assert phase["processed_pixel_fraction"] == 0.0
    np.testing.assert_array_equal(loaded.emap.phase_id, original_ids)


@pytest.mark.parametrize("step", [3, 4, 6])
def test_cross_phase_steps_preserve_the_receiving_grain_orientation(step):
    """
    Apply each supplementary predicate without averaging across phase boundaries.

    Parameters
    ----------
    step : int
        Boundary fragment, invalid hull, or minimum pixel count cleanup pass.
    """
    phase_map = np.ones((20, 60), dtype=int)
    if step == 3:
        phase_map[6:8, 5:11] = 2
    elif step == 4:
        phase_map[8, 5:46] = 2
    else:
        phase_map[6:9, 5:8] = 2
    loaded = _loaded_map(phase_map)
    original_rotations = loaded.emap.rotations.data.copy()
    transferred = np.flatnonzero(phase_map.ravel() == 2)

    analyzer = EBSDmap(
        ebsd_data=loaded, graph_only=True, vf_min=0.0,
        boundary_artifact_fraction_min=1.0 if step == 6 else 0.85,
    )

    phase_a = _phase_data(analyzer, 1)
    phase_b = _phase_data(analyzer, 2)
    assert len(phase_a["graph"]) == 1
    assert len(phase_b["graph"]) == 0
    node = next(iter(phase_a["graph"].nodes.values()))
    assert node["npix"] == phase_map.size
    np.testing.assert_array_equal(phase_a["idx"], np.arange(phase_map.size))
    np.testing.assert_allclose(
        Orientation(node["ori_av"], phase_a["cs"]).angle_with(
            Orientation(original_rotations[0], phase_a["cs"]), degrees=True,
        ),
        0.0,
        atol=1e-5,
    )
    expected = np.tile(node["ori_av"], (transferred.size, 1))
    np.testing.assert_array_equal(
        phase_a["graph"].graph["rotations"][transferred], expected,
    )
    records = analyzer.isolated_cleanup_records
    assert len(records) == 1
    assert records[0]["action"] == "merge"
    assert records[0]["step"] == step
    assert records[0]["source_phase_id"] == 2
    assert records[0]["target_phase_id"] == 1
    np.testing.assert_array_equal(loaded.emap.phase_id, phase_map.ravel())
    np.testing.assert_array_equal(loaded.emap.rotations.data, original_rotations)


def test_full_graph_preserves_phase_grain_ids_when_largest_phase_is_not_first():
    """
    Keep node identity stable between per phase graphs and the full graph.

    Phase seven occupies sixty percent, so sorting the display legend by
    volume fraction must not exchange the grain IDs of phases two and seven.
    """
    phase_map = np.full((10, 20), 7, dtype=int)
    phase_map[:, :8] = 2

    analyzer = EBSDmap(ebsd_data=_loaded_map(phase_map), graph_only=True)

    assert len(analyzer.graph) == 2
    labels = analyzer.graph.graph["label_map"].ravel()
    for phase in analyzer.ms_data:
        for grain_id, node in phase["graph"].nodes(data=True):
            full_node = analyzer.graph.nodes[grain_id]
            assert full_node["phase_id"] == phase["index"]
            assert full_node["source_grain_id"] == grain_id
            np.testing.assert_array_equal(full_node["pixels"], node["pixels"])
            np.testing.assert_array_equal(labels[node["pixels"]], grain_id)


@pytest.mark.parametrize("preserve_target", [True, False])
def test_full_graph_supports_existing_same_phase_merge_operations(preserve_target):
    """
    Keep the symmetry and measured rotations needed by public graph utilities.

    Parameters
    ----------
    preserve_target : bool
        Retain the receiving grain orientation or average the measured pixels.
    """
    rows, columns = np.indices((10, 30))
    phase_ids = np.where(columns.ravel() < 24, 2, 7)
    eulers = np.zeros((300, 3))
    eulers[:, 0] = (columns.ravel() // 8) * np.deg2rad(20)
    emap = CrystalMap(
        rotations=Rotation.from_euler(eulers), phase_id=phase_ids,
        x=columns.ravel().astype(float), y=rows.ravel().astype(float),
        phase_list=PhaseList(
            names=["A", "B"], point_groups=["m-3m", "6/mmm"], ids=[2, 7],
        ),
    )
    loaded = LoadedEBSDData(emap, 10, 30, 1.0, 1.0, 300)
    analyzer = EBSDmap(ebsd_data=loaded, graph_only=True)
    graph = analyzer.graph
    phase_a = _phase_data(analyzer, 2)
    source, target, neighbor = sorted(phase_a["graph"])
    original_pixels = np.sort(np.concatenate((
        graph.nodes[source]["pixels"], graph.nodes[target]["pixels"],
    )))

    merge_nodes(graph, source, target, preserve_target_orientation=preserve_target)

    assert source not in graph
    assert graph.nodes[target]["phase_id"] == 2
    assert graph.nodes[target]["npix"] == 160
    np.testing.assert_array_equal(np.sort(graph.nodes[target]["pixels"]), original_pixels)
    expected_angle = np.deg2rad(20 if preserve_target else 10)
    expected = Orientation.from_euler([expected_angle, 0, 0], symmetry=phase_a["cs"])
    np.testing.assert_allclose(
        Orientation(graph.nodes[target]["ori_av"], phase_a["cs"]).angle_with(
            expected, degrees=True,
        ),
        0.0, atol=1e-5,
    )
    assert "misorientation" in graph[target][neighbor]
    for phase in analyzer.ms_data:
        np.testing.assert_array_equal(
            graph.graph["rotations"][phase["idx"]],
            phase["graph"].graph["rotations"][phase["idx"]],
        )
        np.testing.assert_array_equal(
            graph.graph["symmetry"][phase["index"]].data, phase["cs"].data,
        )


@pytest.mark.parametrize("plot_method", ["plot_grains", "plot_grains_marked"])
def test_grain_plots_skip_phases_without_remaining_nodes(plot_method, monkeypatch):
    """
    Allow public grain figures after cleanup removes every indexed region.

    Parameters
    ----------
    plot_method : str
        Public grain plotting method to exercise.
    monkeypatch : pytest.MonkeyPatch
        Suppress the interactive display while retaining actual rendering code.
    """
    phase_map = np.full((10, 10), -1, dtype=int)
    phase_map.flat[[0, 1, 10, 99]] = 1
    analyzer = EBSDmap(
        ebsd_data=_loaded_map(phase_map), show_plot=False, show_hist=False,
    )

    def no_interactive_display():
        """
        Keep rendering local to the test without opening a GUI.
        """
        return None

    monkeypatch.setattr(plt, "show", no_interactive_display)
    try:
        getattr(analyzer, plot_method)()
    finally:
        plt.close("all")


@pytest.mark.parametrize("plot_method", ["plot_grains", "plot_grains_marked"])
def test_largest_grain_plot_handles_fewer_grains_than_requested(monkeypatch, plot_method):
    """
    Render the surviving grain and ignore a fully absorbed second phase.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Suppress interactive display without replacing the plotting code.
    plot_method : str
        Plot the remaining grain geometry or its label and axes.
    """
    phase_map = np.ones((15, 15), dtype=int)
    phase_map[6:8, 5:11] = 2
    analyzer = EBSDmap(
        ebsd_data=_loaded_map(phase_map), show_plot=False, show_hist=False,
    )
    calls = []

    def record_display():
        """
        Record the actual grain figure instead of opening a GUI.
        """
        calls.append(plt.gca().get_title())

    monkeypatch.setattr(plt, "show", record_display)
    try:
        getattr(analyzer, plot_method)()
        assert len(calls) == 1
        assert calls[0].startswith("Phase #1")
    finally:
        plt.close("all")
