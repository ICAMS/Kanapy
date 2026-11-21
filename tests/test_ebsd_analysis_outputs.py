"""
Regression tests for experimental EBSD analysis and plotting outputs.
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Orientation
from scipy.spatial import ConvexHull
from scipy.stats import lognorm, vonmises

from kanapy.core.rve_stats import get_grain_geom
from kanapy.texture import EBSDmap, _graph_points_to_image


@pytest.fixture(autouse=True)
def close_figures():
    """
    Release the real Matplotlib figures produced by each regression test.

    Keeping cleanup here prevents figures from one test affecting another.
    """
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def analyzed_map():
    """
    Analyze the supplied EBSD fixture with all automatic plots disabled.

    This exercises orientation construction and statistics without substitutes.

    Returns
    -------
    EBSDmap
        Complete analysis of the real ANG map.
    """
    fixture = (Path(__file__).resolve().parents[1] / "examples" / "fixtures"
               / "ebsd_316L_500x500.ang")
    return EBSDmap(str(fixture), show_plot=False, show_hist=False)


def test_imported_orientations_keep_phase_symmetry_and_source_rotations(analyzed_map):
    """
    Retain crystal symmetry without changing the imported numerical rotations.

    Identity symmetry makes later ODF reconstruction use the wrong crystal grid.

    Parameters
    ----------
    analyzed_map : EBSDmap
        Analysis of the real fixture.
    """
    phase = analyzed_map.ms_data[0]
    assert phase["ori"].symmetry.name == "m-3m"
    assert phase["ori"].symmetry == phase["cs"]
    source = analyzed_map.emap[analyzed_map.emap.phase_id == phase["index"]]
    expected = Orientation.from_euler(source.orientations.in_euler_fundamental_region())
    confident = np.ones(expected.size, dtype=bool)
    if "ci" in analyzed_map.emap.prop:
        confident = source.prop["ci"] >= 0.1
    np.testing.assert_allclose(phase["ori"].data[confident], expected.data[confident])
    assert plt.get_fignums() == []


def test_grain_euler_metadata_roundtrips_final_graph_orientations(analyzed_map):
    """
    Expose final grain orientations as sequential Euler angle rows in degrees.

    Parameters
    ----------
    analyzed_map : EBSDmap
        Analysis of the real fixture with complete grain statistics.
    """
    phase = analyzed_map.ms_data[0]
    graph = phase["graph"]
    angles = phase["grain_euler_angles"]
    assert list(angles) == [f"grain {index}" for index in range(1, len(graph) + 1)]
    restored = Orientation.from_euler(
        list(angles.values()), symmetry=phase["cs"], degrees=True,
    )
    expected = np.array([node["ori_av"] for node in graph.nodes.values()])
    np.testing.assert_allclose(np.abs(np.sum(restored.data * expected, axis=1)), 1.)


def test_graph_statistics_and_axes_use_kanapy_geometry(analyzed_map):
    """
    Apply Kanapy geometry to every grain without rotating its image.

    Expected ellipses are fitted directly to scaled row and column indices.

    Parameters
    ----------
    analyzed_map : EBSDmap
        Complete analysis of the real ANG map.
    """
    phase = analyzed_map.ms_data[0]
    graph = phase["graph"]
    expected_angles = []
    for node in graph.nodes.values():
        rows, columns = np.unravel_index(node["pixels"], graph.graph["label_map"].shape)
        points = np.column_stack((rows * analyzed_map.dx, columns * analyzed_map.dy))
        hull = ConvexHull(points)
        _, _, major, _ = get_grain_geom(hull.points[hull.vertices], method="ellipsis", two_dim=True)
        expected_angles.append(np.arccos(major[0]))
        np.testing.assert_allclose(node["hull"].points, points)
        np.testing.assert_allclose(
            _graph_points_to_image(graph, node["center"]),
            [np.mean(columns), np.mean(rows)],
        )
    np.testing.assert_allclose(phase["om_data"], expected_angles)


@pytest.mark.parametrize("method", ["plot_pf", "plot_pf_proj"])
def test_pole_figures_use_original_retained_phase_identifier(method):
    """
    Use phase seven after filtering phases zero and two from the retained rows.

    Real plot titles expose lookup against the unfiltered phase positions.

    Parameters
    ----------
    method : str
        Conventional or projected pole figure method to exercise.
    """
    phases = PhaseList(
        names=["Removed zero", "Removed two", "Retained seven"],
        point_groups=["1", "1", "m-3m"], ids=[0, 2, 7],
    )
    ebsd = EBSDmap.__new__(EBSDmap)
    ebsd.emap = SimpleNamespace(phases=phases)
    orientations = Orientation.from_euler(
        np.array([[0.1, 0.2, 0.3], [0.4, 0.6, 0.8], [1.0, 0.5, 0.2]]),
        symmetry=phases[7].point_group.laue,
    )
    ebsd.ms_data = [{"index": 7, "ori": orientations}]

    getattr(ebsd, method)()

    assert len(plt.get_fignums()) == 1
    titles = [axis.get_title() for axis in plt.gcf().axes if axis.get_title()]
    assert len(titles) == 2
    assert all("Retained seven" in title for title in titles)


def test_histograms_preserve_density_and_doubled_angle_fit(analyzed_map):
    """
    Plot three measured distributions with their stored scientific fits.

    The angular fit must transform its argument and density back to radians.

    Parameters
    ----------
    analyzed_map : EBSDmap
        Analysis with stored arrays and distribution parameters.
    """
    phase = analyzed_map.ms_data[0]

    analyzed_map.plot_histograms(iphase=0)

    assert len(plt.get_fignums()) == 3
    for figure_number, prefix in zip(plt.get_fignums(), ("gs", "ar", "om")):
        axis = plt.figure(figure_number).axes[0]
        values = phase[f"{prefix}_data"]
        params = phase[f"{prefix}_param"]
        line = axis.lines[0]
        if prefix == "om":
            density, edges = np.histogram(values, bins=40, density=True)
            x = np.linspace(0, np.pi, 200)
            y = 2 * vonmises.pdf(2 * x - np.pi, params[0], loc=params[1])
        else:
            counts, full_edges = np.histogram(values, bins=20)
            singletons = np.flatnonzero(counts == 1)
            upper = full_edges[singletons[0]] if len(singletons) else np.max(values)
            density, edges = np.histogram(values, bins=20, range=(full_edges[0], upper), density=True)
            x = np.linspace(full_edges[0], upper, 150)
            y = lognorm.pdf(x, params[0], loc=params[1], scale=params[2])
        np.testing.assert_allclose([patch.get_height() for patch in axis.patches], density)
        np.testing.assert_allclose([patch.get_x() for patch in axis.patches], edges[:-1])
        np.testing.assert_allclose(line.get_xdata(), x)
        np.testing.assert_allclose(line.get_ydata(), y)


def test_histogram_selection_uses_retained_phase_position():
    """
    Plot all retained rows by default or one row by its retained list position.

    Noncontiguous original identifiers must not become list indices.
    """
    first = {
        "index": 2, "gs_data": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        "gs_param": np.array([0.2, 0.0, 2.0]),
        "ar_data": np.array([1.1, 1.5, 2.0]),
        "ar_param": np.array([0.3, 0.0, 1.5]),
        "om_data": np.array([0.2, 0.7, 1.0]),
        "om_param": np.array([1.0, 0.2]),
    }
    second = dict(first, index=7, gs_data=np.array([4.0, 4.0, 5.0, 5.0, 6.0, 6.0]))
    ebsd = EBSDmap.__new__(EBSDmap)
    ebsd.ms_data = [first, second]

    ebsd.plot_histograms(iphase=1)

    assert len(plt.get_fignums()) == 3
    np.testing.assert_allclose(plt.figure(1).axes[0].lines[0].get_xdata()[[0, -1]], [4, 6])
    plt.close("all")

    ebsd.plot_histograms()

    assert len(plt.get_fignums()) == 6
    np.testing.assert_allclose(plt.figure(1).axes[0].lines[0].get_xdata()[[0, -1]], [1, 3])
    np.testing.assert_allclose(plt.figure(4).axes[0].lines[0].get_xdata()[[0, -1]], [4, 6])


def test_constructor_histograms_match_explicit_plotting(analyzed_map):
    """
    Produce the same three histograms during import and on explicit request.

    This catches duplicate figures or divergent plotting paths in the constructor.

    Parameters
    ----------
    analyzed_map : EBSDmap
        Analysis with automatic plotting disabled.
    """
    analyzed_map.plot_histograms()
    expected = []
    for number in plt.get_fignums():
        axis = plt.figure(number).axes[0]
        expected.append((axis.get_title(), axis.lines[0].get_xydata().copy(),
                         [patch.get_height() for patch in axis.patches]))
    plt.close("all")

    fixture = (Path(__file__).resolve().parents[1] / "examples" / "fixtures"
               / "ebsd_316L_500x500.ang")
    EBSDmap(str(fixture), show_plot=False, show_hist=True)

    assert len(plt.get_fignums()) == 3
    for number, (title, curve, density) in zip(plt.get_fignums(), expected):
        axis = plt.figure(number).axes[0]
        assert axis.get_title() == title
        np.testing.assert_allclose(axis.lines[0].get_xydata(), curve)
        np.testing.assert_allclose([patch.get_height() for patch in axis.patches], density)
