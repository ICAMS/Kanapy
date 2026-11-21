"""
Verify shared measured phase assignment before phase specific graph cleanup.
"""

import numpy as np
import pytest
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
from scipy.spatial import cKDTree

from kanapy.texture import EBSDmap
from kanapy.ebsd_io import LoadedEBSDData
from kanapy.ebsd_hex_grid import make_regular_grid

def _make_three_phase_map(hex_grid=False, unindexed_pixel=None, extra_minor_pixel=None):
    """
    Create two large regions and two isolated minority phase measurements.

    The hundred measured points contain 60 percent A, 38 percent B, and
    2 percent C. Both C points lie beside only one retained phase, so their
    expected reassignment does not depend on how equal distances are ordered.

    Parameters
    ----------
    hex_grid : bool, optional
        Offset alternate scan rows by half a horizontal step.
    unindexed_pixel : int, optional
        Replace one measured phase label with the unindexed label.
    extra_minor_pixel : int, optional
        Add a third minority pixel to exercise the exact fraction cutoff.

    Returns
    -------
    LoadedEBSDData
        Actual orix data suitable for the public analyzer constructor.
    """
    phase_ids = np.full(100, 2, dtype=int)
    phase_ids[1:61] = 1
    phase_ids[[0, 99]] = 3
    if extra_minor_pixel is not None:
        phase_ids[extra_minor_pixel] = 3
    if unindexed_pixel is not None:
        phase_ids[unindexed_pixel] = -1

    rows = np.repeat(np.arange(10), 10)
    x = np.tile(np.arange(10, dtype=float), 10)
    y_step = np.sqrt(3.0) / 2.0 if hex_grid else 1.0
    if hex_grid:
        x += 0.5 * (rows % 2)
    y = rows * y_step

    euler_angles = np.tile([0.3, 0.4, 0.5], (100, 1))
    euler_angles[phase_ids == 2] = [0.8, 0.6, 0.2]
    euler_angles[phase_ids == 3] = [1.4, 0.9, 0.7]
    phases = PhaseList(
        phases=[
            Phase(name="A", point_group="m-3m"),
            Phase(name="B", point_group="m-3m"),
            Phase(name="C", point_group="m-3m"),
        ],
        ids=[1, 2, 3],
    )
    emap = CrystalMap(
        rotations=Rotation.from_euler(euler_angles),
        x=x,
        y=y,
        phase_id=phase_ids,
        phase_list=phases,
        prop={"ci": 0.5 + 0.004 * np.arange(100)},
    )
    columns = 20 if hex_grid else 10
    return LoadedEBSDData(
        emap=emap,
        sh_x=10,
        sh_y=columns,
        dx=0.5 if hex_grid else 1.0,
        dy=y_step,
        npx=10 * columns,
        hex_grid=hex_grid,
        original_npx=100,
    )


def _read_processed_map(analyzer):
    """
    Assemble phase ownership from the actual analyzed pixel subsets.

    Overlapping subsets would represent the same map position as multiple
    phases and are rejected before returning the assembled map.

    Parameters
    ----------
    analyzer : EBSDmap
        Analyzer with completed per phase graph and orientation data.

    Returns
    -------
    tuple of numpy.ndarray
        Phase labels and quaternions on the common analysis grid.
    """
    labels = np.full(analyzer.npx, -1, dtype=int)
    quaternions = np.full((analyzer.npx, 4), np.nan)
    occupied = np.zeros(analyzer.npx, dtype=bool)
    for data in analyzer.ms_data:
        indices = data["idx"]
        assert not np.any(occupied[indices]), "Retained phases overlap spatially"
        occupied[indices] = True
        labels[indices] = data["index"]
        quaternions[indices] = data["ori"].data
        graph_pixels = []
        for node in data["graph"].nodes.values():
            graph_pixels.extend(node["pixels"])
        np.testing.assert_array_equal(np.sort(graph_pixels), np.sort(indices))
    return labels, quaternions


def test_square_reassigns_only_minor_phase_and_keeps_original_measurements():
    """
    Fill minority pixels without changing retained measurements or raw data.

    The corner minority pixels have different neighboring phases, which
    prevents a global majority phase from absorbing both positions.
    """
    loaded = _make_three_phase_map()
    original_phases = loaded.emap.phase_id.copy()
    original_rotations = loaded.emap.rotations.data.copy()
    original_confidence = loaded.emap.prop["ci"].copy()

    analyzer = EBSDmap(
        ebsd_data=loaded,
        graph_only=True,
        gs_min=1,
    )
    labels, quaternions = _read_processed_map(analyzer)

    expected = np.full(100, 2, dtype=int)
    expected[:61] = 1
    np.testing.assert_array_equal(labels, expected)
    fractions = {data["index"]: data["vf"] for data in analyzer.ms_data}
    assert fractions == pytest.approx({1: 0.60, 2: 0.38})
    retained = original_phases != 3
    angles = Orientation(quaternions[retained], Oh).angle_with(
        Orientation(original_rotations[retained], Oh), degrees=True
    )
    np.testing.assert_allclose(angles, 0.0, atol=1e-5)
    angles = Orientation(quaternions[[0, 99]], Oh).angle_with(
        Orientation(original_rotations[[1, 98]], Oh), degrees=True
    )
    np.testing.assert_allclose(angles, 0.0, atol=1e-5)
    for data in analyzer.ms_data:
        phase_id = data["index"]
        target = 0 if phase_id == 1 else 99
        nearest_sources = {1, 10} if phase_id == 1 else {89, 98}
        expected_fraction = 0.61 if phase_id == 1 else 0.39
        provenance = data["graph"].graph["phase_preprocessing"]
        assert provenance["reassigned_pixels"] == [target]
        assert provenance["original_phase_ids"] == [3]
        assert provenance["processed_pixel_fraction"] == pytest.approx(expected_fraction)
        assert provenance["original_pixel_fraction"] == pytest.approx(fractions[phase_id])
        source = provenance["source_measurement_indices"][0]
        assert source in nearest_sources
        assert original_phases[source] == phase_id
        assert analyzer.ci_map[target] == pytest.approx(original_confidence[source])
    np.testing.assert_allclose(analyzer.ci_map[retained], original_confidence[retained])
    np.testing.assert_array_equal(loaded.emap.phase_id, original_phases)
    np.testing.assert_array_equal(loaded.emap.rotations.data, original_rotations)
    np.testing.assert_array_equal(loaded.emap.prop["ci"], original_confidence)


@pytest.mark.parametrize("target_spacing", [None, 1.0])
def test_hex_uses_one_shared_map_and_original_phase_fractions(target_spacing):
    """
    Resample retained phases together and fill omitted phase positions once.

    Default and explicitly spaced grids must both keep two distinct phase
    regions. The fraction threshold is based on the original hundred points.

    Parameters
    ----------
    target_spacing : float or None
        Explicit physical grid spacing or the default orix dimensions.
    """
    loaded = _make_three_phase_map(hex_grid=True)
    original_phases = loaded.emap.phase_id.copy()
    original_rotations = loaded.emap.rotations.data.copy()
    original_confidence = loaded.emap.prop["ci"].copy()

    analyzer = EBSDmap(
        ebsd_data=loaded,
        graph_only=True,
        gs_min=1,
        hex_grid_target_spacing=target_spacing,
    )
    labels, quaternions = _read_processed_map(analyzer)

    assert set(np.unique(labels)) == {1, 2}
    assert labels[0] == 1
    assert labels[-1] == 2
    assert analyzer.grid_origin_x == pytest.approx(0.0)
    assert analyzer.grid_origin_y == pytest.approx(0.0)
    fractions = {data["index"]: data["vf"] for data in analyzer.ms_data}
    assert fractions == pytest.approx({1: 0.60, 2: 0.38})
    for phase_id, original_index in [(1, 1), (2, 98)]:
        angles = Orientation(quaternions[labels == phase_id], Oh).angle_with(
            Orientation(original_rotations[original_index], Oh), degrees=True
        )
        np.testing.assert_allclose(angles, 0.0, atol=1e-5)
    phase_a = next(data for data in analyzer.ms_data if data["index"] == 1)
    provenance = phase_a["graph"].graph["phase_preprocessing"]
    origin_record = provenance["reassigned_pixels"].index(0)
    assert provenance["original_phase_ids"][origin_record] == 3
    source = provenance["source_measurement_indices"][origin_record]
    assert source in {1, 10}
    assert original_phases[source] == 1
    assert analyzer.ci_map[0] == pytest.approx(original_confidence[source])
    if target_spacing is None:
        assert (analyzer.sh_x, analyzer.sh_y) == (10, 20)
        assert analyzer.dx == pytest.approx(0.5)
        assert analyzer.dy == pytest.approx(np.sqrt(3.0) / 2.0)
    else:
        assert (analyzer.sh_x, analyzer.sh_y) == (8, 10)
        assert analyzer.dx == pytest.approx(1.0)
        assert analyzer.dy == pytest.approx(1.0)
    np.testing.assert_array_equal(loaded.emap.phase_id, original_phases)
    np.testing.assert_array_equal(loaded.emap.rotations.data, original_rotations)
    np.testing.assert_array_equal(loaded.emap.prop["ci"], original_confidence)


@pytest.mark.parametrize("hex_grid", [False, True])
def test_unindexed_measurement_is_not_reassigned_as_a_minor_phase(hex_grid):
    """
    Preserve the distinction between unindexed data and an omitted phase.

    The unindexed position lies inside A and would be silently filled if
    preprocessing treated every excluded label as a small indexed phase.

    Parameters
    ----------
    hex_grid : bool
        Whether the same measured regions use staggered rows.
    """
    loaded = _make_three_phase_map(hex_grid=hex_grid, unindexed_pixel=50)
    analyzer = EBSDmap(
        ebsd_data=loaded,
        graph_only=True,
        gs_min=1,
    )
    labels, _ = _read_processed_map(analyzer)

    index = 101 if hex_grid else 50
    assert labels[index] == -1
    assert labels[0] == 1
    assert labels[-1] == 2
    assert loaded.emap.phase_id[50] == -1


def test_hex_extent_keeps_boundary_positions_of_the_filtered_phase():
    """
    Keep the original scan bounds when only minority pixels reach its edges.

    Building a grid from retained source coordinates alone would crop both
    horizontal edges and change the physical meaning of every column.
    """
    loaded = _make_three_phase_map(hex_grid=True)
    x = loaded.emap.x.copy()
    x[[0, 99]] = [-1.0, 10.5]
    loaded.emap = CrystalMap(
        rotations=loaded.emap.rotations,
        x=x,
        y=loaded.emap.y,
        phase_id=loaded.emap.phase_id,
        phase_list=loaded.emap.phases,
        prop={"ci": np.ones(100)},
    )
    loaded.sh_y = 24
    loaded.npx = 240

    analyzer = EBSDmap(
        ebsd_data=loaded,
        graph_only=True,
        gs_min=1,
        hex_grid_target_spacing=0.5,
    )

    assert analyzer.grid_origin_x == pytest.approx(-1.0)
    assert analyzer.grid_origin_x + (analyzer.sh_y - 1) * analyzer.dx == pytest.approx(10.5)
    assert analyzer.sh_y == 24
    labels, _ = _read_processed_map(analyzer)
    assert labels[0] == 1
    assert labels[-1] == 2


@pytest.mark.parametrize("hex_grid", [False, True])
def test_filtering_every_phase_returns_no_analysis_without_resampling_donors(hex_grid):
    """
    Return an empty phase result when the requested cutoff excludes all phases.

    Empty interpolation input must not reach a nearest point lookup.

    Parameters
    ----------
    hex_grid : bool
        Whether measured rows are staggered.
    """
    loaded = _make_three_phase_map(hex_grid=hex_grid)

    analyzer = EBSDmap(ebsd_data=loaded, graph_only=True, vf_min=0.7)

    assert analyzer.ms_data == []


def make_voting_map(hex_grid=False, x_spacing=1.0, unindexed=False):
    """
    Place a filtered pixel between several retained phases on a real map.

    B is the largest phase globally, while A has more equally close donors
    around the filtered D measurement on both square and hexagonal grids.

    Parameters
    ----------
    hex_grid : bool, optional
        Arrange alternate rows at half step offsets.
    x_spacing : float, optional
        Physical column spacing for the square scan.
    unindexed : bool, optional
        Replace the vertical square neighbors with unindexed measurements.

    Returns
    -------
    tuple
        Loaded EBSD data, original filtered pixel, target grid pixel, and
        nearest source indices of the winning phase.
    """
    rows, cols = np.indices((11, 11))
    x = cols.ravel().astype(float) * x_spacing
    y = rows.ravel().astype(float)
    phase_ids = np.full(121, 2, dtype=int)
    phase_ids[:22] = 1
    phase_ids[-22:] = 3
    if hex_grid:
        x += 0.5 * (rows.ravel() % 2)
        y *= np.sqrt(3.0) / 2.0
        source_pixel, target_pixel = 49, 98
        phase_ids[[48, 50, 59]] = 1
        phase_ids[[37, 38]] = 2
        phase_ids[60] = 3
        winners = {48, 50, 59}
        width = 22
    else:
        source_pixel = target_pixel = 60
        phase_ids[[49, 59]] = 1
        phase_ids[61] = 2
        phase_ids[71] = 3
        winners = {49, 59}
        width = 11
        if x_spacing > 1.0:
            phase_ids[[49, 71]] = 2
            phase_ids[[59, 61]] = 1
            winners = {49, 71}
        if unindexed:
            phase_ids[[49, 71]] = -1
            winners = {59, 61}
    phase_ids[source_pixel] = 4
    eulers = np.tile([0.3, 0.4, 0.5], (121, 1))
    eulers[phase_ids == 2] = [0.8, 0.6, 0.2]
    eulers[phase_ids == 3] = [1.4, 0.9, 0.7]
    phases = PhaseList(
        phases=[Phase(name=name, point_group="m-3m") for name in ("A", "B", "C", "D")],
        ids=[1, 2, 3, 4],
    )
    emap = CrystalMap(
        rotations=Rotation.from_euler(eulers), x=x, y=y,
        phase_id=phase_ids, phase_list=phases,
        prop={"ci": 0.5 + 0.003 * np.arange(121)},
    )
    loaded = LoadedEBSDData(
        emap=emap, sh_x=11, sh_y=width,
        dx=0.5 if hex_grid else x_spacing,
        dy=np.sqrt(3.0) / 2.0 if hex_grid else 1.0,
        npx=11 * width, original_npx=121, hex_grid=hex_grid,
    )
    return loaded, source_pixel, target_pixel, winners


@pytest.mark.parametrize(
    "options", [{}, {"hex_grid": True}, {"x_spacing": 2.0}, {"unindexed": True}],
    ids=["square", "hex", "physical-distance", "unindexed"],
)
def test_nearest_phase_votes_copy_one_complete_measurement_record(options):
    """
    Resolve equal nearest distances before any subsequent grain cleanup.

    Final ownership may change when a tiny retained grain is absorbed. The
    preprocessing source indices preserve which measured donor won this step.

    Parameters
    ----------
    options : dict
        Spatial arrangement of the real measured scan.
    """
    loaded, omitted, target, winners = make_voting_map(**options)
    raw_phases = loaded.emap.phase_id.copy()
    raw_rotations = loaded.emap.rotations.data.copy()
    raw_confidence = loaded.emap.prop["ci"].copy()

    analyzer = EBSDmap(ebsd_data=loaded, graph_only=True, gs_min=1)

    source = analyzer._analysis_source_indices[target]
    assert source in winners
    assert raw_phases[omitted] == 4
    assert analyzer._analysis_original_phase_ids[target] == 4
    assert analyzer._analysis_confidence[target] == raw_confidence[source]
    if options.get("hex_grid"):
        xy = np.column_stack([loaded.emap.x, loaded.emap.y])
        _, _, queries = make_regular_grid(xy, nx=22, ny=11)
        original_sources = cKDTree(xy).query(queries, k=1)[1]
        unaffected = raw_phases[original_sources] != 4
        donors = np.flatnonzero(raw_phases != 4)
        ordinary_sources = donors[cKDTree(xy[donors]).query(queries, k=1)[1]]
        np.testing.assert_array_equal(
            analyzer._analysis_source_indices[unaffected], ordinary_sources[unaffected],
        )
    else:
        retained = np.flatnonzero(raw_phases != 4)
        np.testing.assert_array_equal(analyzer._analysis_source_indices[retained], retained)
    if options.get("unindexed"):
        np.testing.assert_array_equal(analyzer._analysis_phase_ids[[49, 71]], [-1, -1])
    np.testing.assert_array_equal(loaded.emap.phase_id, raw_phases)
    np.testing.assert_array_equal(loaded.emap.rotations.data, raw_rotations)
    np.testing.assert_array_equal(loaded.emap.prop["ci"], raw_confidence)


def test_phase_exactly_at_three_percent_survives_filtering_before_cleanup():
    """
    Keep a phase at the original fraction cutoff even when cleanup removes it.
    """
    loaded = _make_three_phase_map(extra_minor_pixel=98)

    analyzer = EBSDmap(ebsd_data=loaded, graph_only=True, gs_min=1)

    phases = {phase["index"]: phase for phase in analyzer.ms_data}
    assert set(phases) == {1, 2, 3}
    assert phases[3]["vf"] == pytest.approx(0.03)
    np.testing.assert_array_equal(analyzer._analysis_source_indices[[0, 98, 99]], [0, 98, 99])
