"""
Verify distance priority and phase voting when filling omitted measurements.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from kanapy.phase_assignment import nearest_phase_sources


def test_equal_distance_candidates_vote_by_phase():
    """
    Select the repeated phase from four equally near measurements.
    """
    xy = np.array([[-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
    phase_ids = np.array([2, 1, 3, 1])
    source = nearest_phase_sources(xy, phase_ids, np.arange(4), [[0., 0.]])
    assert phase_ids[source[0]] == 1
    assert source[0] in (1, 3)


def test_vote_tie_selects_only_a_phase_with_the_most_votes():
    """
    Resolve a two two one vote without selecting the one vote phase.
    """
    xy = np.array([[.6, .8], [-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
    phase_ids = np.array([3, 2, 2, 1, 1])
    first = nearest_phase_sources(xy, phase_ids, np.arange(5), [[0., 0.]])
    second = nearest_phase_sources(xy, phase_ids, np.arange(5), [[0., 0.]])
    assert phase_ids[first[0]] in (1, 2)
    np.testing.assert_array_equal(first, second)


def test_vote_includes_more_than_six_equal_distance_candidates():
    """
    Count the whole closest ring rather than a fixed number of neighbors.
    """
    xy = np.array([[3, 4], [4, 3], [4, -3], [3, -4],
                   [-3, -4], [-4, -3], [-4, 3], [-3, 4]], dtype=float)
    phase_ids = np.array([1, 1, 1, 3, 2, 2, 2, 2])
    result = nearest_phase_sources(xy, phase_ids, np.arange(8), [[0., 0.]])
    assert phase_ids[result[0]] == 2


@pytest.mark.parametrize("scale", [1.e-10, 1., 1.e8])
def test_farther_majority_never_overrides_the_nearest_measurement(scale):
    """
    Keep real distance differences at small and large physical scales.

    Parameters
    ----------
    scale : float
        Coordinate scaling that must not turn farther points into a tie.
    """
    xy = scale * np.array([[1., 0.], [0., 1.0000001],
                           [-1.0000001, 0.], [0., -1.0000001]])
    phase_ids = np.array([3, 1, 1, 1])
    source = nearest_phase_sources(xy, phase_ids, np.arange(4), [[0., 0.]])
    assert source[0] == 0


def test_single_phase_keeps_existing_nearest_orientation_donors():
    """
    Preserve the original k one choice for single phase workflows.
    """
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    queries = np.array([[.5, .5], [.5, 0.], [1., .5], [.1, .2]])
    expected = cKDTree(xy).query(queries, k=1, workers=-1)[1]
    result = nearest_phase_sources(xy, np.ones(4, dtype=int), np.arange(4), queries)
    np.testing.assert_array_equal(result, expected)


def test_only_selected_original_donors_participate():
    """
    Ignore a closer omitted measurement and return an original source index.
    """
    xy = np.array([[0., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    phase_ids = np.array([9, 3, 1, 2, 1])
    donors = np.array([1, 2, 3, 4])
    result = nearest_phase_sources(xy, phase_ids, donors, [[0., 0.]])
    assert result[0] in (2, 4)
    assert phase_ids[result[0]] == 1


def test_one_donor_and_empty_query_batch_are_supported():
    """
    Handle the smallest donor set and a batch with nothing to fill.
    """
    xy = np.array([[0., 0.], [1., 0.]])
    result = nearest_phase_sources(xy, np.array([1, 2]), np.array([1]), [[0., 0.]])
    np.testing.assert_array_equal(result, [1])
    empty = nearest_phase_sources(xy, np.array([1, 2]), np.array([1]), np.empty((0, 2)))
    assert empty.shape == (0,)
