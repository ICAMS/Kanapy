"""
Exercise JSON output for scientific statistics.
"""

import json

import numpy as np
import pytest

from kanapy.core.input_output import write_stats


def test_write_stats_serializes_scientific_values_and_path_filename(tmp_path):
    """
    Preserve numeric arrays, scalar values, and paths in a JSON file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    stats = {
        "dimensions": np.array([20, 30, 40]),
        "fractions": np.array([0.25, 0.75]),
        "grain_count": np.int64(7),
        "mean_size": np.float64(2.5),
        "source": tmp_path / "input.ang",
    }
    write_stats(stats, tmp_path / "statistics")
    written = json.loads((tmp_path / "statistics.json").read_text())
    assert written == {
        "dimensions": [20, 30, 40],
        "fractions": [0.25, 0.75],
        "grain_count": 7,
        "mean_size": 2.5,
        "source": str(tmp_path / "input.ang"),
    }


def test_write_stats_keeps_compact_default_and_allows_indent(tmp_path):
    """
    Preserve the original formatting unless indentation is requested.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    write_stats({"size": 2}, "compact", path=tmp_path)
    write_stats({"size": 2}, "indented", path=tmp_path, indent=2)
    assert (tmp_path / "compact.json").read_text() == '{"size": 2}'
    assert (tmp_path / "indented.json").read_text() == '{\n  "size": 2\n}'


def test_write_stats_rejects_unsupported_values(tmp_path):
    """
    Retain the normal JSON error for an unsupported object.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    with pytest.raises(TypeError):
        write_stats({"unknown": object()}, "unsupported", path=tmp_path)
