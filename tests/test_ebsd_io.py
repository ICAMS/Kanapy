"""
Tests for loading ANG geometry and declared property names.
"""

from types import SimpleNamespace
from pathlib import Path
import warnings

import numpy as np
import pytest

from kanapy.ebsd_io import (
    EBSDDataLoader,
    _read_declared_ang_property_names,
    read_scan_step_from_ang_header,
)


def test_loader_uses_validated_declared_ang_property_names(tmp_path, monkeypatch):
    """
    Preserve custom ANG property names and their array order.

    The header declaration is valid only when its core columns and row width agree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the ANG fixture.
    monkeypatch : pytest.MonkeyPatch
        Substitute the external orix file read.
"""
    path = tmp_path / "custom.ang"
    path.write_text(
        "# GRID: SqrGrid\n"
        "# XSTEP: 2.0\n"
        "# YSTEP: 3.0\n"
        "# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase_Index, My Metric\n"
        "0 0 0 0 0 10 0.5 1 7\n",
        encoding="utf-8",
    )
    iq = np.array([10.0])
    ci = np.array([0.5])
    metric = np.array([7.0])
    emap = SimpleNamespace(
        prop={"iq": iq, "ci": ci, "unknown_1": metric},
        shape=(1, 1), dx=2.0, dy=3.0, size=1,
        phases=SimpleNamespace(ids=[1, -1]),
    )

    def fake_load(filepath):
        """
Return an orix shaped map for the supplied ANG file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            ANG path supplied to the external loader.

        Returns
        -------
        SimpleNamespace
            Map with generated property keys.
        """
        assert str(filepath) == str(path)
        warnings.warn("Number of columns, 9, does not match expected", UserWarning)
        return emap

    monkeypatch.setattr("kanapy.ebsd_io.io.load", fake_load)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = EBSDDataLoader().load(str(path))

    assert not caught
    assert list(loaded.emap.prop) == ["iq", "ci", "my_metric"]
    assert loaded.emap.prop["my_metric"] is metric
    assert (loaded.sh_x, loaded.sh_y, loaded.npx) == (1, 1, 1)
    assert (loaded.source_grid, loaded.source_xstep, loaded.source_ystep) == (
        "SqrGrid", 2.0, 3.0,
    )


def test_malformed_column_declaration_keeps_normal_orix_handling(tmp_path, monkeypatch):
    """
    Leave generated property keys and column warnings intact for bad headers.

    The first data row has fewer columns than the declaration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the ANG fixture.
    monkeypatch : pytest.MonkeyPatch
        Substitute the external orix file read.
    """
    path = tmp_path / "malformed.ang"
    path.write_text(
        "# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase_Index, My Metric\n"
        "0 0 0 0 0 10 0.5 1\n",
        encoding="utf-8",
    )
    emap = SimpleNamespace(
        prop={"unknown_1": np.array([7.0])},
        shape=(1, 1), dx=1.0, dy=1.0, size=1,
        phases=SimpleNamespace(ids=[1]),
    )

    def fake_load(filepath):
        """
Return an orix shaped map while emitting its column warning.

        Parameters
        ----------
        filepath : str or pathlib.Path
            ANG path supplied to the external loader.

        Returns
        -------
        SimpleNamespace
            Map with an unchanged generated property key.
        """
        warnings.warn("Number of columns, 8, does not match expected", UserWarning)
        return emap

    monkeypatch.setattr("kanapy.ebsd_io.io.load", fake_load)

    assert _read_declared_ang_property_names(path) is None
    with pytest.warns(UserWarning, match="Number of columns"):
        loaded = EBSDDataLoader().load(str(path))

    assert list(loaded.emap.prop) == ["unknown_1"]


def test_loader_converts_edax_symmetry_codes_when_orix_rejects_them(tmp_path, monkeypatch):
    """
    Retry a rejected ANG map after replacing supported EDAX symmetry codes.

    The original input file must remain untouched.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the ANG fixture.
    monkeypatch : pytest.MonkeyPatch
        Substitute the external orix file read.
    """
    path = tmp_path / "edax.ang"
    original = "# Symmetry 43\n# Symmetry 62\n0 0 0 0 0 0 0 1\n"
    path.write_text(original, encoding="utf-8")
    emap = SimpleNamespace(
        prop={}, shape=(1, 2), dx=1.0, dy=1.0, size=1,
        phases=SimpleNamespace(ids=[1]),
    )
    attempts = []

    def fake_load(filepath):
        """
Reject numeric symmetry and accept translated point group names.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Source or temporary ANG path.

        Returns
        -------
        SimpleNamespace
            Map after symmetry translation.
        """
        content = Path(filepath).read_text(encoding="utf-8")
        attempts.append(content)
        if "# Symmetry 43" in content:
            raise ValueError("Symmetry must be of type valid point group")
        assert "# Symmetry m-3m" in content
        assert "# Symmetry 6/mmm" in content
        return emap

    monkeypatch.setattr("kanapy.ebsd_io.io.load", fake_load)

    loaded = EBSDDataLoader().load(str(path))

    assert len(attempts) == 2
    assert path.read_text(encoding="utf-8") == original
    assert loaded.hex_grid is True
    assert (loaded.original_npx, loaded.npx) == (1, 2)


def test_scan_header_reads_steps_and_stops_at_data(tmp_path):
    """
    Read geometry only from comment lines before ANG data.

    A later comment cannot change the source grid metadata.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the ANG fixture.
    """
    path = tmp_path / "scan.ang"
    path.write_text(
        "# GRID: HexGrid\n# XSTEP: 5.0\n# YSTEP: 4.330127\n"
        "0 0 0 0 0 0 0 1\n# XSTEP: 99\n",
        encoding="utf-8",
    )

    assert read_scan_step_from_ang_header(path) == {
        "grid": "HexGrid", "xstep": 5.0, "ystep": 4.330127,
    }
    assert read_scan_step_from_ang_header(path.with_suffix(".txt")) == {
        "grid": None, "xstep": None, "ystep": None,
    }
