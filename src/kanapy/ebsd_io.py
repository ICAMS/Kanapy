"""
Load EBSD maps with ANG property names and source scan geometry.

The loader keeps the map and physical grid metadata together for graph analysis.
"""

import logging
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orix import io


_ANG_CORE_COLUMN_POSITIONS = {
    0: "phi1",
    1: "phi",
    2: "phi2",
    3: "x",
    4: "y",
    7: "phase_index",
}

_ANG_PROPERTY_NAME_ALIASES = {
    "sem": "detector_signal",
}


def _normalize_ang_column_name(column_name: str) -> str:
    """
    Normalize an ANG column label to a property name.

    The normalized names follow the names used by orix for standard TSL
    properties while retaining meaningful names for custom columns.

    Parameters
    ----------
    column_name : str
        Column label read from the ANG header.

    Returns
    -------
    str
        Normalized property name.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().casefold())
    normalized = normalized.strip("_")
    return _ANG_PROPERTY_NAME_ALIASES.get(normalized, normalized)


def _read_declared_ang_property_names(filepath: str | Path) -> list[str] | None:
    """
    Read validated property names from a custom ANG header.

    A result is returned only when the declared columns, the first data row,
    and the core EBSD column positions agree. This keeps warning suppression
    limited to files whose mapping is unambiguous.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Candidate ANG file.

    Returns
    -------
    list[str] or None
        Property names in file order, or ``None`` when no validated custom
        declaration is available.
    """
    source = Path(filepath)
    if source.suffix.casefold() != ".ang" or not source.is_file():
        return None

    declared_columns = None
    data_column_count = None

    with source.open("r", encoding="utf-8", errors="replace") as file_handle:
        for line_number, line in enumerate(file_handle):
            if line_number >= 1_000:
                break

            if line.startswith("#"):
                label, separator, value = line[1:].partition(":")
                if separator and label.strip().casefold() == "column_headers":
                    declared_columns = [item.strip() for item in value.split(",")]
                continue

            if line.strip():
                data_column_count = len(line.split())
                break

    if not declared_columns or data_column_count != len(declared_columns):
        return None

    normalized_columns = [
        _normalize_ang_column_name(column_name)
        for column_name in declared_columns
    ]
    if len(normalized_columns) <= max(_ANG_CORE_COLUMN_POSITIONS):
        return None

    for position, expected_name in _ANG_CORE_COLUMN_POSITIONS.items():
        if normalized_columns[position] != expected_name:
            return None

    property_names = [
        column_name
        for position, column_name in enumerate(normalized_columns)
        if position not in _ANG_CORE_COLUMN_POSITIONS
    ]
    if not all(property_names) or len(property_names) != len(set(property_names)):
        return None

    return property_names


def _load_orix_map(filepath: str | Path, suppress_column_warning: bool) -> Any:
    """
    Load an EBSD map with optional suppression of a validated column warning.

    The warning is suppressed only after this module has independently
    validated the custom ANG column declaration.

    Parameters
    ----------
    filepath : str or pathlib.Path
        EBSD file passed to orix.
    suppress_column_warning : bool
        Whether to suppress the unsupported column count warning from the
        orix ANG reader.

    Returns
    -------
    object
        Loaded EBSD map returned by orix.
    """
    if not suppress_column_warning:
        return io.load(filepath)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Number of columns, .*",
            category=UserWarning,
        )
        return io.load(filepath)


def _apply_declared_ang_property_names(
    emap: Any, property_names: list[str]
) -> None:
    """
    Replace generated orix property names with declared ANG names.

    Property arrays are moved between dictionary keys without copying their
    large numeric buffers.

    Parameters
    ----------
    emap : object
        Crystal map returned by orix.
    property_names : list[str]
        Validated property names in file order.

    Raises
    ------
    ValueError
        If orix returned a different number of property arrays.
    """
    properties = emap.prop
    current_names = list(properties.keys())
    if len(current_names) != len(property_names):
        raise ValueError(
            "ANG property count does not match the validated header: "
            f"orix returned {len(current_names)}, header declares "
            f"{len(property_names)}"
        )

    property_arrays = [properties.pop(name) for name in current_names]
    for name, values in zip(property_names, property_arrays):
        properties.setdefault(name, values)


def _write_ang_with_point_group_names(
    src_file: str | Path, dst_file: str | Path
) -> bool:
    """
    Write an ANG file with EDAX symmetry codes converted to point group names.

    Parameters
    ----------
    src_file : str or pathlib.Path
        Source ANG file.
    dst_file : str or pathlib.Path
        Destination ANG file.

    Returns
    -------
    bool
        True if at least one symmetry line was converted.
    """
    symmetry_codes = {
        "43": "m-3m",
        "62": "6/mmm",
    }
    changed = False

    with open(src_file, "r", encoding="utf-8", errors="replace") as src:
        with open(dst_file, "w", encoding="utf-8", newline="") as dst:
            for line in src:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "#" and parts[1] == "Symmetry":
                    code = parts[-1]
                    point_group = symmetry_codes.get(code)
                    if point_group is not None:
                        pos = line.rfind(code)
                        line = line[:pos] + point_group + line[pos + len(code) :]
                        changed = True
                dst.write(line)

    return changed


def _load_ebsd_map(filepath: str | Path) -> Any:
    """
    Load an EBSD map with a fallback for older orix ANG parsing.

    Parameters
    ----------
    filepath : str or pathlib.Path
        EBSD file path.

    Returns
    -------
    object
        Loaded EBSD map returned by orix.
    """
    property_names = _read_declared_ang_property_names(filepath)
    suppress_column_warning = property_names is not None

    try:
        emap = _load_orix_map(filepath, suppress_column_warning)
    except ValueError as exc:
        message = str(exc)
        if "must be of type" not in message or "valid point group" not in message:
            raise

        src_file = Path(filepath)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dst_file = Path(tmp_dir) / src_file.name
            changed = _write_ang_with_point_group_names(src_file, dst_file)
            if not changed:
                raise

            logging.warning(
                "Retrying ANG import after converting EDAX symmetry codes "
                "to point group names for this orix version."
            )
            emap = _load_orix_map(str(dst_file), suppress_column_warning)

    if property_names is not None:
        _apply_declared_ang_property_names(emap, property_names)

    return emap


@dataclass
class LoadedEBSDData:
    """
    Hold an orix map and its source grid geometry.

    A staggered hex scan retains its original point count while ``npx``
    represents the rectangular shape prepared for graph analysis.

    Attributes
    ----------
    emap : object
        Loaded orix crystal map.
    sh_x, sh_y : int
        Dimensions of the orix map shape.
    dx, dy : float
        Map spacing reported by orix.
    npx : int
        Pixel count in the rectangular map shape.
    hex_grid : bool
        Whether the map shape and source point count differ.
    original_npx : int or None
        Source point count before resampling.
    source_grid : str or None
        Grid type declared in the ANG header.
    source_xstep, source_ystep : float or None
        Physical scan steps declared in the ANG header.
    """

    emap: Any  # orix CrystalMap object
    sh_x: int
    sh_y: int
    dx: float
    dy: float
    npx: int
    hex_grid: bool = False
    original_npx: int | None = None
    source_grid: str | None = None
    source_xstep: float | None = None
    source_ystep: float | None = None


class EBSDDataLoader:
    """
    Load EBSD maps and collect their geometry.

    The resulting data object can be passed to graph analysis without another
    file read.
    """

    def __init__(self):
        """
        Initialize the logger used for map import diagnostics.
        """
        self.logger = logging.getLogger(__name__)

    def load(self, filepath: str) -> LoadedEBSDData:
        """
        Load an EBSD map and extract its geometry.

        ANG scan steps are retained so staggered hexagonal maps can be
        resampled in physical coordinates during graph construction.

        Parameters
        ----------
        filepath : str
            Path to the EBSD data file.

        Returns
        -------
        LoadedEBSDData
            Loaded map and source geometry metadata.
        """
        emap = _load_ebsd_map(filepath)

        sh_x, sh_y = emap.shape
        dx, dy = emap.dx, emap.dy
        original_npx = emap.size
        scan_step = read_scan_step_from_ang_header(filepath)

        expected_size = sh_x * sh_y
        hex_grid = expected_size != original_npx
        if hex_grid:
            npx = expected_size
            self.logger.warning(
                f"Size of map ({original_npx} px) does not match its shape: "
                f"({sh_x}, {sh_y}). Assuming staggered hex grid of EBSD map."
            )
        else:
            npx = original_npx

        valid_phase_ids = [pid for pid in emap.phases.ids if pid != -1]
        self.logger.info(f"Imported EBSD map with {len(valid_phase_ids)} phases.")

        return LoadedEBSDData(
            emap=emap,
            sh_x=sh_x,
            sh_y=sh_y,
            dx=dx,
            dy=dy,
            npx=npx,
            hex_grid=hex_grid,
            original_npx=original_npx,
            source_grid=scan_step.get("grid"),
            source_xstep=scan_step.get("xstep"),
            source_ystep=scan_step.get("ystep"),
        )


def read_scan_step_from_ang_header(ang_path: str | Path) -> dict[str, str | float | None]:
    """
    Read the source grid and scan steps from an ANG header.

    These physical steps describe the source coordinates used when a staggered
    hex scan is resampled for graph analysis.

    Parameters
    ----------
    ang_path : str or pathlib.Path
        Candidate ANG input file.

    Returns
    -------
    dict
        Grid type and optional x and y scan steps.
    """
    ang_path = Path(ang_path)
    metadata = {"grid": None, "xstep": None, "ystep": None}
    if not ang_path.exists() or ang_path.suffix.lower() != ".ang":
        return metadata

    with ang_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break

            parts = stripped[1:].strip().split()
            if len(parts) < 2:
                continue

            key = parts[0].rstrip(":").upper()
            value = parts[1]
            if key == "GRID":
                metadata["grid"] = value
            elif key == "XSTEP":
                metadata["xstep"] = float(value)
            elif key == "YSTEP":
                metadata["ystep"] = float(value)

    return metadata
