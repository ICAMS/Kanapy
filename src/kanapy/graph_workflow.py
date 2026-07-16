"""
In memory EBSD graph workflow objects.

This module keeps graph construction results as Python objects and makes file
outputs explicit choices by the caller.
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .texture import (
    DEFAULT_BOUNDARY_ARTIFACT_FRACTION_MIN,
    DEFAULT_BOUNDARY_ARTIFACT_SIZE_FACTOR,
    DEFAULT_CONNECTIVITY,
    DEFAULT_MAX_MISORIENTATION_DEGREES,
    DEFAULT_MIN_VOLUME_FRACTION,
    DEFAULT_SIMILAR_ORIENTATION_MERGE_DEGREES,
    EBSDmap,
    CONVEX_HULL_MIN_POINTS,
    MIN_GRAIN_SIZE_PIXELS,
)


EBSD_GRAPH_FILENAME = "ebsd_graph.pkl"
GRAPH_SUMMARY_FILENAME = "ebsd_graph_summary.json"
GRAPH_OVERLAY_FILENAME = "graph_overlay.png"
GRAPH_IPF_MAP_FILENAME = "graph_step0_ipf_map.png"


@dataclass
class EBSDGraphConfig:
    """
    Store user configurable thresholds for EBSD graph construction.

    Defaults match the existing graph workflow while keeping each threshold
    visible to callers.
    """

    min_phase_pixel_fraction: float = DEFAULT_MIN_VOLUME_FRACTION
    max_misorientation_degrees: float = DEFAULT_MAX_MISORIENTATION_DEGREES
    min_grain_size_pixels: float = MIN_GRAIN_SIZE_PIXELS
    connectivity: int = DEFAULT_CONNECTIVITY
    boundary_artifact_size_factor: float = DEFAULT_BOUNDARY_ARTIFACT_SIZE_FACTOR
    boundary_artifact_fraction_min: float = DEFAULT_BOUNDARY_ARTIFACT_FRACTION_MIN
    convex_hull_min_points: int = CONVEX_HULL_MIN_POINTS
    similar_orientation_merge_degrees: float = (
        DEFAULT_SIMILAR_ORIENTATION_MERGE_DEGREES
    )

    def __post_init__(self) -> None:
        """
        Validate graph threshold settings.

        Convex hull construction needs at least three points in two
        dimensions.

        Raises
        ------
        ValueError
            If ``convex_hull_min_points`` is lower than three.
        """
        if self.convex_hull_min_points < CONVEX_HULL_MIN_POINTS:
            raise ValueError(
                f"convex_hull_min_points must be at least {CONVEX_HULL_MIN_POINTS}"
            )

    def to_dict(self) -> dict[str, Union[float, int]]:
        """
        Return graph construction thresholds as plain values.

        The dictionary is used in summaries and serialized handoff metadata.

        Returns
        -------
        dict
            Graph threshold and connectivity settings.
        """
        return {
            "min_phase_pixel_fraction": float(self.min_phase_pixel_fraction),
            "max_misorientation_degrees": float(self.max_misorientation_degrees),
            "min_grain_size_pixels": float(self.min_grain_size_pixels),
            "connectivity": int(self.connectivity),
            "boundary_artifact_size_factor": float(
                self.boundary_artifact_size_factor
            ),
            "boundary_artifact_fraction_min": float(
                self.boundary_artifact_fraction_min
            ),
            "convex_hull_min_points": int(self.convex_hull_min_points),
            "similar_orientation_merge_degrees": float(
                self.similar_orientation_merge_degrees
            ),
        }


@dataclass
class EBSDGraphOutputOptions:
    """
    Store optional graph output settings.

    No files are written unless their corresponding option is enabled. The
    example script opts into the four paper handoff files.
    """

    output_dir: Optional[Union[str, Path]] = None
    write_pickle: bool = False
    write_summary_json: bool = False
    write_graph_overlay: bool = False
    write_ipf_map: bool = False
    write_step_figures: bool = False
    clear_output_dir: bool = False
    pickle_filename: str = EBSD_GRAPH_FILENAME
    summary_filename: str = GRAPH_SUMMARY_FILENAME
    graph_overlay_filename: str = GRAPH_OVERLAY_FILENAME
    ipf_map_filename: str = GRAPH_IPF_MAP_FILENAME

    @property
    def any_file_output_enabled(self) -> bool:
        """
        Return whether any graph output file is enabled.

        Returns
        -------
        bool
            True when a configured output writes to disk.
        """
        return (
            self.write_pickle
            or self.write_summary_json
            or self.write_graph_overlay
            or self.write_ipf_map
            or self.write_step_figures
        )

    @property
    def requires_output_dir(self) -> bool:
        """
        Return whether the configured options need an output directory.

        Returns
        -------
        bool
            True when an output directory is required.
        """
        return self.any_file_output_enabled or self.clear_output_dir

    def resolved_output_dir(self) -> Path:
        """
        Return the configured output directory as a path.

        Raises
        ------
        ValueError
            If an output directory is required but was not configured.

        Returns
        -------
        pathlib.Path
            Resolved output directory path.
        """
        if self.output_dir is None:
            raise ValueError("output_dir is required when graph file output is enabled")

        return Path(self.output_dir)


@dataclass
class EBSDGraphResult:
    """
    Store an EBSD graph handoff in memory.

    Downstream Python functions can consume this object directly instead of
    reading a PKL, JSON, or PNG file.
    """

    graph: Any
    phase_data: dict[str, Any]
    phase_index: int
    phase_name: str
    rgb_image: np.ndarray
    npx: int
    sh_x: int
    sh_y: int
    source_input_file: Optional[Path] = None
    config: EBSDGraphConfig = field(default_factory=EBSDGraphConfig)
    elapsed_time_seconds: Optional[float] = None
    ebsd: Optional[EBSDmap] = None
    written_files: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_phase_data(
        cls,
        phase_data: dict[str, Any],
        npx: int,
        sh_x: int,
        sh_y: int,
        source_input_file: Optional[Union[str, Path]],
        config: EBSDGraphConfig,
        elapsed_time_seconds: Optional[float] = None,
        ebsd: Optional[EBSDmap] = None,
    ) -> "EBSDGraphResult":
        """
        Build a graph result from one ``EBSDmap.ms_data`` phase row.

        Parameters
        ----------
        phase_data : dict
            Phase row containing the generated graph and RGB IPF image.
        npx : int
            Total number of pixels in the EBSD map.
        sh_x : int
            Map height in pixels.
        sh_y : int
            Map width in pixels.
        source_input_file : str or pathlib.Path or None
            Input EBSD file used to build the graph.
        config : EBSDGraphConfig
            Graph construction thresholds.
        elapsed_time_seconds : float, optional
            Runtime for graph construction.
        ebsd : EBSDmap, optional
            Originating EBSD map object.

        Returns
        -------
        EBSDGraphResult
            In memory graph handoff object.
        """
        rgb_image = _phase_rgb_image(phase_data, npx, sh_x, sh_y)
        source_path = None
        if source_input_file is not None:
            source_path = Path(source_input_file)

        return cls(
            graph=phase_data["graph"],
            phase_data=phase_data,
            phase_index=int(phase_data["index"]),
            phase_name=str(phase_data["name"]),
            rgb_image=rgb_image,
            npx=int(npx),
            sh_x=int(sh_x),
            sh_y=int(sh_y),
            source_input_file=source_path,
            config=config,
            elapsed_time_seconds=elapsed_time_seconds,
            ebsd=ebsd,
        )

    def to_payload(self) -> dict[str, Any]:
        """
        Return the full Python payload for optional pickle output.

        Returns
        -------
        dict
            Serializable handoff payload containing the graph object.
        """
        return {
            "graph": self.graph,
            "phase": {
                "index": self.phase_index,
                "name": self.phase_name,
            },
            "rgb_image": self.rgb_image,
            "npx": self.npx,
            "sh_x": self.sh_x,
            "sh_y": self.sh_y,
            "source_input_file": (
                str(self.source_input_file) if self.source_input_file else None
            ),
            "thresholds": self.config.to_dict(),
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "graph_node_count": self.graph.number_of_nodes(),
            "graph_edge_count": self.graph.number_of_edges(),
        }

    def to_summary(self) -> dict[str, Any]:
        """
        Return a compact JSON friendly graph summary.

        Returns
        -------
        dict
            JSON serializable graph metadata.
        """
        summary = {
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "graph_node_count": self.graph.number_of_nodes(),
            "graph_edge_count": self.graph.number_of_edges(),
            "connectivity": int(self.config.connectivity),
            "source_input_file": (
                str(self.source_input_file) if self.source_input_file else None
            ),
            "thresholds": self.config.to_dict(),
            "elapsed_time_seconds": self.elapsed_time_seconds,
        }

        for key, path in self.written_files.items():
            summary[key] = Path(path).name

        return summary


def _phase_rgb_image(
    phase_data: dict[str, Any],
    npx: int,
    sh_x: int,
    sh_y: int,
) -> np.ndarray:
    """
    Return the phase RGB image normalized to zero to one.

    Parameters
    ----------
    phase_data : dict
        Phase data containing an ``rgb_im`` field.
    npx : int
        Total number of map pixels.
    sh_x : int
        Map height in pixels.
    sh_y : int
        Map width in pixels.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``(sh_x, sh_y, 3)``.
    """
    rgb_image = np.asarray(phase_data["rgb_im"], dtype=float)
    if rgb_image.shape == (npx, 3):
        rgb_image = rgb_image.reshape((sh_x, sh_y, 3))
    if rgb_image.max(initial=0.0) > 1.0:
        rgb_image = rgb_image / 255.0

    return rgb_image


def select_major_phase(ebsd: EBSDmap) -> tuple[int, dict[str, Any]]:
    """
    Return the retained phase with the largest volume fraction.

    Parameters
    ----------
    ebsd : EBSDmap
        Analyzed EBSD map object.

    Raises
    ------
    ValueError
        If no phase passed the configured volume fraction filter.

    Returns
    -------
    tuple
        Phase list index and selected phase data dictionary.
    """
    if len(ebsd.ms_data) == 0:
        raise ValueError("No EBSD phase passed the volume fraction filter.")

    phase_rows = list(enumerate(ebsd.ms_data))
    return max(phase_rows, key=lambda row: row[1]["vf"])


def write_graph_result_outputs(
    result: EBSDGraphResult,
    output_options: EBSDGraphOutputOptions,
) -> dict[str, Path]:
    """
    Write selected graph handoff files.

    Parameters
    ----------
    result : EBSDGraphResult
        In memory graph result to serialize or plot.
    output_options : EBSDGraphOutputOptions
        File output settings.

    Returns
    -------
    dict
        Mapping from output kind to written path.
    """
    if not output_options.any_file_output_enabled:
        return {}

    output_dir = output_options.resolved_output_dir()
    if output_options.clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: dict[str, Path] = {}

    if output_options.write_pickle:
        pickle_path = output_dir / output_options.pickle_filename
        with pickle_path.open("wb") as file_handle:
            pickle.dump(
                result.to_payload(),
                file_handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        written_files["pickle"] = pickle_path
        result.written_files["pickle"] = pickle_path

    if output_options.write_graph_overlay:
        overlay_path = output_dir / output_options.graph_overlay_filename
        _write_graph_overlay(result, overlay_path)
        written_files["graph_overlay"] = overlay_path
        result.written_files["graph_overlay"] = overlay_path

    if output_options.write_ipf_map:
        ipf_path = output_dir / output_options.ipf_map_filename
        _write_ipf_map(result, ipf_path)
        written_files["ipf_map"] = ipf_path
        result.written_files["ipf_map"] = ipf_path

    if output_options.write_summary_json:
        summary_path = output_dir / output_options.summary_filename
        with summary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(result.to_summary(), file_handle, indent=2)
            file_handle.write("\n")
        written_files["summary_json"] = summary_path
        result.written_files["summary_json"] = summary_path

    return written_files


def _node_center(
    graph: Any,
    node_id: int,
    image_shape: tuple[int, int],
) -> Optional[tuple[float, float]]:
    """
    Return a graph node center in image pixel coordinates.

    Parameters
    ----------
    graph : networkx.Graph
        Microstructure graph containing the node.
    node_id : int
        Graph node ID.
    image_shape : tuple of int
        EBSD map shape as ``(rows, columns)``.

    Returns
    -------
    tuple or None
        Center as ``(x, y)`` image coordinates, or None if no center exists.
    """
    node = graph.nodes[node_id]
    center = node.get("center")
    if center is not None:
        dx = graph.graph.get("dx", 1.0)
        dy = graph.graph.get("dy", 1.0)
        return float(center[1]) / dy, float(center[0]) / dx

    pixels = node.get("pixels")
    if pixels is None or len(pixels) == 0:
        return None

    rows, cols = np.unravel_index(np.asarray(pixels, dtype=int), image_shape)
    return float(np.mean(cols)), float(np.mean(rows))


def _figure_size_for_image(image_shape: tuple[int, ...]) -> tuple[float, float]:
    """
    Return a figure size that follows the image aspect ratio.

    Parameters
    ----------
    image_shape : tuple of int
        Image shape as ``(rows, columns)`` or ``(rows, columns, channels)``.

    Returns
    -------
    tuple
        Figure size in inches.
    """
    rows, cols = image_shape[:2]
    return 8.0 * cols / rows, 8.0


def _format_image_axes(ax: Any, image_shape: tuple[int, ...]) -> None:
    """
    Format an EBSD image axis with compact pixel coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis containing an EBSD image.
    image_shape : tuple of int
        Image shape as ``(rows, columns)`` or ``(rows, columns, channels)``.
    """
    rows, cols = image_shape[:2]
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)


def _save_tight_figure(fig: Any, output_path: Path) -> None:
    """
    Save a figure with compact margins.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    output_path : pathlib.Path
        Output image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.03)


def _write_ipf_map(result: EBSDGraphResult, output_path: Path) -> None:
    """
    Write the EBSD IPF map PNG.

    Parameters
    ----------
    result : EBSDGraphResult
        Graph result containing the IPF RGB image.
    output_path : pathlib.Path
        PNG output path.
    """
    fig, ax = plt.subplots(figsize=_figure_size_for_image(result.rgb_image.shape))
    ax.imshow(result.rgb_image, origin="upper")
    ax.set_title("EBSD map", fontsize=16, fontfamily="DejaVu Sans", pad=14)
    _format_image_axes(ax, result.rgb_image.shape)
    fig.tight_layout(pad=0.3)
    _save_tight_figure(fig, output_path)
    plt.close(fig)


def _write_graph_overlay(result: EBSDGraphResult, output_path: Path) -> None:
    """
    Write the final graph overlay PNG.

    Parameters
    ----------
    result : EBSDGraphResult
        Graph result containing the final graph and IPF RGB image.
    output_path : pathlib.Path
        PNG output path.
    """
    graph = result.graph
    image_shape = result.rgb_image.shape[:2]
    fig, ax = plt.subplots(figsize=_figure_size_for_image(result.rgb_image.shape))
    ax.imshow(result.rgb_image, origin="upper")

    for node_a, node_b in graph.edges():
        center_a = _node_center(graph, node_a, image_shape)
        center_b = _node_center(graph, node_b, image_shape)
        if center_a is None or center_b is None:
            continue

        xa, ya = center_a
        xb, yb = center_b
        ax.plot([xa, xb], [ya, yb], color="black", linewidth=0.7)

    for node_id in graph.nodes:
        center = _node_center(graph, node_id, image_shape)
        if center is None:
            continue

        x, y = center
        ax.plot(x, y, "ko", markersize=2.5)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    ax.set_title(
        f"Final graph ({node_count} nodes, {edge_count} edges)",
        fontsize=16,
        fontfamily="DejaVu Sans",
        pad=14,
    )
    _format_image_axes(ax, result.rgb_image.shape)
    fig.tight_layout(pad=0.3)
    _save_tight_figure(fig, output_path)
    plt.close(fig)


def load_ebsd_graph(path: Union[str, Path]) -> dict[str, Any]:
    """
    Load an optional EBSD graph pickle handoff.

    Parameters
    ----------
    path : str or pathlib.Path
        Pickle file to read.

    Returns
    -------
    dict
        Graph handoff payload.
    """
    with Path(path).open("rb") as file_handle:
        return pickle.load(file_handle)


def build_ebsd_graph(
    input_file: Union[str, Path],
    config: Optional[EBSDGraphConfig] = None,
    output_options: Optional[EBSDGraphOutputOptions] = None,
) -> EBSDGraphResult:
    """
    Build an EBSD graph and return an in memory result object.

    Parameters
    ----------
    input_file : str or pathlib.Path
        EBSD input file used for graph construction.
    config : EBSDGraphConfig, optional
        Graph construction thresholds and connectivity.
    output_options : EBSDGraphOutputOptions, optional
        Optional file output settings.

    Raises
    ------
    FileNotFoundError
        If the EBSD input file does not exist.

    Returns
    -------
    EBSDGraphResult
        In memory graph result for downstream Python functions.
    """
    graph_config = config or EBSDGraphConfig()
    graph_output_options = output_options or EBSDGraphOutputOptions()
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"EBSD input file not found: {input_path}")

    if graph_output_options.requires_output_dir:
        output_dir = graph_output_options.resolved_output_dir()
        if graph_output_options.clear_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    ebsd = EBSDmap(
        str(input_path),
        vf_min=graph_config.min_phase_pixel_fraction,
        gs_min=graph_config.min_grain_size_pixels,
        max_angle=graph_config.max_misorientation_degrees,
        connectivity=graph_config.connectivity,
        boundary_artifact_size_factor=graph_config.boundary_artifact_size_factor,
        boundary_artifact_fraction_min=graph_config.boundary_artifact_fraction_min,
        convex_hull_min_points=graph_config.convex_hull_min_points,
        similar_orientation_merge_degrees=(
            graph_config.similar_orientation_merge_degrees
        ),
        show_plot=False,
        show_grains=False,
        show_hist=False,
        show_graph=False,
    )

    _, selected_phase_data = select_major_phase(ebsd)
    elapsed_time_seconds = time.perf_counter() - start_time
    result = EBSDGraphResult.from_phase_data(
        phase_data=selected_phase_data,
        npx=ebsd.npx,
        sh_x=ebsd.sh_x,
        sh_y=ebsd.sh_y,
        source_input_file=input_path,
        config=graph_config,
        elapsed_time_seconds=elapsed_time_seconds,
        ebsd=ebsd,
    )

    write_options = graph_output_options
    if graph_output_options.clear_output_dir:
        write_options = replace(graph_output_options, clear_output_dir=False)

    write_graph_result_outputs(result, write_options)
    return result
