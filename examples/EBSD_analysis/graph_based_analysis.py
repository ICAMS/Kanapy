"""
Analyze an EBSD map through grain graphs and statistics, before 3D RVE generation.

The workflow creates an ``EBSDGraphResult`` with equivalent diameters, aspect
ratios, tilt angles, and their fitted distributions for every retained phase.
By default, phases below 3% are skipped. Multiple retained phases are segmented
together using four neighboring pixels, with each grain restricted to one phase.
Grain nodes store their phase ID and name, and only grains from the same phase
can be merged. All retained phases appear on one EBSD map, with nodes and edges
colored by phase and gray edges between adjacent grains of different phases.
The full graph is available as result.full_graph and in the graph pickle.

The output switches below enable graph files, statistics.json, histograms,
misorientation and IPF maps, segmentation images, and pole figures. Results go
to 2D_graph_result/ beside this script, which is cleared before each run by
default. Figures are saved without interactive previews. Run in an environment
with Kanapy installed. graph_to_rve.py runs the complete EBSD to RVE workflow.
The grain graph is saved as 00_graph_overlay.png so it sorts first by filename;
when show_figures is enabled, the graph is also the first preview.

Authors: Jun Xue, Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

October 2026
"""

from dataclasses import replace
from pathlib import Path

import kanapy as knpy
from kanapy.core.plotting import capture_figures
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# input and EBSD reconstruction settings
fname = BASE_DIR / "../fixtures/ebsd_316L_500x500.ang"
vf_min = 0.03
max_angle = 5.0  # degrees
min_size = 10.0  # pixels
connectivity = 4
boundary_artifact_size_factor = 2.0
boundary_artifact_fraction_min = 0.85
convex_hull_min_points = 3
merge_angle = None  # use max_angle for similar grain merges
hex_grid_target_spacing = None

# optional files and interactive previews
output_dir = BASE_DIR / "2D_graph_result"
clear_output_dir = True
write_graph_pickle = True
write_graph_summary = True
write_graph_overlay = True
write_ipf_map = True
write_statistics_json = True
write_analysis_figures = True
write_step_figures = False
show_figures = False


def write_statistics(path, result):
    """
    Save experimental distributions and fitting conventions for retained phases.

    The top level statistics describe the selected phase, as in the complete
    RVE example. Multiphase data also include a separate entry for every phase.

    Parameters
    ----------
    path : pathlib.Path
        Destination JSON file.
    result : kanapy.EBSDGraphResult
        Complete analysis containing per phase statistics.
    """
    statistics = result.to_statistics()
    if len(result.ebsd.ms_data) > 1:
        statistics["phases"] = []
        for phase in result.ebsd.ms_data:
            phase_result = replace(
                result, graph=phase["graph"], phase_data=phase,
                phase_index=int(phase["index"]), phase_name=phase["name"],
            )
            statistics["phases"].append(phase_result.to_statistics())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    knpy.write_stats(statistics, path.name, path=path.parent, indent=2)


def print_graph_handoff_summary(result):
    """
    Print graph counts and grain statistics for every retained phase.

    Medians describe the measured grain distributions. The file list contains
    only outputs enabled for this analysis.

    Parameters
    ----------
    result : kanapy.graph_workflow.EBSDGraphResult
        Complete EBSD analysis result.
    """
    print("----------------------------------------")
    print("EBSD Analysis Result")
    for phase in result.ebsd.ms_data:
        graph = phase["graph"]
        print(f"Phase: #{phase['index']} ({phase['name']})")
        print(f"Graph nodes: {graph.number_of_nodes()}; edges: {graph.number_of_edges()}")
        print(f"Equivalent diameter median: {np.median(phase['gs_data']):.3f} um")
        print(f"Aspect ratio median: {np.median(phase['ar_data']):.3f}")
        print(f"Tilt angle median: {np.degrees(np.median(phase['om_data'])):.3f} degrees")

    if result.written_files:
        print("Written files:")
        for output_name, output_path in result.written_files.items():
            print(f"  {output_name}: {output_path}")
    else:
        print("Written files: none")

    if result.elapsed_time_seconds is not None:
        print(f"Graph and statistics time: {result.elapsed_time_seconds:.2f} s")

    print("----------------------------------------")


def main():
    """
    Run the complete two dimensional EBSD analysis and optional exports.

    Kanapy provides the statistics and plotting methods. The example stops
    before descriptor construction, packing, voxelization, or orientation fitting.

    Returns
    -------
    kanapy.EBSDGraphResult
        Graphs and experimental statistics for subsequent Python analysis.
    """
    if not show_figures:
        plt.switch_backend("Agg")
    graph_config = knpy.EBSDGraphConfig(
        min_phase_pixel_fraction=vf_min,
        max_misorientation_degrees=max_angle,
        min_grain_size_pixels=min_size,
        connectivity=connectivity,
        boundary_artifact_size_factor=boundary_artifact_size_factor,
        boundary_artifact_fraction_min=boundary_artifact_fraction_min,
        convex_hull_min_points=convex_hull_min_points,
        similar_orientation_merge_degrees=max_angle if merge_angle is None else merge_angle,
        hex_grid_target_spacing=hex_grid_target_spacing,
    )
    output_options = knpy.EBSDGraphOutputOptions(
        output_dir=output_dir,
        clear_output_dir=clear_output_dir,
        write_pickle=write_graph_pickle,
        write_summary_json=write_graph_summary,
        write_graph_overlay=write_graph_overlay,
        graph_overlay_filename="00_graph_overlay.png",
        write_ipf_map=write_ipf_map,
        write_step_figures=write_step_figures,
    )

    result = knpy.build_ebsd_graph(
        input_file=fname,
        config=graph_config,
        output_options=output_options,
        include_statistics=True,
    )
    if show_figures:
        with plt.ioff():
            figure, _ = result.ebsd.plot_graph_overlay(iphase=None)
        plt.close(figure)
    if write_statistics_json:
        path = output_dir / "statistics.json"
        write_statistics(path, result)
        result.written_files["statistics_json"] = path
    if write_analysis_figures:
        files = []
        ebsd = result.ebsd
        for position, phase in enumerate(ebsd.ms_data):
            stage = "ebsd_histograms"
            if len(ebsd.ms_data) > 1:
                stage += f"_phase_{phase['index']}"
            with capture_figures(output_dir, stage, files, show=show_figures):
                ebsd.plot_histograms(iphase=position)
        for stage, plot in [
            ("ebsd_misorientation", ebsd.plot_mo_map),
            ("ebsd_ipf", ebsd.plot_ipf_map),
            ("ebsd_segmentation", ebsd.plot_segmentation),
            ("ebsd_pole_figures", ebsd.plot_pf),
            ("ebsd_ipf_projection", ebsd.plot_pf_proj),
        ]:
            with capture_figures(output_dir, stage, files, show=show_figures):
                plot()
        for path in files:
            result.written_files[path.stem] = path
    if write_graph_summary:
        knpy.write_stats(
            result.to_summary(), output_options.summary_filename, path=output_dir, indent=2,
        )
    print_graph_handoff_summary(result)
    return result


if __name__ == "__main__":
    main()
