"""
Build a reusable EBSD graph handoff from an EBSD map.

This example returns an ``EBSDGraphResult`` instance and writes only the graph
handoff files explicitly enabled below.
"""

from pathlib import Path
import sys

repo_src = Path(__file__).resolve().parents[2] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

import kanapy as knpy


def print_graph_handoff_summary(result):
    """
    Print a concise summary of the graph handoff.

    The terminal output confirms graph construction and lists only the files
    explicitly requested by this example.

    Parameters
    ----------
    result : kanapy.graph_workflow.EBSDGraphResult
        In memory graph construction result.
    """
    graph = result.graph
    print("----------------------------------------")
    print("EBSD Graph Result")
    print(f"Phase: #{result.phase_index} ({result.phase_name})")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")

    if result.written_files:
        print("Written files:")
        for output_name, output_path in result.written_files.items():
            print(f"  {output_name}: {output_path}")
    else:
        print("Written files: none")

    if result.elapsed_time_seconds is not None:
        print(f"Elapsed time: {result.elapsed_time_seconds:.2f} s")

    print("----------------------------------------")


def main():
    """
    Build the EBSD graph handoff for downstream analysis.

    The API returns an in memory ``EBSDGraphResult``. PKL, JSON, overlay PNG,
    and IPF map PNG files are written here only because the example explicitly
    enables them.
    """
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "p558_250x_1.ang"
    output_dir = base_dir / "2D_graph_result"

    graph_config = knpy.EBSDGraphConfig(
        min_phase_pixel_fraction=0.03,
        max_misorientation_degrees=5.0,
        min_grain_size_pixels=20.0,
        connectivity=4,
        boundary_artifact_size_factor=2.0,
        boundary_artifact_fraction_min=0.85,
        convex_hull_min_points=3,
        similar_orientation_merge_degrees=5.0,
    )
    output_options = knpy.EBSDGraphOutputOptions(
        output_dir=output_dir,
        clear_output_dir=True,
        write_pickle=True,
        write_summary_json=True,
        write_graph_overlay=True,
        write_ipf_map=True,
        write_step_figures=False,
    )

    result = knpy.build_ebsd_graph(
        input_file=input_file,
        config=graph_config,
        output_options=output_options,
    )
    print_graph_handoff_summary(result)


if __name__ == "__main__":
    main()
