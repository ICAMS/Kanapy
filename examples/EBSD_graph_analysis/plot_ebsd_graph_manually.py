"""
Read an EBSD map and visualize the resulting 2D microstructure graph.

The graph nodes represent detected grains or grain regions. Graph edges mark
neighboring nodes in the EBSD pixel map.
"""

from pathlib import Path
import sys

import numpy as np

repo_src = Path(__file__).resolve().parents[2] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

import kanapy as knpy


def get_graph_summary_lines(ms_data):
    """Create a concise summary of the EBSD graph.

    Parameters
    ----------
    ms_data : dict
        Phase data dictionary from ``ebsd.ms_data``.

    Returns
    -------
    list[str]
        Summary lines for terminal and text-file output.
    """
    graph = ms_data["graph"]
    degrees = np.array([degree for _, degree in graph.degree], dtype=float)
    npix = np.array([node["npix"] for _, node in graph.nodes.items()], dtype=float)

    lines = [
        "*** EBSD graph information ***",
        f"Phase: #{ms_data['index']} ({ms_data['name']})",
        f"Nodes: {graph.number_of_nodes()}",
        f"Edges: {graph.number_of_edges()}",
        f"Mean node degree: {degrees.mean():.2f}",
        f"Median node area: {np.median(npix):.1f} pixels",
    ]

    if "graph_initial" in ms_data:
        initial_graph = ms_data["graph_initial"]
        lines.append(
            f"Initial nodes before graph cleanup: {initial_graph.number_of_nodes()}"
        )

    if "merge_debug" in ms_data:
        performed_merges = []
        for row in ms_data["merge_debug"]:
            if row.get("will_merge", False):
                performed_merges.append(row)
        lines.append(f"Recorded graph merges: {len(performed_merges)}")

    return lines


def select_major_phase(ebsd):
    """Return the retained phase with the largest volume fraction.

    Parameters
    ----------
    ebsd : kanapy.texture.EBSDmap
        EBSD map object.

    Returns
    -------
    tuple[int, dict]
        Phase list index and phase data dictionary.
    """
    if len(ebsd.ms_data) == 0:
        raise ValueError("No EBSD phase passed the volume fraction filter.")

    phase_rows = list(enumerate(ebsd.ms_data))
    return max(phase_rows, key=lambda row: row[1]["vf"])


def write_graph_summary(ms_data, out_file):
    """Print and save graph summary information.

    Parameters
    ----------
    ms_data : dict
        Phase data dictionary from ``ebsd.ms_data``.
    out_file : pathlib.Path
        Text file where the summary is saved.
    """
    lines = get_graph_summary_lines(ms_data)
    text = "\n".join(lines)
    print(f"\n{text}")
    out_file.write_text(text + "\n", encoding="utf-8")


def main():
    """Run the EBSD graph visualization example."""
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "p558_250x_1.ang"
    result_dir = base_dir / "plot_ebsd_graph_manually_result"
    result_dir.mkdir(exist_ok=True)

    overlay_file = result_dir / "ebsd_graph_overlay.png"
    summary_file = result_dir / "ebsd_graph_summary.txt"

    vf_min = 0.03       # minimum phase volume fraction to keep
    max_angle = 5.0     # maximum misorientation angle within one grain in degrees
    min_size = 20.0     # minimum grain size in pixels
    connectivity = 4    # use edge sharing pixel neighbors

    if not input_file.exists():
        raise FileNotFoundError(
            f"EBSD input file not found: {input_file}. "
            "Set input_file to the path of an ANG or CTF file before running this example."
        )

    ebsd = knpy.EBSDmap(
        str(input_file),
        vf_min=vf_min,
        gs_min=min_size,
        max_angle=max_angle,
        connectivity=connectivity,
        show_plot=False,
        show_grains=False,
        show_hist=False,
    )

    iphase, ms_data = select_major_phase(ebsd)
    ebsd.plot_graph_overlay(
        iphase=iphase,
        save_path=overlay_file,
        show=False,
    )
    write_graph_summary(ms_data, summary_file)

    print(f"Graph overlay saved to: {overlay_file}")
    print(f"Graph summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
