"""
Draw grain contacts with distinct colors for each retained phase.
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np


def phase_graph_colors(phases):
    """
    Assign black first, then distinct colors without cycling a short palette.

    Parameters
    ----------
    phases : list of dict
        Phase IDs in decreasing original volume fraction order.

    Returns
    -------
    dict
        Phase IDs mapped to RGB tuples.
    """
    colors = [to_rgb(color) for color in (
        "black", "#0072B2", "#D55E00", "#228833", "#AA3377", "#00A6A6", "#B59A00",
    )]
    if len(phases) > len(colors):
        candidates = []
        for name in ("tab20", "tab20b", "tab20c", "Dark2", "Set1"):
            for color in plt.get_cmap(name).colors:
                color = tuple(color[:3])
                luminance = np.dot(color, [0.2126, 0.7152, 0.0722])
                if 0.18 <= luminance <= 0.75 and color not in candidates:
                    candidates.append(color)
        while len(colors) < len(phases):
            distances = np.linalg.norm(
                np.asarray(candidates)[:, None] - np.asarray(colors)[None, :], axis=2,
            )
            selected = int(np.argmax(np.min(distances, axis=1)))
            colors.append(candidates.pop(selected))
    return {phase["id"]: colors[index] for index, phase in enumerate(phases)}


def plot_multiphase_graph_overlay(
        graph, rgb_image, *, show=False, show_ids=False, marker_size=3.0,
        line_width=1.0, add_title=True, phase_colors=None):
    """
    Plot every retained phase and dashed contacts across phase boundaries.

    Parameters
    ----------
    graph : networkx.Graph
        Combined graph with global node IDs, pixels, phase IDs and phase metadata.
    rgb_image : numpy.ndarray
        Normalized RGB background on the common analysis grid.
    show : bool, optional
        Display the figure; otherwise return it without opening a window.
    show_ids : bool, optional
        Annotate nodes with their original phase and grain IDs.
    marker_size, line_width : float, optional
        Node marker diameter and edge width in points.
    add_title : bool, optional
        Add grain and contact counts above the image.
    phase_colors : dict, optional
        Color overrides keyed by original phase ID.

    Returns
    -------
    tuple
        Matplotlib figure and axes. The caller owns saving and closing them.
    """
    phases = sorted(graph.graph["phases"], key=lambda phase: (-phase["vf"], phase["id"]))
    colors = phase_graph_colors(phases)
    if phase_colors is not None:
        colors.update(phase_colors)
    rows, columns = rgb_image.shape[:2]
    centers = {}
    for node_id, node in graph.nodes(data=True):
        pixels = np.asarray(node.get("pixels", []), dtype=int)
        if pixels.size:
            pixel_rows, pixel_columns = np.unravel_index(pixels, (rows, columns))
            centers[node_id] = (pixel_columns.mean(), pixel_rows.mean())
        elif node.get("center") is not None:
            # import lazily so texture can use this plotter without a cycle
            from .texture import _graph_points_to_image
            centers[node_id] = _graph_points_to_image(graph, node["center"])

    fig, ax = plt.subplots(figsize=(8.0 * columns / rows, 8.0))
    ax.imshow(rgb_image, origin="upper")
    halo = [path_effects.Stroke(linewidth=line_width + 0.7, foreground="white"),
            path_effects.Normal()]
    legend = []
    for phase in phases:
        phase_id = phase["id"]
        color = colors[phase_id]
        for first, second in graph.edges:
            if first not in centers or second not in centers:
                continue
            if graph.nodes[first]["phase_id"] != phase_id or graph.nodes[second]["phase_id"] != phase_id:
                continue
            points = np.asarray([centers[first], centers[second]])
            line, = ax.plot(points[:, 0], points[:, 1], color=color, linewidth=line_width, zorder=3)
            line.set_path_effects(halo)
        for node_id, (x, y) in centers.items():
            node = graph.nodes[node_id]
            if node["phase_id"] != phase_id:
                continue
            ax.plot(x, y, "o", color=color, markersize=marker_size,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=5)
            if show_ids:
                source_id = node.get("source_grain_id", node_id)
                ax.text(x, y, f"{phase_id}:{source_id}", color="white", fontsize=8,
                        ha="center", va="center", zorder=6)
        legend.append(Line2D([0], [0], color=color, marker="o", markersize=4,
                             label=f"Phase #{phase_id} ({phase['name']}): {phase['vf']:.1%}"))

    has_cross_phase = False
    for first, second in graph.edges:
        if first not in centers or second not in centers:
            continue
        if graph.nodes[first]["phase_id"] == graph.nodes[second]["phase_id"]:
            continue
        points = np.asarray([centers[first], centers[second]])
        line, = ax.plot(points[:, 0], points[:, 1], color="#555555", linestyle="--",
                        linewidth=line_width, zorder=2)
        line.set_path_effects(halo)
        has_cross_phase = True
    if has_cross_phase:
        legend.append(Line2D([0], [0], color="#555555", linestyle="--",
                             label="Cross phase boundary"))
    if add_title:
        ax.set_title(f"Final graph: {len(graph)} grains, {graph.number_of_edges()} contacts")
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    if legend:
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=min(3, max(1, int(fig.get_figwidth() / 3.5))), fontsize=11, framealpha=1.0)
    fig.tight_layout(pad=0.3)
    if show:
        plt.show()
    return fig, ax
