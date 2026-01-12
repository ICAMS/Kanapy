"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import logging
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from kanapy.core import get_grain_geom
from scipy.stats import lognorm, vonmises
from scipy.spatial import ConvexHull, KDTree
from scipy.special import legendre, beta
from scipy.optimize import fminbound
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import mark_boundaries
from orix import io
from orix import plot as ox_plot
from orix.quaternion import Orientation, symmetry
from orix.quaternion.symmetry import Symmetry
from orix.sampling import get_sample_fundamental
from orix.vector import Miller
from abc import ABC
from collections import Counter, defaultdict
import os, glob



def get_distinct_colormap(N, cmap='prism'):
    """
    Generate N visually distinct colors as an RGB colormap.

    Parameters:
    - N: int, number of colors
    - seed: optional int, random seed for reproducibility

    Returns:
    - cmap: list of N RGB tuples in [0, 1]
    """
    colors = plt.get_cmap(cmap, N)
    col_ = [colors(i)[:3] for i in range(N)]
    return col_


def neighbors(r, c, connectivity=8):
    """
    Return the neighboring coordinates of a cell in a 2D grid.

    Parameters
    ----------
    r : int
        Row index of the cell.
    c : int
        Column index of the cell.
    connectivity : int, optional
        Type of connectivity. Options are:
        - 1 or 4 : 4-connected neighbors (up, down, left, right)
        - 8      : 8-connected neighbors (includes diagonals)
        Default is 8.

    Returns
    -------
    neighbors : list of tuple
        List of (row, column) tuples representing neighboring cells.

    Examples
    --------
    >>> neighbors(2, 3, connectivity=4)
    [(3, 3), (1, 3), (2, 4), (2, 2)]

    >>> neighbors(2, 3)
    [(3, 2), (3, 3), (3, 4), (2, 2), (2, 4), (1, 2), (1, 3), (1, 4)]
    """
    if connectivity == 1 or connectivity == 4:
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    else:
        return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                if not (i == 0 and j == 0)]


def merge_nodes(G, node1, node2):
    """
    Merge the pixel and attribute data of node1 into node2 in a graph

    This function performs the following steps:
    1. Concatenates the pixel lists of node1 and node2 and updates node2's 'pixels' attribute
    2. Updates the average orientation 'ori_av' and pixel count 'npix' for node2
    3. Updates the convex hull for node2 if it exists
    4. Adds edges from node1's neighbors to node2, avoiding duplicates
    5. Removes node1 from the graph
    6. Updates the graph's 'label_map' to replace all occurrences of node1 with node2

    Parameters
    ----------
    G : networkx.Graph
        Graph object where nodes represent grains and have attributes:
        - 'pixels' : array of pixel indices
        - 'ori_av' : average orientation
        - 'npix' : number of pixels
        - optional 'hull' : ConvexHull object
        The graph also contains:
        - 'label_map' : 2D array of node labels
        - 'dx', 'dy' : pixel spacing in x and y directions
    node1 : int
        Node ID to be merged and removed
    node2 : int
        Node ID to merge into and retain

    Notes
    -----
    - Modifies the graph `G` in place
    - Updates 'pixels', 'ori_av', 'npix', 'hull' for node2
    - Updates 'label_map' to replace node1 with node2
    - Adds new edges from node1 to node2 while avoiding duplicates
    """
    # merge pixel lists of node 1 into node 2, delete node 1
    G.nodes[node2]['pixels'] = np.concatenate((G.nodes[node1]['pixels'], G.nodes[node2]['pixels']))
    ntot = len(G.nodes)
    G.nodes[node2]['ori_av'] = (G.nodes[node1]['ori_av'] * G.nodes[node1]['npix'] +
                                G.nodes[node2]['ori_av'] * G.nodes[node2]['npix']) / ntot
    G.nodes[node2]['npix'] = ntot  # update length
    if 'hull' in G.nodes[node2].keys():
        # update hull if it exists already
        sh = G.graph['label_map'].shape
        pts = np.array(np.unravel_index(G.nodes[node2]['pixels'], sh), dtype=float).T
        pts[:, 0] *= G.graph['dx']  # convert pixel distances to micron
        pts[:, 1] *= G.graph['dy']
        G.nodes[node2]['hull'] = ConvexHull(pts)
    # add new edges (will ignore if edge already exists)
    for neigh in G.adj[node1]:
        if node2 != neigh:
            G.add_edge(node2, neigh)
    G.remove_node(node1)  # remove grain1 and all its edges
    # update label map
    ix, iy = np.nonzero(G.graph['label_map'] == node1)
    G.graph['label_map'][ix, iy] = node2


def find_largest_neighbor(G, node):
    """
    Find the largest neighboring node of a given node in a graph

    This function iterates over all neighbors of the specified node and returns
    the neighbor with the largest 'npix' (number of pixels).

    Parameters
    ----------
    G : networkx.Graph
        Graph object where nodes have an attribute 'npix' representing their size.
    node : int
        Node ID for which the largest neighbor is to be found.

    Returns
    -------
    int
        Node ID of the largest neighbor.

    Notes
    -----
    - Raises a ValueError if the node has no neighbors.
    - Raises a ValueError if the graph contains circular edges (neighbor is the node itself).
    - Does not modify the graph.
    """
    # find the largest neighbor of given grain
    size_ln = 0  # size of largest neighbor grain
    num_ln = -1  # ID of largest neighbor grain
    for neigh in G.adj[node]:
        if G.nodes[neigh]['npix'] > size_ln:
            size_ln = G.nodes[neigh]['npix']
            num_ln = neigh  # number of largest neighbor grain
    if num_ln < 0:
        raise ValueError(f'Grain {node} has no neighbors.')
    if num_ln == node:
        raise ValueError(f'Corrupted graph with circular edges: {node, num_ln}.')
    return num_ln


def find_sim_neighbor(G, nn):
    """
    Find the neighboring node most similar in orientation to the given node

    This function computes the misorientation angles between the given node and
    all its neighbors, and returns the neighbor with the smallest misorientation.

    Parameters
    ----------
    G : networkx.Graph
        Graph object where nodes have an 'ori_av' attribute representing average orientation.
        The graph also contains a 'symmetry' key in G.graph used for orientation calculations.
    nn : int
        Node ID for which the most similar neighbor is to be found.

    Returns
    -------
    tuple
        A tuple (neighbor_id, angle) where:
        - neighbor_id : int, ID of the neighbor with smallest misorientation
        - angle : float, misorientation angle with the given node

    Notes
    -----
    - Uses the `Orientation` class for misorientation calculations
    - Assumes all neighbors have a valid 'ori_av' attribute
    - Does not modify the graph
    """
    sym = G.graph['symmetry']
    ori0 = Orientation(G.nodes[nn]['ori_av'], sym)
    on = []
    ng = []
    for neigh in G.adj[nn]:
        ang = ori0.angle_with(Orientation(G.nodes[neigh]['ori_av'], sym))[0]
        on.append(ang)
        ng.append(neigh)
    nn = np.argmin(on)
    return ng[nn], on[nn]


def summarize_labels(label_array, rotations, wanted_labels=None):
    """
    Summarize labeled pixels with average orientation and statistics

    This function computes, for each label in `label_array` (or a subset of `wanted_labels`):
    - Number of pixels
    - Pixel indices
    - Average orientation vector
    - Standard deviation of orientation

    Parameters
    ----------
    label_array : ndarray of shape (H, W)
        2D array of integer labels.
    rotations : ndarray of shape (H*W, D)
        Orientation data corresponding to each pixel (e.g., emap.rotations.data).
    wanted_labels : sequence of int, optional
        List of label IDs to summarize. If None, all labels in `label_array` are used.

    Returns
    -------
    list of tuples
        Each tuple is `(label, info_dict)` where `info_dict` contains:
        - 'npix' : int, number of pixels
        - 'pixels' : array of 1D indices in `label_array.ravel()`
        - 'ori_av' : ndarray of shape (D,), average orientation vector
        - 'ori_std' : ndarray of shape (D,), standard deviation of orientation

    Notes
    -----
    - Uses `np.add.reduceat` and sorting for efficient computation
    - The returned list preserves the order of `wanted_labels` if provided,
      otherwise follows sorted unique labels from `label_array`
    - Does not modify input arrays
    """
    lab = label_array.ravel()
    N = lab.size
    rot = rotations  # shape (N, D)

    # Gruppierung: alle Pixel nach Label sortieren
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]

    # Eindeutige Labels + Segmentstarts + Counts
    uniq, starts, counts = np.unique(lab_sorted, return_index=True, return_counts=True)
    ends = starts + counts

    # Rotations in derselben Reihenfolge sortieren
    rot_sorted = rot[order]

    # Mittelwerte und Standardabweichungen pro Segment (reduceat ist sehr schnell)
    sums = np.add.reduceat(rot_sorted, starts, axis=0)
    means = sums / counts[:, None]

    sq = rot_sorted * rot_sorted
    sums_sq = np.add.reduceat(sq, starts, axis=0)
    vars_ = sums_sq / counts[:, None] - means**2
    stds = np.sqrt(np.maximum(vars_, 0.0))

    # Pixelindizes pro Label (als Listen von 1D-Indizes)
    pixels_list = np.split(order, starts[1:])  # Liste der Segmente in uniq-Reihenfolge

    # Auswahl / Re-Ordering auf gewünschte Labels
    if wanted_labels is None:
        labels_out = uniq
        idx = np.arange(len(uniq))
    else:
        labels_out = np.asarray(wanted_labels)
        pos = {int(l): i for i, l in enumerate(uniq)}
        idx = np.array([pos[int(l)] for l in labels_out], dtype=int)

    nodes = []
    for lbl, i in zip(labels_out, idx):
        info = {
            "npix": int(counts[i]),
            "pixels": pixels_list[i],              # 1D-Indices in label_array.ravel()
            "ori_av": means[i],                   # shape (D,)
            "ori_std": stds[i],                   # shape (D,)
        }
        nodes.append((int(lbl), info))
    return nodes


def build_graph_from_labeled_pixels(label_array, emap, phase, connectivity=8):
    """
    Build a graph representation of a microstructure from labeled pixels

    Each node in the graph represents a grain (label) and stores pixel indices,
    average orientation, and orientation statistics. Edges connect neighboring grains
    based on pixel adjacency.

    Parameters
    ----------
    label_array : ndarray of shape (H, W)
        2D array of integer labels representing grains.
    emap : object
        Orientation map object containing:
        - `rotations.data` : orientation vectors for each pixel
        - `phases.point_groups` : symmetry information for each phase
        - `dx`, `dy` : pixel spacing in x and y directions
    phase : int
        Phase index to select the corresponding symmetry from `emap.phases.point_groups`.
    connectivity : int, optional
        Connectivity criterion for neighbors (4 or 8). Default is 8.

    Returns
    -------
    networkx.Graph
        Graph object with:
        - Nodes representing grains, storing 'pixels', 'ori_av', 'ori_std', 'npix'
        - Edges representing neighboring grains (shared boundaries)
        - Graph attributes: 'label_map', 'symmetry', 'dx', 'dy'

    Notes
    -----
    - Uses `summarize_labels` to extract node properties from labeled pixels
    - Loops over all pixels to add edges between neighboring grains
    - The function preserves the shape and indices of `label_array`
    """
    # t1 = time.time()
    nodes = summarize_labels(label_array, emap.rotations.data)
    # t2 = time.time()
    # print(f'Time for extracting nodes: {t2 - t1}')
    # print(f'Building microstructure graph with {len(nodes)} nodes (grains).')
    G = nx.Graph(label_map=label_array, symmetry=emap.phases.point_groups[phase],
                 dx=emap.dx, dy=emap.dy)
    G.add_nodes_from(nodes)
    # t3 = time.time()
    # print(f'Time for building graph: {t3 - t2}')

    # print('Adding edges (grain boundaries) to microstructure graph.')
    rows, cols = label_array.shape
    for x in range(rows):
        for y in range(cols):
            label_here = label_array[x, y]
            for px, py in neighbors(x, y, connectivity=connectivity):
                if 0 <= px < rows and 0 <= py < cols:
                    neighbor_label = label_array[px, py]
                    if neighbor_label != label_here:
                        G.add_edge(label_here, neighbor_label)
    # t4 = time.time()
    # print(f'Time for adding edges: {t4 - t3}')
    return G


def visualize_graph(G, node_size=100, fs=12):
    """
    Visualize a microstructure graph using a spring layout

    This function draws the nodes and edges of the graph with labels,
    using a spring layout for positioning. Node color, edge color,
    node size, and font size can be adjusted.

    Parameters
    ----------
    G : networkx.Graph
        Graph object representing the microstructure. Nodes should have meaningful labels.
    node_size : int, optional
        Size of the nodes in the plot. Default is 100.
    fs : int, optional
        Font size for node labels. Default is 12.

    Notes
    -----
    - Uses NetworkX's `spring_layout` for node positioning
    - Uses Matplotlib to render the plot
    - Does not modify the graph
    """
    pos = nx.spring_layout(G, seed=42)  # positioning
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=node_size, font_size=fs)
    plt.show()


def export_graph(G, filename, format="graphml"):
    """
    Export a microstructure graph to a file in the specified format

    This function writes the NetworkX graph to disk in either GraphML or GEXF format.

    Parameters
    ----------
    G : networkx.Graph
        Graph object representing the microstructure.
    filename : str
        Path to the output file.
    format : str, optional
        File format for export. Supported options are 'graphml' and 'gexf'. Default is 'graphml'.

    Notes
    -----
    - Raises a ValueError if an unsupported format is provided
    - Does not modify the graph
    """
    if format == "graphml":
        nx.write_graphml(G, filename)
    elif format == "gexf":
        nx.write_gexf(G, filename)
    else:
        raise ValueError("Only 'graphml' or 'gexf' formats are supported.")


def find_similar_regions(array, tolerance=0.087, connectivity=1):
    """
    Identify connected regions of similar values in a 2D array

    This function labels all connected pixels in the input array whose values
    differ by at most `tolerance`. Connectivity can be 4- or 8-connected.

    Parameters
    ----------
    array : ndarray of shape (H, W)
        2D input array with values to segment.
    tolerance : float, optional
        Maximum allowed difference between connected values. Default is 0.087.
    connectivity : int, optional
        Connectivity criterion for neighbors: 1 for 4-connectivity, 2 for 8-connectivity. Default is 1.

    Returns
    -------
    labeled_array : ndarray of shape (H, W)
        Array where each connected region of similar values has a unique label.
    num_features : int
        Total number of connected regions found.

    Notes
    -----
    - Uses a depth-first search to find connected regions.
    - Does not modify the input array.
    - Assumes `neighbors(i, j, connectivity)` returns valid neighboring indices.
    """

    array = np.asarray(array)
    visited = np.full(array.shape, False, dtype=bool)
    labeled_array = np.zeros(array.shape, dtype=int)
    current_label = 1
    rows, cols = array.shape
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                ref_val = array[r, c]
                stack = [(r, c)]
                while stack:
                    i, j = stack.pop()
                    if (0 <= i < rows) and (0 <= j < cols) and not visited[i, j]:
                        if abs(array[i, j] - ref_val) <= tolerance:
                            visited[i, j] = True
                            labeled_array[i, j] = current_label
                            stack.extend(neighbors(i, j, connectivity))
                current_label += 1
    return labeled_array, current_label - 1


def calc_error(odf_ref, odf_test, res=10.):
    """
    Compute the normalized difference between two orientation distribution functions (ODFs)

    This function evaluates the ODFs on a sample grid in fundamental space and
    computes the L1 norm of the difference after normalization.

    Parameters
    ----------
    odf_ref : object
        Reference ODF object. Must have `.orientations.symmetry` and `.evaluate()` method.
    odf_test : object
        Test ODF object to compare. Must have the same symmetry as `odf_ref`.
    res : float, optional
        Resolution for sampling the fundamental space grid. Default is 10.

    Returns
    -------
    float
        L1 error between the normalized ODFs.

    Notes
    -----
    - Raises a RuntimeError if the symmetries of the two ODFs do not match.
    - Uses `get_sample_fundamental` to sample the fundamental zone and `Orientation` for evaluation.
    - Does not modify the input ODFs.
    """
    cs = odf_ref.orientations.symmetry
    if cs != odf_test.orientations.symmetry:
        raise RuntimeError("Symmetries of ODF's do not match.")
    so3g = get_sample_fundamental(resolution=res, point_group=cs)
    so3g = Orientation(so3g.data, symmetry=cs)
    p1 = odf_ref.evaluate(so3g)
    p2 = odf_test.evaluate(so3g)
    err = np.sum(np.abs(p1/np.sum(p1) - p2/np.sum(p2)))
    return err


def calc_orientations(odf, nori, res=None):
    """
    Generate a set of orientations sampled from an orientation distribution function (ODF)

    This function uses Monte Carlo sampling to generate `nori` orientations according to
    the probabilities defined by the input ODF. Sampling is performed in the fundamental
    zone of the crystal symmetry.

    Parameters
    ----------
    odf : object
        Orientation distribution function object. Must have `.orientations.symmetry` and `.evaluate()` method.
    nori : int
        Number of orientations to generate.
    res : float, optional
        Angular resolution in degrees for sampling the fundamental zone grid.
        If None, a resolution between 2° and 5° is selected based on ODF half-width.

    Returns
    -------
    Orientation
        Orientation object containing `nori` orientations sampled from the ODF.

    Notes
    -----
    - Raises a RuntimeError if the Monte Carlo sampling does not converge after 200 iterations.
    - Logs a warning if the SO3 grid resolution is too coarse for the requested number of orientations.
    - Uses `get_sample_fundamental` to generate the sampling grid and `Orientation` to store results.
    - Does not modify the input ODF.
    """
    oq = np.zeros((nori, 4))
    indstart = 0
    rem = nori
    cs = odf.orientations.symmetry
    hw = odf.halfwidth
    if res is None:
        res = np.clip(np.rad2deg(hw), 2, 5)
    so3g = get_sample_fundamental(resolution=res, point_group=cs)  # generate fine mesh in fund. region; res in degree!
    so3g = Orientation(so3g.data, symmetry=cs)
    if so3g.size < 5*nori:
        logging.warning(f'Resolution of SO3 grid in "calc_orientations" is too coarse: {res}°.\n'
                        f'Only {so3g.size} grid points available for {nori} desired orientations.')
    val = odf.evaluate(so3g)  # value of ODF at each gridpoint
    # do MC sampling of ODF to generate nori orientations
    ctr = 0
    while rem > 0 and ctr < 200:
        rn = np.random.rand(so3g.size)
        vn = val * np.linalg.norm(rn) * nori / np.linalg.norm(val)
        ori = so3g[vn >= rn]
        no = min(ori.size, rem)
        oq[indstart:indstart+no, :] = ori.data[0:no, :]
        indstart += no
        rem -= no
        ctr += 1
    if ctr >= 200:
        raise RuntimeError(f'Monte Carlo algorithm in "calc_orientations" did not converge for resolution {res}.\n'
                           f'Orientations found {indstart} orientations. Target was {nori}.')
    return Orientation(oq, symmetry=cs)


def odf_est(ori, odf, nstep=50, step=0.5, halfwidth=None, verbose=False):
    """
    Estimate an ODF from a set of orientations using iterative half-width adjustment

    This function generates a new orientation distribution function (ODF) from
    input orientations by iteratively adjusting the half-width to minimize the
    error relative to a reference ODF.

    Parameters
    ----------
    ori : Orientation
        Input orientations used to estimate the ODF.
    odf : ODF
        Reference orientation distribution function for error comparison.
    nstep : int, optional
        Maximum number of iterations for adjusting the half-width. Default is 50.
    step : float, optional
        Increment step in degrees for half-width adjustment. Default is 0.5.
    halfwidth : float, optional
        Initial half-width in radians. If None, a value based on `odf.halfwidth` is used.
    verbose : bool, optional
        If True, prints iteration details. Default is False.

    Returns
    -------
    ODF
        Estimated orientation distribution function corresponding to input orientations.

    Notes
    -----
    - Uses `calc_error` to compare the estimated ODF with the reference ODF.
    - Iteratively increases half-width until the error starts to increase.
    - The returned ODF corresponds to the last half-width before error increased.
    - Does not modify the input orientations or reference ODF.
    """
    e0 = 1e8
    st_rad = np.radians(step)
    hwmin = np.radians(2)
    hwmax = np.radians(30) - 0.5 * nstep * st_rad
    if halfwidth is None:
        hw = np.clip(odf.halfwidth - 0.5 * nstep * st_rad, hwmin, hwmax)
        if verbose:
            print(f'Initial halfwidth set to {np.degrees(hw)}')
    else:
        hw = np.clip(halfwidth - 2 * st_rad, hwmin, hwmax)
        if verbose:
            print(f'Initial halfwidth is {np.degrees(hw)}')
    for c in range(nstep):
        todf = ODF(ori, halfwidth=hw)
        e = calc_error(odf, todf)
        if verbose:
            print(f'Iteration {c}: error={e}, halfwidth={np.degrees(hw)}°')
        if e > e0:
            break
        else:
            e0 = e
        hw += st_rad
    # return last value before error increased
    hw -= st_rad
    todf = ODF(ori, halfwidth=hw)
    return todf


def plot_pole_figure(orientations, phase, vector=None,
                     size=None, alpha=None):
    """
    Plot an inverse pole figure and pole density function for given orientations

    This function visualizes the orientation distribution of a crystal phase by
    plotting the stereographic projection of specified poles and the corresponding
    pole density function (PDF).

    Parameters
    ----------
    orientations : Orientation
        Orientation object containing crystal orientations to plot.
    phase : object
        Crystal phase object containing symmetry information and name.
    vector : array-like of shape (3,), optional
        Miller indices of the pole vector to plot. Default is [0, 0, 1].
    size : float, optional
        Marker size for scatter plot. Automatically scaled if None.
    alpha : float, optional
        Marker transparency for scatter plot. Automatically scaled if None.

    Notes
    -----
    - Uses Matplotlib for plotting with stereographic projection.
    - The first subplot shows the inverse pole figure (scatter of poles).
    - The second subplot shows the pole density function.
    - Marker size and transparency are scaled inversely with the number of orientations.
    - Does not modify the input `orientations` or `phase`.
    """
    # plot inverse pole figure
    # <111> poles in the sample reference frame
    if vector is None:
        vector = [0, 0, 1]
    assert isinstance(orientations, Orientation)
    if orientations.size < 1:
        logging.warning(f'No orientations provided: {orientations.size}.')
        scf = 1.0
    else:
        scf = 1.0 / np.sqrt(orientations.size)
    if size is None:
        size = np.clip(250*scf, 0.25, 25)
    if alpha is None:
        alpha = np.clip(4*scf, 0.05, 0.5)
    t_ = Miller(uvw=vector, phase=phase).symmetrise(unique=True)
    t_all = orientations.inv().outer(t_)
    fig = plt.figure(figsize=(9, 12))
    ax1 = fig.add_subplot(211, projection="stereographic")
    ax1.scatter(t_all, s=size, alpha=alpha)
    ax1.set_labels("X", "Y", None)
    ax1.set_title(phase.name + r" $\left<" +
                  f"{vector[0]}{vector[1]}{vector[2]}" +
                  r"\right>$ PF")
    ax2 = fig.add_subplot(212, projection="stereographic")
    ax2.pole_density_function(t_all)
    ax2.set_labels("X", "Y", None)
    ax2.set_title(phase.name + r" $\left<" +
                  f"{vector[0]}{vector[1]}{vector[2]}" +
                  r"\right>$ PDF")
    plt.show()


def plot_pole_figure_proj(orientations, phase, vector=None,
                          title=None, alpha=None, size=None):
    """
    Plot an inverse pole figure (IPF) projection and pole density function for EBSD data

    This function visualizes crystal orientations along a specified sample direction
    using an IPF projection. The first subplot shows the scatter of rotated poles,
    and the second subplot shows the corresponding pole density function (PDF).

    Parameters
    ----------
    orientations : Orientation
        Orientation object containing crystal orientations to plot.
    phase : object
        Crystal phase object with symmetry information and name.
    vector : array-like of shape (3,), optional
        Sample direction to project (Miller indices). Default is [0, 0, 1].
    title : str, optional
        Title for the plot. If None, automatically uses the principal axis of `vector`.
    alpha : float, optional
        Marker transparency for scatter plot. Automatically scaled if None.
    size : float, optional
        Marker size for scatter plot. Automatically scaled if None.

    Notes
    -----
    - Uses Matplotlib with stereographic projection (IPF) for plotting.
    - Marker size and transparency are scaled inversely with the number of orientations.
    - The first subplot shows the inverse pole figure (scatter of rotated poles).
    - The second subplot shows the pole density function (PDF).
    - Does not modify the input `orientations` or `phase`.
    """
    # Some sample direction, v
    if vector is None:
        vector = [0, 0, 1]
    v = Miller(vector)
    if title is None:
        v_title = ["X", "Y", "Z"][np.argmax(vector)]
    else:
        v_title = title
    if orientations.size < 1:
        logging.warning(f'No orientations provided: {orientations.size}.')
        scf = 1.0
    else:
        scf = 1.0 / np.sqrt(orientations.size)
    if size is None:
        size = np.clip(250*scf, 0.25, 25)
    if alpha is None:
        alpha = np.clip(4*scf, 0.05, 0.5)
    assert isinstance(orientations, Orientation)
    # Rotate sample direction v into every crystal orientation O
    t_ = orientations * v

    # Set IPDF range
    vmin, vmax = (0, 3)

    subplot_kw = {"projection": "ipf", "symmetry": phase.point_group.laue, "direction": v}
    fig = plt.figure(figsize=(9, 8))

    ax0 = fig.add_subplot(211, **subplot_kw)
    ax0.scatter(t_, s=size, alpha=alpha)
    _ = ax0.set_title(f"EBSD data, {phase.name}, {v_title}")

    ax2 = fig.add_subplot(212, **subplot_kw)
    ax2.pole_density_function(t_, vmin=vmin, vmax=vmax)
    _ = ax2.set_title(f"EBSD data, {phase.name}, {v_title}")

    plt.show()

def find_orientations_fast(ori1: Orientation, ori2: Orientation, tol: float = 1e-3) -> np.ndarray:
    """
    Find closest matches in ori1 for each orientation in ori2 using KDTree

    Parameters
    ----------
    ori1 : Orientation
        Orientation database (e.g., EBSD orientations).
    ori2 : Orientation
        Orientations to match (e.g., grain mean orientations).
    tol : float
        Angular tolerance in radians.

    Returns
    -------
    matches : np.ndarray
        Array of indices in ori1 matching each entry in ori2; -1 if no match found.
    """
    # Get quaternions
    q1 = ori1.data
    q2 = ori2.data

    tree = KDTree(q2)  # KDTree in 4D quaternion space for ori2 (typically, regular SO3 grid)
    dists, indices = tree.query(q1)  # For every orientation in ori1, look for nearest neighbor in ori2 (grid)

    # create list of length or ori2, and store counts and indices of nearest neighbors in ori1
    matches = np.zeros(ori2.size, dtype=int)
    neigh_dict = dict()
    for i, idx in enumerate(indices):
        matches[idx] += 1
        if idx in neigh_dict.keys():
            neigh_dict[idx].append(i)
        else:
            neigh_dict[idx] = [i]

    return matches, neigh_dict


def texture_reconstruction(ns, ebsd=None, ebsdfile=None, orientations=None,
                          grainsfile=None, grains=None, kernel=None, kernel_halfwidth=5,
                          res_low=5, res_high=25, res_step=2, lim=5, verbose=False):
    """
    Reconstruct a reduced ODF from EBSD or orientation data

    This function systematically reconstructs an orientation distribution function (ODF)
    using a given number of orientations or grains in a representative volume element (RVE).
    The reconstructed ODF approximates the misorientation distribution of the input data.
    Follows the method described in Biswas et al. (https://doi.org/10.1107/S1600576719017138).

    Parameters
    ----------
    ns : int
        Number of reduced orientations/grains in the RVE.
    ebsd : EBSDmap, optional
        EBSD map containing orientations for a single phase.
    ebsdfile : str, optional
        Path to a *.mat file with EBSD data (not yet supported).
    orientations : Orientation, optional
        Predefined orientations to use instead of EBSD data.
    grainsfile : str, optional
        Path to estimated grains file (*.mat). Not used for kernel estimation.
    grains : object, optional
        Grain data. Not used for kernel estimation.
    kernel : DeLaValleePoussinKernel, optional
        Kernel for ODF estimation. Default: halfwidth=kernel_halfwidth.
    kernel_halfwidth : float, optional
        Halfwidth in degrees for default DeLaValleePoussin kernel. Default is 5°.
    res_low : float, optional
        Minimum resolution (in degrees) of orientation grid for reconstruction. Default 5°.
    res_high : float, optional
        Maximum resolution (in degrees) of orientation grid. Default 25°.
    res_step : float, optional
        Step size (in degrees) for grid resolution. Default 2°.
    lim : int, optional
        Maximum number of consecutive worsening errors before stopping. Default 5.
    verbose : bool, optional
        If True, print progress and error information. Default False.

    Returns
    -------
    orired_f : Orientation
        Reduced set of orientations in the fundamental region.
    odfred_f : ODF
        Reconstructed ODF using the reduced orientations.
    ero : float
        L1 error between reference ODF and reconstructed ODF.
    res : float
        Grid resolution (degrees) at which the minimum error was achieved.

    Notes
    -----
    - Uses Monte Carlo sampling and kernel smoothing to generate reduced orientations.
    - Only supports single-phase EBSD data.
    - Kernel estimation from grains or files is not yet supported; default kernel is used.
    - The reconstructed ODF approximates the misorientation distribution of the input data.
    """
    ori = None
    psi = None
    if ebsdfile is not None:
        raise ModuleNotFoundError('Option "ebsdMatFile" is not yet supported.')
        # ind = args.index('ebsdMatFile') + 1
        # ebsd = loadmat(args[ind])
        # ebsd_var = list(ebsd.keys())[0]
        # ebsd = ebsd[ebsd_var]
        # assert len(np.unique(ebsd.phaseId)) == 1, 'EBSD has multiple phases'
        # ori = ebsd.orientations
    if ebsd is not None:
        if not isinstance(ebsd, EBSDmap):
            raise TypeError('Argument "ebsd" must be of type EBSDmap.')
        assert len(ebsd.emap.phases.ids) == 1, 'EBSD has multiple phases'
        if ori is not None:
            logging.warning('Both arguments "ebsd" and "ori" are given, using EBSD map orientations.')
        else:
            ori = ebsd.emap.orientations
    if orientations is not None:
        if not isinstance(orientations, Orientation):
            raise TypeError('Argument "orientations" must be of type Orientation,')
        if ori is not None:
            logging.warning('Both EBSD map and orientations are given, using EBSD.')
        else:
            ori = orientations

    if grainsfile is not None:
        logging.warning('Estimation of optimal kernel from grain files not supported. '
                        'DeLaValleePoussinKernel with halfwidth 5° will be used.\n'
                        'Please use kanapy-mtex for support of optimal kernels.')
        # ind = args.index('grainsMatFile') + 1
        # grains = loadmat(args[ind])
        # grains_var = list(grains.keys())[0]
        # grains = grains[grains_var]
        # assert len(np.unique(grains.phaseId)) == 1, 'Grains has multiple phases'
        # print('Optimum kernel estimated from mean orientations of grains')
        # psi = calcKernel(grains.meanOrientation)
    if grains is not None:
        logging.warning('Estimation of optimal kernel from grains not supported. '
                        'DeLaValleePoussinKernel with halfwidth 5° will be used.\n'
                        'Please use kanapy-mtex for support of optimal kernels.')
        #assert len(np.unique(grains.phaseId)) == 1, 'Grains has multiple phases'
        #print('Optimum kernel estimated from mean orientations of grains')
        #psi = calcKernel(grains.meanOrientation)
    if kernel is not None:
        if not isinstance(kernel, DeLaValleePoussinKernel):
            raise TypeError('Only kernels of type "DeLaValeePoussinKernel" are supported.')
        psi = kernel
    if psi is None:
        psi = DeLaValleePoussinKernel(halfwidth=np.radians(kernel_halfwidth))

    # Step 1: Create reference odf from given orientations with proper kernel
    odf = ODF(ori, kernel=psi)
    cs = ori.symmetry

    ero = 10.
    e_mod = []
    hw_stored = None
    for hw in np.arange(res_low, res_high + res_step, res_step):
        # Step 2: create equispaced grid of orientations
        S3G = get_sample_fundamental(resolution=hw, point_group=cs)  # resolution in degrees! ori.SS not considered
        S3G = Orientation(S3G.data, symmetry=cs)
        # Step 3: calculate number of orientations close to each grid point (0 if no orientation close to GP)
        # count close orientations from EBSD map for each grid point, and get list of neighbor indices
        M, neighs = find_orientations_fast(ori, S3G, tol=np.radians(0.5))
        ictr = np.nonzero(M > 0)[0]  # indices of gridpoints with non-zero counts
        weights = M[ictr]  # create weights from non-zero counts of EBSD orientations at gridpoints
        weights = weights / np.sum(weights)  # calculate weights
        # Step 4: Calculate scaling factor hval such that sum of all int(weights*hval) = ns
        lval = 0
        hval = float(ns)
        ifc = 1.0
        ihval = np.sum(np.round(hval * weights))
        while (hval - lval > hval * 1e-15 or ihval < ns) and ihval != ns:
            if ihval < ns:
                hval_old = hval
                hval = hval + ifc * (hval - lval) / 2.0
                lval = hval_old
                ifc = ifc * 2.0
            else:
                hval = (lval + hval) / 2.0
                ifc = 1.0
            ihval = np.sum(np.round(hval * weights))
        screen = np.round(weights * hval)  # number of orientations associated to each grid point
        diff = np.sum(screen) - ns  # difference to desired number of orientations
        weights_loc = np.argsort(weights)
        co = 1
        while diff > 0:
            if screen[weights_loc[co]] > 0:
                screen[weights_loc[co]] = screen[weights_loc[co]] - 1
                diff = np.sum(screen) - ns
            co = co + 1

        # Step 5: Subdivide orientations around grid points into screen orientations
        #         and estimate mean orientation for each group or take orientation of grid point
        #         if only one orientation needs to be generated
        ori_red = np.zeros((ns, 4))
        octr = 0
        for i, no in enumerate(screen):
            nl = neighs[ictr[i]]
            if len(nl) < no:
                raise RuntimeError(f'{len(nl)} < {no} @ {i}, ictr: {ictr[i]}, weight: {weights[i]}')
            if 0.5 < no < 1.5:
                ori_red[octr, :] = S3G[ictr[i]].data
                octr += 1
                assert np.isclose(no, 1), f'no: {no}, should be 1'
            elif no >= 1.5:
                # split orientations in EBSD map matching to one grid point according to required number of
                # orientations at this point
                ind = np.linspace(0, len(nl), int(no)+1, dtype=int)
                idx = np.split(np.array(nl), ind[1:-1])
                ho = octr
                for j in idx:
                    ori_red[octr, :] = np.mean(ori[j].data, axis=0)
                    octr += 1
                if not np.isclose(no, octr-ho):
                    print(f'len_nl: {len(nl)}, len_idx: {len(idx)}')
                    raise RuntimeError(f'no: {no}, but increment is only #{octr-ho}')
        # create Orientations from quaternion array
        ori_f = Orientation(ori_red, symmetry=cs)
        assert not np.isnan(ori_f.data).any()
        assert not np.isinf(ori_f.data).any()

        # Step 6: Compute reduced ODF
        odfred = odf_est(ori_f, odf, halfwidth=hw_stored, verbose=verbose)
        hw_stored = odfred.halfwidth

        # Step 7: Compute error for kernel optimization
        er = calc_error(odf, odfred)
        if verbose:
            print(f'Resolution: {hw}, Error: {er}, Reduced HW: {np.degrees(odfred.halfwidth)}°')
        # store best results and evaluate stopping criterion
        if er < ero:
            orired_f = ori_f
            odfred_f = odfred
            ero = er
            res = hw
        e_mod.append(er)
        if len(e_mod) - np.argmin(e_mod) > lim:
            break
    orired_f = orired_f.in_euler_fundamental_region()
    return orired_f, odfred_f, ero, res


class Kernel(ABC):
    """
    Abstract base class for kernels used in orientation distribution estimation.

    Attributes
    ----------
    A : ndarray
        Flattened array of kernel parameters. Initialized to empty array if not provided.
    """
    def __init__(self, A=None):
        self.A = np.array(A).flatten() if A is not None else np.array([])

    @property
    def bandwidth(self):
        """
        Returns the bandwidth of the kernel.

        The bandwidth is defined as the number of elements in `A` minus one.

        Returns
        -------
        int
            Bandwidth of the kernel.
        """
        return len(self.A) - 1

    @bandwidth.setter
    def bandwidth(self, L):
        """
        Set the bandwidth of the kernel.

        Truncates the kernel parameter array `A` to have at most `L + 1` elements.

        Parameters
        ----------
        L : int
            Desired bandwidth.
        """
        self.A = self.A[:min(L + 1, len(self.A))]

    def __str__(self):
        """
        Return a human-readable string representation of the kernel.

        Displays the kernel type and its halfwidth in degrees.

        Returns
        -------
        str
            String describing the kernel.
        """
        return f"custom, halfwidth {np.degrees(self.halfwidth()):.2f}°"

    def __eq__(self, other):
        """
        Check if two kernels are equal.

        Comparison is based on the truncated arrays of kernel parameters `A` up to
        the minimum bandwidth of the two kernels. Returns True if the relative
        difference is less than 1e-6.

        Parameters
        ----------
        other : Kernel
            Another kernel to compare against.

        Returns
        -------
        bool
            True if kernels are considered equal, False otherwise.
        """
        L = min(self.bandwidth, other.bandwidth)
        return np.linalg.norm(self.A[:L + 1] - other.A[:L + 1]) / np.linalg.norm(self.A) < 1e-6

    def __mul__(self, other):
        """
        Multiply two kernels element-wise with scaling by (2l + 1).

        Only the first `L + 1` elements are used, where L is the minimum bandwidth
        of the two kernels. Returns a new Kernel instance with the resulting array.

        Parameters
        ----------
        other : Kernel
            Another kernel to multiply with.

        Returns
        -------
        Kernel
            New kernel resulting from element-wise multiplication and scaling.
        """
        L = min(self.bandwidth, other.bandwidth)
        l = np.arange(L + 1)
        return Kernel(self.A[:L + 1] * other.A[:L + 1] / (2 * l + 1))

    def __pow__(self, p):
        """
        Raise kernel elements to a given power with scaling.

        Each element of the kernel parameter array `A` is scaled by (2l + 1),
        raised to the power `p`, and then rescaled by (2l + 1).

        Parameters
        ----------
        p : float
            Power to raise the kernel elements to.

        Returns
        -------
        Kernel
            New kernel resulting from the power operation with scaling.
        """
        l = np.arange(self.bandwidth + 1)
        return Kernel(((self.A / (2 * l + 1)) ** p) * (2 * l + 1))

    def norm(self):
        """
        Compute the L2 norm of the squared kernel elements.

        Returns
        -------
        float
            L2 norm of the squared elements of the kernel array `A`.
        """
        return np.linalg.norm(self.A ** 2)

    def cutA(self, fft_accuracy=1e-2):
        """
        Truncate the kernel array `A` based on the specified FFT accuracy.

        Elements of `A` are scaled by 1/(l^2), and elements smaller than
        a threshold determined from `fft_accuracy` are removed.

        Parameters
        ----------
        fft_accuracy : float, optional
            Desired FFT accuracy (default is 1e-2).

        Notes
        -----
        The first element is never truncated.
        """
        epsilon = fft_accuracy / 150
        A_mod = self.A / (np.arange(1, len(self.A) + 1) ** 2)
        idx = np.where(A_mod[1:] <= max(min([np.min(A_mod[1:]), 10 * epsilon]), epsilon))[0]
        if idx.size > 0:
            self.A = self.A[:idx[0] + 2]

    def halfwidth(self):
        """
        Compute the halfwidth of the kernel.

        The halfwidth is determined by finding the angle `omega` that minimizes
        the squared difference between K(1) and 2*K(cos(omega/2)).

        Returns
        -------
        float
            Halfwidth angle in radians.
        """
        def error_fn(omega):
            """
            Compute the squared difference used to determine kernel halfwidth.

            Parameters
            ----------
            omega : float
                Angle in radians to evaluate the kernel function.

            Returns
            -------
            float
                Squared difference: (K(1) - 2*K(cos(omega/2)))**2
            """
            return (self.K(1) - 2 * self.K(np.cos(omega / 2))) ** 2

        return fminbound(error_fn, 0, 3 * np.pi / 4)

    def K(self, co2):
        """
        Evaluate the kernel function at a given squared cosine value.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine squared value(s), will be clipped to [-1, 1].

        Returns
        -------
        float or ndarray
            Value(s) of the kernel function.
        """
        co2 = np.clip(co2, -1, 1)
        omega = 2 * np.arccos(co2)
        return self._clenshawU(self.A, omega)

    def K_orientations(self, orientations_ref, orientations):
        """
        Evaluate the kernel function for the misorientation angles between two sets of orientations.

        Parameters
        ----------
        orientations_ref : Orientation
            Reference set of orientations.
        orientations : Orientation
            Target set of orientations to compare.

        Returns
        -------
        ndarray
            Kernel values corresponding to misorientation angles.
        """
        misangles = orientations.angle_with(orientations_ref)
        co2 = np.cos(misangles / 2)
        return self.K(co2)

    def RK(self, d):
        """
        Evaluate the kernel using the Clenshaw-L method for given cosine distances.

        Parameters
        ----------
        d : array_like
            Cosine of angles (distance metric) in the range [-1, 1].

        Returns
        -------
        ndarray
            Kernel values for the given distances.
        """
        d = np.clip(d, -1, 1)
        return self._clenshawL(self.A, d)

    def RRK(self, dh, dr):
        """
        Evaluate the kernel on two sets of cosine distances using a reduced rotational kernel.

        Parameters
        ----------
        dh : array_like
            Cosines of angles for the first set, clipped to [-1, 1].
        dr : array_like
            Cosines of angles for the second set, clipped to [-1, 1].

        Returns
        -------
        ndarray
            2D array of kernel values for each combination of dh and dr.
        """
        dh = np.clip(dh, -1, 1)
        dr = np.clip(dr, -1, 1)
        L = self.bandwidth
        result = np.zeros((len(dh), len(dr)))

        if len(dh) < len(dr):
            for i, dh_i in enumerate(dh):
                Plh = [legendre(l)(dh_i) for l in range(L + 1)]
                result[i, :] = self._clenshawL(np.array(Plh) * self.A, dr)
        else:
            for j, dr_j in enumerate(dr):
                Plr = [legendre(l)(dr_j) for l in range(L + 1)]
                result[:, j] = self._clenshawL(np.array(Plr) * self.A, dh)
        result[result < 0] = 0
        return result

    def _clenshawU(self, A, omega):
        """
        Evaluate the kernel using the Clenshaw algorithm in the Chebyshev U basis.

        Parameters
        ----------
        A : array_like
            Coefficients of the kernel.
        omega : array_like
            Angles in radians.

        Returns
        -------
        ndarray
            Kernel values evaluated at the given angles.
        """
        omega = omega / 2
        res = np.ones_like(omega) * A[0]
        for l in range(1, len(A)):
            term = np.cos(2 * l * omega) + np.cos(omega) * np.cos((2 * l - 1) * omega) + \
                   (np.cos(omega) ** 2)
            res += A[l] * term
        return res

    def _clenshawL(self, A, x):
        """
        Evaluate the kernel using the Clenshaw algorithm in the Legendre basis.

        Parameters
        ----------
        A : array_like
            Coefficients of the kernel.
        x : array_like
            Input values where the kernel is evaluated (should be within [-1, 1]).

        Returns
        -------
        ndarray or float
            Kernel values evaluated at the given input(s).
        """
        b_next, b_curr = 0.0, 0.0
        x2 = 2 * x
        for a in reversed(A[1:]):
            b_next, b_curr = b_curr, a + x2 * b_curr - b_next
        return A[0] + x * b_curr - b_next

    def calc_fourier(self, L, max_angle=np.pi, fft_accuracy=1e-2):
        """
        Compute the Fourier coefficients of the kernel up to order L.

        Parameters
        ----------
        L : int
            Maximum order of the Fourier coefficients.
        max_angle : float, optional
            Upper limit of integration in radians (default is pi).
        fft_accuracy : float, optional
            Threshold below which coefficients are considered negligible and
            computation stops early (default is 1e-2).

        Returns
        -------
        ndarray
            Array of Fourier coefficients of length <= L+1.
        """
        A = []
        small = 0
        for l in range(L + 1):
            def integrand(omega):
                return self.K(np.cos(omega / 2)) * np.sin((2 * l + 1) * omega / 2) * np.sin(omega / 2)

            coeff, _ = quad(integrand, 0, max_angle, limit=2000)
            coeff *= 2 / np.pi
            A.append(coeff)
            if abs(coeff) < fft_accuracy:
                small += 1
            else:
                small = 0
            if small == 10:
                break
        return np.array(A)

    def plot_K(self, n_points=200):
        """
        Plot the kernel function K as a function of misorientation angle.

        Parameters
        ----------
        n_points : int, optional
            Number of points used for plotting the kernel function (default is 200).
        """
        omega = np.linspace(0, np.pi, n_points)
        co2 = np.cos(omega / 2)
        values = self.K(co2)
        plt.figure()
        plt.plot(np.degrees(omega), values)
        plt.xlabel("Misorientation angle (degrees)")
        plt.ylabel("K(cos(omega/2))")
        plt.title("Kernel Function")
        plt.grid(True)
        plt.show()


class DeLaValleePoussinKernel(Kernel):
    """
    De La Vallee Poussin kernel class for orientation distribution functions

    Parameters
    ----------
    kappa : float, optional
        Shape parameter of the kernel
    halfwidth : float, optional
        Halfwidth in radians; overrides kappa if provided
    bandwidth : int, optional
        Maximum degree of the series expansion

    Attributes
    ----------
    A : ndarray
        Series coefficients of the kernel
    kappa : float
        Shape parameter of the kernel
    C : float
        Normalization constant
    """
    def __init__(self, kappa=None, halfwidth=None, bandwidth=None):
        if halfwidth is not None:
            kappa = 0.5 * np.log(0.5) / np.log(np.cos(0.5*halfwidth))
        elif kappa is None:
            kappa = 90

        self.kappa = kappa
        L = bandwidth if bandwidth is not None else round(kappa)
        C = beta(1.5, 0.5) / beta(1.5, kappa + 0.5)
        self.C = C

        A = np.ones(L + 1)
        A[1] = kappa / (kappa + 2)

        for l in range(1, L - 1):
            A[l + 1] = ((kappa - l + 1) * A[l - 1] - (2 * l + 1) * A[l]) / (kappa + l + 2)

        for l in range(0, L + 1):
            A[l] *= (2 * l + 1)

        super().__init__(A)
        self.cutA()

    def K(self, co2):
        """
        Evaluate the De La Vallee Poussin kernel function.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine of half the misorientation angle. Values will be clipped to [-1, 1].

        Returns
        -------
        float or ndarray
            Kernel value(s) evaluated at the input `co2`.
        """
        co2 = np.clip(co2, -1, 1)
        return self.C * co2 ** (2 * self.kappa)

    def DK(self, co2):
        """
        Evaluate the derivative of the De La Vallee Poussin kernel with respect to misorientation.

        Parameters
        ----------
        co2 : float or ndarray
            Cosine of half the misorientation angle. Values should be in [-1, 1].

        Returns
        -------
        float or ndarray
            Derivative of the kernel function evaluated at the input `co2`.
        """
        return -self.C * self.kappa * np.sqrt(1 - co2 ** 2) * co2 ** (2 * self.kappa - 1)

    def RK(self, t):
        """
        Evaluate the reduced kernel function R_K at a given input.

        Parameters
        ----------
        t : float or ndarray
            Input value, typically representing a normalized distance or cosine value.

        Returns
        -------
        float or ndarray
            Value of the reduced kernel function R_K at `t`.
        """
        return (1 + self.kappa) * ((1 + t) / 2) ** self.kappa

    def DRK(self, t):
        """
        Evaluate the derivative of the reduced kernel function R_K at a given input.

        Parameters
        ----------
        t : float or ndarray
            Input value, typically representing a normalized distance or cosine value.

        Returns
        -------
        float or ndarray
            Value of the derivative of R_K at `t`.
        """
        return self.kappa * (1 + self.kappa) * ((1 + t) / 2) ** (self.kappa - 1) / 2

    def halfwidth(self):
        """
        Compute the halfwidth of the DeLaValleePoussin kernel.

        The halfwidth is the misorientation angle ω where the kernel value
        drops to half its maximum.

        Returns
        -------
        float
            Halfwidth angle in radians.
        """
        return 2 * np.arccos(0.5 ** (1 / (2 * self.kappa)))


class ODF(object):
    """
    Estimate an Orientation Distribution Function (ODF) from a set of orientations
    using kernel density estimation.

    Parameters
    ----------
    orientations : orix.quaternion.Orientation
        Input orientation set.
    halfwidth : float, optional
        Halfwidth of the kernel in radians (default: 10 degrees).
    weights : array-like, optional
        Weights for each orientation. If None, uniform weights are used.
    kernel : Kernel instance, optional
        Kernel function to use. If None, DeLaValleePoussinKernel is used.
    exact : bool, optional
        If False and orientation count > 1000, approximation using grid is applied.

    Attributes
    ----------
    orientations : orix.quaternion.Orientation
        Orientation set stored in the ODF.
    weights : ndarray
        Normalized weights of the orientations.
    kernel : Kernel
        Kernel used for density estimation.
    halfwidth : float
        Halfwidth of the kernel in radians.
    """
    def __init__(self, orientations, halfwidth=np.radians(10), weights=None, kernel=None, exact=False):

        if orientations.size == 0:
            raise ValueError("Orientation set is empty.")

        if weights is None:
            weights = np.ones(orientations.size) / orientations.size

        # Set up kernel
        if kernel is None:
            kernel = DeLaValleePoussinKernel(halfwidth=halfwidth)
        hw = kernel.halfwidth()

        # Gridify if too many orientations and not exact
        if orientations.size > 1000 and not exact:
            # Placeholder: replace with proper gridify function if needed
            # Currently using simple thinning and weighting
            step = max(1, orientations.size // 1000)
            orientations = orientations[::step]
            weights = weights[::step]
            weights = weights / np.sum(weights)

        self.orientations = orientations
        self.weights = weights
        self.kernel = kernel
        self.halfwidth = hw

    def evaluate(self, ori):
        """
        Evaluate the ODF at given orientations.

        Parameters
        ----------
        ori : Orientation
            Orientation(s) at which to evaluate the ODF.

        Returns
        -------
        values : ndarray
            ODF values at the specified orientations.
        """
        values = np.zeros(ori.size)
        for o in self.orientations:
            values += self.kernel.K_orientations(o, ori)
        return values


class EBSDmap:
    """
    Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate
    synthetic RVEs

    Parameters
    ----------
    fname : str
        Filename including path to EBSD file.
    gs_min : float, optional
        Minimum grain size in pixels. Grains smaller than this are disregarded
        for statistical analysis. Default is 10.0.
    vf_min : float, optional
        Minimum volume fraction. Phases with smaller values are disregarded.
        Default is 0.03.
    max_angle : float, optional
        Maximum misorientation angle (degrees) used for grain merging.
        Default is 5.0.
    connectivity : int, optional
        Connectivity for grain identification. Default is 8.
    show_plot : bool, optional
        If True, plots microstructure maps. Default is True.
    show_hist : bool, optional
        If True, plots histograms of grain statistics. Default follows `show_plot`.
    felzenszwalb : bool, optional
        If True, applies Felzenszwalb segmentation. Default is False.
    show_grains : bool, optional
        If True, plots grain labeling. Default is False.
    hist : bool, optional
        Deprecated. Use `show_hist` instead.
    plot : bool, optional
        Deprecated. Use `show_plot` instead.

    Attributes
    ----------
    emap : object
        Loaded EBSD map object.
    sh_x, sh_y : int
        Shape of EBSD map in pixels.
    dx, dy : float
        Pixel size in microns.
    npx : int
        Total number of pixels in the map.
    ms_data : list of dict
        Phase-specific microstructure information including:
        - name : str — phase name
        - vf : float — volume fraction
        - index : int — phase index
        - ori : Orientation — grain orientations
        - cs : crystal symmetry / Laue group
        - mo_map : ndarray — misorientation field
        - rgb_im : ndarray — IPF color image
        - ngrains : int — number of grains
        - gs_param, ar_param, om_param : ndarray — grain size, aspect ratio, main axis angle parameters
        - gs_data, ar_data, om_data : ndarray — raw distributions
        - gs_moments, ar_moments, om_moments : list — distribution moments
        - graph : object — microstructure graph
    ci_map : ndarray
        Optional confidence index map if available.
    ngrains : int
        Total number of grains after merging/pruning.
    """

    def __init__(self, fname, gs_min=10.0, vf_min=0.03, max_angle=5.0, connectivity=8,
                 show_plot=True, show_hist=None, felzenszwalb=False, show_grains=False,
                 hist=None, plot=None):


        def reassign(pix, ori, phid, bads):
            """
            Reassign the orientation of a pixel based on neighboring pixels of the same phase.

            Parameters
            ----------
            pix : int
                Index of the pixel to be reassigned.
            ori : ndarray
                Array of orientations for all pixels.
            phid : ndarray
                Array of phase IDs for all pixels.
            bads : set
                Set of indices corresponding to pixels that could not be reassigned previously.

            Returns
            -------
            None
                Modifies `ori` in-place. If no valid neighbor is found, `pix` is added to `bads`.
            """
            phase = phid[pix]
            ix = -1
            if pix - 1 >= 0 and phid[pix - 1] == phase and pix - 1 not in bads:
                ix = pix - 1
            elif pix + 1 < self.npx and phid[pix + 1] == phase and pix + 1 not in bads:
                ix = pix + 1
            elif pix - self.sh_x >= 0 and phid[pix - self.sh_x - 1] == phase and pix - self.sh_x not in bads:
                pix = pix - self.sh_x
            elif pix + self.sh_x < self.npx and phid[pix + self.sh_x + 1] == phase and pix + self.sh_x not in bads:
                ix = pix + self.sh_x
            if ix >= 0:
                ori[pix] = ori[ix]  # pixel orientation is reassigned
            else:
                bads.add(pix)  # no valid neighbor found add pix again to list

        # interpret parameters
        if plot is not None:
            show_plot = plot
            logging.warning('Use of "plot" is depracted, use argument "show_plot" instead.')
        if hist is not None:
            show_hist = hist
            logging.warning('Use of "hist" is depracted, use argument "show_plot" instead.')
        if show_hist is None:
            show_hist = show_plot
        max_angle *= np.pi / 180
        self.ms_data = []
        # read EBSD map and return the orix object
        self.emap = io.load(fname)
        self.sh_x, self.sh_y = self.emap.shape  # shape of EBSD map in pixels
        self.dx, self.dy = self.emap.dx, self.emap.dy  # distance in micron b/w pixels
        self.npx = self.emap.size  # total number of pixels in EBSD map
        if self.sh_x * self.sh_y != self.npx:
            raise ValueError(f"Size of map ({self.npx} px) does not match its shape: {self.sh_x, self.sh_y}")

        # determine number of phases and generate histogram
        Nphase = len(self.emap.phases.ids)  # number of phases
        offs = 0 if 0 in self.emap.phases.ids else 1  # in CTX maps, there is no phase "0"
        phist = np.histogram(self.emap.phase_id, Nphase + offs)
        print(f'Imported EBSD map with {len(self.emap.phases.ids)} phases.')

        # read phase names and calculate volume fractions and plots if active
        for n_ph, ind in enumerate(self.emap.phases.ids):
            if ind == -1:
                continue
            vf = phist[0][n_ph + offs] / self.npx
            if vf < vf_min:
                continue
            data = dict()  # initialize data dictionary
            data['vf'] = vf
            data['name'] = self.emap.phases.names[n_ph]
            data['index'] = ind
            data['len_x'] = self.sh_x
            data['len_y'] = self.sh_y
            data['delta_x'] = self.dx
            data['delta_y'] = self.dy

            # generate phase-specific orientations
            ori_e = self.emap[self.emap.phase_id == ind].orientations.in_euler_fundamental_region()
            data['ori'] = Orientation.from_euler(ori_e)
            data['cs'] = self.emap.phases[ind].point_group.laue
            # assign bad pixels to one neighbor
            # identify non-indexed pixels and pixels with low confidence index (CI)
            if 'ci' in self.emap.prop.keys():
                val = self.emap.prop['ci']
                if len(val) == self.npx:
                    self.ci_map = val
                else:
                    self.ci_map = np.zeros(self.npx)
                    self.ci_map[self.emap.phase_id == 0] = val
                bad_pix = set(np.nonzero(val < 0.1)[0])
            else:
                bad_pix = set()
            if len(bad_pix) > 0:
                niter = 0
                print(f'Initial number of bad pixels: {len(bad_pix)}')
                nbad = len(bad_pix)
                while len(bad_pix) > 0 and niter < 2 * nbad:
                    bp = bad_pix.pop()
                    reassign(bp, data['ori'], self.emap.phase_id, bad_pix)
                    niter += 1
                print(f'After {niter} loops: number of bad pixels: {len(bad_pix)}')

            # calculate misorientation field
            val = data['ori'].angle
            if len(val) == self.npx:
                bmap = val
            else:
                bmap = np.zeros(self.npx)
                bmap[self.emap.phase_id == ind] = val
            data['mo_map'] = bmap

            # Get IPF colors
            ipf_key = ox_plot.IPFColorKeyTSL(data['cs'])
            if data['ori'].size == self.npx:
                rgb_val = ipf_key.orientation2color(data['ori'])
            else:
                rgb_val = np.zeros((self.npx, 3))
                rgb_val[self.emap.phase_id == ind, :] = ipf_key.orientation2color(data['ori'])
            data['rgb_im'] = np.reshape(rgb_val, (self.sh_x, self.sh_y, 3))

            # generate map with grain labels
            print('Identifying regions of homogeneous misorientations and assigning them to grains.')
            labels, n_regions = find_similar_regions(bmap.reshape((self.sh_x, self.sh_y)),
                                                     tolerance=max_angle, connectivity=connectivity)
            print(f"Phase #{data['index']} ({data['name']}): Identified Grains: {n_regions}")

            # build and visualize graph of unfiltered map
            print('Building microstructure graph.')
            ms_graph = build_graph_from_labeled_pixels(labels, self.emap, n_ph)
            ms_graph.name = 'Graph of microstructure'
            print('Starting to simplify microstructure graph.')

            # graph pruning step 1: remove grains that have no convex hull
            # and merge regions with similar misorientation
            grain_set = set(ms_graph.nodes)
            rem_grains = len(grain_set)
            while rem_grains > 0:
                num = grain_set.pop()  # get random ID of grain and remove it from the list
                nd = ms_graph.nodes[num]  # node to be considered
                rem_grains = len(grain_set)
                pts = np.array(np.unravel_index(nd['pixels'], (self.sh_x, self.sh_y)), dtype=float).T
                pts[:, 0] *= self.dx  # convert pixel distances to micron
                pts[:, 1] *= self.dy
                try:
                    hull = ConvexHull(pts)
                    ms_graph.nodes[num]['hull'] = hull
                except Exception as e:
                    # grain has no convex hull
                    num_ln = find_largest_neighbor(ms_graph, num)
                    merge_nodes(ms_graph, num, num_ln)
                    continue
                # search for neighbors with similar orientation
                sim_neigh, ori_neigh = find_sim_neighbor(ms_graph, num)
                if ori_neigh <= max_angle:
                    merge_nodes(ms_graph, num, sim_neigh)
            self.ngrains = len(ms_graph.nodes)
            print(f'After merging of similar regions, {self.ngrains} grains left.')

            # graph pruning step 2: merge small grains into their largest neighbor grain
            grain_set = set(ms_graph.nodes)
            rem_grains = len(grain_set)
            while rem_grains > 0:
                num = grain_set.pop()  # get random ID of grain and remove it from the list
                rem_grains = len(grain_set)
                if ms_graph.nodes[num]['npix'] < gs_min:
                    num_ln = find_largest_neighbor(ms_graph, num)
                    merge_nodes(ms_graph, num, num_ln)
            self.ngrains = len(ms_graph.nodes)
            data['ngrains'] = self.ngrains
            print(f'After elimination of small grains, {self.ngrains} grains left.')

            # Extract grain statistics and axes
            arr_a = []
            arr_b = []
            arr_eqd = []
            arr_om = []
            for num, node in ms_graph.nodes.items():
                hull = node['hull']
                eqd = 2.0 * (hull.volume / np.pi) ** 0.5  # area of convex hull approximates grain better than pixels
                pts = hull.points[hull.vertices]  # outer nodes of grain
                # analyze geometry of point cloud
                ea, eb, va, vb = get_grain_geom(pts, method='ellipsis', two_dim=True)  # std: 'ellipsis', failsafe: 'raw'
                # assert ea >= eb
                if eb < 0.01 * ea:
                    logging.warning(f'Grain {num} has too high aspect ratio: main axes: {ea}, {eb}')
                    eb = 0.01 * ea
                sc_fct = eqd / np.sqrt(ea**2 + eb**2)  # rescale axes of ellipsis to ensure consistency with eqd
                ea *= sc_fct
                eb *= sc_fct
                # assert np.dot(va, vb) < 1.e-9
                # assert np.isclose(np.linalg.norm(va), 1.0)
                omega = np.arccos(va[0])  # angle of major axis against y-axis of map in range [0, pi]
                node['max_dia'] = ea
                node['min_dia'] = eb
                node['equ_dia'] = eqd
                node['maj_ax'] = va
                node['min_ax'] = vb
                node['omega'] = omega
                node['center'] = hull.points.mean(axis=0)
                arr_a.append(ea)
                arr_b.append(eb)
                arr_eqd.append(eqd)
                arr_om.append(omega)
            arr_om = np.array(arr_om)
            arr_a = np.array(arr_a)
            arr_b = np.array(arr_b)
            arr_eqd = np.array(arr_eqd)
            print('\n--------------------------------------------------------')
            print('Statistical microstructure parameters in pixel map ')
            print('--------------------------------------------------------')
            print(np.median(arr_a), np.std(arr_a))
            print(np.median(arr_b), np.std(arr_b))
            print(np.median(arr_eqd), np.std(arr_eqd))

            # calculate equivalent diameters
            doffs = 0.
            deq_log = np.log(arr_eqd)
            dscale = np.exp(np.median(deq_log))
            dsig = np.std(deq_log)
            data['gs_param'] = np.array([dsig, doffs, dscale])
            data['gs_data'] = arr_eqd
            data['gs_moments'] = [dscale, dsig]
            if show_hist:
                fig, ax = plt.subplots()
                hc, hb = np.histogram(arr_eqd, bins=20)
                x0 = hb[0]
                ind = np.nonzero(hc == 1)[0]  # find first bin with count == 1
                if len(ind) > 0:
                    x1 = hb[ind[0]]
                else:
                    x1 = max(arr_eqd)
                x = np.linspace(x0, x1, 150)
                y = lognorm.pdf(x, dsig, loc=doffs, scale=dscale)
                ax.plot(x, y, '-r', label='lognorm fit')
                ax.hist(arr_eqd, bins=20, range=(x0, x1), density=True, label='data')
                plt.legend()
                plt.title('Histogram of grain equivalent diameters')
                plt.xlabel('Equivalent diameter (micron)')
                plt.ylabel('Normalized frequency')
                plt.show()

            # grain aspect ratio
            asp = arr_a / arr_b
            aoffs = 0.
            asp_log = np.log(asp)
            ascale = np.exp(np.median(asp_log))  # lognorm.median(asig, loc=aoffs, scale=ascale)
            asig = np.std(asp_log)  # lognorm.std(asig, loc=aoffs, scale=ascale)
            data['ar_param'] = np.array([asig, aoffs, ascale])
            data['ar_data'] = asp
            data['ar_moments'] = [ascale, asig]
            if show_hist:
                # plot distribution of aspect ratios
                fig, ax = plt.subplots()
                hc, hb = np.histogram(asp, bins=20)
                x0 = hb[0]
                ind = np.nonzero(hc == 1)[0]  # find first bin with count == 1
                if len(ind) > 0:
                    x1 = hb[ind[0]]
                else:
                    x1 = max(asp)
                x = np.linspace(x0, x1, 150)
                y = lognorm.pdf(x, asig, loc=aoffs, scale=ascale)
                ax.plot(x, y, '-r', label='lognorm fit')
                ax.hist(asp, bins=20, range = (x0, x1), density=True, label='data')
                plt.legend()
                plt.title('Histogram of grain aspect ratio')
                plt.xlabel('aspect ratio')
                plt.ylabel('normalized frequency')
                plt.show()

            # angles of main axis
            # fit von Mises distribution (circular normal distribution) to data
            kappa, oloc, oscale = vonmises.fit(2*arr_om - np.pi)
            med_om = vonmises.median(kappa, loc=oloc)  # scale parameter has no effect on vonmises distribution
            std_om = vonmises.std(kappa, loc=oloc)
            data['om_param'] = np.array([kappa, oloc])
            data['om_data'] = arr_om
            data['om_moments'] = [med_om, std_om]
            if show_hist:
                fig, ax = plt.subplots()
                x = np.linspace(-np.pi, np.pi, 200)  # np.amin(omega), np.amax(omega), 150)
                y = vonmises.pdf(x, kappa, loc=oloc)
                ax.plot(0.5*(x+np.pi), 2 * y, '-r', label='von Mises fit')
                ax.hist(arr_om, bins=40, density=True, label='data')
                plt.legend()
                plt.title('Histogram of tilt angles of major axes')
                plt.xlabel('angle (rad)')
                plt.ylabel('normalized frequency')
                plt.show()

            print(f"Analyzed microstructure of phase #{data['index']} ({data['name']}) with {self.ngrains} grains.")
            print(f'Median values: equiv. diameter: {dscale.round(3)} micron, ' +
                  f'aspect ratio: {ascale.round(3)}, ' +
                  f'tilt angle: {(med_om * 180 / np.pi).round(3)}°')
            print(f'Std. dev: equivalent diameter: {dsig.round(3)} micron, ' +
                  f'aspect ratio: {asig.round(3)}, ' +
                  f'tilt angle: {(std_om * 180 / np.pi).round(3)}°')
            data['graph'] = ms_graph
            self.ms_data.append(data)

        if show_plot:
            self.plot_mo_map()
            self.plot_ipf_map()
            self.plot_segmentation()
            self.plot_pf()

        if show_grains:
            self.plot_grains()

        if felzenszwalb:
            self.plot_felsenszwalb()
        return

    def calcORI(self, Ng, iphase=0, shared_area=None, nbins=12,
                res_low=5, res_high=25, res_step=2, lim=5,
                verbose=False, full_output=False):
        """
        Estimate optimum kernel half-width and produce reduced set of
        orientations for given number of grains

        Parameters
        ----------
        Ng : int
            Numbr of grains for which orientation is requested.
        iphase : int, optional
            Phase for which data is requested. The default is 0.
        shared_area : array, optional
            Grain boundary data. The default is None.
        nbins : int, optional
            number of bins for GB texture. The default is 12.

        Returns
        -------
        ori : (Ng, 3)-array
            Array with Ng Euler angles.

        """
        ms = self.ms_data[iphase]
        orired, odfred, ero, res = texture_reconstruction(Ng, orientations=ms['ori'],
                                                          res_low=res_low, res_high=res_high,
                                                          res_step=res_step, lim=lim,
                                                          verbose=verbose)

        if shared_area is None:
            if full_output:
                return orired, odfred, ero, res
            else:
                return orired
        else:
            raise ModuleNotFoundError('Shared area is not implemented yet in pure Python version.\n'
                                      'Use kanapy-mtex for this option.')
            #orilist, ein, eout, mbin = \
            #    self.eng.gb_textureReconstruction(ms['grains'], orired,
            #                                      matlab.double(shared_area), nbins, nargout=4)
            #return np.array(self.eng.Euler(orilist))

    def showIPF(self):
        """
        Plot the inverse pole figure (IPF) color key for all phases in the EBSD map
        """
        for i in self.emap.phases.ids:
            pg = self.emap.phases[i].point_group.laue
            fig = plt.figure(figsize=(8, 8))
            ax0 = fig.add_subplot(111, projection="ipf", symmetry=pg, zorder=2)
            ax0.plot_ipf_color_key(show_title=False)
            ax0.patch.set_facecolor("None")
            plt.show()

    def plot_ci_map(self):
        """
        Plot the confidence index (CI) map of the EBSD data if available
        """
        if 'ci' in self.emap.prop.keys():
            plt.imshow(self.ci_map.reshape((self.sh_x, self.sh_y)))
            plt.title('CI values in EBSD map')
            plt.colorbar(label="CI")
            plt.show()

    def plot_pf(self, vector=None):
        """
        Plot pole figures for all phases using the specified sample direction

        Parameters
        ----------
        vector : array-like, optional
            Sample reference vector for the pole figure. Default is [0, 0, 1].

        Notes
        -----
        Plots <111> poles in the sample reference frame for each phase.
        """
        # plot inverse pole figure
        # <111> poles in the sample reference frame
        if vector is None:
            vector = [0, 0, 1]
        pids = np.array(self.emap.phases.ids)
        pids = pids[pids >= 0]
        for n_ph, data in enumerate(self.ms_data):
            orientations = data['ori']
            plot_pole_figure(orientations, self.emap.phases[pids[n_ph]], vector=vector)
            #t_ = Miller(uvw=vector, phase=self.emap.phases[ind]).symmetrise(unique=True)
            #t_all = orientations.inv().outer(t_)
            #fig = plt.figure(figsize=(8, 8))
            #ax = fig.add_subplot(111, projection="stereographic")
            #ax.scatter(t_all)
            #ax.set_labels("X", "Y", None)
            #ax.set_title(data['name'] + r" $\left<001\right>$ PF")
            #plt.show()

    def plot_pf_proj(self, vector=None):
        """
        Plot projected pole figures for all phases using the specified sample direction

        Parameters
        ----------
        vector : array-like, optional
            Sample reference vector for the projected pole figure. Default is [0, 0, 1].

        Notes
        -----
        Uses a projection method to visualize <111> poles in the sample reference frame.
        """
        # plot inverse pole figure
        # <111> poles in the sample reference frame
        if vector is None:
            vector = [0, 0, 1]
        pids = np.array(self.emap.phases.ids)
        pids = pids[pids >= 0]
        for n_ph, data in enumerate(self.ms_data):
            orientations = data['ori']
            plot_pole_figure_proj(orientations, self.emap.phases[pids[n_ph]], vector=vector)

    def plot_grains_marked(self):
        """
        Plot grains with labels and major/minor axes for all phases

        Notes
        -----
        Each grain is annotated with its number. Major axes are plotted in black,
        minor axes in red. The plot uses a distinct colormap for different grains.
        """
        # plot grain with numbers and axes
        for data in self.ms_data:
            ngr = data['ngrains']
            cols = get_distinct_colormap(ngr)
            cmap = LinearSegmentedColormap.from_list('segs', cols, N=ngr)
            plt.imshow(data['graph'].graph['label_map'] / ngr, cmap=cmap)
            for num, node in data['graph'].nodes.items():
                ctr = data['graph'].nodes[num]['center']
                plt.annotate(str(num), xy=(ctr[1], ctr[0]))
                pts = np.zeros((4, 2))
                pts[0, :] = node['center'] - node['max_dia'] * node['maj_ax']
                pts[1, :] = node['center'] + node['max_dia'] * node['maj_ax']
                pts[2, :] = node['center'] - node['min_dia'] * node['min_ax']
                pts[3, :] = node['center'] + node['min_dia'] * node['min_ax']
                pts[:, 0] /= self.dx
                pts[:, 1] /= self.dy
                plt.plot(pts[0:2, 1], pts[0:2, 0], color='k')
                plt.plot(pts[2:4, 1], pts[2:4, 0], color='red')
            plt.title(f"Phase #{data['index']} ({data['name']}): Grain labels and axes: {ngr}")
            plt.colorbar(label='Grain Number')
            plt.show()

    def plot_mo_map(self):
        """
        Plot the misorientation map for all phases

        Notes
        -----
        Shows the misorientation angle of each pixel with respect to a reference orientation.
        """
        for data in self.ms_data:
            plt.imshow(data['mo_map'].reshape((self.sh_x, self.sh_y)))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation angle wrt reference")
            plt.colorbar(label="Misorientation (rad)")
            plt.show()

    def plot_segmentation(self, show_mo=True, show_ipf=True):
        """
        Plot segmentation results for all phases

        Parameters
        ----------
        show_mo : bool, optional
            If True, overlay grain boundaries on the misorientation map. Default is True.
        show_ipf : bool, optional
            If True, overlay grain boundaries on the IPF map. Default is True.

        Notes
        -----
        Uses similarity-based segmentation to highlight grain boundaries on the maps.
        """
        # plot segmentation results
        for data in self.ms_data:
            if show_mo:
                gscale_map = data['mo_map'].reshape((self.sh_x, self.sh_y)) / np.pi
                plt.imshow(mark_boundaries(gscale_map, data['graph'].graph['label_map']))
                plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation map with similarity segmentation")
                plt.show()
            if show_ipf:
                plt.imshow(mark_boundaries(data['rgb_im'], data['graph'].graph['label_map']))
                plt.title(f"Phase #{data['index']} ({data['name']}): IPF map with similarity segmentation")
                plt.show()

    def plot_felsenszwalb(self):
        """
        Apply Felzenszwalb segmentation to the misorientation map and plot the results

        Notes
        -----
        Segments the misorientation map using the Felzenszwalb algorithm and overlays
        the segment boundaries for visualization.
        """
        from skimage.segmentation import felzenszwalb
        for data in self.ms_data:
            gscale_map = data['mo_map'].reshape((self.sh_x, self.sh_y)) / np.pi
            labels_fz = felzenszwalb(gscale_map, scale=300, sigma=0.6, min_size=25)  # sc=300, sig=0.8, min_s=25
            plt.imshow(mark_boundaries(gscale_map, labels_fz))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation map with Felzenszwalb segmentation")
            plt.show()

    def plot_graph(self):
        """
        Visualize the microstructure graph for all phases

        Notes
        -----
        Uses the `visualize_graph` function to display nodes and connections of each phase's graph.
        """
        # visualize graph
        for data in self.ms_data:
            visualize_graph(data['graph'])

    def plot_ipf_map(self):
        """
        Plot the IPF (inverse pole figure) map for all phases

        Notes
        -----
        Pixels belonging to other phases are set to black. Each phase is visualized separately.
        """
        # plot EBSD maps for all phase
        # set pixels of other phases to black
        for data in self.ms_data:
            fig = self.emap.plot(
                data['rgb_im'].reshape(self.npx, 3),
                return_figure=True,
                figure_kwargs={"figsize": (12, 8)},
            )
            fig.show()

    def plot_grains(self, N=5):
        """
        Plot the N largest grains with one-hot maps, convex hulls, and principal axes

        Parameters
        ----------
        N : int, optional
            Number of largest grains to plot for each phase. Default is 5.

        Notes
        -----
        Each grain is visualized with its pixels, convex hull, and major/minor axes.
        Major axes are plotted in cyan, minor axes in green.
        """
        # select N largest gains and create a one-hot-plot for each grain
        # showing its pixels, the vertices of the convex hull and the convex hull itself
        # together with the principal axes of the grain
        for data in self.ms_data:
            glist = [0]
            slist = [0]
            for num, node in data['graph'].nodes.items():
                hs = node['equ_dia']
                if hs > min(slist):
                    glist.append(num)
                    slist.append(hs)
                    if len(glist) > N:
                        i = np.argmin(slist)
                        glist.pop(i)
                        slist.pop(i)
            for num in glist:
                pix_c = np.unravel_index(data['graph'].nodes[num]['pixels'], (self.sh_x, self.sh_y))
                oh_grain = np.zeros((self.sh_x, self.sh_y))
                oh_grain[pix_c] = 0.3
                node = data['graph'].nodes[num]
                ind = node['hull'].vertices
                oh_grain[pix_c[0][ind], pix_c[1][ind]] = 1
                plt.imshow(oh_grain, cmap='gray')
                # add convex hull
                for i in range(len(ind) - 1):
                    ix = [ind[i], ind[i + 1]]
                    plt.plot(pix_c[1][ix], pix_c[0][ix], color='y')
                ix = [ind[i + 1], ind[0]]
                plt.plot(pix_c[1][ix], pix_c[0][ix], color='y')
                # add axes
                ctr = node['center']
                pts = np.zeros((4, 2))
                pts[0, :] = ctr - node['max_dia'] * node['maj_ax']
                pts[1, :] = ctr + node['max_dia'] * node['maj_ax']
                pts[2, :] = ctr - node['min_dia'] * node['min_ax']
                pts[3, :] = ctr + node['min_dia'] * node['min_ax']
                pts[:, 0] /= self.dx
                pts[:, 1] /= self.dy
                plt.plot(pts[0:2, 1], pts[0:2, 0], color='cyan')
                plt.plot(pts[2:4, 1], pts[2:4, 0], color='green')
                plt.title(f"Phase #{data['index']} ({data['name']}): Grain #{num}")
                plt.show()


def get_ipf_colors(ori_list, color_key=0):
    """
    Get RGB colors for a list of orientations in radians

    Parameters
    ----------
    ori_list : (N, 3) ndarray
        List of N Euler angles in radians

    color_key : int, optional
        Index of the IPF color key to use. Default is 0

    Returns
    -------
    colors : (N, 3) ndarray
        RGB values corresponding to each orientation

    Notes
    -----
    Assumes cubic crystal symmetry and cubic specimen symmetry
    """

    # get colors
    if not isinstance(ori_list, Orientation):
        ori_list = Orientation.from_euler(ori_list)
    ckey = ox_plot.IPFColorKeyTSL(ori_list.symmetry)
    ocol = ckey.orientation2color(ori_list)
    return ocol


def createOriset(num, ang, omega, hist=None, shared_area=None,
                 cs=None, degree=True, Nbase=10000, resolution=None,
                 res_low=5, res_high=25, res_step=2, lim=5,
                 verbose=False, full_output=False):
    """
    Create a set of Euler angles according to an ODF defined by input orientations and kernel half-width

    Parameters
    ----------
    num : int
        Number of Euler angles to generate
    ang : (3,) or (M, 3) array or Orientation
        Input set of Euler angles (in degrees or radians) defining the ODF
    omega : float
        Kernel half-width (in degrees or radians)
    hist : array, optional
        Histogram of MDF. Default is None
    shared_area : array, optional
        Shared area between pairs of grains. Default is None
    cs : Symmetry, optional
        Crystal symmetry group. Default is 'm3m'
    degree : bool, optional
        If True, input angles and omega are in degrees. Default is True
    Nbase : int, optional
        Base number of orientations for artificial ODF. Default is 10000
    resolution : float, optional
        Resolution for orientation generation. If None, derived from omega
    res_low, res_high, res_step, lim : int, optional
        Parameters for texture reconstruction
    verbose : bool, optional
        If True, prints progress messages. Default is False
    full_output : bool, optional
        If True, returns additional reconstruction outputs. Default is False

    Returns
    -------
    ori : (num, 3) ndarray
        Reduced set of Euler angles

    Notes
    -----
    Can generate artificial ODFs if input set is small. Grain-boundary-texture
    reconstruction with hist or shared_area requires the kanapy-mtex module.
    """
    # prepare parameters
    if hist is not None or shared_area is not None:
        raise ModuleNotFoundError('The grain-boundary-texture module is currently only available in kanapy-mtex.')
    if cs is None or not isinstance(cs, Symmetry):
        logging.warning('Crystal Symmetry "cs" must be provided as Symmetry object.')
        print(type(cs))
    if resolution is None:
        if degree:
            resolution = omega
        else:
            resolution = np.rad2deg(omega)  # resolution must be given in degrees
    if degree:
        omega = np.deg2rad(omega)
    if isinstance(ang, list):
        if degree:
            ang = np.deg2rad(ang)
        else:
            ang = np.array(ang)
        ang = Orientation.from_euler(ang, cs)
    else:
        assert isinstance(ang, Orientation)    
    # psi = DeLaValleePoussinKernel(halfwidth=omega)
    if ang.size < Nbase/100:
        # create artificial ODF for monomodal texture or small orientation set
        if verbose:
            logging.info(f'Creating artificial ODF centered around orientation {ang} with '
                         f'kernel halfwidth: {omega}')
        odf = ODF(ang, halfwidth=omega)
        if verbose:
            logging.info(f'Creating {Nbase} orientation from artificial ODF.')
        ori = calc_orientations(odf, Nbase, res=resolution)
        assert ori.size == Nbase
    else:
        ori = ang
    logging.info(f'Texture reconstruction generating {num} orientations from {ori.size} inputs.'
                 f'Kernel halfwidth: {omega}')
    ori_red, odfred, ero, res = texture_reconstruction(num, orientations=ori,
                                                   kernel_halfwidth=omega,
                                                   res_low=res_low, res_high=res_high,
                                                   res_step=res_step, lim=lim,
                                                   verbose=verbose)
    if hist is None:
        if full_output:
            return ori_red, odfred, ero, res
        else:
            return ori_red
    else:
        pass
        #if shared_area is None:
        #    raise ValueError('Microstructure.shared_area must be provided if hist is given.')
        #orilist, ein, eout, mbin = \
        #    eng.gb_textureReconstruction(matlab.double(hist), ori,
        #                                 matlab.double(shared_area), len(hist), nargout=4)
        #return np.array(eng.Euler(orilist))


def createOrisetRandom(num, omega=None, hist=None, shared_area=None, cs=None, Nbase=None):
    """
    Create a set of Euler angles for a random texture

    Parameters
    ----------
    num : int
        Number of Euler angles to generate
    omega : float, optional
        Kernel half-width in degrees. Default is 7.5
    hist : array, optional
        Histogram of MDF. Default is None
    shared_area : array, optional
        Shared area between pairs of grains. Default is None
    cs : Symmetry, optional
        Crystal symmetry group. Default is 'm3m'
    Nbase : int, optional
        Base number of orientations for random texture. Default is 5000

    Returns
    -------
    ori : (num, 3) ndarray
        Set of randomly distributed Euler angles

    Notes
    -----
    Unlike `createOriset`, this function directly generates `num` random orientations
    without reducing a larger artificial EBSD set. Grain-boundary-texture reconstruction
    using `hist` or `shared_area` requires the kanapy-mtex module.
    """
    if hist is not None or shared_area is not None:
        raise ModuleNotFoundError('The grain-boundary-texture module is currently only available in kanapy-mtex.')
    if cs is None or not isinstance(cs, Symmetry):
        logging.warning('Crystal Symmetry "cs" must be provided as Symmetry object.')
        print(type(cs))

    ori = Orientation.random(num, cs).in_euler_fundamental_region()
    if hist is None:
        return ori
    else:
        pass
    # ori = eng.project2FundamentalRegion(ori)
    #    if shared_area is None:
    #        raise ValueError('Shared grain boundary area (geometry["GBarea"]) must be provided if hist is given.')
    #    orilist, ein, eout, mbin = \
    #        eng.gb_textureReconstruction(matlab.double(hist), ori,
    #                                     matlab.double(shared_area), len(hist), nargout=4)
    #    return np.array(eng.Euler(orilist))

def segment_microstructure(
    json_path: str,
    *,
    th_cut_deg: float = 5.0,
    th_merge_deg: float = 7.5,
    min_voxels: int = 5,
    print_topk_rag: int = 10,
    tiny_cutoff: int = 5,
    verbose: bool = True,
) -> None:
    """
    Segment all snapshots in a  microstructure JSON in-place.

    - Snapshot 0 is assumed already segmented and is kept as reference for phase_id mapping.
    - Snapshots 1..last are re-segmented from voxel orientations using:
        Step 1: 6-neighbor voxel graph + ORIX misorientation on edges
        Step 2: threshold cut (<= th_cut_deg) + connected components
        Step 3: segment-RAG boundary stats + ORIX mean orientation per segment
        Step 4: merge/prune via union-find:
            Rule A: merge adjacent segments if mean-orientation miso <= th_merge_deg
            Rule B: absorb tiny components (<= min_voxels) into best neighbor using root-mean miso
        Step 5: diagnostics (post-merge boundary stats)
        Step 6/7: write voxel grain_id + rebuild grains[] with ORIX mean orientation

    This function overwrites `json_path` (no new file).

    Parameters
    ----------
    json_path : str
        Path to the JSON file to update in-place.
    th_cut_deg : float
        Edge cut threshold in degrees (keep edges <= th_cut_deg).
    th_merge_deg : float
        Merge threshold in degrees for Rule A (mean-orientation miso).
    min_voxels : int
        Tiny component cutoff (<= min_voxels) for Rule B.
    print_topk_rag : int
        Number of highest-contact boundaries printed for diagnostics.
    tiny_cutoff : int
        Diagnostics: count segments with <= tiny_cutoff voxels after Step 2.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    None
        Updates the JSON file in-place.
    """

    def _p(*args):
        if verbose:
            print(*args)

    # -----------------------------
    # helpers: union-find
    # -----------------------------
    def _uf_find(parent: np.ndarray, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> bool:
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # -----------------------------
    # Load JSON
    # -----------------------------
    _p("=== LOAD JSON ===")
    with open(json_path, "r") as f:
        data = json.load(f)

    if "microstructure" not in data:
        raise KeyError("Top-level key 'microstructure' not found.")

    micro = data["microstructure"]
    n_snaps = len(micro)
    if n_snaps < 2:
        _p("Nothing to do: only one snapshot found.")
        return

    # Snapshot 0 reference phase mapping
    snap0 = micro[0]
    if "grains" not in snap0:
        raise KeyError("Snapshot 0 must contain 'grains' for phase mapping.")
    grainid_to_phase = {int(g["grain_id"]): int(g.get("phase_id", 0)) for g in snap0["grains"]}

    _p("File:", json_path)
    _p("Number of snapshots:", n_snaps)
    _p("Will update snapshots:", list(range(1, n_snaps)))

    # -----------------------------
    # Process each snapshot INC = 1..last
    # -----------------------------
    for inc in range(1, n_snaps):
        _p("\n" + "=" * 90)
        _p(f"=== PROCESS SNAPSHOT {inc}  time={micro[inc].get('time', None)} ===")
        _p("=" * 90)

        snap = micro[inc]
        voxels = snap.get("voxels", None)
        if voxels is None:
            raise KeyError(f"Snapshot {inc} missing 'voxels'.")

        N = len(voxels)
        if N == 0:
            _p(f"Snapshot {inc}: empty voxels; skipping.")
            continue

        # -----------------------------------------------------------------------------
        # STEP 1 — BUILD VOXEL GRAPH (6-neigh) + MISORIENTATION EDGE WEIGHTS (ORIX)
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 1: VOXEL GRAPH + MISORIENTATION (ORIX) ===")

        idx_to_pos = {}
        eulers = np.empty((N, 3), dtype=float)
        voxel_vol = np.empty(N, dtype=float)
        old_gid = np.empty(N, dtype=int)

        for pos, v in enumerate(voxels):
            ijk = tuple(int(x) for x in v["voxel_index"])  # 1-based indices
            idx_to_pos[ijk] = pos
            eulers[pos, :] = np.asarray(v["orientation"], dtype=float)
            voxel_vol[pos] = float(v["voxel_volume"])
            old_gid[pos] = int(v["grain_id"])

        edges_pos_u = []
        edges_pos_v = []

        for (i, j, k), pos in idx_to_pos.items():
            nbs = [
                (i - 1, j, k), (i + 1, j, k),
                (i, j - 1, k), (i, j + 1, k),
                (i, j, k - 1), (i, j, k + 1),
            ]
            for nb in nbs:
                nb_pos = idx_to_pos.get(nb, None)
                if nb_pos is None:
                    continue
                if nb_pos > pos:  # store undirected edge once
                    edges_pos_u.append(pos)
                    edges_pos_v.append(nb_pos)

        edges_pos_u = np.asarray(edges_pos_u, dtype=int)
        edges_pos_v = np.asarray(edges_pos_v, dtype=int)
        E = len(edges_pos_u)

        _p("N voxels:", N)
        _p("E edges (6-neigh undirected):", E)

        cs = symmetry.Oh  # cubic m-3m
        O = Orientation.from_euler(eulers, symmetry=cs, direction="lab2crystal", degrees=False)
        w_deg = np.asarray(O[edges_pos_u].angle_with(O[edges_pos_v], degrees=True), dtype=float)

        _p("Misorientation (deg): min/mean/max =",
           float(w_deg.min()), float(w_deg.mean()), float(w_deg.max()))
        _p("=== STEP 1 COMPLETE ===")

        # -----------------------------------------------------------------------------
        # STEP 2 — INITIAL CUT + CONNECTED COMPONENTS => INITIAL SEGMENTS
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 2: INITIAL CUT SEGMENTATION ===")
        _p("TH_CUT (deg):", th_cut_deg)

        keep = (w_deg <= th_cut_deg)
        u = edges_pos_u[keep]
        v = edges_pos_v[keep]
        _p("edges kept:", int(keep.sum()), "edges cut:", int((~keep).sum()))

        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        vals = np.ones(rows.shape[0], dtype=np.int8)
        A = coo_matrix((vals, (rows, cols)), shape=(N, N))

        n_seg, labels = connected_components(A, directed=False, connection="weak")
        sizes = np.bincount(labels, minlength=n_seg)
        sizes_sorted = np.sort(sizes)[::-1]

        _p("n_initial_segments:", int(n_seg))
        _p("largest 10 sizes:", sizes_sorted[:10].tolist())
        _p("smallest 10 sizes:", sizes_sorted[-10:].tolist())
        _p(f"segments with <= {tiny_cutoff} voxels:", int((sizes <= tiny_cutoff).sum()))
        _p("=== STEP 2 COMPLETE ===")

        # -----------------------------------------------------------------------------
        # STEP 3 — BUILD SEGMENT-RAG + MEAN ORIENTATIONS (ORIX)
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 3: BUILD SEGMENT-RAG ===")
        _p("segments:", int(n_seg), "labels min/max:", int(labels.min()), int(labels.max()))

        seg_size = sizes.astype(int)

        seg_mean_O = [None] * n_seg
        for s in range(n_seg):
            mask = (labels == s)
            seg_mean_O[s] = O[mask].mean()

        # Boundary aggregation using voxel edges
        rag_contact = {}   # (sa,sb)->int
        rag_miso_sum = {}  # (sa,sb)->float (voxel-edge miso sum)

        for i in range(E):
            a = int(labels[int(edges_pos_u[i])])
            b = int(labels[int(edges_pos_v[i])])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            key = (a, b)
            rag_contact[key] = rag_contact.get(key, 0) + 1
            rag_miso_sum[key] = rag_miso_sum.get(key, 0.0) + float(w_deg[i])

        rag_miso_mean = {k: rag_miso_sum[k] / rag_contact[k] for k in rag_contact}
        _p("segment-RAG edges:", len(rag_contact))

        if print_topk_rag > 0:
            top_edges = sorted(rag_contact.items(), key=lambda kv: kv[1], reverse=True)[:print_topk_rag]
            _p(f"\nTop {print_topk_rag} neighbors by boundary contact count:")
            for (sa, sb), cnt in top_edges:
                _p(f"  seg {sa} -- seg {sb} : contacts={cnt}, boundary_mean_miso_deg={rag_miso_mean[(sa, sb)]:.3f}")

        _p("=== STEP 3 COMPLETE ===")

        # -----------------------------------------------------------------------------
        # STEP 4 — MERGE / PRUNE USING UNION-FIND
        #   Rule A: merge neighbors if mean-orientation miso <= th_merge_deg
        #   Rule B: absorb tiny components (<= min_voxels) into best neighbor
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 4: MERGE / PRUNE SEGMENTS ===")
        _p("TH_MERGE (deg):", th_merge_deg, "MIN_VOXELS:", min_voxels)

        parent = np.arange(n_seg, dtype=int)
        rank = np.zeros(n_seg, dtype=int)

        # Rule A
        merge_A = 0
        for (sa, sb) in rag_contact.keys():
            miso_meanO = float(np.asarray(seg_mean_O[sa].angle_with(seg_mean_O[sb], degrees=True)).squeeze())
            if miso_meanO <= th_merge_deg:
                if _uf_union(parent, rank, sa, sb):
                    merge_A += 1
        _p("Rule A merges performed:", merge_A)

        # Build root members after Rule A
        root_members = {}
        for s in range(n_seg):
            r = _uf_find(parent, s)
            root_members.setdefault(r, []).append(s)

        # Root sizes
        root_size = {r: int(sum(seg_size[s] for s in members)) for r, members in root_members.items()}
        tiny_roots = [r for r, sz in root_size.items() if sz <= min_voxels]
        _p("tiny components after Rule A:", len(tiny_roots), "roots:", tiny_roots)

        # Root mean orientation (ORIX) from voxel sets
        root_mean_O = {}
        for r, members in root_members.items():
            mask = np.isin(labels, members)
            root_mean_O[r] = O[mask].mean()

        # Rule B: absorb each tiny root
        merge_B = 0
        for r0 in tiny_roots:
            r = _uf_find(parent, r0)  # update in case previous unions changed it
            if r not in root_size:
                continue
            if root_size[r] > min_voxels:
                continue

            best = None  # (miso_root_deg, -contact, nbr_root)

            for (sa, sb), contact in rag_contact.items():
                ra = _uf_find(parent, sa)
                rb = _uf_find(parent, sb)
                if ra == rb:
                    continue
                if ra != r and rb != r:
                    continue

                nbr = rb if ra == r else ra

                miso_root = float(np.asarray(root_mean_O[r].angle_with(root_mean_O[nbr], degrees=True)).squeeze())
                score = (miso_root, -int(contact), nbr)
                if (best is None) or (score < best):
                    best = score

            if best is None:
                continue

            _, _, nbr_root = best
            if _uf_union(parent, rank, r, nbr_root):
                merge_B += 1

                # Update bookkeeping (sizes + mean orientations) for stability
                newr = _uf_find(parent, r)
                oldr = nbr_root if newr == r else r  # whichever got absorbed

                root_size[newr] = root_size.get(r, 0) + root_size.get(nbr_root, 0)
                root_size.pop(oldr, None)

                # Recompute mean orientation for new root (safe and correct)
                # (voxel mask uses ORIGINAL segment labels; rebuild members by scanning is OK at this scale)
                members_new = [s for s in range(n_seg) if _uf_find(parent, s) == newr]
                mask_new = np.isin(labels, members_new)
                root_mean_O[newr] = O[mask_new].mean()

        _p("Rule B merges performed:", merge_B)

        # Finalize merged voxel labels (compact 0..n_final-1)
        root_to_new = {}
        labels_merged = np.empty_like(labels)
        next_id = 0
        for pos in range(N):
            s = int(labels[pos])
            r = _uf_find(parent, s)
            if r not in root_to_new:
                root_to_new[r] = next_id
                next_id += 1
            labels_merged[pos] = root_to_new[r]

        n_final = next_id
        final_sizes = np.bincount(labels_merged, minlength=n_final)

        _p("\nAfter merging:")
        _p("n_final_segments:", n_final)
        _p("sum(final_sizes) == N:", int(final_sizes.sum()), "==", N)
        _p("sorted final sizes:", np.sort(final_sizes)[::-1].tolist())
        _p("segments with <= 5 voxels:", int((final_sizes <= 5).sum()))
        _p("=== STEP 4 COMPLETE ===")

        # -----------------------------------------------------------------------------
        # STEP 5 — VALIDATE (post-merge boundary diagnostic)
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 5: POST-MERGE RAG VALIDATION ===")
        rag2_contact = {}
        rag2_sum = {}

        for i in range(E):
            a = int(labels_merged[int(edges_pos_u[i])])
            b = int(labels_merged[int(edges_pos_v[i])])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            key = (a, b)
            rag2_contact[key] = rag2_contact.get(key, 0) + 1
            rag2_sum[key] = rag2_sum.get(key, 0.0) + float(w_deg[i])

        rag2_mean = {k: rag2_sum[k] / rag2_contact[k] for k in rag2_contact}
        _p("post-merge RAG edges:", len(rag2_contact))

        if print_topk_rag > 0:
            top2 = sorted(rag2_contact.items(), key=lambda kv: kv[1], reverse=True)[:print_topk_rag]
            _p(f"\nTop {print_topk_rag} post-merge neighbors by contact:")
            for (a, b), cnt in top2:
                _p(f"  seg {a} -- seg {b} : contacts={cnt}, boundary_mean_miso_deg={rag2_mean[(a,b)]:.3f}")

        _p("=== STEP 5 COMPLETE ===")

        # -----------------------------------------------------------------------------
        # STEP 6/7 — UPDATE SNAPSHOT: voxels[*].grain_id + rebuild grains[]
        # -----------------------------------------------------------------------------
        _p("\n=== STEP 6/7: UPDATE JSON (voxels + grains) ===")

        # Write new grain_id (1-based) into voxels
        new_gid = labels_merged.astype(int) + 1
        for pos in range(N):
            voxels[pos]["grain_id"] = int(new_gid[pos])

        n_grains_out = int(new_gid.max())
        _p("n_final grains:", n_grains_out)
        _p("first 10 new grain_id:", new_gid[:10].tolist())

        # Rebuild grains[]
        new_grains = []
        for k in range(1, n_grains_out + 1):
            mask = (new_gid == k)
            if not np.any(mask):
                continue

            gv = float(voxel_vol[mask].sum())

            Omean = O[mask].mean()
            e_mean = np.asarray(Omean.to_euler(), dtype=float).reshape(3)

            phases = [grainid_to_phase[int(og)] for og in old_gid[mask] if int(og) in grainid_to_phase]
            phase_id = Counter(phases).most_common(1)[0][0] if phases else 0

            new_grains.append({
                "grain_id": int(k),
                "phase_id": int(phase_id),
                "grain_volume": gv,
                "orientation": [float(e_mean[0]), float(e_mean[1]), float(e_mean[2])],
            })

        new_grains.sort(key=lambda g: g["grain_id"])
        snap["grains"] = new_grains

        _p("rebuilt grains[] -> n_grains:", len(new_grains))
        _p("total grain_volume:", float(sum(g["grain_volume"] for g in new_grains)))
        _p("=== STEP 6/7 COMPLETE ===")

    # -----------------------------
    # Overwrite the same JSON file
    # -----------------------------
    _p("\n=== WRITE JSON (IN-PLACE) ===")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    _p("Updated in-place:", json_path)

def step1_extract_gid_arrays(json_path: str, t: int):
    """
    Step 1: Build aligned grain-id arrays for snapshots t and t+1 using voxel_index.

    Parameters
    ----------
    json_path : str
        Path to the microstructure JSON.
    t : int
        Snapshot index (tracks from t -> t+1).

    Returns
    -------
    gid_t : np.ndarray, shape (N,)
        Grain IDs for snapshot t aligned by voxel positions.
    gid_tp1 : np.ndarray, shape (N,)
        Grain IDs for snapshot t+1 aligned by voxel positions (same ordering as gid_t).
    idx_list : list[tuple[int,int,int]]
        The ordered voxel_index list that defines the alignment.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    micro = data["microstructure"]
    snap_t = micro[t]
    snap_tp1 = micro[t + 1]

    vox_t = snap_t["voxels"]
    vox_tp1 = snap_tp1["voxels"]

    # Build index -> grain_id dict for each snapshot
    map_t = {tuple(v["voxel_index"]): int(v["grain_id"]) for v in vox_t}
    map_tp1 = {tuple(v["voxel_index"]): int(v["grain_id"]) for v in vox_tp1}

    # Ensure same voxel set (critical assumption)
    idx_set_t = set(map_t.keys())
    idx_set_tp1 = set(map_tp1.keys())
    if idx_set_t != idx_set_tp1:
        missing_in_tp1 = idx_set_t - idx_set_tp1
        missing_in_t = idx_set_tp1 - idx_set_t
        raise ValueError(
            f"Voxel index sets differ between snapshots {t} and {t+1}.\n"
            f"Missing in t+1: {len(missing_in_tp1)}\n"
            f"Missing in t  : {len(missing_in_t)}"
        )

    # Define a stable order (sort by k, then j, then i OR your preferred)
    idx_list = sorted(idx_set_t)

    gid_t = np.array([map_t[idx] for idx in idx_list], dtype=np.int32)
    gid_tp1 = np.array([map_tp1[idx] for idx in idx_list], dtype=np.int32)

    return gid_t, gid_tp1, idx_list
def step2_overlap_counts(gid_t: np.ndarray, gid_tp1: np.ndarray):
    """
    Step 2: Compute overlap counts C(g_t, g_tp1).

    Parameters
    ----------
    gid_t, gid_tp1 : np.ndarray, shape (N,)
        Aligned grain-id arrays for snapshots t and t+1.

    Returns
    -------
    C : np.ndarray, shape (Gt+1, Gtp1+1)
        Overlap count matrix where C[i,j] = number of voxels shared.
        Indices are grain IDs directly (so row/col 0 unused if IDs start at 1).
    row_sum : np.ndarray
        Total voxels per grain in t (row sums).
    col_sum : np.ndarray
        Total voxels per grain in t+1 (col sums).
    top_pairs : list[tuple]
        Sorted list of (count, i, j) for the largest overlaps.
    """
    if gid_t.shape != gid_tp1.shape:
        raise ValueError("gid arrays must have same shape")

    Gt = int(gid_t.max())
    Gp = int(gid_tp1.max())

    # Build overlap matrix (IDs assumed positive ints)
    C = np.zeros((Gt + 1, Gp + 1), dtype=np.int32)

    # Fast accumulation (vectorized via bincount on combined index)
    # combined_index = i*(Gp+1) + j
    combined = gid_t.astype(np.int64) * (Gp + 1) + gid_tp1.astype(np.int64)
    bc = np.bincount(combined, minlength=(Gt + 1) * (Gp + 1))
    C[:, :] = bc.reshape((Gt + 1, Gp + 1))

    row_sum = C.sum(axis=1)  # voxels per grain at t
    col_sum = C.sum(axis=0)  # voxels per grain at t+1

    # Extract top overlaps ignoring 0 row/col
    pairs = []
    for i in range(1, Gt + 1):
        for j in range(1, Gp + 1):
            c = int(C[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    return C, row_sum, col_sum, pairs
def print_step2_summary(C, row_sum, col_sum, top_pairs):
    print("\n=== STEP 2: OVERLAP MATRIX SUMMARY ===")
    print("C shape:", C.shape, "(row 0/col 0 unused)")
    print("Total voxels check:", int(C.sum()))
    print("Grains t   :", int((row_sum[1:] > 0).sum()), "max id:", len(row_sum) - 1)
    print("Grains t+1 :", int((col_sum[1:] > 0).sum()), "max id:", len(col_sum) - 1)
    print("Overlap matrix:")
    print(C)

    print(f"\n Overlaps (count, g_t -> g_t+1, frac_of_t, frac_of_t+1):")
    for c, i, j in top_pairs:
        frac_t = c / row_sum[i] if row_sum[i] else 0.0
        frac_p = c / col_sum[j] if col_sum[j] else 0.0
        print(f"  {c:4d}   {i:2d} -> {j:2d}    {frac_t:6.3f}    {frac_p:6.3f}")

    # For each grain in t, print its best match in t+1
    print("\nBest match per grain in t (by overlap count):")
    for i in range(1, len(row_sum)):
        if row_sum[i] == 0:
            continue
        j_best = int(np.argmax(C[i, :]))
        c_best = int(C[i, j_best])
        frac_t = c_best / row_sum[i]
        frac_p = c_best / col_sum[j_best] if col_sum[j_best] else 0.0
        print(f"  g_t={i:2d}: best g_t+1={j_best:2d}  count={c_best:4d}  frac_t={frac_t:6.3f}  frac_p={frac_p:6.3f}")
def step3_track_from_overlap(C, row_sum, col_sum, *,
                             tau_parent=0.30,   # child must come >=30% from a parent to count as parent
                             tau_child=0.30,    # parent must send >=30% of its mass to child to count as child
                             allow_multi_parent=True):
    """
    Build tracking relations from overlap matrix.

    Returns
    -------
    parents_of_child : dict[int, list[tuple[int, float, float, int]]]
        child j -> list of (parent i, frac_of_parent, frac_of_child, count)
    best_parent : dict[int, int]
        child j -> parent i with max overlap
    best_child  : dict[int, int]
        parent i -> child j with max overlap
    events : dict[str, list]
        merge/split/birth/death/continue records
    """
    Gt  = C.shape[0] - 1
    Gp1 = C.shape[1] - 1

    # Candidate parent links per child
    parents_of_child = defaultdict(list)
    children_of_parent = defaultdict(list)

    for i in range(1, Gt + 1):
        if row_sum[i] == 0:
            continue
        for j in range(1, Gp1 + 1):
            c = int(C[i, j])
            if c == 0:
                continue
            f_parent = c / row_sum[i]
            f_child  = c / col_sum[j] if col_sum[j] else 0.0

            # Keep links that are meaningful from either side
            if (f_parent >= tau_child) or (f_child >= tau_parent):
                parents_of_child[j].append((i, f_parent, f_child, c))
                children_of_parent[i].append((j, f_parent, f_child, c))

    # Best matches by raw overlap
    best_parent = {}
    for j in range(1, Gp1 + 1):
        i_best = int(np.argmax(C[:, j]))
        best_parent[j] = i_best if i_best != 0 else None

    best_child = {}
    for i in range(1, Gt + 1):
        j_best = int(np.argmax(C[i, :]))
        best_child[i] = j_best if j_best != 0 else None

    # Event classification
    events = {"continue": [], "merge": [], "split": [], "birth": [], "death": []}

    # Births: child has no meaningful parent links
    for j in range(1, Gp1 + 1):
        if col_sum[j] == 0:
            continue
        if j not in parents_of_child:
            events["birth"].append(j)

    # Deaths: parent has no meaningful child links
    for i in range(1, Gt + 1):
        if row_sum[i] == 0:
            continue
        if i not in children_of_parent:
            events["death"].append(i)

    # Continuations + merges (child perspective)
    for j, plist in parents_of_child.items():
        # sort by overlap count descending
        plist = sorted(plist, key=lambda x: x[3], reverse=True)
        if len(plist) == 1:
            i = plist[0][0]
            # mutual best → strong continuation candidate
            if best_child.get(i, None) == j and best_parent.get(j, None) == i:
                events["continue"].append((i, j, plist[0]))
        else:
            # multiple parents → merge
            if allow_multi_parent:
                parents = [p[0] for p in plist]
                events["merge"].append((parents, j, plist))

    # Splits (parent perspective): one parent has multiple strong children
    for i, clist in children_of_parent.items():
        # keep only strong-from-parent links
        strong = [x for x in clist if x[1] >= tau_child]
        if len(strong) >= 2:
            strong = sorted(strong, key=lambda x: x[3], reverse=True)
            children = [x[0] for x in strong]
            events["split"].append((i, children, strong))

    return parents_of_child, best_parent, best_child, events

def step4_assign_tracked_ids(
    row_sum: np.ndarray,
    col_sum: np.ndarray,
    CONTINUE,
    MERGE,
    SPLIT,
    BIRTH,
    DEATH,
    *,
    next_track_id_start: int | None = None,
    prefer_merge_parent="largest_overlap",   # or "largest_parent"
    verbose: bool = True):
    """
    Step 4: Assign persistent/tracked IDs for grains at (t+1).

    Inputs are Step 3 outputs with the following formats:
      CONTINUE: [(i, j, (j, ft, fp, c)), ...]
      MERGE   : [([i1,i2,...], j, [(i, ft, fp, c), ...]), ...]
      SPLIT   : [(i, [j1,j2,...], [(j, ft, fp, c), ...]), ...]
      BIRTH   : [j, j, ...]
      DEATH   : [i, i, ...]

    Returns
    -------
    track_of_child : dict[int,int]
        Mapping child grain-id j (at t+1) -> tracked_id (persistent).
    events4 : dict
        Human-readable summary of what got assigned and why.
    next_track_id : int
        Next available tracked id (for the next time step).
    """

    def _p(*a):
        if verbose:
            print(*a)

    # Determine ID range at time t
    Gt = len(row_sum) - 1
    Gp = len(col_sum) - 1

    # Start new tracked IDs after max existing t-id unless user overrides
    if next_track_id_start is None:
        next_track_id = Gt + 1
    else:
        next_track_id = int(next_track_id_start)

    # Helper: parse structures into convenient dicts
    # cont_map: parent i -> child j
    cont_map = {i: j for (i, j, _) in CONTINUE}
    cont_inv = {j: i for (i, j, _) in CONTINUE}

    merge_children = {child_j: (parents, details) for (parents, child_j, details) in MERGE}

    # split_children_of_parent: parent i -> list of (child_j, c, ft, fp)
    split_children_of_parent = {}
    for (i, child_list, details) in SPLIT:
        # details is [(j, ft, fp, c), ...]
        dd = []
        for (j, ft, fp, c) in details:
            dd.append((j, int(c), float(ft), float(fp)))
        # sort by overlap count desc
        dd.sort(key=lambda x: x[1], reverse=True)
        split_children_of_parent[i] = dd

    # Track assignment output
    track_of_child: dict[int, int] = {}
    assigned_children = set()

    # Event summary
    events4 = {
        "continue": [],
        "merge": [],
        "split": [],
        "birth": [],
        "unassigned_children": [],
    }

    # ------------------------------------------------------------------
    # Priority 1: CONTINUE (1-1 stable identity) => child keeps parent id
    # ------------------------------------------------------------------
    for (i, j, info) in CONTINUE:
        track_of_child[j] = int(i)
        assigned_children.add(j)
        events4["continue"].append((i, j, info))

    # ------------------------------------------------------------------
    # Priority 2: MERGE (many -> one)
    # Choose ONE "dominant" parent to carry the ID.
    # ------------------------------------------------------------------
    for child_j, (parents, details) in merge_children.items():
        if child_j in assigned_children:
            continue  # already labeled by CONTINUE (rare but safe)

        # details: [(i, ft, fp, c), ...]
        # pick dominant parent
        if prefer_merge_parent == "largest_parent":
            # parent with largest row_sum (size at t)
            dom = max(parents, key=lambda i: row_sum[int(i)])
        else:
            # parent with largest overlap c into this child
            dom = max(details, key=lambda t: t[3])[0]  # t=(i,ft,fp,c)
            dom = int(dom)

        track_of_child[child_j] = int(dom)
        assigned_children.add(child_j)
        events4["merge"].append((parents, child_j, dom, details))

    # ------------------------------------------------------------------
    # Priority 3: SPLIT (one -> many)
    # Keep parent id for the strongest child; give new IDs to other children.
    # ------------------------------------------------------------------
    for parent_i, child_info in split_children_of_parent.items():
        # child_info sorted by overlap count desc: [(j, c, ft, fp), ...]
        if len(child_info) == 0:
            continue

        # strongest child
        j_main = int(child_info[0][0])

        # if not already assigned, give it the parent id
        if j_main not in assigned_children:
            track_of_child[j_main] = int(parent_i)
            assigned_children.add(j_main)

        # all remaining children get new IDs if still unassigned
        new_children = []
        for (j, c, ft, fp) in child_info[1:]:
            j = int(j)
            if j in assigned_children:
                continue
            track_of_child[j] = int(next_track_id)
            new_children.append((j, next_track_id, c, ft, fp))
            assigned_children.add(j)
            next_track_id += 1

        events4["split"].append((int(parent_i), int(j_main), child_info, new_children))

    # ------------------------------------------------------------------
    # Priority 4: BIRTH (no parent) => new tracked IDs
    # ------------------------------------------------------------------
    for j in BIRTH:
        j = int(j)
        if j in assigned_children:
            continue
        track_of_child[j] = int(next_track_id)
        events4["birth"].append((j, next_track_id))
        assigned_children.add(j)
        next_track_id += 1

    # ------------------------------------------------------------------
    # Safety: any remaining child grains (not covered by events) => new IDs
    # ------------------------------------------------------------------
    for j in range(1, Gp + 1):
        if col_sum[j] == 0:
            continue
        if j not in assigned_children:
            track_of_child[j] = int(next_track_id)
            events4["unassigned_children"].append((j, next_track_id))
            next_track_id += 1

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _p("\n=== STEP 4: TRACKED-ID ASSIGNMENT SUMMARY ===")
    _p("Children (t+1) grains:", int((col_sum[1:] > 0).sum()), "max id:", Gp)
    _p("Assigned children:", len(track_of_child))
    _p("Next track id:", next_track_id)

    _p("\nCONTINUE assignments:", len(events4["continue"]))
    for (i, j, info) in events4["continue"]:
        _p(f"  child {j:2d} keeps parent id {i:2d}   overlap={info[3]}")

    _p("\nMERGE assignments:", len(events4["merge"]))
    for (parents, child_j, dom, details) in events4["merge"]:
        det = sorted(details, key=lambda t: t[3], reverse=True)
        det_str = ", ".join([f"p{i}:{c}" for (i, ft, fp, c) in det])
        _p(f"  child {child_j:2d} gets dominant parent id {dom:2d}   parents={parents}  ({det_str})")

    _p("\nSPLIT assignments:", len(events4["split"]))
    for (pi, jmain, child_info, new_children) in events4["split"]:
        _p(f"  parent {pi:2d} -> main child {jmain:2d} keeps id {pi:2d}")
        for (j, new_id, c, ft, fp) in new_children:
            _p(f"    extra child {j:2d} gets NEW id {new_id:2d}   overlap={c}")

    _p("\nBIRTH assignments:", len(events4["birth"]))
    for (j, new_id) in events4["birth"]:
        _p(f"  child {j:2d} is BIRTH -> NEW id {new_id:2d}")

    if events4["unassigned_children"]:
        _p("\nUnassigned children -> NEW ids:", len(events4["unassigned_children"]))
        for (j, new_id) in events4["unassigned_children"]:
            _p(f"  child {j:2d} -> NEW id {new_id:2d}")

    # show final mapping (compact)
    _p("\nFinal mapping child grain_id (t+1) -> tracked_id:")
    for j in sorted(track_of_child):
        _p(f"  {j:2d} -> {track_of_child[j]:2d}")

    return track_of_child, events4, next_track_id

def step5_apply_tracking_to_snapshot(
    json_path: str,
    t: int,
    track_of_child: dict[int, int],   # child grain_id (t+1) -> tracked_grain_id
    *,
    events: dict,                     # events from step3_track_from_overlap
    best_parent: dict[int, int | None],
    C: np.ndarray,
    row_sum: np.ndarray,
    col_sum: np.ndarray,
    tracked_key: str = "tracked_grain_id",
    parent_key: str = "parent_grain_id",
    also_update_grains: bool = True,
    verbose: bool = True):
    """
    Step 5 (v2):
      - voxels: write tracked_grain_id only
      - grains : write tracked_grain_id + parent_grain_id
          * CONTINUE: parent_grain_id = parent tracked id (int)
          * SPLIT   : parent_grain_id = parent tracked id (int) for ALL split children
          * MERGE   : parent_grain_id = [parent tracked ids...] (list[int])
          * BIRTH   : parent_grain_id = None
          * UNASSIGNED: Option A -> parent_grain_id = best_parent[j] (tracked id) if overlap>0 else None
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    micro = data["microstructure"]
    if not (0 <= t < len(micro) - 1):
        raise ValueError(f"t must satisfy 0 <= t < {len(micro)-1}")

    snap = micro[t + 1]
    vox = snap["voxels"]

    # -----------------------------
    # 1) Build lineage maps (child grain_id -> parent tracked id(s))
    # Parents in events are already in the "parent space" of the overlap matrix:
    # - for t=0: parents are snapshot0 grain_id
    # - for t>=1: parents are snapshot t tracked_grain_id
    # -----------------------------
    merge_child_to_parents = {}   # child_j -> [parent_ids...]
    for (parents, child_j, plist) in events.get("merge", []):
        merge_child_to_parents[int(child_j)] = [int(p) for p in parents]

    split_child_to_parent = {}   # child_j -> parent_i
    for (parent_i, child_list, strong_details) in events.get("split", []):
        pi = int(parent_i)
        for cj in child_list:
            split_child_to_parent[int(cj)] = pi

    continue_child_to_parent = {}  # child_j -> parent_i
    for (parent_i, child_j, info) in events.get("continue", []):
        continue_child_to_parent[int(child_j)] = int(parent_i)

    birth_set = set(int(j) for j in events.get("birth", []))

    # -----------------------------
    # 2) Apply tracked IDs to voxels (identity label)
    # -----------------------------
    missing_gids = set()
    tracked_ids = np.empty(len(vox), dtype=int)

    for i, v in enumerate(vox):
        gid = int(v["grain_id"])
        tid = track_of_child.get(gid, None)
        if tid is None:
            missing_gids.add(gid)
            tid = -1
        v[tracked_key] = int(tid)     # ✅ voxel: tracked_grain_id only
        tracked_ids[i] = int(tid)

    # -----------------------------
    # 3) Apply lineage + identity to grains[] (if present)
    # -----------------------------
    if also_update_grains and snap.get("grains") is not None:
        for g in snap["grains"]:
            child_gid = int(g["grain_id"])

            # identity
            child_tid = int(track_of_child.get(child_gid, -1))
            g[tracked_key] = child_tid

            # lineage
            if child_gid in birth_set:
                g[parent_key] = None

            elif child_gid in merge_child_to_parents:
                # MERGE: list of parents (tracked ids)
                g[parent_key] = [int(p) for p in merge_child_to_parents[child_gid]]

            elif child_gid in split_child_to_parent:
                # SPLIT: scalar parent
                g[parent_key] = int(split_child_to_parent[child_gid])

            elif child_gid in continue_child_to_parent:
                # CONTINUE: scalar parent
                g[parent_key] = int(continue_child_to_parent[child_gid])

            else:
                # UNASSIGNED:
                # Option A -> use best_parent if overlap exists, else None
                p = best_parent.get(child_gid, None)
                if p is None or int(p) == 0 or int(col_sum[child_gid]) == 0:
                    g[parent_key] = None
                else:
                    # Ensure this is a real overlap (protect against argmax on all-zeros col)
                    if int(C[int(p), child_gid]) > 0:
                        g[parent_key] = int(p)
                    else:
                        g[parent_key] = None

    if verbose:
        print("=== STEP 5 (v2): APPLY TRACKING + LINEAGE ===")
        print(f"json_path      : {json_path}")
        print(f"applied to     : snapshot {t+1} (t={t} -> t+1)")
        print(f"voxel key      : {tracked_key} (identity)")
        print(f"grain keys     : {tracked_key} + {parent_key} (lineage)")
        print(f"voxels updated : {len(vox)}")
        if missing_gids:
            print(f"WARNING: unmapped grain_id(s) in t+1 -> wrote -1 for {len(missing_gids)} grains:", sorted(missing_gids)[:20])
        else:
            print("OK: all voxels mapped.")

        c = Counter(tracked_ids.tolist())
        if -1 in c:
            c.pop(-1)
        print("unique tracked ids:", len(c), "top5:", c.most_common(5))

        # quick lineage counts (grains only)
        if also_update_grains and snap.get("grains") is not None:
            n_merge = sum(isinstance(g.get(parent_key, None), list) for g in snap["grains"])
            n_scalar = sum(isinstance(g.get(parent_key, None), int) for g in snap["grains"])
            n_none = sum(g.get(parent_key, None) is None for g in snap["grains"])
            print(f"lineage @ grains: scalar={n_scalar}, merge(list)={n_merge}, none={n_none}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return tracked_ids


def track_all_grains_across_snapshots(
        folder_path: str,
        *,
        json_name: str = "data_object.json",
        tracked_key: str = "tracked_grain_id",
        parent_key: str = "parent_grain_id",
        tau_parent: float = 0.20,
        tau_child: float = 0.20,
        allow_multi_parent: bool = True,
        prefer_merge_parent: str = "largest_overlap",
        verbose: bool = True):
    """
    Track grains across all snapshots, leaving snapshot 0 unchanged (baseline).

    - Snapshot 0: only grain_id (no tracked_key written)
    - Snapshot t>=1: has both grain_id (local) and tracked_key (persistent)
    - For each transition t->t+1:
        parents use tracked_key if present else grain_id
        children use grain_id
        tracked_key is written into snapshot t+1
    """

    def _p(*a):
        if verbose:
            print(*a)

    def print_full_matrix(C, *, precision=0):
        with np.printoptions(
                threshold=np.inf,  # never summarize with "..."
                linewidth=200,  # wider lines before wrapping
                edgeitems=30,  # irrelevant when threshold=inf, but fine
                suppress=True,
                precision=precision):
            print("Overlap Matrix:")
            print(C)

    # -----------------------------
    # Resolve JSON path
    # -----------------------------
    if os.path.isfile(folder_path):
        json_path = folder_path
    else:
        candidate = os.path.join(folder_path, json_name)
        if os.path.isfile(candidate):
            json_path = candidate
        else:
            hits = sorted(glob.glob(os.path.join(folder_path, "*.json")))
            if not hits:
                raise FileNotFoundError(f"No JSON found in: {folder_path}")
            json_path = hits[0]

    with open(json_path, "r") as f:
        data = json.load(f)

    if "microstructure" not in data:
        raise KeyError("Top-level key 'microstructure' not found.")
    micro = data["microstructure"]
    n_snaps = len(micro)
    if n_snaps < 2:
        raise ValueError("Need at least 2 snapshots to track.")

    # -----------------------------
    # Baseline: snapshot 0 untouched
    # next_track_id starts after snapshot0 grain_id max
    # -----------------------------
    snap0 = micro[0]
    if "voxels" not in snap0 or not snap0["voxels"]:
        raise KeyError("Snapshot 0 missing voxels.")

    max0 = max(int(v["grain_id"]) for v in snap0["voxels"])
    next_track_id = max0 + 1

    _p("=== TRACK GRAINS ACROSS ALL SNAPSHOTS ===")
    _p("File:", json_path)
    _p(f"Snapshots: {n_snaps} (will track 0->1->...->last)")
    _p(f"Init snapshot 0 baseline: grain_id only, max_id={max0}, next_track_id={next_track_id}\n")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _aligned_array_for_snapshot(snap, idx_list, key, fallback_key):
        # build dict: voxel_index -> value
        m = {tuple(v["voxel_index"]): int(v.get(key, v.get(fallback_key))) for v in snap["voxels"]}
        return np.array([m[idx] for idx in idx_list], dtype=np.int32)

    def _unique_count_nonneg(arr):
        u = np.unique(arr)
        u = u[u >= 1]
        return int(u.size)

    tracked_counts = [0] * n_snaps
    tracked_counts[0] = _unique_count_nonneg(
        np.array([int(v["grain_id"]) for v in micro[0]["voxels"]], dtype=np.int32)
    )

    totals = Counter()
    per_step_events = []

    # -----------------------------
    # Main loop
    # -----------------------------
    for t in range(0, n_snaps - 1):
        _p("=" * 90)
        _p(f"TRACK: snapshots {t} -> {t + 1}")
        _p("=" * 90)

        # Step 1 gives idx_list + validates voxel sets match
        _p("[STEP 1] aligned arrays")
        _, _, idx_list = step1_extract_gid_arrays(json_path, t)
        N = len(idx_list)

        # Reload to read current state (because step5 modifies json)
        with open(json_path, "r") as f:
            data = json.load(f)
        micro = data["microstructure"]
        snap_t = micro[t]
        snap_tp1 = micro[t + 1]

        has_tracked = bool(snap_t.get("voxels")) and (tracked_key in snap_t["voxels"][0])

        # Parents: tracked_key if exists else grain_id (baseline at t=0)
        gid_t = _aligned_array_for_snapshot(snap_t, idx_list, tracked_key, "grain_id")
        # Children: always local grain_id
        gid_tp1 = _aligned_array_for_snapshot(snap_tp1, idx_list, "grain_id", "grain_id")

        _p("  N voxels      :", N)
        _p("  parents ids   :", int(gid_t.min()), "->", int(gid_t.max()),
           f"({'tracked' if tracked_key in snap_t['voxels'][0] else 'grain_id baseline'})")
        _p("  children gids :", int(gid_tp1.min()), "->", int(gid_tp1.max()))

        # Step 2
        _p("[STEP 2] overlap")
        C, row_sum, col_sum, _ = step2_overlap_counts(gid_t, gid_tp1)
        _p("  C shape       :", C.shape, " total voxels:", int(C.sum()))
        if int(C.sum()) != N:
            raise RuntimeError(f"Overlap sum mismatch at t={t}: C.sum()={int(C.sum())} vs N={N}")
        #_p("=" * 40)
        #print_full_matrix(C)
        #_p("=" * 40)

        # Step 3
        _p("[STEP 3] events")
        parents_of_child, best_parent, best_child, events = step3_track_from_overlap(
            C, row_sum, col_sum,
            tau_parent=tau_parent,
            tau_child=tau_child,
            allow_multi_parent=allow_multi_parent,
        )
        c_continue = len(events["continue"])
        c_merge = len(events["merge"])
        c_split = len(events["split"])
        c_birth = len(events["birth"])
        c_death = len(events["death"])
        #_p("MERGE:", events["merge"])
        #_p("SPLITS:", events["split"])
        _p(f"  CONTINUE: {c_continue}  MERGE: {c_merge}  SPLIT: {c_split}  BIRTH: {c_birth}  DEATH: {c_death}")

        # Step 4 (child grain_id -> persistent id)
        _p("[STEP 4] assign persistent IDs")
        track_of_child, _, next_track_id = step4_assign_tracked_ids(
            row_sum=row_sum,
            col_sum=col_sum,
            CONTINUE=events["continue"],
            MERGE=events["merge"],
            SPLIT=events["split"],
            BIRTH=events["birth"],
            DEATH=events["death"],
            next_track_id_start=next_track_id,
            prefer_merge_parent=prefer_merge_parent,
            verbose=False,
        )
        _p("  children mapped:", len(track_of_child))
        _p("  next_track_id  :", next_track_id)

        # Step 5: write tracked_key into snapshot t+1 only
        _p(f"[STEP 5] write {tracked_key} into snapshot {t + 1}")

        tracked_ids = step5_apply_tracking_to_snapshot(json_path=json_path, t=t,
                                                       track_of_child=track_of_child,
                                                       events=events,  # from step3
                                                       best_parent=best_parent,  # from step3
                                                       C=C, row_sum=row_sum, col_sum=col_sum,
                                                       tracked_key=tracked_key,
                                                       parent_key=parent_key,
                                                       also_update_grains=True,
                                                       verbose=False, )

        ok = not np.any(tracked_ids == -1)
        _p("  OK: all voxels mapped." if ok else "  WARNING: some voxels mapped to -1")

        # refresh counts
        with open(json_path, "r") as f:
            data = json.load(f)
        micro = data["microstructure"]
        tids_tp1 = np.array([int(v.get(tracked_key, -1)) for v in micro[t + 1]["voxels"]], dtype=np.int32)
        tracked_counts[t + 1] = _unique_count_nonneg(tids_tp1)
        top5 = [(k, v) for k, v in Counter(tids_tp1.tolist()).most_common(5) if k != -1]
        _p(f"  unique tracked ids: {tracked_counts[t + 1]}  top5:", top5)

        per_step_events.append({
            "t": t, "tp1": t + 1,
            "continue": c_continue, "merge": c_merge, "split": c_split, "birth": c_birth, "death": c_death,
            "tracked_count_tp1": tracked_counts[t + 1],
        })
        totals.update({"continue": c_continue, "merge": c_merge, "split": c_split, "birth": c_birth, "death": c_death})

    # Summary
    _p("\n" + "=" * 90)
    _p("SUMMARY (totals across all transitions)")
    _p("=" * 90)
    _p(dict(totals))

    # Plot
    # plt.figure()
    # plt.bar(np.arange(n_snaps), tracked_counts)
    # plt.xlabel("Snapshot index")
    # plt.ylabel(f"# unique {tracked_key} (snapshot 0 uses grain_id baseline)")
    # plt.title("Tracked grain IDs across snapshots")
    # plt.tight_layout()
    # plt.show()

    _p(f"\nDONE: updated in-place: {json_path}")

    return {
        "json_path": json_path,
        "tracked_key": tracked_key,
        "tracked_counts": tracked_counts,
        "per_step_events": per_step_events,
        "totals": dict(totals),
    }
