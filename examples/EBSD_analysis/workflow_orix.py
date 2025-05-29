import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from kanapy import bbox
from scipy.spatial import ConvexHull
from orix import io, plot, quaternion
from orix.vector import Miller


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
    if connectivity == 1 or connectivity == 4:
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    else:
        return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                if not (i == 0 and j == 0)]


def merge_nodes(G, node1, node2):
    # merge pixel lists
    G.nodes[node2]['pixels'] = np.concatenate((G.nodes[node1]['pixels'], G.nodes[node2]['pixels']))
    G.nodes[node2]['npix'] = len(G.nodes)  # update length
    if 'hull' in G.nodes[node2].keys():
        # update hull if it exists already
        sh = G.graph['label_map'].shape
        pts = np.array(np.unravel_index(G.nodes[node2]['pixels'], sh)).T
        G.nodes[node2]['hull'] = ConvexHull(pts)
    # add new edges (will ignore if edge already exists)
    for neigh in G.adj[node1]:
        if node2 != neigh:
            G.add_edge(node2, neigh)
    G.remove_node(num)  # remove grain1 and all its edges
    # update label map
    ix, iy = np.nonzero(G.graph['label_map'] == num)
    G.graph['label_map'][ix, iy] = num_ln


def find_largest_neighbor(G, node):
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


def find_similar_regions(array, tolerance=0.087, connectivity=1):
    """
    Identifies connected regions of similar values in a 2D array.

    Parameters:
        array (ndarray): 2D NumPy array with values.
        tolerance (float): Max allowed difference between connected values.
        connectivity (int): 1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        labeled_array (ndarray): 2D array of region labels.
        num_features (int): Total number of connected regions found.
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


def build_graph_from_labeled_pixels(label_array, connectivity=8):
    labels, counts = np.unique(label_array, return_counts=True)
    nodes = []
    for i, lbl in enumerate(labels):
        info_dict = dict()
        info_dict['npix'] = counts[i]
        ix, iy = np.nonzero(label_array == lbl)
        info_dict['pixels'] = np.ravel_multi_index((ix, iy), label_array.shape)
        nodes.append((lbl, info_dict))

    G = nx.Graph(label_map=label_array)
    G.add_nodes_from(nodes)

    rows, cols = label_array.shape
    for x in range(rows):
        for y in range(cols):
            label_here = label_array[x, y]
            for px, py in neighbors(x, y, connectivity=8):
                if 0 <= px < rows and 0 <= py < cols:
                    neighbor_label = label_array[px, py]
                    if neighbor_label != label_here:
                        G.add_edge(label_here, neighbor_label)
    return G


def visualize_graph(G, node_size=100, fs=12):
    pos = nx.spring_layout(G, seed=42)  # positioning
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=node_size, font_size=fs)
    plt.show()


def export_graph(G, filename, format="graphml"):
    if format == "graphml":
        nx.write_graphml(G, filename)
    elif format == "gexf":
        nx.write_gexf(G, filename)
    else:
        raise ValueError("Only 'graphml' or 'gexf' formats are supported.")


fname = "ebsd_316L_500x500.ang"

show_plot = True  # show plots
vf_min = 0.03  # minimum volume fraction of phases to be considered
max_angle = 5 * np.pi / 180  # maximum misoriantation angle within one grain
min_size = 10  # minim grain size in pixels
connectivity = 8  # take into account diagonal neighbors

emap = io.load(fname)
sh_x, sh_y = emap.shape

# determine number of phases and generate histogram
Nphase = len(emap.phases.ids)  # number of phases
offs = 0 if 0 in emap.phases.ids else 1  # in CTX maps, there is no phase "0"
npx = emap.size  # total number of pixels in EBSD map
if sh_x * sh_y != npx:
    raise ValueError(f"Size of map ({npx} px) does not match its shape: {sh_x, sh_y}")

phist = np.histogram(emap.phase_id, Nphase + offs)

if show_plot and 'ci' in emap.prop.keys():
    val = emap.prop['ci']
    if len(val) == npx:
        bmap = val
    else:
        bmap = np.zeros(npx)
        bmap[emap.phase_id == 0] = val
    plt.imshow(bmap.reshape((sh_x, sh_y)))
    plt.title('CI values in EBSD map')
    plt.colorbar(label="CI")
    plt.show()

# read phase names and calculate volume fractions and plots if active
ms_data = []
for i, ind in enumerate(emap.phases.ids):
    if ind == -1:
        continue
    vf = phist[0][i + offs] / npx
    if vf < vf_min:
        continue
    data = dict()  # initialize data dictionary
    data['vf'] = vf
    data['name'] = emap.phases.names[i]
    data['index'] = ind

    # generate phase-specific orientations
    ori_e = emap[emap.phase_id == ind].orientations.in_euler_fundamental_region()
    data['ori'] = quaternion.Orientation.from_euler(ori_e)
    data['cs'] = emap.phases[ind].point_group.laue

    # plot different EBSD data maps
    if show_plot:
        # plot misorientation field
        val = data['ori'].angle
        if len(val) == npx:
            bmap = val
        else:
            bmap = np.zeros(npx)
            bmap[emap.phase_id == ind] = val
        plt.imshow(bmap.reshape((sh_x, sh_y)))
        plt.title('Misorientation angle wrt reference')
        plt.colorbar(label="Misorientation (rad)")
        plt.show()

        # plot EBSD map for current phase
        # Get IPF colors
        ipf_key = plot.IPFColorKeyTSL(data['cs'])
        rgb_val = ipf_key.orientation2color(data['ori'])
        # set pixels of other phases to black
        rgb_all = np.zeros((npx, 3))
        rgb_all[emap.phase_id == ind] = rgb_val
        fig = emap.plot(
            rgb_all,
            return_figure=True,
            figure_kwargs={"figsize": (12, 8)},
        )
        fig.show()

        # plot inverse pole figure
        # <111> poles in the sample reference frame
        t_fe = Miller(uvw=[1, 1, 1], phase=emap.phases[ind]).symmetrise(unique=True)
        t_fe_all = data['ori'].inv().outer(t_fe)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="stereographic")
        ax.scatter(t_fe_all)
        ax.set_labels("X", "Y", None)
        ax.set_title(data['name'] + r" $\left<111\right>$ PF")
        plt.show()

    # generate map with grain labels
    val = data['ori'].angle
    if len(val) == npx:
        array = val
    else:
        array = np.zeros(npx)
        array[emap.phase_id == data['index']] = val
    labels, n_regions = find_similar_regions(array.reshape((sh_x, sh_y)),
                                             tolerance=max_angle, connectivity=connectivity)

    # build and visualize graph of unfiltered map
    ms_graph = build_graph_from_labeled_pixels(labels)
    ms_graph.name = 'Graph of microstructure'
    ngrains_raw = len(ms_graph.nodes)
    print(f"Phase #{data['index']} ({data['name']}): Identified Grains: {ngrains_raw}")
    # plot labeled grains
    cols = get_distinct_colormap(ngrains_raw, cmap='prism')
    cmap = LinearSegmentedColormap.from_list('segs', cols, N=ngrains_raw)
    plt.imshow(ms_graph.graph['label_map'] / ngrains_raw, cmap=cmap)
    plt.title(f"Phase #{data['index']} ({data['name']}): Raw Grains: {ngrains_raw}")
    plt.colorbar(label='Grain Number')
    plt.show()
    # show graph
    visualize_graph(ms_graph)

    # graph pruning step 1: merge small grains into their largest neighbor grain
    grain_set = set(ms_graph.nodes)
    rem_grains = len(grain_set)
    while rem_grains > 0:
        num = grain_set.pop()  # get random ID of grain and remove it from the list
        rem_grains = len(grain_set)
        if ms_graph.nodes[num]['npix'] < min_size:
            num_ln = find_largest_neighbor(ms_graph, num)
            merge_nodes(ms_graph, num, num_ln)

    ngrains = len(ms_graph.nodes)
    print(f'After elimination of small grains, {ngrains} grains left.')

    # graph pruning step 2: remove grains that have no convex hull (pixels along GBs)
    grain_set = set(ms_graph.nodes)
    rem_grains = len(grain_set)
    while rem_grains > 0:
        num = grain_set.pop()  # get random ID of grain and remove it from the list
        nd = ms_graph.nodes[num]  # node to be considered
        rem_grains = len(grain_set)
        pts = np.array(np.unravel_index(nd['pixels'], (sh_x, sh_y))).T
        try:
            hull = ConvexHull(pts)
            ms_graph.nodes[num]['hull'] = hull
            continue  # grain has convex hull
        except Exception as e:
            num_ln = find_largest_neighbor(ms_graph, num)
            merge_nodes(ms_graph, num, num_ln)
            nd2 = ms_graph.nodes[num_ln]

    ngrains = len(ms_graph.nodes)
    print(f'After elimination of non-convex grains, {ngrains} grains left.')
    # plot filtered labeled grains
    cols = get_distinct_colormap(ngrains)
    cmap = LinearSegmentedColormap.from_list('segs', cols, N=ngrains)
    plt.imshow(ms_graph.graph['label_map'] / ngrains, cmap=cmap)
    plt.title(f"Phase #{data['index']} ({data['name']}): Filtered Grains: {ngrains}")
    plt.colorbar(label='Grain Number')
    plt.show()
    # visualize purified graph
    visualize_graph(ms_graph)

    # Extract grain statistics and axes
    arr_a = []
    arr_b = []
    arr_eqd = []
    for num, node in ms_graph.nodes.items():
        hull = node['hull']
        eqd = 2.0 * (hull.volume/np.pi) ** 0.5
        pts = hull.points[hull.vertices]  # outer nodes of grain
        # find bounding box to hull points
        ea, eb, va, vb = bbox(pts, two_dim=True, return_vector=True)
        node['max_dia'] = ea
        node['min_dia'] = eb
        node['equ_dia'] = eqd
        node['maj_ax'] = va
        node['min_ax'] = vb
        node['center'] = np.mean(hull.points, axis=0)
        arr_a.append(ea)
        arr_b.append(eb)
        arr_eqd.append(eqd)
    print('\n--------------------------------------------------------')
    print('Statistical microstructure parameters in pixel map ')
    print('--------------------------------------------------------')
    print(np.median(arr_a), np.std(arr_a))
    print(np.median(arr_b), np.std(arr_b))
    print(np.median(arr_eqd), np.std(arr_eqd))

    # plot grains with axes
    cols = get_distinct_colormap(ngrains)
    cmap = LinearSegmentedColormap.from_list('segs', cols, N=ngrains)
    plt.imshow(ms_graph.graph['label_map'] / ngrains, cmap=cmap)
    for num, node in ms_graph.nodes.items():
        ctr = ms_graph.nodes[num]['center']
        plt.annotate(str(num), xy=(ctr[1], ctr[0]))
        pts = np.zeros((4, 2))
        pts[0, :] = node['center'] - node['max_dia'] * node['maj_ax']
        pts[1, :] = node['center'] + node['max_dia'] * node['maj_ax']
        pts[2, :] = node['center'] - node['min_dia'] * node['min_ax']
        pts[3, :] = node['center'] + node['min_dia'] * node['min_ax']
        plt.plot(pts[0:2, 1], pts[0:2, 0], color='k')
        plt.plot(pts[2:4, 1], pts[2:4, 0], color='red')
    plt.title(f"Phase #{data['index']} ({data['name']}): Grain labels and axes: {ngrains}")
    plt.colorbar(label='Grain Number')
    plt.show()

    ms_data.append(data)
