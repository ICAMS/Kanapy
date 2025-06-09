"""
Tools for analysis of EBSD maps in form of .ang files

@author: Alexander Hartmaier, Abhishek Biswas, ICAMS, Ruhr-Universität Bochum
"""
import os
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from kanapy.util import MTEX_DIR, ROOT_DIR, MAIN_DIR
from kanapy.rve_stats import bbox
from orix import io, quaternion
from orix import plot as ox_plot
from orix.vector import Miller
from scipy.stats import lognorm, vonmises
from scipy.spatial import ConvexHull
from skimage.segmentation import mark_boundaries


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
    sym = G.graph['symmetry']
    ori0 = quaternion.Orientation(G.nodes[nn]['ori_av'], sym)
    on = []
    ng = []
    for neigh in G.adj[nn]:
        ang = ori0.angle_with(quaternion.Orientation(G.nodes[neigh]['ori_av'], sym))[0]
        on.append(ang)
        ng.append(neigh)
    nn = np.argmin(on)
    return ng[nn], on[nn]


def build_graph_from_labeled_pixels(label_array, emap, phase, connectivity=8):
    labels, counts = np.unique(label_array, return_counts=True)
    nodes = []
    for i, lbl in enumerate(labels):
        info_dict = dict()
        info_dict['npix'] = counts[i]
        ix, iy = np.nonzero(label_array == lbl)
        ind = np.ravel_multi_index((ix, iy), label_array.shape)
        info_dict['pixels'] = ind
        info_dict['ori_av'] = np.mean(emap.rotations.data[ind, :], axis=0)
        info_dict['ori_std'] = np.std(emap.rotations.data[ind, :], axis=0)
        nodes.append((lbl, info_dict))

    print(f'Building microstructure graph with {len(nodes)} nodes (grains).')
    G = nx.Graph(label_map=label_array, symmetry=emap.phases.point_groups[phase],
                 dx=emap.dx, dy=emap.dy)
    G.add_nodes_from(nodes)

    print('Adding edges (grain boundaries) to microstructure graph.')
    rows, cols = label_array.shape
    for x in range(rows):
        for y in range(cols):
            label_here = label_array[x, y]
            for px, py in neighbors(x, y, connectivity=connectivity):
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


class EBSDmap:
    """Class to store attributes and methods to import EBSD maps
    and filter out their statistical data needed to generate
    synthetic RVEs
    """

    def __init__(self, fname, gs_min=10.0, vf_min=0.03, max_angle=5.0, connectivity=8,
                 show_plot=True, show_hist=None, felzenszwalb=False, show_grains=False,
                 hist=None, plot=None):
        """
        Generate microstructural data from EBSD maps

        Parameters
        ----------
        fname : str
            filname incl. path to EBSD file.
        matname : str, optional
            Name of material, depracted. The default is None.
        gs_min : int, optional
            Minimum grain size in pixels, smaller grains will be disregarded
            for the statistical analysis. The default is 3.
        vf_min : int, optional
            Minimum volume fracture, phases with smaller values will be
            disregarded. The default is 0.03.
        plot : bool, optional
            Plot microstructures. The default is True.

        Returns
        -------
        None.
        
        Attributes
        ----------
        ms_data : list of dict
            List of dictionaries with phase specific microstructure
            information.  
            name : name of phase  
            vf : volume fraction
            ngrain : number of grains in phase
            ori : matlab object with grain orientations
            cs : matlab object with crystal structure
            grains : matlab object with grains in each phase
            omega : orintations of major grain axis
            gs_param : statistical grain size parameters
            gs_data : grain sizesvonmises
            ar_param
            ar_data
            om_param
            om_data
            
            
            
        eng : handle for Matlab engine with MTEX data

        """

        def reassign(pix, ori, phid, bads):
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
            data['ori'] = quaternion.Orientation.from_euler(ori_e)
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
                # area = node['npix'] * self.dx * self.dy
                # eqd = 2.0 * (area / np.pi)**0.5
                pts = hull.points[hull.vertices]  # outer nodes of grain
                # find bounding box to hull points
                ea, eb, va, vb = bbox(pts, two_dim=True, return_vector=True)
                # assert np.dot(va, vb) < 1.e-9
                # assert np.isclose(np.linalg.norm(va), 1.0)
                omega = np.acos(va[0])  # angle of major axis against y-axis of map [0, pi]
                node['max_dia'] = ea
                node['min_dia'] = eb
                node['equ_dia'] = eqd
                node['maj_ax'] = va
                node['min_ax'] = vb
                node['omega'] = omega
                node['center'] = np.mean(hull.points, axis=0)
                arr_a.append(ea)
                arr_b.append(eb)
                arr_eqd.append(eqd)
                arr_om.append(omega)
            arr_om = np.array(arr_om)
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
                x = np.linspace(min(arr_eqd), max(arr_eqd), 150)
                y = lognorm.pdf(x, dsig, loc=doffs, scale=dscale)
                ax.plot(x, y, '-r', label='fit')
                ax.hist(arr_eqd, bins=20, density=True, label='data')
                plt.legend()
                plt.title('Histogram of grain equivalent diameters')
                plt.xlabel('Equivalent diameter (micron)')
                plt.ylabel('Normalized frequency')
                plt.show()

            # grain aspect ratio
            asp = np.zeros_like(arr_a)
            for i, va in enumerate(arr_a):
                xu = max(va, arr_b[i])
                xl = min(va, arr_b[i])
                asp[i] = xu / xl
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
                x = np.linspace(np.amin(asp), np.amax(asp), 150)
                y = lognorm.pdf(x, asig, loc=aoffs, scale=ascale)
                ax.plot(x, y, '-r', label='fit lognorm')
                ax.hist(asp, bins=20, density=True, label='density')
                plt.legend()
                plt.title('Histogram of grain aspect ratio')
                plt.xlabel('aspect ratio')
                plt.ylabel('normalized frequency')
                plt.show()

            # angles of main axis
            # fit von Mises distribution (circular normal distribution) to data
            omega_p = 2.0 * arr_om - np.pi  # rescale angles from [0, pi] to [-pi,pi] for von Mises distr. fit
            kappa, oloc, oscale = vonmises.fit(omega_p)
            med_om = vonmises.median(kappa, loc=oloc)  # scale parameter has no effect on vonmises distribution
            std_om = vonmises.std(kappa, loc=oloc)
            data['om_param'] = np.array([kappa, oloc])
            data['om_data'] = arr_om  # omega_p
            data['om_moments'] = [med_om, std_om]
            if show_hist:
                fig, ax = plt.subplots()
                x = np.linspace(-np.pi, np.pi, 200)  # np.amin(omega), np.amax(omega), 150)
                y = vonmises.pdf(x, kappa, loc=oloc)
                ax.plot(0.5 * (x + np.pi), 2 * y, '-r', label='fit')
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
            self.plot_inverse_pole_figure()

        if show_grains:
            self.plot_grains()

        if felzenszwalb:
            self.plot_felsenszwalb()
        return

    def plot_ci_map(self):
        if 'ci' in self.emap.prop.keys():
            plt.imshow(self.ci_map.reshape((self.sh_x, self.sh_y)))
            plt.title('CI values in EBSD map')
            plt.colorbar(label="CI")
            plt.show()

    def plot_inverse_pole_figure(self):
        # plot inverse pole figure
        # <111> poles in the sample reference frame
        for n_ph, ind in enumerate(self.emap.phases.ids):
            data = self.ms_data[n_ph]
            t_fe = Miller(uvw=[0, 0, 1], phase=self.emap.phases[ind]).symmetrise(unique=True)
            t_fe_all = data['ori'].inv().outer(t_fe)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="stereographic")
            ax.scatter(t_fe_all)
            ax.set_labels("X", "Y", None)
            ax.set_title(data['name'] + r" $\left<001\right>$ PF")
            plt.show()

    def plot_grains_marked(self):
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
        for data in self.ms_data:
            plt.imshow(data['mo_map'].reshape((self.sh_x, self.sh_y)))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation angle wrt reference")
            plt.colorbar(label="Misorientation (rad)")
            plt.show()

    def plot_segmentation(self, show_mo=True, show_ipf=True):
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
        from skimage.segmentation import felzenszwalb
        for data in self.ms_data:
            gscale_map = data['mo_map'].reshape((self.sh_x, self.sh_y)) / np.pi
            labels_fz = felzenszwalb(gscale_map, scale=300, sigma=0.6, min_size=25)  # sc=300, sig=0.8, min_s=25
            plt.imshow(mark_boundaries(gscale_map, labels_fz))
            plt.title(f"Phase #{data['index']} ({data['name']}): Misorientation map with Felzenszwalb segmentation")
            plt.show()

    def plot_graph(self):
        # visualize graph
        for data in self.ms_data:
            visualize_graph(data['graph'])

    def plot_ipf_map(self):
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
