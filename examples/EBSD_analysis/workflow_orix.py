import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from orix import io, plot, quaternion
from orix.vector import Miller
from copy import deepcopy


def neighbors(r, c, connectivity):
    nlist = []
    if connectivity == 1:
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    else:
        return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                if not (i == 0 and j == 0)]


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


fname = "ebsd_316L_500x500.ang"

show_plot = True  # show plots
vf_min = 0.03  # minimum volume fraction of phases to be considered
max_angle = 5 * np.pi / 180  # maximum misoriantation angle within one grain
min_size = 5  # minim grain size in pixels
connectivity = 2  # take into account diagonal neighbors


emap = io.load(fname)
h, w = emap.shape

# determine number of phases and generate histogram
Nphase = len(emap.phases.ids)  # number of phases
offs = 0 if 0 in emap.phases.ids else 1  # in CTX maps, there is no phase "0"
npx = emap.size  # total number of pixels in EBSD map
if h*w != npx:
    raise ValueError(f"Size of map ({npx} px) does not match its shape: {h, w}")

phist = np.histogram(emap.phase_id, Nphase + offs)

if show_plot and 'ci' in emap.prop.keys():
    val = emap.prop['ci']
    if len(val) == npx:
        bmap = val
    else:
        bmap = np.zeros(npx)
        bmap[emap.phase_id == 0] = val
    plt.imshow(bmap.reshape((h, w)))
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

    val = data['ori'].angle
    if len(val) == npx:
        array = val
    else:
        array = np.zeros(npx)
        array[emap.phase_id == data['index']] = val
    labels, n_regions = find_similar_regions(array.reshape((h, w)),
                                             tolerance=max_angle, connectivity=connectivity)
    grains, counts = np.unique(labels, return_counts=True)
    print(f"Phase #{data['index']} ({data['name']}): Identified Grains: {n_regions}")

    # merge small grains into neighbor grains with most neighbor pixels
    nlab = deepcopy(labels)
    for ng, ct in enumerate(counts):
        if ct >= min_size:
            continue
        ix, iy = np.nonzero(labels == grains[ng])
        assert len(ix) == ct
        nlist = []
        for i in range(ct):
            neigh = neighbors(ix[i], iy[i], connectivity)
            for pos in neigh:
                if 0 <= pos[0] < h and 0 <= pos[1] < w:
                    nlist.append(labels[pos])
        nn, num = np.unique(nlist, return_counts=True)
        new_grain = nn[np.argmax(num)]
        nlab[ix, iy] = new_grain
    grains, counts = np.unique(nlab, return_counts=True)
    ngrains = len(grains)
    print(f'After elimination of small grains, {ngrains} grains left.')
    # redistribute grain numbers in proper sequence
    for i, ng in enumerate(grains):
        ix, iy = np.nonzero(nlab == ng)
        labels[ix, iy] = i + 1

    """Extract grain statistics"""

    if show_plot:
        # plot misorientation field
        val = data['ori'].angle
        if len(val) == npx:
            bmap = val
        else:
            bmap = np.zeros(npx)
            bmap[emap.phase_id == ind] = val
        plt.imshow(bmap.reshape((h, w)))
        plt.title('Misorientation angle wrt reference')
        plt.colorbar(label="Misorientation (rad)")
        plt.show()

        # plot identified grains
        plt.imshow(labels, cmap='flag')
        plt.title(f"Phase #{data['index']} ({data['name']}): Identified Grains: {ngrains}")
        plt.colorbar(label='Grain Number')
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
    ms_data.append(data)
