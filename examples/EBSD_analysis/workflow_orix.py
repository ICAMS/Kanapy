import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from orix import io, plot, quaternion
from orix.vector import Miller


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

    def neighbors(r, c):
        if connectivity == 1:
            return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        else:
            return [(r + i, c + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                    if not (i == 0 and j == 0)]

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
                            stack.extend(neighbors(i, j))

                current_label += 1

    return labeled_array, current_label - 1


fname = "ebsd_316L_500x500.ang"

show_plot = True
vf_min = 0.03
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
                                             tolerance=0.087, connectivity=2)
    print(f"Phase #{data['index']} ({data['name']}): Identified Grains: {n_regions}")

    """Filter out small grains and extract statistics"""

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
        plt.title(f"Phase #{data['index']} ({data['name']}): Identified Grains: {n_regions}")
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
