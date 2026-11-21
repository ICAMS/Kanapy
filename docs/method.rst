=========
Modeling
=========

---------
Geometry
---------
Grains in polycrystalline microstructures can be approximated by ellipsoids. To generate synthetic microstructures, packing the particles (ellipsoids) which follow a particular size distribution into a pre-defined domain becomes the objective. The general framework employed in this regard is the collision detection and response system for particles under random motion in the box. Each particle :math:`i` in the domain is defined as an ellipsoid in the three-dimensional Euclidean space :math:`\mathbb{R}^3`, and random position and velocity vectors :math:`\mathbf{r}^i`, :math:`\mathbf{v}^i` are assigned to it. During their motion, the particles interact with each other and with the simulation box. The interaction between particles can be modeled by breaking it down into stages of collision detection and response, and the interaction between the particles and the simulation box can be modeled by evaluating if the particle crosses the boundaries of the box. If periodicity is enabled periodic images on the opposite boundaries of the box are created, otherwise the particle position and velocity vectors have to be updated to mimic the bouncing back effect.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Two layered collision detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`n` ellipsoids in the domain, the order of computational complexity would be :math:`O(n^2)`, since each ellipsoid in the domain is checked for collision with every other ellipsoid. A two-layered collision detection scheme is implemented to overcome this limitation. The outer layer uses an Octree data structure to segregate the ellipsoids into sub-domains; this process is done recursively until there are only a few ellipsoids left in each sub-domain. The inner layer consists of a bounding spheres hierarchy, wherein ellipsoids of each Octree sub-domain are tested for collision only when their corresponding bounding spheres overlap. This effectively reduces the number of collision checks and thus the order of computational complexity to :math:`O(nlog(n))`. The general framework of collision detection response systems with two-layer spatial-partitioning data structures has the following features:

* Recursively decompose the given domain into sub-domains based on the Octree data structure.
* Perform collision tests between bounding spheres of ellipsoids belonging to the same sub-domain.
* Test for ellipsoid overlap condition only if the bounding spheres overlap. 
* Update the position and the velocity vectors :math:`\mathbf{r}` and :math:`\mathbf{v}` based on the collision response.
* Test for collision with the simulation domain and create periodic images on the opposite boundaries or mimick the bouncing against the wall effect.

""""""""""""""""""""""""""""""""
Layer 1: Octree data structure
""""""""""""""""""""""""""""""""
To ensure efficient collision checks an Octree data structure is initialized on the simulation box. With pre-defined limits for Octree sub-division and particle assignment, the Octree trunk gets divided into sub-branches recursively. Thus, by only performing collision checks between particles belonging to a particular sub-branch, the overall simulation time is reduced.

.. figure:: /figs/octree.png
    :align: center
    
    **Figure**: Simulation domain and its successive sub-branches on two levels, where particles are represented by red filled circles (left). 
    Octree data structure depicting three levels of sub-divisions of the tree trunk (right).

""""""""""""""""""""""""""""""""""""
Layer 2: Bounding sphere hierarchy
""""""""""""""""""""""""""""""""""""
.. figure:: /figs/layers.png
    :align: center
    
    **Figure**: Upper layer consists of the Octree sub-branch with particles (left), and lower layer is defined by 
    bounding spheres for particles :math:`i, j` with radii :math:`a^i, a^j` respectively (right).

^^^^^^^^^^^^^^^^^^^
Overlap detection
^^^^^^^^^^^^^^^^^^^
The actual overlap of two static ellipsoids is determined by the algebraic separation condition developed by `Wang (2001)`_. Consider two ellipsoids :math:`\mathcal{A}: \mathbf{X}^T \mathbf{A} \mathbf{X} = 0` and :math:`\mathcal{B}: \mathbf{X}^T \mathbf{B} \mathbf{X} = 0` in :math:`\mathbb{R}^3`, where :math:`\mathbf{X} = [x, y, z, 1]^T`, the characteristic equation is given as,

.. math::

    f(\lambda) = det(\lambda \: \mathbf{A} + \mathbf{B}) = 0

Wang et al. have established that the equation has at least two negative roots, and depending on the nature of the remaining two roots, the separation conditions are given as,

* :math:`\mathbf{A}` and :math:`\mathbf{B}` are separated if :math:`f(\lambda) = 0` has two distinct positive roots.
* :math:`\mathbf{A}` and :math:`\mathbf{B}` touch externally if :math:`f(\lambda) = 0` has a positive double root.
* :math:`\mathbf{A}` and :math:`\mathbf{B}` overlap for all other cases.


.. _Wang (2001): https://www.sciencedirect.com/science/article/pii/S0167839601000498

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Particle (Ellipsoid) packing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined simulation box size and the ellipsoid size distribution are used for creating the simulation box and ellipsoids. The simulation begins by randomly placing ellipsoids of null volume inside the box, and each ellipsoid is given a random velocity vector for movement. As the simulation proceeds, the ellipsoids grow in size along their axes and also collide with one another updating their position and velocities. The simulation terminates once all the ellipsoids have reached their defined volumes; the process is depicted pictorially in the figure below.

.. figure:: /figs/packing.png
    :align: center
    
    **Figure**: Ellipsoid packing simulation with partice interactions at three different timesteps.

.. note:: Since the application is microstructure generation, where all grains have a predefined tilt angle, the 
          angular velocity vector :math:`(\mathbf{w})` is not considered for the ellipsoids and thus their orientations are constrained. 


--------
Texture
--------
In this section a brief summary of the Orientation Distribution Function (ODF) reconstruction is presented. A detailed description of this can be found in `Biswas (2020)`_ as a :math:`L_1` minimization scheme. Furthermore, an orientation assignment algorithm is presented which takes the grain boundary texture into consideration during the assignment process. 

.. _Biswas (2020): https://scripts.iucr.org/cgi-bin/paper?ks5643

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ODF reconstruction with discrete orientations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Crystallographic texture can be represented in the form of a continuous functions called ODF i.e., :math:`f:SO(3) \rightarrow \mathbb{R}`. With the availability of Electron Back Scatter Diffraction (EBSD) equipment, the polycrystalline materials are easily characterized in the form of measured crystallographic orientations :math:`g_i, \ i=1,...,N`. 

The measured orientations :math:`g_i` are used to estimate the ODF by a kernel density estimation `Hielscher (2013)`_. A bell shaped kernel function :math:`\psi: \ [0,\pi] \rightarrow \mathbb{R}` is placed at each :math:`g_i`, which when combined together estimates the ODF as

.. math::

    f(g) = \frac{1}{N}\sum_{i=1}^{N} \psi_{\kappa} (\omega (g_i, g) ), \quad g \in SO(3)    

where :math:`\omega(g_i , g)` is the disorientation angle between the orientations :math:`g_i` and :math:`g`. It is vital to note here that the estimated ODF :math:`f` heavily depends on the choice of the kernel function :math:`\psi`. Keeping this in mind, the de la Vall\'ee Poussin kernel :math:`\psi_\kappa` is used for ODf reconstruction within Kanapy. Please refer to `Schaeben (1997)`_ for a detailed description of the de la Vall\'ee Poussin kernel function as well as its advantages.

Within the numerical modeling framework for polycrystalline materials, the micromechanical modeling requires a reduced number :math:`N^\prime` of discrete orientations :math:`{g^\prime}_i` to be assigned to grains in the RVE. The ODF of the reduced number of orientations is given as, 

.. math::

    {f^\prime}(g) = \frac{1}{N^\prime}\sum_{i=1}^{N^\prime} \psi_{\kappa^\prime} (\omega ({g^\prime}_i, g) ), \quad g \in SO(3)


Since :math:`N^\prime \ll N`, the the kernel shape parameter :math:`\kappa` must be optimized (:math:`\kappa^\prime`) such that the ODF estimated from :math:`{g^\prime}_i` should be close to the input ODF :math:`f`. To quantify the difference between them an :math:`L_1` error can be defined on the fixed :math:`SO(3)` grid as

.. math::

    \parallel f(g) - {f^\prime} (q) \parallel_{1} = \int_{SO(3)} \big | f(q) - {f^\prime} (q) \big | \text{d} \hspace{2pt} q

.. _Hielscher (2013): https://www.sciencedirect.com/science/article/pii/S0047259X13000419
.. _Schaeben (1997): https://onlinelibrary.wiley.com/doi/abs/10.1002/1521-3951%28199704%29200%3A2%3C367%3A%3AAID-PSSB367%3E3.0.CO%3B2-I

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Orientation assignment process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In addition to the crystallographic texture, polycrystalline material also have grain boundary texture which is represented in the form of the misorientation distribution function (MDF). Similar to ODF, MDF can be estimated on :math:`\Delta \phi:SO(3) \rightarrow \mathbb{R}` due to the disorientation (:math:`\Delta g_i, \ i=1,...,N_g`) at grain boundary segments. These can be used to assign the discrete orientations obtained after ODF reconstruction to the grains in the RVE generated by Kanapy's geometry module. Both the orientations and the disorientations play different roles in the mechanical behavior of the material, a detailed discussion of which can be found in `Biswas (2019)`_. 

Other than the crystallographic orientation of the grains, the dimension of the grain boundary is also a key aspect. Therefore, to incorporate the effect of the grain boundary dimension, :math:`\Delta g` at each segment is weighted as per the segment dimension, as suggested in `Kocks (2000)`_. Since the EBSD data is a 2D image this weighting factor is estimated as :math:`w_i = s_i/S`, where :math:`s_i` is corresponding segment length and :math:`S = \sum_i^{N_g} s_i`.  

The disorientation :math:`\Delta g` can be represented in axis-angle notation. And within Kanapy's algorithms we focus on the angle (:math:`\omega`) part of :math:`\Delta g`, commonly referred to as the disorientation angle. To imitate the statistical distribution from experiments, a Monte-Carlo scheme is suggested in `Miodownik (1999)`_. This is used here in a :math:`L_1` minimization framework similar to the ODF reconstruction discussed earlier. 

The assignment algorithm begins by randomly assigning orientations obtained from ODF reconstruction to the grains in the RVE. The disorientation angle distribution is estimated in the present configuration including the weights (due to the corresponding grain boundaries) and the :math:`L_1` error is estimated. The orientations are then exchanged between two grains modifying the configuration. The :math:`L_1` error is estimated for the modified configuration, and compared with that of the previous configuration. If the error is minimized between the two configurations, then the orientations are retained, else the orientations are flipped to revert back to the previous configuration.    

.. _Biswas (2019): https://onlinelibrary.wiley.com/doi/full/10.1002/adem.201900275
.. _Kocks (2000): https://www.cambridge.org/de/academic/subjects/engineering/materials-science/texture-and-anisotropy-preferred-orientations-polycrystals-and-their-effect-materials-properties?format=PB&isbn=9780521794206
.. _Miodownik (1999): https://www.sciencedirect.com/science/article/pii/S1359645499001378


------------
EBSD reading
------------

Kanapy reads two dimensional EBSD maps and reconstructs their grains as graphs.
The analysis is handled by :class:`kanapy.texture.EBSDmap`. ``ebsd.graph``
contains all retained grains and their adjacency, including phase boundaries.
Phase graphs, initial segmentation, and cleanup records remain available in
``ebsd.ms_data`` for statistics and RVE generation. These objects can be used
directly without writing intermediate files.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Phase selection and scan grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Phase fractions are calculated from the original phase pixel counts divided
by the total number of input measurements. Unindexed measurements are not
analyzed as a phase. The default ``vf_min=0.03`` retains phases occupying at
least 3% of the input map, including phases exactly at that threshold.
Original phase IDs are preserved even when some phases are filtered out.

Rectangular scans use their existing pixel grid. Hexagonal scans have staggered
measurement rows and are resampled onto a rectangular grid before segmentation.
The resampling path depends on the number of phases remaining after filtering:

* With one retained phase and no explicit target spacing, Kanapy keeps the
  existing resampling from that phase's measurements. This path retains its
  filling behavior and does not preserve masks for phases removed by the
  fraction filter.
* With multiple retained phases, Kanapy creates one grid for the complete map.
  Phase labels, confidence indices, and orientations use the same nearest
  measurement at each grid position. The retained phases are segmented together
  on this grid, with connectivity restricted to pixels of the same phase.

Setting ``hex_grid_target_spacing`` explicitly selects the shared grid path
with the requested isotropic physical spacing, including for a single phase.
Otherwise the rectangular shape reported by orix is retained.

When confidence indices are available, the bad pixel repair step uses actual
grid neighbors from the same phase. Both segmentation and graph construction
use the resulting pixel orientations.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Initial region construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Regions are formed by :func:`kanapy.texture.find_similar_regions_by_misorientation`.
A pixel pair is connected when both pixels belong to the same retained phase and
their symmetry reduced misorientation is at most the selected threshold.
The default threshold is 5 degrees. The implementation evaluates neighboring
pixel pairs in batches and labels the connected components of their sparse
adjacency graph. This implements the same connectivity rule as breadth first
region growing. Multiple retained phases share one component labeling pass,
using each phase's own crystal symmetry for the orientation comparisons.
With one retained phase, the existing segmentation path is preserved.

The default ``connectivity=4`` uses horizontal and vertical pixel neighbors.
Setting ``connectivity=8`` also includes diagonal neighbors. The chosen
connectivity is used for segmentation, grain adjacency, and boundary analysis.
For multiple retained phases, filtered and unindexed pixels remain background
in the complete label map. Each phase's separate analysis map also masks out
the other phases. The single phase hex filling behavior described above is unchanged.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Graph construction and phase data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`kanapy.texture.build_graph_from_labeled_pixels` creates one node for
each labeled region. Each node stores its pixel indices, pixel count
(``npix``), mean orientation (``ori_av``), and orientation spread (``ori_std``).
The graph stores the grain label map, pixel rotations used for reconstruction,
crystal symmetry, and grid geometry. The complete multiphase graph stores
symmetries keyed by phase ID. Every grain node has ``phase_id`` and
``phase_name`` attributes; the individual phase graphs also carry these fields.

The mean orientation is computed by :func:`kanapy.texture.mean_orientation_data`.
Pixel quaternions are normalized, symmetry equivalent orientations are aligned
to a reference, and the ``q`` versus ``-q`` ambiguity is handled before
averaging. The mean quaternion is the dominant eigenvector of the quaternion
accumulator matrix.

Edges connect adjacent labeled regions, including grains from different phases.
Region label ``0`` denotes background and is not a grain node. Grain IDs are
unique across the complete map. In a multiphase graph, edges carry an
``is_phase_boundary`` flag. These edges record adjacency; they do not introduce
a physical interface model. No additional phase nodes are created.

Cleanup selects only neighbors in the same phase. A direct attempt to merge
nodes tagged with different phases raises an error. Interface edges are kept
when either endpoint merges with a neighbor of its own phase. Orientation
comparisons and orientation averaging use the symmetry of the relevant phase;
interface edges do not receive a same phase misorientation value.

^^^^^^^^^^^^^
Graph cleanup
^^^^^^^^^^^^^

Cleanup proceeds in the following order. Merge records retain the source
grain, target grain, and the criterion used.

1. **Boundary artifacts.**
   :func:`kanapy.texture.get_node_boundary_stats` measures the boundary pixel
   fraction and local geometry of each node.
   :func:`kanapy.texture.merge_boundary_artifact_nodes` considers nodes with
   at most ``max(int(boundary_artifact_size_factor * gs_min), 40)`` pixels
   and a boundary fraction of at least ``boundary_artifact_fraction_min``.
   The defaults are ``2.0`` and ``0.85``. A candidate is merged into its
   crystallographically closest neighbor in the same phase, preserving the
   target orientation. This pass is based on the boundary criterion and does
   not impose the later 5 degree similarity threshold.
2. **Invalid convex hulls.** Regions that cannot form a valid convex hull are
   absorbed into their largest neighbor in the same phase when one exists.
   The target orientation is preserved.
3. **Similar orientations.** Adjacent grains with mean orientations differing
   by at most ``similar_orientation_merge_degrees`` are merged; the default
   is 5 degrees. Their mean orientation is recomputed from the combined
   reconstruction pixel orientations. Neighbors are reconsidered after merges.
4. **Small grains.** Nodes below ``gs_min`` are absorbed into their largest
   neighbor in the same phase when one exists, preserving the target
   orientation. The default ``gs_min`` is 10 pixels. The current size is
   checked before each merge, so a queued grain that has already grown to
   the threshold is retained.

The final validation requires every remaining grain to have a valid convex
hull. The minimum number of points for a hull is controlled by
``convex_hull_min_points``, whose default is 3.

^^^^^^^^^^^^^^^^
Grain statistics
^^^^^^^^^^^^^^^^

With statistics enabled, Kanapy extracts equivalent diameters, aspect ratios,
and tilt angles from the cleaned grains using its existing geometric
conventions. For convex hull area :math:`A`, the equivalent diameter is
:math:`d_{eq} = 2\sqrt{A / \pi}`. Aspect ratio is the ratio of the fitted
major and minor ellipse axes. Tilt is :math:`\omega = \arccos(v_{a,0})`,
where :math:`v_a` is the major axis direction in Kanapy's scaled row and
column coordinates. This geometric angle is distinct from crystallographic
misorientation.

Diameter and aspect ratio use lognormal parameters. The tilt distribution
uses a von Mises fit to :math:`2\omega - \pi`. Raw arrays, fitted
parameters, and moments are kept in the phase data.

Direct ``EBSDmap`` analysis includes these statistics unless
``graph_only=True``. The :func:`kanapy.graph_workflow.build_ebsd_graph`
interface skips statistics by default; pass ``include_statistics=True``
when they are needed for RVE generation or further analysis.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Result objects and optional files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`kanapy.graph_workflow.build_ebsd_graph` returns an
:class:`kanapy.graph_workflow.EBSDGraphResult` instance. Reconstruction
settings are supplied through :class:`kanapy.graph_workflow.EBSDGraphConfig`.
For development, install the checkout with ``python -m pip install -e .``
from the repository root before starting Python:

.. code-block:: python

    import kanapy as knpy

    config = knpy.EBSDGraphConfig(
        min_phase_pixel_fraction=0.03,
        max_misorientation_degrees=5.0,
        min_grain_size_pixels=10.0,
        connectivity=4,
        similar_orientation_merge_degrees=5.0,
    )
    result = knpy.build_ebsd_graph(
        'examples/fixtures/ebsd_316L_500x500.ang',
        config=config,
    )
    graph = result.graph
    ebsd = result.ebsd

``result.graph`` and ``result.phase_data`` refer to the retained phase with
the largest original pixel fraction. ``result.full_graph`` returns the complete
graph from ``result.ebsd.graph``, including phase boundaries. With one retained
phase, it is the same object as ``result.graph``. Individual phase graphs remain
available through ``result.ebsd.ms_data``. The original phase ID is
``result.phase_index``; ``result.selected_phase_position`` is its position
in the retained phase list.

No files are written by default. Supply an
:class:`kanapy.graph_workflow.EBSDGraphOutputOptions` instance to enable
pickle, JSON summary, graph overlay, IPF image, or diagnostic step figures.
For multiple retained phases, the pickle payload adds ``full_graph`` and a
``phases`` list containing every phase graph and its metadata. The JSON summary
includes ``full_graph_node_count``, ``full_graph_edge_count``, and
``phase_boundary_edge_count``, as well as counts for each phase. The original
top level graph fields continue to describe the selected phase.

^^^^^^^^^^^^^^
Graph plotting
^^^^^^^^^^^^^^

:meth:`kanapy.texture.EBSDmap.plot_graph_overlay` draws the retained grain
graphs on one EBSD inverse pole figure map. With one phase, the nodes and
edges are black. With multiple phases, nodes and edges within each phase use
phase colors. Edges across phase boundaries are gray and have a separate
legend entry.
Node positions come from stored centers or, when statistics are skipped,
the mean image coordinates of their pixels.

The default ``iphase=None`` includes all retained phases. An integer selects
a position in ``ebsd.ms_data``. The optional ``phase_colors`` mapping uses
original phase IDs as keys. ``show_ids=True`` labels multiphase nodes as
``phase_id:node_id``.

Using the result from the previous example, save a combined overlay without
opening an interactive window:

.. code-block:: python

    ebsd.plot_graph_overlay(
        iphase=None, save_path='graph_overlay.png', show=False,
    )

For direct analysis, ``EBSDmap(..., show_graph=True)`` displays the same
combined overlay. ``plot_ipf_map(iphase=None)`` displays the retained phases
together without the graph.

^^^^^^^^^^^^^^^^^
Example workflows
^^^^^^^^^^^^^^^^^

The examples keep experiment parameters and output choices in the scripts.

``examples/EBSD_analysis/graph_based_analysis.py``
    Runs complete two dimensional EBSD analysis without generating an RVE.
    Graph files remain available in its ``2D_graph_result`` directory:
    ``ebsd_graph.pkl``, ``ebsd_graph_summary.json``, ``00_graph_overlay.png``,
    and ``graph_ipf_map.png``. ``statistics.json`` adds equivalent diameters,
    aspect ratios, tilt angles, fitted parameters, and units. Its top level
    statistics describe the selected phase; for multiple retained phases,
    a ``phases`` list includes each phase's separate statistics. Histograms,
    misorientation and IPF maps, segmentation images, pole figures, and IPF
    projections are saved automatically. With multiple retained phases,
    histogram filenames include the original phase ID. Statistics and plotting use Kanapy's
    existing methods. Output switches are defined at the top of the script;
    diagnostic step figures remain disabled by default. The example recreates
    its result directory on each run and returns the complete result object.
    The graph image sorts first by filename and is the first interactive
    preview when ``show_figures=True``.

``examples/EBSD_to_RVE/graph_to_rve.py``
    Runs the complete workflow from EBSD input, graph reconstruction, and
    grain statistics through descriptor construction, RVE initialization,
    packing, voxelization, orientation generation, and voxel export. Result
    and EBSD instances are passed directly between stages. Figures are saved
    and closed automatically in the script's ``outputs`` directory. Matching
    filenames are overwritten, unrelated files remain, and ``manifest.json``
    lists the current run's outputs. ``--show`` enables interactive previews;
    ``--extra-outputs`` enables additional polyhedral and Abaqus exports.
    The graph overlay is saved and optionally displayed before the histograms
    and remaining EBSD figures.

``examples/EBSD_to_RVE/convert_ang2rve.py``
    Preserves the original Kanapy example's complete EBSD to RVE computation
    sequence, with updated comments and paths relative to the script. Its
    plots remain interactive, with the grain graph displayed first: close each
    window to continue. The single voxel JSON is written directly beside the
    script in ``EBSD_to_RVE``, for example ``Iron fcc_voxels.json``. This example
    retains its selection of the first phase in ``ebsd.ms_data`` for the RVE
    statistics. The graph example's files remain in its ``outputs`` directory.

The two graph examples default to four pixel neighbors, a 10 pixel small
grain threshold, and a 3% phase fraction threshold. All three examples use
the currently installed Kanapy implementation. Run them from the repository root:

.. code-block:: console

    python examples/EBSD_analysis/graph_based_analysis.py
    python examples/EBSD_to_RVE/graph_to_rve.py
    python examples/EBSD_to_RVE/convert_ang2rve.py

Multiphase EBSD analysis and combined plotting do not change
``graph_to_rve.py`` phase selection: it generates an RVE for the retained
phase with the largest fraction. It does not construct a multiphase RVE from
all measured phases.
