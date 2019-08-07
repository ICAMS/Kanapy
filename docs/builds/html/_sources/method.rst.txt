=========
Modeling
=========

Grains in polycrystalline microstructures can be approximated by ellipsoids. To generate synthetic microstructures, packing the particles (ellipsoids) which follow a particular size distribution into a pre-defined domain becomes the objective. The general framework employed in this regard is the collision detection and response system for particles under random motion in the box. Each particle :math:`i` in the domain is defined as an ellipsoid in three-dimensional Euclidean space :math:`\mathbb{R}^3`, and random position and velocity vectors :math:`\mathbf{r}^i`, :math:`\mathbf{v}^i` are assigned to it. During their motion, the particles interact with each other and with the simulation box. The interaction between particles can be modeled by breaking it down into stages of collision detection and response. And the interaction between particles and the simulation box can be modeled by evaluating if the particle crosses the boundaries of the box. If periodicity is enabled periodic images on the opposite boundaries of the box are created, else the particle position and velocity vectors have to be updated to mimic the bouncing back effect.

--------------------------------
Two layered collision detection
--------------------------------

For :math:`n` ellipsoids in the domain, the order of computational complexity would be :math:`O(n^2)`, since each ellipsoid in the domain is checked for collision with every other ellipsoid. A two layered collision detection scheme is implemented to overcome this limitation. The outer layer uses an Octree data structure to segregate the ellipsoids into sub-domains, this process is done recursively until there are only a few ellipsoids in each sub-domain. The inner layer consists of bounding spheres hierarchy, wherein ellipsoids of each Octree sub-domain are tested for collision only when their corresponding bounding spheres overlap. This effectively reduces the number of collision checks and thus the order of computational complexity to :math:`O(nlog(n))`. The general framework of collision detection response systems with the two layer spatial partitioning data structures has the following features:

* Recursively decompose the given domain into sub-domains based on the Octree data structure.
* Perform collision test between bounding spheres of ellipsoids belonging to the same sub-domain.
* Test for ellipsoid overlap condition only if the bounding spheres overlap. 
* Update the position and velocity vectors :math:`\mathbf{r}` and :math:`\mathbf{v}` based on collision response.
* Test for collision with simulation domain and create periodic images on the opposite boundaries (or) mimick the bouncing against the wall effect.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Layer 1: Octree data structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To ensure efficient collision checks an Octree data structure is initialized on the simulation box. With pre-defined limits for Octree sub-division and particle assignment, the Octree trunk gets divided into sub-branches recursively. Thus, by only performing collision checks between particles belonging to a particular sub-branch, the overall simulation time is reduced.

.. figure:: /figs/octree.png
    :align: center
    
    **Figure**: Simulation domain and its successive sub-branches on two levels, where particles are represented by red filled circles (left). 
    Octree data structure depicting three levels of sub-divisions of the tree trunk (right).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Layer 2: Bounding sphere hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: /figs/layers.png
    :align: center
    
    **Figure**: Upper layer consists of the Octree sub-branch with particles (left) and lower layer is defined by 
    bounding spheres for particles :math:`i, j` with radii :math:`a^i, a^j` respectively (right).

--------------------
Overlap detection
--------------------
Actual overlap of two static ellipsoids is determined by the algebraic separation condition developed by `Wang (2001)`_. Consider two ellipsoids :math:`\mathcal{A}: \mathbf{X}^T \mathbf{A} \mathbf{X} = 0` and :math:`\mathcal{B}: \mathbf{X}^T \mathbf{B} \mathbf{X} = 0` in :math:`\mathbb{R}^3`, where :math:`\mathbf{X} = [x, y, z, 1]^T`, the characteristic equation is given as,

.. math::

    f(\lambda) = det(\lambda \: \mathbf{A} + \mathbf{B}) = 0

Wang et al. have established that the equation has at least two negative roots and depending on the nature of the remaining two roots the separation conditions are given as,

* :math:`\mathbf{A}` and :math:`\mathbf{B}` are separated if :math:`f(\lambda) = 0` has two distinct positive roots.
* :math:`\mathbf{A}` and :math:`\mathbf{B}` touch externally if :math:`f(\lambda) = 0` has a positive double root.
* :math:`\mathbf{A}` and :math:`\mathbf{B}` overlap for all other cases.


.. _Wang (2001): https://www.sciencedirect.com/science/article/pii/S0167839601000498

-----------------------------
Particle (Ellipsoid) packing
-----------------------------
User defined simulation box size and ellipsoid size distribution are used for creating simulation box and ellipsoids. The simulation begins by randomly placing ellipsoids of null volume inside the box and each ellipsoid is given a random velocity vector for movement. As the simulation proceeds the ellipsoids grow in size along their axes and also collide with one another updating their position and velocities. The simulation terminates once all the ellipsoids have reached their defined volumes, the process is depicted pictorially in the figure below.

.. figure:: /figs/packing.png
    :align: center
    
    **Figure**: Ellipsoid packing simulation with partice interactions at three different timesteps.

.. note:: Since the application is microstructure generation, where all grains have a predefined tilt angle, the 
          angular velocity vector :math:`(\mathbf{w})` is not considered for the ellipsoids and thus their orientations are constrained. 


