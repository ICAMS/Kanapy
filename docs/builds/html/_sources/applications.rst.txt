=============
Applications
=============

------------------------------------
Microstructure with equiaxed grains
------------------------------------

Voronoi_ and Laguerre_ tessellations are some of the popular methods for generating polycrystalline microstructures. These approaches require positions and weights as input parameters for generating tessellation cells that resemble grains of a polycrystal. In this regard, the proposed particle packing approach can be used to generate the required information. Microstructures with equiaxed grains are best approximated by spheres, and after obtaining the necessary packing fraction, positions and radii can be outputted. 

.. _Voronoi: https://en.wikipedia.org/wiki/Voronoi_diagram
.. _Laguerre: https://en.wikipedia.org/wiki/Power_diagram


.. figure:: /figs/sphere_app.png
    :align: center
    
    **Figure**: An example of sphere packing (left), radical Voronoi tesselation (center) and FE tetrahedral mesh (right).

In this example, to obtain a high packing fraction, the box size is set as the volume sum of all the spheres. This definition of the box size will inevitably result in an overlap of spheres at the final stages of the simulation. This is because the packing fraction of :math:`100\%` is unachievable without the occurrence of overlaps. A reasonable amount of overlap is accepted, as the input data obtained from microstructure characterization techniques (like EBSD) are also approximations. Hence, for further post-processing, it is suggested to choose the time step at which the spheres are tightly packed and at which there is the least amount of overlap. The remaining empty spaces will get assigned to the closest sphere when it is sent to the tessellation and meshing routine. Complete freedom in selecting the desired time step of the simulation to be sent for further processing is one of the highlights of this approach.

Using the information obtained from sphere packing, radical Voronoi tessellation can be performed by Neper. And FE discretization of the tessellation cells by tetrahedral elements can also be done within Neper - `Quey (2011)`_ with assistance from the FE mesh generating software Gmsh - `Geuzaine (2009)`_.

.. _Quey (2011): https://doi.org/10.1016/j.cma.2011.01.002
.. _Geuzaine (2009): https://doi.org/10.1002/nme.2579

------------------------------------
Microstructure with elongated grains
------------------------------------

Microstructures generated through conventional manufacturing processes like rolling and state-of-the-art processes like additive manufacturing (AM) consist of both elongated and equiaxed grains. In the context of AM, this is due to the complex solidification process, which occurs as a result of constant re-heating and re-cooling of previously melted layers during the manufacturing. The morphology of the grains along with the texture plays an important role in the resulting mechanical behavior. Hence, modeling such complex microstructures is vital, and in this regard the proposed approach of random ellipsoid packing is used. 

.. figure:: /figs/ellipsoid_app.png
    :align: center
    
    **Figure**: An example of ellipsoidal packing (left), Voxelization - FE hexahedral mesh (right).

The ellipsoid packing process is similar to that of sphere packing as described earlier. Once the ellipsoids are tightly packed with minimal acceptable overlaps, they are processed further for meshing. Since Voronoi tessellations cannot be applied to anisotropic particles such as ellipsoids, a voxel-based mesh-generating routine that utilizes convex hull for discretizing the RVE is employed. The voxelization (meshing) routine is made of 2 stages. The number of voxels in the RVE has a direct influence on the FE solution and the FE simulation time and must hence be meticulously chosen. The choice is not arbitrary, as it is constrained by how well the grains of the RVE are represented with respect to their geometry and the FEM simulation time. 

---------------------
Simulation benchmarks
---------------------
To estimate the performance of Kanapy with respect to its particle packing and voxelization routines, multiple realizations of both sphere and ellipsoid packing was performed on a 2.60 GHz Intel Xeon CPU. Figure 1 depicts graphically the average CPU time recorded over 10 simulation runs for each realization of particle packing. The data is tabulated in Table 1. A significant difference in the variation of CPU time can be observed between spheres and ellipsoids of the same number. The difference arises from the fact that, two layered collision detection is sufficient to estimate if spheres collide. But ellipsoid collision detection requires an additional computational step of solving the characteristic equation as described in :ref:`Overlap detection`. 

.. figure:: /figs/CPUtime_analysis_packing.png
    :align: center
    
    **Figure 1**: Log-log plot depicting the performance of the particle packing routine in terms of average CPU time (in seconds) recorded for different realizations of packing of spheres and ellipsoids.
    
.. list-table:: **Table 1**: Particle packing simulation tests for spheres and ellipsoids. The CPU time is averaged over 10 simulation runs.
   :widths: 10 10 10
   :header-rows: 2

   * - Number of
     - CPU time (s)
     - 
   * - particles
     - Sphere
     - Ellipsoids    
   * - 50
     - 50.63
     - 72.29 
   * - 100
     - 91.15     
     - 163.88
   * - 500
     - 362.32     
     - 618.73
   * - 1000
     - 704.13
     - 1174.94
   * - 5000
     - 3119.56
     - 4635.41
    
Figure 2 depicts the CPU time recorded for the kanapy's voxelization routine. The CPU time is estimated for various combinations of number of grains and number of voxels. These results are tabulated in Table 2.  

.. figure:: /figs/CPUtime_analysis_voxelization.png
    :align: center
    
    **Figure 2**: Plot depicting the performance of the voxelization routine in terms of CPU time (in seconds) for various combinations of grain numbers and total voxel numbers.     
     
.. list-table:: **Table 2**: Voxelization routine performance data for various combinations of grain numbers and total voxel numbers.
   :widths: 15 10 10 10
   :header-rows: 2

   * - Number of voxels
     - 
     - CPU time (s)
     - 
   * - 
     - 100 grains
     - 500 grains
     - 1000 grains
   * - :math:`20^3` = 8000
     - 7.09
     -
     -    
   * - :math:`30^3` = 27,000
     - 11.33
     - 
     - 
   * - :math:`40^3` = 64,000
     - 18.9
     - 53.63
     - 
   * - :math:`50^3` = 125,000
     - 28.26
     - 78.33
     - 122.19
   * - :math:`60^3` = 216,000
     - 44.79 
     - 123.36 
     - 186.54
   * - :math:`70^3` = 343,000 
     - 68.65 
     - 186.49 
     - 282.17
   * - :math:`80^3` = 512,000 
     -  
     - 257.95 
     - 378.13 
   * - :math:`90^3` = 729,000 
     -  
     - 389.48 
     - 528.95 
   * - :math:`100^3` = 1,000,000 
     - 
     - 
     - 683.30      
     
