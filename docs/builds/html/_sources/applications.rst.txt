=============
Applications
=============

------------------------------------
Microstructure with equiaxed grains
------------------------------------

Voronoi_ and Laguerre_ tessellations are some of the popular methods for generating polycrystalline microstructures. These approaches require positions and weights as input parameters for generating tessellation cells that resemble grains of a polycrystal. In this regard, the proposed particle packing approach can be used to generate the required information. Microstructure with equiaxed grains are best approximated by spheres and after obtaining the necessary packing fraction, positions and radii can be outputted. 

.. _Voronoi: https://en.wikipedia.org/wiki/Voronoi_diagram
.. _Laguerre: https://en.wikipedia.org/wiki/Power_diagram


.. figure:: /figs/sphere_app.png
    :align: center
    
    **Figure**: An example of sphere packing (left), radical voronoi tesselation (center) and FE tetrahedral mesh (right).

In this example, to obtain high packing fraction the box size is set as the volume sum of all the spheres. This definition of the box size will inevitably result in overlap of spheres at the final stages of the simulation. This is because packing fraction of :math:`100\%` is unachievable without the occurrence of overlaps. Reasonable amount of overlap is accepted, as the input data obtained from microstructure characterization techniques (like EBSD) are also approximations. Hence for further post-processing, it is suggested to choose the time step at which the spheres are tightly packed and there is least amount of overlap. The remaining empty spaces will get assigned to the closest sphere when it is sent to the tessellation and meshing routine. Complete freedom in selecting the desired time step of the simulation to be sent for further processing is one of the highlights of this approach.

Using the information obtained from sphere packing, radical Voronoi tessellation can be performed by Neper. And FE discretization of the tessellation cells by tetrahedral elements can also be done within Neper - `Quey (2011)`_ with assistance from FE mesh generating software Gmsh - `Geuzaine (2009)`_.

.. _Quey (2011): https://doi.org/10.1016/j.cma.2011.01.002
.. _Geuzaine (2009): https://doi.org/10.1002/nme.2579

------------------------------------
Microstructure with elongated grains
------------------------------------

Microstructures generated through conventional manufacturing processes like rolling and state of the art processes like additively manufacturing (AM) consists of both elongated and equiaxed grains. In the context of AM, this is due to the complex solidification process, which occurs as a result of constant re-heating and re-cooling of previously melted layers during the manufacturing. The morphology of the grains along with texture plays an important role in the resulting mechanical behavior. Hence, modeling such complex microstructures is vital and in this regard the proposed approach of random ellipsoid packing is used. 

.. figure:: /figs/ellipsoid_app.png
    :align: center
    
    **Figure**: An example of ellipsoidal packing (left), Voxelization - FE hexahedral mesh (right).

The ellipsoid packing process is similar to that of sphere packing described earlier. Once the ellipsoids are tightly packed with minimal acceptable overlaps, they are processed further for meshing. Since Voronoi tessellations cannot be applied to anisotropic particles such as ellipsoids, a voxel based mesh generating routine that utilizes convex hull for discretizing the RVE is employed. The voxelization (meshing) routine is made up of 2 stages. The number of voxels in the RVE has a direct influence on the FE solution and the FE simulation time, hence it must be meticulously chosen. The choice is not arbitrary, as it is constrained by how well the grains of the RVE are represented with respect to their geometry and the FEM simulation time. 

