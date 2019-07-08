---
title: 'kanapy: A Python package for generating complex synthetic polycrystalline microstructures'
tags:
  - Python
  - Microstructure modeling
  - Additive manufacturing
  - Particle packing
  - Collision detection
  - Spatial partitioning
  - Voxelization
authors:
  - name: Mahesh R.G. Prasad
    orcid: 0000-0003-2546-0161
    affiliation: 1
  - name: Napat Vajragupta
    orcid: 0000-0001-6085-7493
    affiliation: 1
  - name: Alexander Hartmaier
    orcid: 0000-0002-3710-1169
    affiliation: 1
affiliations:
 - name: Interdisciplinary Centre for Advanced Materials Simulation, Ruhr-Universität Bochum, Universitätsstr. 150,
44801 Bochum, Germany.
   index: 1
date: 01 July 2019
bibliography: paper.bib
---

# Summary


In the context of process-structure-property relationships, the contribution of microstructure
to material properties is substantial. Thus a micromechanical modeling approach
allows us to understand the influence of microstructural features on macroscopic mechanical
behavior through numerical simulations. At the center of this approach lies the modeling of
synthetic microstructures that mimick the important aspects such as grain morphologies
and crystallographic orientations. Several methods have been developed in this regard,
these include Spatial tessellations, Cellular automata, Monte-Carlo Potts models and Phase field
models.

With the advent of additive manufacturing, the processing steps usually result in
rather complex microstructural features such as elongated grains in metals. Most of the currently 
available software packages to generate synthetic microstructures
lack either in the representativeness or the efficiency. Probabilistic method
like spatial tessellations [@Quey2011] provide sufficiently accurate representations for simple
grain morphologies and size distributions, but it cannot capture complex
morphologies i.e, irregular shaped grains. Another approach that is widely used
is the Random Sequential Addition [@Groeber2014, @Vajragupta2014], it overcomes
the shortcomings of tessellation based methods with its ability to model convex and non-convex
grain morphologies, but the computational expense is high as space filling by random
addition of particles is not efficient for higher volume fractions [@Zhang2013].

``kanapy`` is a python package for generating complex synthetic microstructures
based on time-driven collision detection approach for packing ellipsoids. It can be
classified as a probabilistic method and is directed towards the microstructure
modeling community. It employs a two layer collision detection scheme, wherein the outer layer
utilizes an Octree spatial partitioning data structure to estimate which particles should
be checked for collision. The inner layer consists of bounding spheres hierarchy,
which carries out the collision detection only if the bounding spheres between two
particles overlap. The actual collision detection between two static ellipsoidal
particles is determined by employing the algebraic separation condition developed
by Wang et al. [@Wang2001]. Using the in-built voxelization routine complex
microstructures like those found in additively manufactured components can be easily created.

``kanapy`` is a modular package and it gives the user the flexibility to create individual
work flows for generating specific microstructures. The modules can be executed independent
of one another as it provides easy data storage and handling. It is based on
existing implementations of *k*-d tree and Convex hull from the Scipy package together
with various Numpy array operations. A pure python Octree data structure is implemented
in ``kanapy`` for efficient collision handling. And the performance critical part of the actual
collision detection code is implemented in C++ (using the Eigen library [@eigenweb])
with python bindings generated using header-only library pybind11 [@pybind11].
Examples for generating microstructure with equiaxed and elongated grains are detailed
in the documentation.

# Acknowledgements

This work was sponsored in part by Siemens AG - Corporate Technology.

# References
