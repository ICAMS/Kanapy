---
title: 'Kanapy-v2.0: Synthetic polycrystalline microstructure generator with geometry and texture'
tags:
  - Python
  - MATLAB
  - Synthetic microstructure
  - ODF reconstruction
  - Disorientation angle distribution
authors:
- name: Abhishek Biswas
  orcid: 0000-0001-6984-7144
  affiliation: 1
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
 - name: Interdisciplinary Centre for Advanced Materials Simulation, Ruhr-Universität Bochum, Universitätsstr. 150, 44801 Bochum, Germany.
   index: 1
date: 06 February 2020
bibliography: paper.bib
---

# Summary
In the previous version of ``Kanapy``  [@prasad2019kanapy], an efficient modeling strategy for generating the geometry of the synthetic microstructures using statistical data was presented. Some of the novel features provided include: modeling complex microstructures consisting of irregular grain shapes, particle (grain) packing through collision detection and response system, and collision handling through a two-layer collision detection scheme. Synthetic microstructures are a key aspect of the micromechanical modeling approach for the prediction of mechanical properties. Apart from microstructure geometry, a vital component of polycrystalline microstructures that has substantial influence on the material behavior is texture [@kocks1998texture] [@bunge1993texture]. Therefore, a synthetic microstructure can be considered incomplete without the information of texture. This is addressed in the current version of ``Kanapy`` by including new efficient texture reduction and orientation assignment algorithms to the already existing package.

Texture in polycrystalline materials consists of two major aspects, the distribution of crystallographic orientations, and the grain boundary texture. Mathematically, the distribution of crystallographic orientations is represented by an orientation distribution function (ODF). Whereas, the grain boundary texture consists of both misorientation between two neighbouring grains, and the orientation of  boundaries with respect to the two lattices [@adams1992measurement]. This is represented by a five parameter space [@saylor2004measuring], out of which we focus on the misorientation angle, where the minimum value of misorientation (considering crystal symmetry) is referred to as disorientation.

Figure \ref{algorithm_flow} shows the complete workflow for generating a synthetic microstructure starting from experimental data obtained through the electron back scatter diffraction (EBSD) characterizations. Modeling, usage and application aspects of Kanapy's geometry module is detailed in [@prasad2019kanapy]. The texture information in the form of crystallographic orientations is incorporated into the synthetic microstructure using the ``ODF reconstruction`` and the ``Orientation assignment`` modules. An $L_1$ minimization scheme, as detailed in [@Biswas:ks5643], is implemented in the ``ODF reconstruction`` module
to systematically reduce the experimentally measured orientations to a smaller number of discrete orientations (equal to the number of grains in the synthetic microstructure).

To assign the reduced discrete orientations to grains in the synthetic microstructure, the ``Orientation assignment`` module uses the disorientation angle distribution obtained from the experimental data (like EBSD). The disorientation angle is measured at each grain boundary and is weighted by the grain boundary dimension (described in [@kocks1998texture] as measured misorientation distribution).  Since EBSD maps are 2D images, the grain boundary perimeter is considered as the weighting factor for experimental data, whereas,  in the case of synthetic microstructures the weighting factor is the shared grain boundary surface area. The $L_1$ norm of the difference between the experimental and the synthetic microstructure's disorientation distribution is chosen as the error estimate. This module uses an $L_1$ minimization scheme based on the Mote-Carlo algorithm suggested in [@miodownik1999boundary]. And the effect of incorporating such distributions in the micromechanical modeling approach is studied in [@biswas2019influence].

As indicated in [@kocks1998texture], for real microstructures, the measured misorientation distribution or the weighted disorientation distribution (as used in ``Kanapy``) can be very different from the  distribution derived from the ODF (texture derived misorientation).  The inclusion of this feature in texture analysis makes ``Kanapy`` a more realistic synthetic misrostructure generator in comparison to other microstructure generators like DREAM.3D [@groeber2014dream].       

Both, the ``ODF reconstruction`` module and the ``Orientation assignment`` module are written as MATLAB (The MathWorks Inc., Natick, USA) functions and they use several MTEX [@bachmann2010texture] functions for texture analysis. The ``ODF reconstruction`` module can be used independently (provided required inputs), whereas the ``Orientation assignment`` module only works if the geometry is generated by Kanapy’s geometry module and the ODF reconstruction is performed. The ``ODF reconstruction`` module can also be used as a standalone MATLAB code outside of Kanapy’s framework.  


![Kanapy v2.0 workflow \label{algorithm_flow} ](../docs/figs/method_flow.jpeg)


# References
