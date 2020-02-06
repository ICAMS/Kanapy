---
title: 'Kanapy-v2.0: A package for generating complex synthetic polycrystalline microstructures with geometry and texture'
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
In the previous version of Kanapy  [@prasad2019kanapy] , an effecient tool for generating synthetic microstructure geometry using the statistical data of grain size was presented. It provided several novel features like collision handling through a two-layer collision detection, modeling complex microstructure consisting of irregular grain shapes,  coupling to softwares like ABAQUS and Neper. Synthetic microstructure is a key aspect of the micomechanical modeling for predicting material behavior numerically. Other than microstructure geometry, texture is another vital component of a polycrystalline microstructure, which has  prominent effects on the behavior of the material [@kocks1998texture] [@bunge1993texture]. Therefore, a synthetic microstructure without the texture information is incomplete.

Texture in polycrystalline material consists of two major aspects, the distribution of crystallographic orientations and grain boundary texture. Mathematically, the distribution of crystallographic orientation in polycrystalline materials is represented by the orientation distribution function (ODF). On the other hand, the grain boundary texture consist of both misorientation between two neighbouring grains and the orientation of the boundaries with respect to the two lattices [@adams1992measurement]. This is represented by a five parameter space [@saylor2004measuring], out of which we focus on the misorientation angle, where the minimum value of misorientation (considering crystal symmetry) is referred to as disorientation.


Figure \ref{algorithm_flow} shows the complete flow for generating a synthetic microstructure starting from an experimental data like electron back scatter diffraction (EBSD), the description about kanapy's geometry module can be found in [@prasad2019kanapy]. The texture information is incorporated in the synthetic microstructure using the ``ODF reconstruction module`` and the ``Orientation assignment module``. The ``ODF reconstruction module`` uses an $L_1$ minimization scheme [@Biswas:ks5643] to systematically reduce the experimentally measured orientations to a smaller number of discrete orientations (equal to the number of grains in the synthetic microstructure).

The ``Orientation assignment module`` uses the disorientation angle distribution from the experiments (like EBSD), where the disorientation angle measure at each grain boundary is weighted by its dimension (described in [@kocks1998texture] as measured misorientation distribution). Since EBSD maps are 2D images the grain boundary perimeter is considered for weighting, whereas in case of the synthetic microstructures the shared grain boundary surface area is used . The $L_1$ norm of the difference between the experimental and the synthetic microstructure disorientation distribution is chosen as the error estimate. The module uses a $L_1$ minimization scheme based on a Mote-Carlo algorithm suggested in [@miodownik1999boundary]. The effect of incorporating such distributions in micromechanical modeling approach is studied in [@biswas2019influence].

As indicated in [@kocks1998texture] for real microstructures, the weighted disorientation distribution or the measured misorientation distribution (used in Kanapy) can be very different from the  distribution derived from the ODF (texture derived misorientation).  This feature makes kanapy a more realistic synthetic misrostructure generator in comparison to other microstructure generators like DREAM.3D [@groeber2014dream].       

The ``ODF reconstruction module`` and the ``Orientation assignment module`` are written as a MATLAB (The MathWorks Inc., Natick, USA) functions and they use several MTEX [@bachmann2010texture] functions for texture analysis. These MATLAB functions are coupled with kanapy's geometry module by creating MATLAB-python link.  


![Flow diagram of Kanapy v2.0\label{algorithm_flow} ](../docs/figs/method_flow.jpeg)


# References
