# -*- coding: utf-8 -*-
"""
Subroutines for analysis of grains in microstructure

@author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany
March 2024
"""
import itertools
import logging
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm


def calc_polygons(rve, mesh, tol=1.e-3):
    """
    Evaluates grain volumes and the shared surface area of grain boundaries
    between neighboring grains in a voxelized microstructure. Generates
    vertices at contact points where three or more grains meet and constructs
    polyhedral convex hulls for these vertices.

    Parameters
    ----------
    rve : kanapy object
        Object containing information about the RVE geometry.
    mesh : kanapy object
        Object containing voxel mesh details and grain assignments for each voxel.
    tol : float, optional
        Tolerance for numerical operations. Default is 1.e-3.

    Returns
    -------
    geometry : dict
        Dictionary containing the calculated geometrical information, including
        vertices, contact points, and convex hulls for grains.

    Notes
    -----
    For periodic structures, large grains spanning both halves of the simulation
    box and touching a boundary may be incorrectly split, which can affect
    geometry calculations.
    """

    def facet_on_boundary(conn):
        """
        Check if given voxel facet lies on any RVE boundary

        Parameters
        ----------
        conn

        Returns
        -------
        face : (3,2)-array of bool
        """
        n1 = mesh.nodes[conn[0] - 1, :]
        n2 = mesh.nodes[conn[1] - 1, :]
        n3 = mesh.nodes[conn[2] - 1, :]
        n4 = mesh.nodes[conn[3] - 1, :]
        face = np.zeros((3, 2), dtype=bool)
        for i in range(3):
            h1 = np.abs(n1[i] - RVE_min[i]) < tol
            h2 = np.abs(n2[i] - RVE_min[i]) < tol
            h3 = np.abs(n3[i] - RVE_min[i]) < tol
            h4 = np.abs(n4[i] - RVE_min[i]) < tol
            face[i, 0] = h1 and h2 and h3 and h4
            h1 = np.abs(n1[i] - RVE_max[i]) < tol
            h2 = np.abs(n2[i] - RVE_max[i]) < tol
            h3 = np.abs(n3[i] - RVE_max[i]) < tol
            h4 = np.abs(n4[i] - RVE_max[i]) < tol
            face[i, 1] = h1 and h2 and h3 and h4
        return face

    def check_neigh(nodes, grains, margin):
        """
        Checks if new vertices are close to existing vertices in neighboring grains
        and merges them if within a given margin

        Parameters
        ----------
        nodes : list
            List of node IDs identified as new vertices.
        grains : set
            Set of grain IDs containing the nodes in the element list.
        margin : float
            Radius within which vertices will be considered identical and united.

        Returns
        -------
        list
            Updated list of node IDs with close neighbors merged.

        Notes
        -----
        For each new vertex, the function computes distances to all vertices of the specified grains.
        If a vertex lies within the specified margin, it is merged with the existing vertex.
        Special treatment is applied for vertices on the RVE boundaries using the `RVE_min` and `RVE_max` values.
        """

        # create set of all vertices of all involved grains
        vset = set()
        for gid in grains:
            vset.update(vert[gid])
        # loop over all combinations
        for i, nid1 in enumerate(nodes):
            npos1 = mesh.nodes[nid1 - 1]
            sur1 = [np.abs(npos1[0] - RVE_min[0]) < tol,
                    np.abs(npos1[1] - RVE_min[1]) < tol,
                    np.abs(npos1[2] - RVE_min[2]) < tol,
                    np.abs(npos1[0] - RVE_max[0]) < tol,
                    np.abs(npos1[1] - RVE_max[1]) < tol,
                    np.abs(npos1[2] - RVE_max[2]) < tol]
            for nid2 in vset:
                npos2 = mesh.nodes[nid2 - 1]
                sur2 = [np.abs(npos2[0] - RVE_min[0]) < tol,
                        np.abs(npos2[1] - RVE_min[1]) < tol,
                        np.abs(npos2[2] - RVE_min[2]) < tol,
                        np.abs(npos2[0] - RVE_max[0]) < tol,
                        np.abs(npos2[1] - RVE_max[1]) < tol,
                        np.abs(npos2[2] - RVE_max[2]) < tol]
                d = np.linalg.norm(npos1 - npos2)
                if d < margin:
                    if (not np.any(sur1)) or (sur1 == sur2):
                        nodes[i] = nid2
                    break
        return nodes

    def get_voxel(pos):
        """
        Determines the voxel indices corresponding to a given position vector

        Parameters
        ----------
        pos : array-like
            Position vector in the 3D space of the RVE.

        Returns
        -------
        v0 : int
            Voxel index along the x-axis.
        v1 : int
            Voxel index along the y-axis.
        v2 : int
            Voxel index along the z-axis.
        """
        v0 = np.minimum(int(pos[0] / voxel_res[0]), rve.dim[0] - 1)
        v1 = np.minimum(int(pos[1] / voxel_res[1]), rve.dim[1] - 1)
        v2 = np.minimum(int(pos[2] / voxel_res[2]), rve.dim[2] - 1)
        return (v0, v1, v2)

    def tet_in_grain(tet, vertices):
        """
        Checks whether all vertices of a given tetrahedron belong to a specified set of vertices

        Parameters
        ----------
        tet : list or array-like
            Indices of the tetrahedron's vertices
        vertices : set or list
            Set of vertices to check against

        Returns
        -------
        bool
            True if all tetrahedron vertices are in the specified set, False otherwise
        """
        return geometry['Vertices'][tet[0]] in vertices and \
            geometry['Vertices'][tet[1]] in vertices and \
            geometry['Vertices'][tet[2]] in vertices and \
            geometry['Vertices'][tet[3]] in vertices

    def vox_in_tet(vox_, tet_):
        """
        Determine whether the center of a voxel lies within a given tetrahedron

        Parameters
        ----------
        vox_ : int or tuple
            Index or identifier of the voxel
        tet_ : list or array-like
            Indices of the tetrahedron's vertices

        Returns
        -------
        contained : bool
            True if the voxel center lies inside the tetrahedron, False otherwise
        """

        v_pos = mesh.vox_center_dict[vox_]
        contained = True
        for node in tet_:
            n_pos = geometry['Points'][node]
            hh = set(tet_)
            hh.remove(node)
            ind_ = list(hh)
            f_pos = geometry['Points'][ind_]
            ctr_ = np.mean(f_pos, axis=0)
            normal = np.cross(f_pos[1, :] - f_pos[0, :], f_pos[2, :] - f_pos[0, :])
            hn = np.linalg.norm(normal)
            if hn > 1.e-5:
                normal /= hn
            dist_to_vox = np.dot(v_pos - ctr_, normal)
            dist_to_node = np.dot(n_pos - ctr_, normal)
            if np.sign(dist_to_vox * dist_to_node) < 0. or \
                    np.abs(dist_to_vox) > np.abs(dist_to_node):
                contained = False
                break
        return contained

    def get_diameter(pts):
        """
        Get estimate of the largest diameter of a set of points

        Parameters
        ----------
        pts : (N, dim) ndarray
            Array of points in dim-dimensional space

        Returns
        -------
        diameter : (dim,) ndarray
            Vector representing the largest extent of the point set along any Cartesian axis
        """
        ind0 = np.argmin(pts, axis=0)  # index of point with lowest coordinate for each Cartesian axis
        ind1 = np.argmax(pts, axis=0)  # index of point with highest coordinate for each Cartesian axis
        v_min = np.array([pts[i, j] for j, i in enumerate(ind0)])  # min. value for each Cartesian axis
        v_max = np.array([pts[i, j] for j, i in enumerate(ind1)])  # max. value for each Cartesian axis
        ind_d = np.argmax(v_max - v_min)  # Cartesian axis along which largest distance occurs
        return pts[ind1[ind_d], :] - pts[ind0[ind_d], :]

    def project_pts(pts, ctr, axis):
        """
        Project points (pts) to plane defined via center (ctr) and normal vector (axis)

        Parameters
        ----------
        pts : (N, dim) ndarray
            Point set in dim dimensions
        ctr : (dim)-ndarray
            Center point of the projection plane
        axis : (dim)-ndarray
            Unit vector for plane normal

        Returns
        -------
        ppt : (N, dim) ndarray
            Points projected to plane with normal axis
        """
        dvec = pts - ctr[None, :]  # distance vector b/w points and center point
        pdist = np.array([np.dot(axis, v) for v in dvec])
        ppt = np.zeros(pts.shape)
        for i, p in enumerate(dvec):
            ppt[i, :] = p - pdist[i] * axis
        return ppt

    # define constants
    voxel_res = np.divide(rve.size, rve.dim)
    voxel_size = voxel_res[0]
    RVE_min = np.min(mesh.nodes, axis=0)
    if np.any(RVE_min > 1.e-3) or np.any(RVE_min < -1.e-3):
        raise ValueError('Irregular RVE geometry: RVE_min = {}'.format(RVE_min))
    RVE_max = np.max(mesh.nodes, axis=0)
    Ng_max = np.max(list(mesh.grain_dict.keys()))  # highest grain number
    Ngr = len(mesh.grain_dict.keys())  # number of grains

    # create dicts for GB facets, including fake facets at surfaces
    geometry = dict()
    grain_facesDict = dict()  # {Grain: faces}
    gb_vox_dict = dict()
    for i in range(1, Ng_max + 7):
        grain_facesDict[i] = dict()

    # generate following objects:
    # outer_faces: {face_id's of outer voxel faces}  (potential GB facets)
    # face_nodes: {face_id: list with 4 nodes}
    # grain_facesDict: {grain_id: {face_id: list with 4 nodes}}
    # gb_vox_dict: {face_id: list with voxels}
    # Loop over all grains in microstructure
    for gid, elset in mesh.grain_dict.items():
        outer_faces = set()
        face_nodes = dict()
        nodeConn = [mesh.voxel_dict[el] for el in elset]  # list of node sets for each voxel in grain

        # For each voxel, re-create its 6 faces, each face is list of 4 nodes
        for nc in nodeConn:
            faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                     [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                     [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]
            # Sort in ascending order
            sorted_faces = [sorted(fc) for fc in faces]
            # create unique face ids by joining node id's
            face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]
            # Create face_nodes = {face_id: nodal connectivity} dictionary
            for enum, fid in enumerate(face_ids):
                if fid not in face_nodes.keys():
                    face_nodes[fid] = faces[enum]
            # Identify outer faces that occur only once and store in outer_faces
            for fid in face_ids:
                if fid not in outer_faces:
                    outer_faces.add(fid)
                else:
                    outer_faces.remove(fid)

        # Update grain_faces_dict= {grain_id: {face_id: facet nodes}
        for of in outer_faces:
            # add face nodes to grain faces dictionary
            conn = face_nodes[of]
            grain_facesDict[gid][of] = conn  # list of four nodes
            # if voxel faces lies on RVE boundary add also to fake grains
            fob = facet_on_boundary(conn)
            for i in range(3):
                for j in range(2):
                    if fob[i, j]:
                        grain_facesDict[Ng_max + 1 + 2 * i + j][of] = conn

    # Find all combinations of grains to check for common area
    # analyse grain_facesDict and create object:
    # gbDict: {f{gid1}_{gid2}: list with 4 nodes shared by grains #gid1 and #gid2}
    # shared_area: [[gid1, gid2, GB area]]
    shared_area = []  # GB area
    gbDict = dict()  # voxel facets on GB
    # Find the shared area and generate gbDict for all pairs of neighboring grains
    combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))
    for cb in combis:
        finter = set(grain_facesDict[cb[0]]).intersection(set(grain_facesDict[cb[1]]))
        if finter:
            ind = set()
            [ind.update(grain_facesDict[cb[0]][key]) for key in finter]
            key = 'f{}_{}'.format(cb[0], cb[1])
            gbDict[key] = ind
            if cb[0] <= Ng_max and cb[1] <= Ng_max:
                # grain facet is not on boundary
                try:
                    hull = ConvexHull(mesh.nodes[list(ind), :])
                    shared_area.append([cb[0], cb[1], hull.area])
                except:
                    sh_area = len(finter) * (voxel_size ** 2)
                    shared_area.append([cb[0], cb[1], sh_area])

    # analyse gbDict to find intersection lines of GB's
    # (triple or quadruple lines) -> edges
    # vertices are end points of edges, represented by
    # nodes in voxelized microstructure
    # created objects:
    # vert: {grain_id: [node_numbers of vertices]}
    # grains_of_vert: {node_number: [grain_id's connected to vertex]}

    # for periodic structures vertices at surfaces should be mirrored!!!
    vert = dict()
    grains_of_vert = dict()
    for i in grain_facesDict.keys():
        vert[i] = set()
    for key0 in gbDict.keys():
        klist = list(gbDict.keys())
        # select grains with list positions before the current one
        while key0 != klist.pop(0):
            pass
        for key1 in klist:
            finter = gbDict[key0].intersection(gbDict[key1])
            if finter:
                if len(finter) == 1:
                    # only one node in intersection of GBs
                    elist = list(finter)
                else:
                    # mulitple nodes in intersection
                    # identify end points of triple or quadruple line
                    edge = np.array(list(finter), dtype=int)
                    rn = mesh.nodes[edge - 1]
                    dmax = 0.
                    for j, rp0 in enumerate(rn):
                        for k, rp1 in enumerate(rn[j + 1:, :]):
                            d = np.sqrt(np.dot(rp0 - rp1, rp0 - rp1))
                            if d > dmax:
                                elist = [edge[j], edge[k + j + 1]]
                                dmax = d
                # select all involved grains
                gr_set = set()
                j = key0.index('_')
                igr = int(key0[1:j])
                if igr <= Ng_max:
                    gr_set.add(igr)
                igr = int(key0[j + 1:])
                if igr <= Ng_max:
                    gr_set.add(igr)
                j = key1.index('_')
                igr = int(key1[1:j])
                if igr <= Ng_max:
                    gr_set.add(igr)
                igr = int(key1[j + 1:])
                if igr <= Ng_max:
                    gr_set.add(igr)
                # check if any neighboring nodes are already in list of
                # vertices. If yes, replace new vertex with existing one
                newlist = check_neigh(elist, gr_set, margin=2 * voxel_size)
                # update grains with new vertex
                for igr in gr_set:
                    vert[igr].update(newlist)
                for nv in newlist:
                    if nv in grains_of_vert.keys():
                        grains_of_vert[nv].update(gr_set)
                    else:
                        grains_of_vert[nv] = gr_set

    # Store grain-based information and do Delaunay tesselation
    # Sort grains w.r.t number of vertices
    num_vert = [len(vert[igr]) for igr in mesh.grain_dict.keys()]
    glist = np.array(list(mesh.grain_dict.keys()), dtype=int)
    glist = list(glist[np.argsort(num_vert)])
    assert len(glist) == Ngr

    # re-sort by keeping neighborhood relations
    grain_sequence = [glist.pop()]  # start sequence with grain with most vertices
    while len(glist) > 0:
        igr = grain_sequence[-1]
        neigh = set()
        for gb in shared_area:
            if igr == gb[0]:
                neigh.add(gb[1])
            elif igr == gb[1]:
                neigh.add(gb[0])
        # remove grains already in list from neighbor set
        neigh.difference_update(set(grain_sequence))
        if len(neigh) == 0:
            # continue with next grain in list
            grain_sequence.append(glist.pop())
        else:
            # continue with neighboring grain that has most vertices
            ind = [glist.index(i) for i in neigh]
            if len(ind) == 0:
                grain_sequence.append(glist.pop())
            else:
                grain_sequence.append(glist.pop(np.amax(ind)))
    if len(grain_sequence) != Ngr or len(glist) > 0:
        raise ValueError(f'Grain list incomplete: {grain_sequence}, remaining elements: {glist}')

    # initialize dictionary for grain information
    grains = dict()
    vertices = np.array([], dtype=int)
    for step, igr in enumerate(grain_sequence):
        add_vert = vert[igr].difference(set(vertices))
        grains[igr] = dict()
        grains[igr]['Vertices'] = np.array(list(vert[igr]), dtype=int) - 1  # indices of voxels at grain vertices
        grains[igr]['Points'] = mesh.nodes[grains[igr]['Vertices']]  # positions of vertices
        center = np.mean(grains[igr]['Points'], axis=0)
        grains[igr]['Center'] = center  # position of grain center
        # initialize values to be incrementally updated later
        grains[igr]['Simplices'] = []
        grains[igr]['Volume'] = 0.
        grains[igr]['Area'] = 0.
        grains[igr]['Phase'] = mesh.grain_phase_dict[igr]  # phase number to which grain belongs

        # Construct incremental Delaunay tesselation of
        # structure given by vertices
        vlist = np.array(list(add_vert), dtype=int) - 1
        vertices = np.append(vertices, list(add_vert))
        if step == 0:
            tetra = Delaunay(mesh.nodes[vlist], incremental=True)
        else:
            try:
                tetra.add_points(mesh.nodes[vlist])
            except:
                vlist = np.array(vertices, dtype=int) - 1
                tetra = Delaunay(mesh.nodes[vlist], incremental=True)

    tetra.close()
    # store global result of tesselation
    geometry['Vertices'] = np.array(vertices, dtype=int) - 1
    geometry['Points'] = tetra.points
    geometry['Simplices'] = tetra.simplices

    # assign simplices (tetrahedra) to grains
    Ntet = len(tetra.simplices)
    print('\nGenerated Delaunay tesselation of grain vertices.')
    print(f'Assigning {Ntet} tetrahedra to grains ...')
    tet_to_grain = np.zeros(Ntet, dtype=int)
    for i, tet in tqdm(enumerate(tetra.simplices)):
        ctr = np.mean(tetra.points[tet], axis=0)
        igr = mesh.grains[get_voxel(ctr)]
        if (igr == 0) and (0 not in grains.keys()):
            continue  # special case of precipit, where grain 0 is assigned to pores
        # test if all vertices of tet belong to that grain
        if not tet_in_grain(tet, grains[igr]['Vertices']):
            # try to find a neighboring grain with all vertices of tet
            neigh_list = set()
            for hv in tet:
                neigh_list.update(grains_of_vert[vertices[hv]])
            match_found = False
            for jgr in neigh_list:
                if tet_in_grain(tet, grains[jgr]['Vertices']):
                    igr = jgr
                    match_found = True
                    break
            if not match_found:
                # get a majority vote. BETTER: split up tet
                # count all voxels in tet
                neigh_list.add(igr)
                neigh_list = list(neigh_list)
                num_vox = []
                for ngr in neigh_list:
                    nv = 0
                    for vox in mesh.grain_dict[ngr]:
                        if vox_in_tet(vox, tet):
                            nv += 1
                    num_vox.append(nv)
                igr = neigh_list[np.argmax(num_vox)]

        tet_to_grain[i] = igr
        # Update grain volume with tet volume
        dv = tetra.points[tet[3]]
        vmat = [tetra.points[tet[0]] - dv,
                tetra.points[tet[1]] - dv,
                tetra.points[tet[2]] - dv]
        grains[igr]['Volume'] += np.abs(np.linalg.det(vmat)) / 6.

    # Keep only facets at boundary or between different grains
    facet_keys = set()
    for i, tet in enumerate(tetra.simplices):
        igr = tet_to_grain[i]
        if (igr == 0) and (0 not in grains.keys()):
            continue  # special case of precipit, where grain 0 is assigned to pores
        for j, neigh in enumerate(tetra.neighbors[i, :]):
            if neigh == -1 or tet_to_grain[neigh] != igr:
                ft = []
                for k in range(4):
                    if k != j:
                        ft.append(tet[k])
                ft = sorted(ft)
                facet_keys.add(f'{ft[0]}_{ft[1]}_{ft[2]}')
                grains[igr]['Simplices'].append(ft)
                # Update grain surface area
                cv = tetra.points[ft[2]]
                avec = np.cross(tetra.points[ft[0]] - cv,
                                tetra.points[ft[1]] - cv)
                grains[igr]['Area'] += np.linalg.norm(avec)

    # perform geometrical analysis of grain structure
    facets = []
    for key in facet_keys:
        hh = key.split('_')
        facets.append([int(hh[0]), int(hh[1]), int(hh[2])])
    geometry['Facets'] = np.array(facets)

    for igr in mesh.grain_dict.keys():
        if grains[igr]['Volume'] < 1.e-5:
            logging.warning(f'No tet assigned to grain {igr}.')
            if grains[igr]['Simplices']:
                nf = len(grains[igr]['Simplices'])
                logging.warning(f'Grain {igr} contains {nf} tets, but no volume')
            grains.pop(igr)
            continue
        grains[igr]['eqDia'] = 2.0 * (3.0 * grains[igr]['Volume']
                                      / (4.0 * np.pi)) ** (1.0 / 3.0)
        # Approximate smallest rectangular cuboid around points of grains
        # to analyse prolate (aspect ratio > 1) and oblate (a.r. < 1) particles correctly
        pts = grains[igr]['Points']
        if len(pts) < 4:
            logging.warning(f'Removing grain {igr} with few vertices: {grains[igr]["Vertices"]}')
            grains.pop(igr)
            continue
        dia = get_diameter(pts)  # approx. of largest diameter of grain
        len_a = np.linalg.norm(dia)  # length of largest side
        if len_a < 1.e-5:
            logging.warning(f'Very small grain {igr} with max. diameter = {len_a}')
        ax_a = dia / len_a  # unit vector along longest side
        ppt = project_pts(pts, grains[igr]['Center'], ax_a)  # project points onto central plane normal to diameter
        trans1 = get_diameter(ppt)  # largest side transversal to long axis
        len_b = np.linalg.norm(trans1)  # length of second-largest side
        ax_b = trans1 / len_b  # unit vector of second axis (normal to diameter)
        ax_c = np.cross(ax_a, ax_b)  # calculate third orthogonal axes of rectangular cuboid
        lpt = project_pts(ppt, np.zeros(3), ax_b)  # project points on third axis
        pdist = np.array([np.dot(ax_c, v) for v in lpt])  # calculate distance of points on third axis
        len_c = np.max(pdist) - np.min(pdist)  # get length of shortest side
        # calculate list of aspect ratios to decide upon most likely rotational symmetry axis
        ar_list = [np.abs(len_a / len_b - 1.0), np.abs(len_b / len_a - 1.0),
                   np.abs(len_b / len_c - 1.0), np.abs(len_c / len_b - 1.0),
                   np.abs(len_c / len_a - 1.0), np.abs(len_a / len_c - 1.0)]
        minval = np.min(ar_list)
        if minval > 0.15:
            # no clear rotational symmetry, choose longest axis as major diameter
            grains[igr]['majDia'] = len_a
            grains[igr]['minDia'] = 0.5 * (len_b + len_c)
        else:
            ind = np.argmin(ar_list)  # identify two axes with aspect ratio closest to 1
            if ind in [0, 1]:
                grains[igr]['majDia'] = len_c  # major diameter along the rotational axis of grain
                grains[igr]['minDia'] = 0.5 * (len_a + len_b)  # minor diameter is average of transversal axes
            elif ind in [2, 3]:
                grains[igr]['majDia'] = len_a
                grains[igr]['minDia'] = 0.5 * (len_c + len_b)
            else:
                grains[igr]['majDia'] = len_b
                grains[igr]['minDia'] = 0.5 * (len_a + len_c)

    geometry['Grains'] = grains
    geometry['GBnodes'] = gbDict
    geometry['GBarea'] = shared_area
    geometry['GBfaces'] = grain_facesDict
    print('Finished generating polyhedral hulls for grains.')

    # Analyse error in polyhedral grain volume
    vref = np.prod(rve.size)
    vtot = 0.
    vtot_vox = 0.
    vunit = vref / np.prod(rve.dim)
    vol_mae = 0.
    for igr, grain in geometry['Grains'].items():
        vg = grain['Volume']
        vtot += vg
        vvox = np.count_nonzero(mesh.grains == igr) * vunit
        vtot_vox += vvox
        vol_mae += np.abs(1. - vg / vvox)
    if np.abs(vtot_vox - vref) > 1.e-5 and not bool(mesh.prec_vf_voxels):
        logging.warning(f'Inconsistent volume of voxelized grains: {vtot_vox}, ' +
                        f'Reference volume: {vref}. Grians missing in polyhedral structure.')
    if np.abs(vtot - vref) > 1.e-5 and not bool(mesh.prec_vf_voxels):
        logging.warning(f'Inconsistent volume of polyhedral grains: {vtot}, ' +
                        f'Reference volume: {vref}')
    print(f'Total volume of RVE: {vref} {rve.units}^3')
    print(f'Total volume of polyhedral grains: {vtot} {rve.units}^3')
    if bool(mesh.prec_vf_voxels):
        geometry['Porosity_grains'] = 1. - vtot / vref
        print(f'Porosity in voxel structure: {mesh.prec_vf_voxels}')
        print(f'Porosity in polyhedral grains: {geometry["Porosity_grains"]}')
    print(f'Mean relative error of polyhedral vs. voxel volume of individual ' +
          f'grains: {round(vol_mae / Ngr, 3)}')
    print(f'for mean volume of grains: {round(vref / Ngr, 3)} {rve.units}^3.')
    return geometry
