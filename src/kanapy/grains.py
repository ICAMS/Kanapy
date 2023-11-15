import itertools
import warnings

import numpy
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean
from tqdm import tqdm


def calcPolygons(RVE_data, nodes_v, elmtSetDict, elmtDict, Ngr,
                 voxels, phases, vox_centerDict, tol=1.e-3,
                 dual_phase=False):
    """
    Evaluates the grain volume and the grain boundary shared surface area
    between neighbouring grains in voxelized microstrcuture. Generate
    vertices as contact points between 3 or more grains. Generate
    polyhedral convex hull for vertices.

    Parameters
    ----------
    tol : TYPE, optional
        DESCRIPTION. The default is 1.e-3.

    Returns
    -------
    grain_facesDict : TYPE
        DESCRIPTION.
    gbDict : TYPE
        DESCRIPTION.
    shared_area : TYPE
        DESCRIPTION.

    ISSUES: for periodic structures, large grains with segments in both
    halves of the box and touching one boundary are split wrongly

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
        n1 = nodes_v[conn[0] - 1, :]
        n2 = nodes_v[conn[1] - 1, :]
        n3 = nodes_v[conn[2] - 1, :]
        n4 = nodes_v[conn[3] - 1, :]
        face = np.zeros((3,2), dtype=bool)
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
        ''' Check if close neighbors of new vertices are already in list
        # nodes: list of nodes identified as new vertices
        # grains: set of grains containing the nodes in elist
        # margin: radius in which vertices will be united'''

        # create set of all vertices of all involved grains
        vset = set()
        for gid in grains:
            vset.update(vert[gid])
        # loop over all combinations
        for i, nid1 in enumerate(nodes):
            npos1 = nodes_v[nid1 - 1]
            sur1 = [np.abs(npos1[0] - RVE_min[0]) < tol,
                    np.abs(npos1[1] - RVE_min[1]) < tol,
                    np.abs(npos1[2] - RVE_min[2]) < tol,
                    np.abs(npos1[0] - RVE_max[0]) < tol,
                    np.abs(npos1[1] - RVE_max[1]) < tol,
                    np.abs(npos1[2] - RVE_max[2]) < tol]
            for nid2 in vset:
                npos2 = nodes_v[nid2 - 1]
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
        Get voxel associated with position vector pos.

        Parameters
        ----------
        pos : TYPE
            DESCRIPTION.

        Returns
        -------
        v0 : TYPE
            DESCRIPTION.
        v1 : TYPE
            DESCRIPTION.
        v2 : TYPE
            DESCRIPTION.

        """
        v0 = np.minimum(int(pos[0] / RVE_data['Voxel_resolutionX']),
                        RVE_data['Voxel_numberX'] - 1)
        v1 = np.minimum(int(pos[1] / RVE_data['Voxel_resolutionY']),
                        RVE_data['Voxel_numberY'] - 1)
        v2 = np.minimum(int(pos[2] / RVE_data['Voxel_resolutionZ']),
                        RVE_data['Voxel_numberZ'] - 1)
        return (v0, v1, v2)

    def tet_in_grain(tet, vertices):
        """

        Parameters
        ----------
        tet
        vertices

        Returns
        -------

        """
        return RVE_data['Vertices'][tet[0]] in vertices and \
               RVE_data['Vertices'][tet[1]] in vertices and \
               RVE_data['Vertices'][tet[2]] in vertices and \
               RVE_data['Vertices'][tet[3]] in vertices

    def vox_in_tet(vox_, tet_):
        """
        Determine whether centre of voxel lies within tetrahedron

        Parameters
        ----------
        vox_
        tet_

        Returns
        -------
        contained : bool
            Voxel lies in tetrahedron
        """

        v_pos = vox_centerDict[vox_]
        contained = True
        for node in tet_:
            n_pos = RVE_data['Points'][node]
            hh = set(tet_)
            hh.remove(node)
            ind_ = list(hh)
            f_pos = RVE_data['Points'][ind_]
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

    # define constants
    voxel_size = RVE_data['Voxel_resolutionX']
    RVE_min = np.amin(nodes_v, axis=0)
    if np.any(RVE_min > 1.e-3) or np.any(RVE_min < -1.e-3):
        raise ValueError('Irregular RVE geometry: RVE_min = {}'.format(RVE_min))
    RVE_max = np.amax(nodes_v, axis=0)
    Ng = np.amax(list(elmtSetDict.keys()))  # highest grain number

    # create dicts for GB facets, including fake facets at surfaces
    grain_facesDict = dict()  # {Grain: faces}
    for i in range(1, Ng + 7):
        grain_facesDict[i] = dict()

    # genrate following objects:
    # outer_faces: {face_id's of outer voxel faces}  (potential GB facets)
    # face_nodes: {face_id: list with 4 nodes}
    # grain_facesDict: {grain_id: {face_id: list with 4 nodes}}
    for gid, elset in elmtSetDict.items():
        outer_faces = set()
        face_nodes = dict()
        nodeConn = [elmtDict[el] for el in elset]  # Nodal connectivity of a voxel

        # For each voxel, re-create its 6 faces
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
                        grain_facesDict[Ng + 1 + 2*i + j][of] = conn

    # Find all combinations of grains to check for common area
    # analyse grain_facesDict and create object:
    # gbDict: {f{gid1}_{gid2}: list with 4 nodes shared by grains #gid1 and #gid2}
    # shared_area: [[gid1, gid2, GB area]]
    shared_area = []  # GB area
    gbDict = dict()  # voxel factes on GB
    # Find the shared area and generate gbDict for all pairs of neighboring grains
    combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))
    for cb in combis:
        finter = set(grain_facesDict[cb[0]]).intersection(set(grain_facesDict[cb[1]]))
        if finter:
            ind = set()
            [ind.update(grain_facesDict[cb[0]][key]) for key in finter]
            key = 'f{}_{}'.format(cb[0], cb[1])
            gbDict[key] = ind
            if cb[0] <= Ng and cb[1] <= Ng:
                # grain facet is not on boundary
                try:
                    hull = ConvexHull(nodes_v[list(ind), :])
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
                    rn = nodes_v[edge - 1]
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
                if igr <= Ng:
                    gr_set.add(igr)
                igr = int(key0[j + 1:])
                if igr <= Ng:
                    gr_set.add(igr)
                j = key1.index('_')
                igr = int(key1[1:j])
                if igr <= Ng:
                    gr_set.add(igr)
                igr = int(key1[j + 1:])
                if igr <= Ng:
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
    num_vert = [len(vert[igr]) for igr in elmtSetDict.keys()]
    glist = np.array(list(elmtSetDict.keys()), dtype=int)
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
        grains[igr]['Vertices'] = np.array(list(vert[igr]), dtype=int) - 1
        grains[igr]['Points'] = nodes_v[grains[igr]['Vertices']]
        center = np.mean(grains[igr]['Points'], axis=0)
        grains[igr]['Center'] = center
        # initialize values to be incrementally updated later
        grains[igr]['Simplices'] = []
        grains[igr]['Volume'] = 0.
        grains[igr]['Area'] = 0.

        # Construct incremental Delaunay tesselation of
        # structure given by vertices
        vlist = np.array(list(add_vert), dtype=int) - 1
        vertices = np.append(vertices, list(add_vert))
        if step == 0:
            tetra = Delaunay(nodes_v[vlist], incremental=True)
        else:
            try:
                tetra.add_points(nodes_v[vlist])
            except:
                #print(f'Incremental Delaunay tesselation failed for grain {step + 1}. Restarting Delaunay process from there')
                vlist = np.array(vertices, dtype=int) - 1
                tetra = Delaunay(nodes_v[vlist], incremental=True)

    tetra.close()
    # store global result of tesselation
    RVE_data['Vertices'] = np.array(vertices, dtype=int) - 1
    RVE_data['Points'] = tetra.points
    RVE_data['Simplices'] = tetra.simplices

    # assign simplices (tetrahedra) to grains
    Ntet = len(tetra.simplices)
    print('\nGenerated Delaunay tesselation of grain vertices.')
    print(f'Assigning {Ntet} tetrahedra to grains ...')
    tet_to_grain = np.zeros(Ntet, dtype=int)
    for i, tet in tqdm(enumerate(tetra.simplices)):
        ctr = np.mean(tetra.points[tet], axis=0)
        igr = voxels[get_voxel(ctr)]
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
                    for vox in elmtSetDict[ngr]:
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
        grains[igr]['Volume'] += np.abs(np.linalg.det(vmat))/6.

    # Keep only facets at boundary or between different grains
    facet_keys = set()
    for i, tet in enumerate(tetra.simplices):
        igr = tet_to_grain[i]
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
    RVE_data['Facets'] = np.array(facets)

    for igr in elmtSetDict.keys():
        if grains[igr]['Volume'] < 1.e-5:
            warnings.warn(f'No tet assigned to grain {igr}.')
            if grains[igr]['Simplices']:
                nf = len(grains[igr]['Simplices'])
                warnings.warn(f'Grain {igr} contains {nf} tets, but no volume')
        # Find the euclidean distance to all surface points from the center
        dists = [euclidean(grains[igr]['Center'], pt) for pt in
                 nodes_v[grains[igr]['Vertices']]]
        if len(dists) == 0:
            warnings.warn(f'Grain {igr} with few vertices: {grains[igr]["Vertices"]}')
            dists = [0.]
        grains[igr]['eqDia'] = 2.0 * (3 * grains[igr]['Volume']
                                      / (4 * np.pi)) ** (1 / 3)
        grains[igr]['majDia'] = 2.0 * np.amax(dists)
        grains[igr]['minDia'] = 2.0 * np.amin(dists)
        if dual_phase:
            grains[igr]['PhaseID'] = phases['Phase number'][igr - 1]
            grains[igr]['PhaseName'] = phases['Phase name'][igr - 1]

    RVE_data['Grains'] = grains
    RVE_data['GBnodes'] = gbDict
    RVE_data['GBarea'] = shared_area
    print('Finished generating polyhedral hulls for grains.')
    vref = RVE_data['RVE_sizeX'] * \
           RVE_data['RVE_sizeY'] * \
           RVE_data['RVE_sizeZ']
    vtot = 0.
    vtot_vox = 0.
    vunit = RVE_data['Voxel_resolutionX'] * \
            RVE_data['Voxel_resolutionY'] * \
            RVE_data['Voxel_resolutionZ']
    vol_mae = 0.
    for igr, grain in RVE_data['Grains'].items():
        vg = grain['Volume']
        vtot += vg
        vvox = np.count_nonzero(voxels == igr) * vunit
        vtot_vox += vvox
        vol_mae += np.abs(1. - vg / vvox)
        #print(f'igr: {igr}, vol={vg}, vox={vvox}')
    vol_mae /= Ngr
    if np.abs(vtot - vref) > 1.e-5:
        warnings.warn(f'Inconsistent volume of polyhedral grains: {vtot}, Reference volume: {vref}')
    print(f'Mean absolute error of polyhedral vs. voxel volume of grains: {vol_mae}')

    return grain_facesDict, shared_area


def get_stats(particle_data, Ngr, RVE_data, nphase, 
              dual_phase=False, phase_list=None):
    """
    Compare the geometries of particles used for packing and the resulting
    grains.

    Parameters
    ----------
    save_files : bool, optional
        Indicate if output is written to file. The default is False.

    Returns
    -------
    output_data : dict
        Statistical information about particle and grain geometries.

    """
    # Analyse geometry of particles used for packing algorithm
    par_eqDia = np.array(particle_data['Equivalent_diameter'])
    if particle_data['Type'] == 'Elongated':
        par_majDia = np.array(particle_data['Major_diameter'])
        par_minDia = np.array(particle_data['Minor_diameter1'])

    # Analyse grain geometries
    grain_eqDia = np.zeros(Ngr)
    grain_majDia = np.zeros(Ngr)
    grain_minDia = np.zeros(Ngr)
    for i, igr in enumerate(RVE_data['Grains'].keys()):
        grain_eqDia[i] = RVE_data['Grains'][igr]['eqDia']
        if particle_data['Type'] == 'Elongated':
            grain_minDia[i] = RVE_data['Grains'][igr]['minDia']
            grain_majDia[i] = RVE_data['Grains'][igr]['majDia']

    output_data_list = []
    if dual_phase:
        # Compute the L1-error between particle and grain geometries for phases
        part_PhaseID = np.array(phase_list)
        grain_PhaseID = np.zeros(Ngr)
        grain_PhaseName = np.zeros(Ngr).astype(str)
        for i, igr in enumerate(RVE_data['Grains'].keys()):
            grain_PhaseID[i] = np.array(RVE_data['Grains'][igr]['PhaseID'])
            grain_PhaseName[i] = RVE_data['Grains'][igr]['PhaseName']
        for iph in range(nphase):
            ind_grn = np.nonzero(grain_PhaseID == iph)
            ind_par = np.nonzero(part_PhaseID == iph)
            
            error = l1_error_est(par_eqDia[ind_par], grain_eqDia[ind_grn])
            print('\n    L1 error phase {} between particle and grain geometries: {}' \
                  .format(iph, round(error, 5)))

            # Create dictionaries to store the data generated
            output_data = {'Number_of_particles/grains': int(len(par_eqDia[ind_par])),
                           'Grain type': particle_data['Type'],
                           'Unit_scale': RVE_data['Units'],
                           'L1-error': error,
                           'Particle_Equivalent_diameter': par_eqDia[ind_par],
                           'Grain_Equivalent_diameter': grain_eqDia[ind_grn]}
            if particle_data['Type'] == 'Elongated':
                output_data['Particle_Major_diameter'] = par_majDia[ind_par]
                output_data['Particle_Minor_diameter'] = par_minDia[ind_par]
                output_data['Grain_Major_diameter'] = grain_majDia[ind_grn]
                output_data['Grain_Minor_diameter'] = grain_minDia[ind_grn]
                output_data['PhaseID'] = grain_PhaseID[ind_grn]
                output_data['PhaseName'] = grain_PhaseName[ind_grn]
            output_data_list.append(output_data)

    else:
        # Compute the L1-error between particle and grain geometries
        error = l1_error_est(par_eqDia, grain_eqDia)
        print('\n    L1 error between particle and grain geometries: {}' \
              .format(round(error, 5)))

        # Create dictionaries to store the data generated
        output_data = {'Number_of_particles/grains': int(len(par_eqDia)),
                       'Grain type': particle_data['Type'],
                       'Unit_scale': RVE_data['Units'],
                       'L1-error': error,
                       'Particle_Equivalent_diameter': par_eqDia,
                       'Grain_Equivalent_diameter': grain_eqDia}
        if particle_data['Type'] == 'Elongated':
            output_data['Particle_Major_diameter'] = par_majDia
            output_data['Particle_Minor_diameter'] = par_minDia
            output_data['Grain_Major_diameter'] = grain_majDia
            output_data['Grain_Minor_diameter'] = grain_minDia
        output_data_list.append(output_data)

    return output_data_list


def l1_error_est(par_eqDia, grain_eqDia):
    r"""
    Evaluates the L1-error between the particle- and output RVE grain
    statistics with respect to Major, Minor & Equivalent diameters.

    .. note:: 1. Particle information is read from (.json) file generated by
                 :meth:`kanapy.input_output.particleStatGenerator`.
                 And RVE grain information is read from the (.json) files
                 generated by :meth:`kanapy.voxelization.voxelizationRoutine`.

              2. The L1-error value is written to the 'output_statistics.json'
                 file.
    """

    print('')
    print('Computing the L1-error between input and output diameter distributions', 
          end="")

    # Concatenate both arrays to compute shared bins
    # NOTE: 'doane' produces better estimates for non-normal datasets
    total_eqDia = np.concatenate([par_eqDia, grain_eqDia])
    shared_bins = np.histogram_bin_edges(total_eqDia, bins='doane')

    # Compute the histogram for particles and grains
    hist_par, _ = np.histogram(par_eqDia, bins=shared_bins)
    hist_gr, _ = np.histogram(grain_eqDia, bins=shared_bins)

    # Normalize the values
    hist_par = hist_par/np.sum(hist_par)
    hist_gr = hist_gr/np.sum(hist_gr)

    # Compute the L1-error between particles and grains
    l1_value = np.sum(np.abs(hist_par - hist_gr))
    return l1_value
