import os
import re
import logging
import numpy as np
from collections import defaultdict
from kanapy.entities import Ellipsoid, Cuboid


def write_dump(Ellipsoids, sim_box):
    """
    Writes the (.dump) file, which can be read by visualization software OVITO.  

    :param Ellipsoids: Contains information of ellipsoids such as its position, axes lengths and tilt angles 
    :type Ellipsoids: list    
    :param sim_box: Contains information of the dimensions of the simulation box
    :type sim_box: :obj:`Cuboid`    
    :param num_particles: Total number of ellipsoids in the simulation box 
    :type num_particles: int    

    .. note:: This function writes (.dump) files containing simulation domain and ellipsoid attribute information. 
    """
    num_particles = len(Ellipsoids)
    cwd = os.getcwd()
    output_dir = cwd + '/dump_files'  # output directory
    dump_outfile = output_dir + '/particle'  # output dump file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(dump_outfile + ".{0}.dump".format(sim_box.sim_ts), 'w') as f:
        f.write('ITEM: TIMESTEP\n')
        f.write('{0}\n'.format(sim_box.sim_ts))
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write('{0}\n'.format(num_particles))
        f.write('ITEM: BOX BOUNDS ff ff ff\n')
        f.write('{0} {1}\n'.format(sim_box.left, sim_box.right))
        f.write('{0} {1}\n'.format(sim_box.bottom, sim_box.top))
        f.write('{0} {1}\n'.format(sim_box.back, sim_box.front))
        f.write(
            'ITEM: ATOMS id x y z AsphericalShapeX AsphericalShapeY AsphericalShapeZ OrientationX OrientationY OrientationZ OrientationW\n')
        for ell in Ellipsoids:
            qw, qx, qy, qz = ell.quat
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(
                ell.id, ell.x, ell.y, ell.z, ell.a, ell.b, ell.c, qx, qy, qz, qw))


def read_dump(file):
    """
    Reads the (.dump) file to extract information for voxelization (meshing) routine    

    :param file: Contains information of ellipsoids generated in the packing routine.
    :type file: document

    :returns: * Cuboid object that represents the RVE.
              * List of ellipsoid objects that represent the grains.
    :rtype: Tuple of python objects (:obj:`Cuboid`, :obj:`Ellipsoid`)
    """
    print('    Reading the .dump file for particle information')

    try:
        # Read the Simulation box dimensions
        with open(file, 'r+') as fd:
            lookup = "ITEM: NUMBER OF ATOMS"
            lookup2 = "ITEM: BOX BOUNDS ff ff ff"
            for num, lines in enumerate(fd, 1):
                if lookup in lines:
                    number_particles = int(next(fd))
                    par_line_num = num + 7

                if lookup2 in lines:
                    valuesX = re.findall(r'\S+', next(fd))
                    RVE_minX, RVE_maxX = list(map(float, valuesX))

                    valuesY = re.findall(r'\S+', next(fd))
                    RVE_minY, RVE_maxY = list(map(float, valuesY))

                    valuesZ = re.findall(r'\S+', next(fd))
                    RVE_minZ, RVE_maxZ = list(map(float, valuesZ))

    except FileNotFoundError:
        raise FileNotFoundError('.dump file not found, execute packing with option save_files=True')

    # Create an instance of simulation box
    sim_box = Cuboid(RVE_minX, RVE_maxY, RVE_maxX, RVE_minY, RVE_maxZ, RVE_minZ)

    # Read the particle shape & position information
    # Create instances for ellipsoids & assign values from dump files
    Ellipsoids = []
    with open(file, "r") as f:
        count = 0
        for num, lines in enumerate(f, 1):
            if num >= par_line_num:

                count += 1
                values = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                iden = count  # ellipsoid 'id'
                a, b, c = values[4], values[5], values[6]  # Semi-major length, Semi-minor length 1 & 2
                x, y, z = values[1], values[2], values[3]
                qx, qy, qz, qw = values[7], values[8], values[9], values[10]
                quat = np.array([qw, qx, qy, qz])
                ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat)  # instance of Ellipsoid class

                # Find the original particle if the current is duplicate
                for c in values[0]:
                    if c == '_':
                        split_str = values[0].split("_")
                        original_id = int(split_str[0])
                        ellipsoid.duplicate = original_id
                        break
                    else:
                        continue

                Ellipsoids.append(ellipsoid)

    return sim_box, Ellipsoids


def export2abaqus(nodes, file, elmtSetDict, voxel_dict, units='mm',
                  grain_facesDict=None, dual_phase=False, thermal=False):
    r"""
    Creates an ABAQUS input file with microstructure morphology information
    in the form of nodes, elements and element sets.

    .. note:: 1. JSON files generated by :meth:`kanapy.voxelization.voxelizationRoutine` are read to generate the ABAQUS (.inp) file.
                 The json files contain:

                 * Node ID and its corresponding coordinates
                 * Element ID with its nodal connectivities
                 * Element sets representing grains (Assembly of elements) 
                 
              2. The nodal coordinates are written out in :math:`mm` or :math:`\mu m` scale, as requested by the user in the input file.
    """
    
    def write_grain_sets():
        # Create element sets for grains
        for k, v in elmtSetDict.items():
            f.write('*ELSET, ELSET=GRAIN{0}_SET\n'.format(k))
            for enum, el in enumerate(v, 1):
                if enum % 16 != 0:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d, ' % el)
                else:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d\n' % el)
        # Create sections
        for k in elmtSetDict.keys():
            f.write(
                '*Solid Section, elset=GRAIN{0}_SET, material=GRAIN{1}_MAT\n'\
                    .format(k, k))
        return
    
    def write_phase_sets():
        # Create element sets for phases
        for k, v in elmtSetDict.items():
            f.write('*ELSET, ELSET=PHASE{0}_SET\n'.format(k))
            for enum, el in enumerate(v, 1):
                if enum % 16 != 0:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d, ' % el)
                else:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d\n' % el)
        # Create sections
        for k in elmtSetDict.keys():
            f.write(
                '*Solid Section, elset=PHASE{0}_SET, material=PHASE{1}_MAT\n'\
                    .format(k, k))
        return
            

    print('')
    print('Writing ABAQUS (.inp) file', end="")
    # select element type
    if grain_facesDict is None:
        if thermal:
            print('Using brick element type C3D8T for coupled structural-thermal analysis.')
            eltype = 'C3D8T'
        else:
            print('Using brick element type C3D8.')
            eltype = 'C3D8'
    else:
        print('Using tet element type SFM3D4.')
        eltype = 'SFM3D4'

    # Factor used to generate nodal cordinates in 'mm' or 'um' scale
    if units == 'mm':
        scale_fact = 1.e-3
    elif units == 'um':
        scale_fact = 1.
    else:
        raise ValueError(f'Units must be either "mm" or "um", not "{0}"'\
                         .format(units))

    with open(file, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('** Nodal coordinates scale in {0}\n'.format(units))
        f.write('*HEADING\n')
        f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=PART-1\n')
        f.write('*Node\n')
        # Create nodes
        for k, v in enumerate(nodes):
            # Write out coordinates in 'mm' or 'um'
            f.write('{0}, {1}, {2}, {3}\n'.format(k + 1, v[0]*scale_fact,
                                        v[1]*scale_fact, v[2]*scale_fact))
        f.write('*ELEMENT, TYPE={0}\n'.format(eltype))
        if grain_facesDict is None:
            # export voxelized structure with regular hex mesh
            # Create Elements
            for k, v in voxel_dict.items():
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                    k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
            if dual_phase:
                write_phase_sets()
            else:
                write_grain_sets()
        else:
            # export smoothened structure with tetrahedral mesh
            fcList = {}
            fcNum = 0
            gr_fcs = defaultdict(list)
            for gid, ginfo in grain_facesDict.items():
                for fc, conn in ginfo.items():
                    if fc not in fcList.keys():
                        fcNum += 1
                        fcList[fc] = fcNum
                        f.write('%d,%d,%d,%d,%d\n' % (fcNum, conn[0], conn[1], conn[2], conn[3]))
                        gr_fcs[gid].append(fcNum)
                    elif fc in fcList.keys():
                        f.write('%d,%d,%d,%d,%d\n' % (fcList[fc], conn[0], conn[1], conn[2], conn[3]))
                        gr_fcs[gid].append(fcList[fc])

            for gid, fcs in gr_fcs.items():
                f.write('*ELSET, ELSET=GRAIN{}_SET\n'.format(gid))
                for enum, el in enumerate(fcs, 1):
                    if enum % 16 != 0:
                        if enum == len(fcs):
                            f.write('%d\n' % el)
                        else:
                            f.write('%d, ' % el)
                    else:
                        if enum == len(fcs):
                            f.write('%d\n' % el)
                        else:
                            f.write('%d\n' % el)

            for gid, fcs in gr_fcs.items():
                f.write('*SURFACE SECTION, ELSET=GRAIN{}_SET\n'.format(gid))
        f.write('*End Part\n')
        f.write('**\n')
        f.write('**\n')
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=PART-1-1, part=PART-1\n')
        f.write('*End Instance\n')
        f.write('*End Assembly\n')
        f.write('**\n')
        if dual_phase:
            f.write('**\n')
            f.write('** MATERIALS\n')
            f.write('**\n')
            for i in range(1,3):
                f.write('**\n')
                f.write('*Material, name=PHASE{}_MAT\n'.format(i))
                f.write('**Include, input=Material{}.inp\n'.format(i))
                f.write('**\n')
    print('---->DONE!\n')
    return


def writeAbaqusMat(ialloy, angles, file=None, path='./', nsdv=200):
    '''
    angles : Euler angles with number of rows= number of grains and
            three columns phi1, Phi, phi2
    ialloy : alloy number in the umat, mod_alloys.f
    nsdv : number of state dependant variables default value is 200
    '''
    nitem = len(angles)
    titem = '{}grains'.format(nitem)

    if file is None:
        cwd = os.getcwd()
        file = cwd + '/abq_px_{0}_materials.inp'.format(titem)
        if os.path.exists(file):
            os.remove(file)  # remove old file if it exists
    with open(file, 'w') as f:
        f.write('**\n')
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(nitem):
            f.write('*Material, name=GRAIN{}_MAT\n'.format(i + 1))
            f.write('*Depvar\n')
            f.write('    {}\n'.format(nsdv))
            f.write('*User Material, constants=4\n')
            f.write('{}, {}, {}, {}\n'.format(float(ialloy),
                                              angles[i, 0],
                                              angles[i, 1],
                                              angles[i, 2]))
    return


def writeAbaqusPhase(grains, nsdv=200):
    with open('Material.inp', 'w') as f:
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(len(grains)):
            f.write('*Material, name=GRAIN{}_mat\n'.format(i + 1))
            f.write('*Depvar\n')
            f.write('    {}\n'.format(nsdv))
            f.write('*User Material, constants=1\n')
            f.write('{}\n'.format(float(grains[i + 1]['PhaseID'])))
    return


def pickle2microstructure(file, path='./'):
    """Read pickled microstructure file.


    Parameters
    ----------
    file : string
        File name of pickled microstructure to be read.
    path : string
        Path under which pickle-files are stored (optional, default: './')

    Returns
    -------
    pcl : Material object
        unpickled microstructure

    """
    import pickle

    if path[-1] != '/':
        path += '/'
    if file is None:
        raise ValueError('Name for pickled microstructure must be given.')
    fname = os.path.normpath(path + file)
    with open(fname, 'rb') as inp:
        pcl = pickle.load(inp)
    return pcl


def import_voxels(file, path='./'):
    import json
    from kanapy.api import Microstructure
    from kanapy.initializations import RVE_creator, mesh_creator
    from kanapy.entities import Simulation_Box

    if path[-1] != '/':
        path += '/'
    if file is None:
        raise ValueError('Name for voxel file must be given.')
    fname = os.path.normpath(path + file)
    data = json.load(open(fname))

    # extract basic model information
    sh = tuple(data['Data']['Shape'])
    nvox = np.prod(sh)
    size = tuple(data['Model']['Size'])
    gr_numbers = np.unique(data['Data']['Values'])
    grain_keys = np.asarray(gr_numbers, dtype=str)
    grains = np.reshape(data['Data']['Values'], sh, order=data['Data']['Order'])
    ph_names = data['Model']['Phase_names']
    nphases = len(ph_names)
    grain_dict = dict()
    grain_phase_dict = dict()
    gr_arr = grains.flatten(order='F')
    if 'Grains' in data.keys():
        if 'Orientation' in data['Grains'][grain_keys[0]].keys():
            grain_ori_dict = dict()
        else:
            grain_ori_dict = None
        phase_vf = np.zeros(nphases)
        ngrain = np.zeros(nphases, dtype=int)
        phases = np.zeros(nvox, dtype=int)
        for igr in gr_numbers:
            ind = np.nonzero(gr_arr == igr)[0]
            nv = len(ind)
            ip = data['Grains'][str(igr)]['Phase']
            phase_vf[ip] += nv
            grain_dict[igr] = ind + 1
            grain_phase_dict[igr] = ip
            ngrain[ip] += 1
            phases[ind] = ip
            if grain_ori_dict is not None:
                grain_ori_dict[igr] = data['Grains'][str(igr)]['Orientation']
        phase_vf /= nvox
        if not np.isclose(np.sum(phase_vf), 1.):
            logging.warning(f'Volume fractions do not add up to 1: {phase_vf}')
    else:
        # no grain-level information in data
        if nphases > 1:
            logging.error('No grain-level information in data file.' +
                          'Cannot extract phase information or orientations.' +
                          'Continuing with single phase model.')
            nphases = 1
            phase_vf = 1.
        for igr in gr_numbers:
            ind = np.nonzero(gr_arr == igr)[0]
            grain_dict[igr] = ind + 1
            grain_phase_dict[igr] = 0
        ngrain = [len(grain_keys)]
        phases = np.zeros(nvox, dtype=int)

    # reconstructing microstructure information for RVE
    stats_dict = {
        'RVE': {'sideX': size[0], 'sideY': size[1], 'sideZ': size[2],
                'Nx': sh[0], 'Ny': sh[1], 'Nz': sh[2]},
        'Simulation': {'periodicity': data['Model']['Periodicity'],
                       'output_units': data['Model']['Units']['Length']},
        'Phase': {'Name': 'Simulanium', 'Volume fraction': 1.0}
    }
    # add phase information and construct list of stats_dict's
    stats_list = []
    for i in range(nphases):
        stats_dict['Phase']['Name'] = ph_names[i]
        stats_dict['Phase']['Volume fraction'] = phase_vf[i]
        stats_list.append(stats_dict)
    # Create microstructure object
    ms = Microstructure('from_voxels')
    ms.name = data['Model']['Material']
    ms.Ngr = np.sum(ngrain)
    ms.nphases = nphases
    ms.descriptor = stats_list
    ms.ngrains = ngrain
    ms.rve = RVE_creator(stats_list, from_voxels=True)
    ms.simbox = Simulation_Box(size)
    # initialize voxel structure (= mesh)
    ms.mesh = mesh_creator(sh)
    ms.mesh.nphases = nphases
    ms.mesh.grains = grains
    ms.mesh.grain_dict = grain_dict
    ms.mesh.grain_ori_dict = grain_ori_dict
    ms.mesh.phases = phases.reshape(sh, order='F')
    ms.mesh.grain_phase_dict = grain_phase_dict
    ms.mesh.ngrains_phase = ngrain
    # import or create mesh
    voxel_dict = dict()
    vox_centerDict = dict()
    if 'Mesh' in data.keys():
        nodes = np.array(data['Mesh']['Nodes']['Values'])
        for i, el in enumerate(data['Mesh']['Voxels']['Values']):
            voxel_dict[i+1] = el
            ind = np.array(el, dtype=int) - 1
            vox_centerDict[i+1] = np.mean(nodes[ind], axis=0)
        ms.mesh.voxel_dict = voxel_dict
        ms.mesh.vox_center_dict = vox_centerDict
        ms.mesh.nodes = nodes
    else:
        ms.mesh.create_voxels(ms.simbox)

    return ms
