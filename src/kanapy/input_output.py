import os
import re
import logging
import numpy as np
from collections import defaultdict
from kanapy.entities import Ellipsoid, Cuboid


def write_dump(Ellipsoids, sim_box):
    """
    Writes the (.dump) files into a sub-directory "dump_files", which can be read by visualization software OVITO
    or imported again into Kanapy to avoid the packing simulation.

    :param Ellipsoids: Contains information of ellipsoids such as its position, axes lengths and tilt angles 
    :type Ellipsoids: list    
    :param sim_box: Contains information of the dimensions of the simulation box
    :type sim_box: :obj:`Cuboid`

    .. note:: This function writes (.dump) files containing simulation domain and ellipsoid attribute information. 
    """
    num_particles = len(Ellipsoids)
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'dump_files')
    dump_outfile = os.path.join(output_dir, 'particle')  # output dump file
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
            'ITEM: ATOMS id x y z AsphericalShapeX AsphericalShapeY AsphericalShapeZ OrientationX OrientationY ' +
            'OrientationZ OrientationW\n')
        for ell in Ellipsoids:
            qw, qx, qy, qz = ell.quat
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'.format(
                ell.id, ell.x, ell.y, ell.z, ell.a, ell.b, ell.c, qx, qy, qz, qw, ell.phasenum))


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
                values: list = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                iden = count  # ellipsoid 'id'
                a, b, c = values[4], values[5], values[6]  # Semi-major length, Semi-minor length 1 & 2
                x, y, z = values[1], values[2], values[3]
                qx, qy, qz, qw = values[7], values[8], values[9], values[10]
                ip = int(values[11])
                quat = np.array([qw, qx, qy, qz])
                ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat, phasenum=ip)  # instance of Ellipsoid class

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


from kanapy.initializations import NodeSets
from collections import defaultdict


from kanapy.initializations import NodeSets
from collections import defaultdict

def export2abaqus(nodes, file, grain_dict, voxel_dict, units='um',
                  gb_area=None, dual_phase=False, thermal=False,
                  ialloy=None, grain_phase_dict=None, periodic=False,
                  bc_type='Uni-axial', load_type='strain', loading_direction='x', value=0.20000, apply_bc=False):
    r"""
    Creates an ABAQUS input file with microstructure morphology information
    in the form of nodes, elements and element sets. If "dual_phase" is true,
    element sets with phase numbers will be defined and assigned to materials
    "PHASE_{phase_id}MAT" plain material definitions for phases will be included.
    Otherwise, it will be assumed that each grain refers to a material
    "GRAIN_{grain_id}MAT. In this case, a "_mat.inp" file with the same name
    trunc will be included, in which the alloy number and Euler angles for each
    grain must be defined.

    Parameters
    ----------
    nodes
    file
    grain_dict
    voxel_dict
    units
    gb_area
    dual_phase
    thermal
    ialloy
    grain_phase_dict
    periodic
    bc_type
    load_type
    loading_direction
    value
    apply_bc : bool, optional
        If True, boundary conditions are written to the Abaqus input file. Default is False.
    """
    def write_node_set(name, nset):
        f.write(name)
        for i, val in enumerate(nset[:-1], start=1):
            if i % 16 == 0:
                f.write(f'{val+1}\n')
            else:
                f.write(f'{val+1}, ')
        f.write(f'{nset[-1]+1}\n')

    def write_grain_sets():
        for k, v in grain_dict.items():
            f.write('*ELSET, ELSET=GRAIN{0}_SET\n'.format(k))
            for enum, el in enumerate(v[:-1], start=1):
                if enum % 16 != 0:
                    f.write('%d, ' % el)
                else:
                    f.write('%d\n' % el)
            f.write('%d\n' % v[-1])
        for k in grain_dict.keys():
            if grain_phase_dict is None or grain_phase_dict[k] < nall:
                f.write(
                    '*Solid Section, elset=GRAIN{0}_SET, material=GRAIN{1}_MAT\n'
                    .format(k, k))
            else:
                f.write(
                    '*Solid Section, elset=GRAIN{0}_SET, material=PHASE{1}_MAT\n'
                    .format(k, grain_phase_dict[k]))
                ph_set.add(grain_phase_dict[k])
        return

    def write_phase_sets():
        for k, v in grain_dict.items():
            f.write('*ELSET, ELSET=PHASE{0}_SET\n'.format(k))
            for enum, el in enumerate(v, start=1):
                if enum % 16 != 0:
                    if enum == len(v):
                        f.write('%d\n' % el)
                    else:
                        f.write('%d, ' % el)
                else:
                    f.write('%d\n' % el)
        for k in grain_dict.keys():
            f.write(
                '*Solid Section, elset=PHASE{0}_SET, material=PHASE{1}_MAT\n'
                .format(k, k))
            ph_set.add(k)
        return

    def write_surface_sets():
        face_sets = {
            1: ['S3', nsets.F1yz],  # X = max
            2: ['S2', nsets.Fx1z],  # Y = max
            3: ['S6', nsets.Fxy1]   # Z = max
        }
        for i in range(1, 4):
            face_label, face_nodes = face_sets[i]
            face_nodes_set = set(face_nodes)
            f.write('*ELSET, ELSET=_Surf-{0}_{1}, internal, instance=PART-1-1\n'.format(i, face_label))
            line_count = 0
            for enum, item in enumerate(voxel_dict.items()):
                k, v = item
                count = sum((nid in face_nodes_set) for nid in v)
                if count == 4:
                    f.write('%d, ' % k)
                    line_count += 1
                    if line_count == 16:
                        f.write('\n')
                        line_count = 0
            if line_count > 0:
                f.write('\n')
                f.write('*Surface, type=ELEMENT, name=Surf-{0}\n'.format(i))
                f.write('_Surf-{0}_{1}, {1}\n'.format(i, face_label))

    def get_node_weights(face_nodes, direction):
        weights = {}
        interior_count = 0
        edge_count = 0
        corner_count = 0
        # Calculate min/max coordinates for the entire mesh
        if direction == 'x':
            y_min, y_max = min(nodes[:, 1]), max(nodes[:, 1])
            z_min, z_max = min(nodes[:, 2]), max(nodes[:, 2])
            for node in face_nodes:
                y = nodes[node, 1]
                z = nodes[node, 2]
                if (abs(y - y_min) < 1e-10 and abs(z - z_min) < 1e-10) or \
                   (abs(y - y_min) < 1e-10 and abs(z - z_max) < 1e-10) or \
                   (abs(y - y_max) < 1e-10 and abs(z - z_min) < 1e-10) or \
                   (abs(y - y_max) < 1e-10 and abs(z - z_max) < 1e-10):
                    weights[node] = 0.25
                    corner_count += 1
                elif abs(y - y_min) < 1e-10 or abs(y - y_max) < 1e-10 or \
                     abs(z - z_min) < 1e-10 or abs(z - z_max) < 1e-10:
                    weights[node] = 0.5
                    edge_count += 1
                else:
                    weights[node] = 1.0
                    interior_count += 1
        elif direction == 'y':
            x_min, x_max = min(nodes[:, 0]), max(nodes[:, 0])
            z_min, z_max = min(nodes[:, 2]), max(nodes[:, 2])
            for node in face_nodes:
                x = nodes[node, 0]
                z = nodes[node, 2]
                if (abs(x - x_min) < 1e-10 and abs(z - z_min) < 1e-10) or \
                   (abs(x - x_min) < 1e-10 and abs(z - z_max) < 1e-10) or \
                   (abs(x - x_max) < 1e-10 and abs(z - z_min) < 1e-10) or \
                   (abs(x - x_max) < 1e-10 and abs(z - z_max) < 1e-10):
                    weights[node] = 0.25
                    corner_count += 1
                elif abs(x - x_min) < 1e-10 or abs(x - x_max) < 1e-10 or \
                     abs(z - z_min) < 1e-10 or abs(z - z_max) < 1e-10:
                    weights[node] = 0.5
                    edge_count += 1
                else:
                    weights[node] = 1.0
                    interior_count += 1
        elif direction == 'z':
            x_min, x_max = min(nodes[:, 0]), max(nodes[:, 0])
            y_min, y_max = min(nodes[:, 1]), max(nodes[:, 1])
            for node in face_nodes:
                x = nodes[node, 0]
                y = nodes[node, 1]
                if (abs(x - x_min) < 1e-10 and abs(y - y_min) < 1e-10) or \
                   (abs(x - x_min) < 1e-10 and abs(y - y_max) < 1e-10) or \
                   (abs(x - x_max) < 1e-10 and abs(y - y_min) < 1e-10) or \
                   (abs(x - x_max) < 1e-10 and abs(y - y_max) < 1e-10):
                    weights[node] = 0.25
                    corner_count += 1
                elif abs(x - x_min) < 1e-10 or abs(x - x_max) < 1e-10 or \
                     abs(y - y_min) < 1e-10 or abs(y - y_max) < 1e-10:
                    weights[node] = 0.5
                    edge_count += 1
                else:
                    weights[node] = 1.0
                    interior_count += 1
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError(f"No valid weights assigned for nodes in direction {direction}")
        print(f"Direction: {direction}, Nodes: Interior={interior_count}, Edges={edge_count}, "
              f"Corners={corner_count}, Total={len(weights)}, Total Weight={total_weight:.6f}")
        # Debug: Print weights and coordinates for first few nodes
        for i, node in enumerate(face_nodes[:5]):  # Print first 5 nodes
            print(f"Node {node}: Weight={weights.get(node, 0.0):.2f}, "
                  f"Coords=({nodes[node, 0]:.6f}, {nodes[node, 1]:.6f}, {nodes[node, 2]:.6f})")
        return weights

    print('')
    print(f'Writing RVE as ABAQUS file "{file}"')
    if gb_area is None:
        if thermal:
            print('Using brick element type C3D8T for coupled structural-thermal analysis.')
            eltype = 'C3D8T'
        else:
            print('Using brick element type C3D8.')
            eltype = 'C3D8'
    else:
        print('Using tet element type SFM3D4.')
        eltype = 'SFM3D4'

    if type(ialloy) is not list:
        ialloy = [ialloy]
    nall = len(ialloy)
    ph_set = set()
    # Convert input units from µm to mm for Abaqus output
    scale_fact = 0.001  # conversion from µm to mm
    nsets = NodeSets(nodes)

    # Calculate RVE edge lengths and face areas in mm
    edge_lengths = {
        'x': (max(nodes[:, 0]) - min(nodes[:, 0])) * scale_fact,
        'y': (max(nodes[:, 1]) - min(nodes[:, 1])) * scale_fact,
        'z': (max(nodes[:, 2]) - min(nodes[:, 2])) * scale_fact
    }
    face_areas = {
        'x': edge_lengths['y'] * edge_lengths['z'],  # yz face
        'y': edge_lengths['x'] * edge_lengths['z'],  # xz face
        'z': edge_lengths['x'] * edge_lengths['y']   # xy face
    }

    with open(file, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('** Nodal coordinates scale in mm\n')
        f.write('*HEADING\n')
        f.write('*PREPRINT,ECHO=NO,HISTORY=NO,MODEL=NO,CONTACT=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.write('*Part, name=PART-1\n')
        f.write('*Node\n')
        for k, v in enumerate(nodes):
            f.write('{0}, {1}, {2}, {3}\n'.format(k + 1, v[0] * scale_fact,
                                                  v[1] * scale_fact, v[2] * scale_fact))
        f.write('*ELEMENT, TYPE={0}\n'.format(eltype))
        if gb_area is None:
            for k, v in voxel_dict.items():
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                    k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
            if dual_phase:
                write_phase_sets()
            else:
                write_grain_sets()
        else:
            fcList = {}
            fcNum = 0
            gr_fcs = defaultdict(list)
            for gid, ginfo in enumerate(gb_area):
                for fc, conn in ginfo.items():
                    if fc not in fcList.keys():
                        fcNum += 1
                        fcList[fc] = fcNum
                        f.write('%d,%d,%d,%d,%d\n' % (fcNum, conn[0], conn[1], conn[2], conn[3]))
                        gr_fcs[gid].append(fcNum)
                    elif fc in fcList.keys():
                        f.write('%d,%d,%d,%d,%d\n' % (fcList[fc], conn[0], conn[1], conn[2], conn[3]))
                        gr_fcs[gid].append(fcNum)
            for gid, fcs in gr_fcs.items():
                f.write('*ELSET, ELSET=GRAIN{}_SET\n'.format(gid))
                for enum, el in enumerate(fcs, start=1):
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
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=PART-1-1, part=PART-1\n')
        f.write('*End Instance\n')
        f.write('**\n')
        f.write('** DEFINE NODE SETS\n')
        f.write('** 1. VERTICES\n')
        f.write(f'*Nset, nset=V000, instance=PART-1-1\n')
        f.write(f'{nsets.V000+1}\n')
        f.write(f'*Nset, nset=V001, instance=PART-1-1\n')
        f.write(f'{nsets.V001+1}\n')
        f.write(f'*Nset, nset=V010, instance=PART-1-1\n')
        f.write(f'{nsets.V010+1}\n')
        f.write(f'*Nset, nset=V100, instance=PART-1-1\n')
        f.write(f'{nsets.V100+1}\n')
        f.write(f'*Nset, nset=V011, instance=PART-1-1\n')
        f.write(f'{nsets.V011+1}\n')
        f.write(f'*Nset, nset=V101, instance=PART-1-1\n')
        f.write(f'{nsets.V101+1}\n')
        f.write(f'*Nset, nset=V110, instance=PART-1-1\n')
        f.write(f'{nsets.V110+1}\n')
        f.write(f'*Nset, nset=V111, instance=PART-1-1\n')
        f.write(f'{nsets.V111+1}\n')
        f.write('*Nset, nset=Vertices, instance=PART-1-1\n')
        f.write(f'{nsets.V000+1}, {nsets.V100+1}, {nsets.V010+1}, {nsets.V001+1}, {nsets.V011+1}, '
                f'{nsets.V101+1}, {nsets.V110+1}, {nsets.V111+1}\n')
        f.write('** 2. EDGES\n')
        write_node_set('*Nset, nset=Ex00, instance=PART-1-1\n', nsets.Ex00)
        write_node_set('*Nset, nset=Ex01, instance=PART-1-1\n', nsets.Ex01)
        write_node_set('*Nset, nset=Ex10, instance=PART-1-1\n', nsets.Ex10)
        write_node_set('*Nset, nset=Ex11, instance=PART-1-1\n', nsets.Ex11)
        write_node_set('*Nset, nset=E0y0, instance=PART-1-1\n', nsets.E0y0)
        write_node_set('*Nset, nset=E0y1, instance=PART-1-1\n', nsets.E0y1)
        write_node_set('*Nset, nset=E1y0, instance=PART-1-1\n', nsets.E1y0)
        write_node_set('*Nset, nset=E1y1, instance=PART-1-1\n', nsets.E1y1)
        write_node_set('*Nset, nset=E00z, instance=PART-1-1\n', nsets.E00z)
        write_node_set('*Nset, nset=E01z, instance=PART-1-1\n', nsets.E01z)
        write_node_set('*Nset, nset=E10z, instance=PART-1-1\n', nsets.E10z)
        write_node_set('*Nset, nset=E11z, instance=PART-1-1\n', nsets.E11z)
        f.write('** 3. FACES\n')
        write_node_set(f'*Nset, nset=Fxy0, instance=PART-1-1\n', nsets.Fxy0)
        write_node_set(f'*Nset, nset=Fxy1, instance=PART-1-1\n', nsets.Fxy1)
        write_node_set(f'*Nset, nset=Fx0z, instance=PART-1-1\n', nsets.Fx0z)
        write_node_set(f'*Nset, nset=Fx1z, instance=PART-1-1\n', nsets.Fx1z)
        write_node_set(f'*Nset, nset=F0yz, instance=PART-1-1\n', nsets.F0yz)
        write_node_set(f'*Nset, nset=F1yz, instance=PART-1-1\n', nsets.F1yz)
        f.write('** 4. FACES_SETS\n')
        write_surface_sets()
        f.write('**\n')
        f.write('*End Assembly\n')
        f.write('**\n')
        f.write('** MATERIALS\n')
        f.write('**\n')
        if dual_phase:
            for i in ph_set:
                f.write('**\n')
                f.write('*Material, name=PHASE{}_MAT\n'.format(i))
                f.write('**Include, input=Material{}.inp\n'.format(i))
                f.write('**\n')
        else:
            for i in ph_set:
                f.write('**\n')
                f.write('*Material, name=PHASE{}_MAT\n'.format(i))
            f.write('**\n')
            f.write('*Include, input={}mat.inp\n'.format(file[0:-8]))
            f.write('**\n')
            f.write('**__________________________________________________________________')
            f.write('**\n')
            f.write('** STEP: Loading\n')
            f.write('**\n')
            f.write('*Step, name=Loading, nlgeom=YES, inc=500000, unsymm=YES, solver=ITERATIVE\n')
            f.write('*Static\n')
            f.write('1, 250, 1e-6, 1\n')
            f.write('**\n')
            f.write('**\n')
            f.write('*CONTROLS, PARAMETER=TIME INCREMENTATION\n')
            f.write('35, 50, 9, 50, 28, 5, 12, 45\n')
            f.write('**\n')
            f.write('*CONTROLS, PARAMETERS=LINE SEARCH\n')
            f.write('10\n')
            f.write('** Originally that was SOLVER CONTROL\n')
            f.write('*SOLVER CONTROL\n')
            f.write('1e-5,200,\n')
            f.write('**\n')
        if apply_bc and bc_type.lower() == 'uni-axial':
            f.write('** BOUNDARY CONDITIONS\n')
            f.write('**\n')
            f.write('** Name: F0yzFix Type: Displacement/Rotation\n')
            f.write('*Boundary\n')
            f.write('F0YZ, 1, 1\n')
            f.write('** Name: Fx0zFix Type: Displacement/Rotation\n')
            f.write('*Boundary\n')
            f.write('FX0Z, 2, 2\n')
            f.write('** Name: Fxy0Fix Type: Displacement/Rotation\n')
            f.write('*Boundary\n')
            f.write('FXY0, 3, 3\n')
            if load_type == 'strain':
                displacement_bc_map = {
                    'x': ('F1YZ', 1, 'disX'),
                    'y': ('FX1Z', 2, 'disY'),
                    'z': ('FXY1', 3, 'disZ'),
                }
                direction = loading_direction.lower()
                if direction in displacement_bc_map:
                    set_name, dof, bc_name = displacement_bc_map[direction]
                    strain = value / 100.0
                    displacement = strain * edge_lengths[direction]
                    f.write(f'** Name: {bc_name} Type: Displacement/Rotation\n')
                    f.write('*Boundary\n')
                    f.write(f'{set_name}, {dof}, {dof}, {displacement:.6f}\n')
            if load_type == 'stress':
                f.write('** LOADS\n')
                f.write('**\n')
                load_bc_map = {
                    'x': ('F1YZ', 1, 'loadX', nsets.F1yz),
                    'y': ('FX1Z', 2, 'loadY', nsets.Fx1z),
                    'z': ('FXY1', 3, 'loadZ', nsets.Fxy1),
                }
                direction = loading_direction.lower()
                if direction in load_bc_map:
                    set_name, dof, bc_name, face_nodes = load_bc_map[direction]
                    face_area = face_areas[direction]  # Use dynamic face area
                    total_force = value * face_area  # MPa * mm² = N
                    weights = get_node_weights(face_nodes, direction)
                    total_weight = sum(weights.values())
                    if total_weight == 0:
                        raise ValueError(f'No effective nodes for stress application on {set_name}.')
                    base_force = total_force / total_weight
                    print(f"Direction: {direction}, Total nodes: {len(face_nodes)}, "
                          f"Total weight: {total_weight:.6f}, Face area: {face_area:.6f} mm², "
                          f"Base force: {base_force:.10f} N, Total force: {total_force:.10f} N")
                    f.write(f'** Name: {bc_name} Type: Concentrated Force\n')
                    f.write('*Cload\n')
                    for node in face_nodes:
                        if node in weights:
                            force_per_node = base_force * weights[node]
                            f.write(f'PART-1-1.{node+1}, {dof}, {force_per_node:.10f}\n')
                        else:
                            print(f"Warning: Node {node} missing weight, skipping")
        elif apply_bc and bc_type.lower() == 'periodic':
            if not periodic:
                raise ValueError("Periodic boundary conditions cannot be applied to a non-periodic RVE.")
            pass
    print('---->DONE!\n')
    return

def writeAbaqusMat(ialloy, angles,
                   file=None, path='./',
                   grain_phase_dict=None,
                   nsdv=200):
    """
    Export Euler angles to Abaqus input deck that can be included in the _geom.inp file. If
    parameter "grain_phase_dict" is given, the phase number for each grain will be used to select
    the corresponding ialloy from a list. If the list ialloy is shorter than the number of phases in
    grain_phase_dict, no angles for phases with no corresponding ialloy will be written.

    Parameters:
    -----------
    ialloy : int or list
        Identifier, alloy number in ICAMS CP-UMAT: mod_alloys.f
    angles : dict or (N, 3)-ndarray
        Dict with Euler angles for each grain or array with number of N rows (= number of grains) and
            three columns phi1, Phi, phi2
    file : str
        Filename, optional (default: None)
    path : str
        Path to save file, option (default: './')
    grain_phase_dict: dict
        Dict with phase for each grain, optional (default: None)
    nsdv : int
        Number of state dependant variables, optional (default: 200)
    """
    if type(ialloy) is not list:
        ialloy = [ialloy]
    nall = len(ialloy)
    if type(angles) is not dict:
        # converting (N, 3) ndarray to dict
        gr_ori_dict = dict()
        for igr, ori in enumerate(angles):
            gr_ori_dict[igr+1] = ori
    else:
        gr_ori_dict = angles
    nitem = len(gr_ori_dict.keys())
    if file is None:
        file = f'abq_px_{nitem}_mat.inp'
    path = os.path.normpath(path)
    file = os.path.join(path, file)
    with open(file, 'w') as f:
        f.write('**\n')
        f.write('** MATERIALS\n')
        f.write('**\n')
        for igr, ori in gr_ori_dict.items():
            if grain_phase_dict is None:
                ip = 0
            else:
                ip = grain_phase_dict[igr]
                if ip > nall - 1:
                    continue
            f.write('*Material, name=GRAIN{}_MAT\n'.format(igr))
            f.write('*Depvar\n')
            f.write('    {}\n'.format(nsdv))
            f.write('*User Material, constants=4\n')
            f.write('{}, {}, {}, {}\n'.format(float(ialloy[ip]),
                                              ori[0], ori[1], ori[2]))
    return

""" Function not used
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
"""

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

    if file is None:
        raise ValueError('Name for pickled microstructure must be given.')
    fname = os.path.join(path, file)
    with open(fname, 'rb') as inp:
        pcl = pickle.load(inp)
    return pcl


def import_voxels(file, path='./'):
    import json
    import copy
    from kanapy.api import Microstructure
    from kanapy.initializations import RVE_creator, mesh_creator
    from kanapy.entities import Simulation_Box

    if file is None:
        raise ValueError('Name for voxel file must be given.')
    fname = os.path.join(path, file)
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
    phases = np.zeros(nvox, dtype=int)
    grain_dict = dict()
    grain_phase_dict = dict()
    gr_arr = grains.flatten(order='C')
    if 'Grains' in data.keys():
        if 'Orientation' in data['Grains'][grain_keys[-1]].keys():
            grain_ori_dict = dict()
        else:
            grain_ori_dict = None
        phase_vf = np.zeros(nphases)
        ngrain = np.zeros(nphases, dtype=int)
        for igr in gr_numbers:
            ind = np.nonzero(gr_arr == igr)[0]
            nv = len(ind)
            ip = data['Grains'][str(igr)]['Phase']
            phase_vf[ip] += nv
            grain_dict[int(igr)] = ind + 1
            grain_phase_dict[int(igr)] = ip
            ngrain[ip] += 1
            phases[ind] = ip
            if grain_ori_dict is not None:
                if 'Orientation' in data['Grains'][str(igr)].keys():
                    grain_ori_dict[igr] = data['Grains'][str(igr)]['Orientation']
                else:
                    grain_ori_dict[igr] = None
        phase_vf /= nvox
        if not np.isclose(np.sum(phase_vf), 1.):
            logging.warning(f'Volume fractions do not add up to 1: {phase_vf}')
    else:
        # no grain-level information in data
        grain_ori_dict = None
        if nphases > 1:
            logging.error('No grain-level information in data file.' +
                          'Cannot extract phase information or orientations.' +
                          'Continuing with single phase model.')
            nphases = 1
        for igr in gr_numbers:
            ind = np.nonzero(gr_arr == igr)[0]
            grain_dict[int(igr)] = ind + 1
            grain_phase_dict[int(igr)] = 0
        ngrain = [len(grain_keys)]
        phase_vf = [1.]

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
        stats_list.append(copy.deepcopy(stats_dict))
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
    ms.mesh.grains = grains
    ms.mesh.grain_dict = grain_dict
    ms.mesh.grain_ori_dict = grain_ori_dict
    ms.mesh.phases = phases.reshape(sh, order='C')
    ms.mesh.grain_phase_dict = grain_phase_dict
    ms.mesh.ngrains_phase = ngrain
    if 0 in ms.mesh.grain_dict.keys():
        porosity = len(ms.mesh.grain_dict[0]) / nvox
        ms.precipit = porosity
        ms.mesh.prec_vf_voxels = porosity
    # import or create mesh
    voxel_dict = dict()
    vox_centerDict = dict()
    if 'Mesh' in data.keys():
        nodes = np.array(data['Mesh']['Nodes']['Values'])
        for i, el in enumerate(data['Mesh']['Voxels']['Values']):
            voxel_dict[i + 1] = el
            ind = np.array(el, dtype=int) - 1
            vox_centerDict[i + 1] = np.mean(nodes[ind], axis=0)
        ms.mesh.voxel_dict = voxel_dict
        ms.mesh.vox_center_dict = vox_centerDict
        ms.mesh.nodes = nodes
    else:
        ms.mesh.create_voxels(ms.simbox)
    print('\n Voxel structure imported.\n')

    return ms


def write_stats(stats, file, path='./'):
    """
    Write statistical descriptors of microstructure to JSON file.

    Parameters
    ----------
    file : string
        File name of pickled microstructure to be read.
    path : string
        Path under which pickle-files are stored (optional, default: './')

    Returns
    -------
    desc : list or dict
        (List of) dict with statistical microstructure descriptors

    """
    import json
    if stats is None:
        raise ValueError('List or dict with microstructure descriptors must be given.')
    if file is None:
        raise ValueError('Name for json file with microstructure descriptors must be given.')
    file = os.path.join(path, file)
    with open(file, 'w') as fp:
        json.dump(stats, fp)



def import_stats(file, path='./'):
    """
    Write statistical descriptors of microstructure to JSON file.

    Parameters
    ----------
    file : string
        File name of pickled microstructure to be read.
    path : string
        Path under which pickle-files are stored (optional, default: './')

    Returns
    -------
    desc : list or dict
        (List of) dict with statistical microstructure descriptors

    """
    import json
    if file is None:
        raise ValueError('Name for json file with microstructure descriptors must be given.')
    file = os.path.join(path, file)
    with open(file, 'r') as inp:
        desc = json.load(inp)
    return desc



