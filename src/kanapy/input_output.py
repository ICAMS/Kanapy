import os
import re
import logging
import numpy as np
from collections import defaultdict
from kanapy.entities import Ellipsoid, Cuboid
from kanapy.initializations import NodeSets


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


def export2abaqus(nodes, file, grain_dict, voxel_dict, units='um',
                  gb_area=None, dual_phase=False, thermal=False,
                  ialloy=None, grain_phase_dict=None, periodicBC=False, crystal_plasticity=False, phase_props=None,
                  value=None, apply_bc=False):
    """
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
                f.write(f'{val + 1}\n')
            else:
                f.write(f'{val + 1}, ')
        f.write(f'{nset[-1] + 1}\n')

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

        # 1) Find grid dimensions
        xs = np.unique(nodes[:, 0])
        ys = np.unique(nodes[:, 1])
        zs = np.unique(nodes[:, 2])
        nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

        # 2) Precompute face‐ID sets in C‐order flattening (Z fastest)
        B = ny * nz
        # Xmax: last block of size B
        x_max = set(range((nx - 1) * B + 1, nx * B + 1))

        # Ymax: in each x‐block, offset by (ny-1)*nz, vary z=0…nz-1
        y_offset = (ny - 1) * nz
        y_max = {
            x * B + y_offset + (z + 1)
            for x in range(nx)
            for z in range(nz)
        }

        # Zmax: in each xy‐slice, ID = x*B + y*nz + nz
        z_max = {
            x * B + y * nz + nz
            for x in range(nx)
            for y in range(ny)
        }

        # 3) Helper to write one ELSET/SURFACE block
        def _emit(face_ids, idx, label):
            f.write(f'*ELSET, ELSET=_Surf-{idx}_{label}, internal, instance=PART-1-1\n')
            # chunk 16 IDs per line
            ids = sorted(face_ids)
            for i in range(0, len(ids), 16):
                chunk = ids[i:i + 16]
                f.write(', '.join(str(e) for e in chunk) + ',\n')
            f.write(f'*Surface, type=ELEMENT, name=Surf-{idx}\n')
            f.write(f'_Surf-{idx}_{label}, {label}\n\n')
            f.write('\n')


        # 4) Emit all three faces
        _emit(x_max, 1, 'S3')  # X = max
        _emit(y_max, 2, 'S2')  # Y = max
        _emit(z_max, 3, 'S6')  # Z = max


    def write_boundary_conditions():
        f.write('** BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: F0yzFix Type: Displacement/Rotation\n')
        f.write('*Boundary\nF0YZ, 1, 1\n')
        f.write('** Name: Fx0zFix Type: Displacement/Rotation\n')
        f.write('*Boundary\nFX0Z, 2, 2\n')
        f.write('** Name: Fxy0Fix Type: Displacement/Rotation\n')
        f.write('*Boundary\nFXY0, 3, 3\n')

    def write_strain_load():
        displacement_bc_map = {
            'x': ('F1YZ', 1, 'disX'),
            'y': ('FX1Z', 2, 'disY'),
            'z': ('FXY1', 3, 'disZ'),
        }
        direction = loading_direction.lower()
        if direction in displacement_bc_map:
            set_name, dof, bc_name = displacement_bc_map[direction]

            # 'value' is a length-6 list; dof=1→X, 2→Y, 3→Z
            vstrain = value[dof - 1]
            strain = vstrain / 100.0  # Convert percentage to decimal

            displacement = edge_lengths[direction] * (np.exp(strain) - 1)  # Logarithmic strain
            print(f"Direction: {direction}, Strain: {strain:.6f}, Edge length: {edge_lengths[direction]:.6f} mm, "
                  f"Displacement: {displacement:.6f} mm")
            f.write(f'** Name: {bc_name} Type: Displacement/Rotation\n')
            f.write('*Boundary\n')
            f.write(f'{set_name}, {dof}, {dof}, {displacement:.6f}\n')

    def write_stress_load():
        f.write('** LOADS\n')
        f.write('**\n')
        load_bc_map = {
            'x': ('SURF-1', 1, 'loadX'),
            'y': ('SURF-2', 2, 'loadY'),
            'z': ('SURF-3', 3, 'loadZ'),
        }
        direction = loading_direction.lower()
        if direction in load_bc_map:
            set_name, dof, bc_name = load_bc_map[direction]

            # 'value' is a length-6 list; dof=1→X, 2→Y, 3→Z
            vstress = value[dof - 1]

            f.write(f'** Name: {bc_name} Type: Pressure\n')
            f.write('*Dload\n')
            f.write(f'{set_name}, P, {-vstress:.6f}\n')

    def write_periodic_load():
        if load_type == 'stress':
            f.write('** BOUNDARY CONDITIONS\n')
            f.write('** \n')
            f.write('** LOADS \n')
            f.write('** \n')
            f.write('** Name: Load Type: Stress BC\n')
            f.write('*Cload \n')

            # map each index to its Abaqus CLOAD pattern
            stress_map = {
                0: "V101,1",  # X-direction
                1: "V011,2",
                2: "V000,3",
                3: "V000,2",
                4: "V101,3",
                5: "V011,1",
            }

            # find all non-zero entries
            nz = [(i, v) for i, v in enumerate(value) if v != 0]
            if len(nz) == 1:
                # only one non-zero → emit just that line
                idx, val = nz[0]
                f.write(f'{stress_map[idx]},{val}\n')
            else:
                # fallback: emit all, same as before
                for idx, val in nz:
                    f.write(f'{stress_map[idx]},{val}\n')

            f.write('** \n')

        elif load_type == 'strain':
            f.write('** BOUNDARY CONDITIONS\n')
            f.write('** \n')
            f.write('** Displacements \n')
            f.write('** \n')
            f.write('** Name: Dis Type: Displacement BC\n')
            f.write('*Boundary \n')

            # map each index to (node, direction) for the *Boundary card
            strain_map = {
                0: ("V101", 1),  # X-direction
                1: ("V011", 2),
                2: ("V000", 3),
                3: ("V000", 2),
                4: ("V101", 3),
                5: ("V011", 1),
            }

            nz = [(i, v) for i, v in enumerate(value) if v != '*']
            if len(nz) == 1:
                idx, val = nz[0]
                node, direction = strain_map[idx]
                # node, start DOF, end DOF, value
                f.write(f'{node}, {direction}, {direction}, {val}\n')
            else:
                # fallback: emit all non-zero entries
                for idx, val in nz:
                    node, direction = strain_map[idx]
                    f.write(f'{node}, {direction}, {direction}, {val}\n')

    def _parse_value():
        """
        Given value: a list/tuple of length 6, each entry either
          • a number (float/int), or
          • the wildcard '*' (meaning “this DOF is free for strain”)
        Returns: load_type, bc_type, direction, magnitude
        """
        if len(value) != 6:
            raise ValueError("value must be length 6")

        # 1) If any entry is '*', it’s a displacement (strain) BC
        if any(v == '*' for v in value):
            load_type = 'strain'
            # find the one “real” number
            for idx, v in enumerate(value):
                if v != '*':
                    magnitude = float(v)
                    break
        else:
            # 2) else it’s stress (force) control; find the non‐zero entry
            load_type = 'stress'
            nonzeros = [(i, v) for i, v in enumerate(value) if v != 0]
            if len(nonzeros) != 1:
                raise ValueError(f"For stress you must supply exactly one non‐zero value, got {nonzeros}")
            idx, magnitude = nonzeros[0]

        # 3) Map index → direction & bc_type
        dir_map = {0: 'x', 1: 'y', 2: 'z', 3: 'xy', 4: 'xz', 5: 'yz'}
        direction = dir_map[idx]
        bc_type = 'uni-axial' if idx < 3 else 'shear'

        return load_type, bc_type, direction, magnitude

    # Parse the value list into load type, bc_type, loading direction,and magnitude.
    if value is None:
        # skip parsing and writing any BC/load for this surface
        load_type = bc_type = loading_direction = mag = None
    else:
        load_type, bc_type, loading_direction, mag = _parse_value()

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
        'z': edge_lengths['x'] * edge_lengths['y']  # xy face
    }

    #####################################
    #### Start writing the .inp file ####
    #####################################

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
            f.write('{0}, {1}, {2}, {3}\n'.format(k + 1, v[0] * scale_fact, v[1] * scale_fact, v[2] * scale_fact))
        f.write('*ELEMENT, TYPE={0}\n'.format(eltype))
        if gb_area is None:
            for k, v in voxel_dict.items():
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                    k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
            if dual_phase:  # Why don't we write both phase sets and grain sets anyway?
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

        if periodicBC and apply_bc:
            f.write('**** ======================================================== \n')
            f.write('**** Left to Right \n')
            # LeftToRight
            f.write('**** \n')
            f.write('**** X-DIR \n')
            for i in range(len(nsets.F0yzP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.F1yzP[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.F0yzP[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V101 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** \n')
            f.write('**** Y-DIR \n')
            for i in range(len(nsets.F0yzP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.F1yzP[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.F0yzP[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V101 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** \n')
            f.write('**** Z-DIR \n')
            for i in range(len(nsets.F0yzP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.F1yzP[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.F0yzP[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V101 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            f.write('**** Bottom to Top \n')
            # BottomToTop
            f.write('**** X-DIR \n')
            for i in range(len(nsets.Fx0zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fx0zP[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.Fx1zP[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1,-1 \n')
                f.write(str(nsets.V011 + 1) + ',1, 1 \n')

            f.write('**** \n')
            f.write('**** Y-DIR \n')
            for i in range(len(nsets.Fx0zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fx0zP[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.Fx1zP[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2,-1 \n')
                f.write(str(nsets.V011 + 1) + ',2, 1 \n')

            f.write('**** \n')
            f.write('**** Z-DIR \n')
            for i in range(len(nsets.Fx0zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fx0zP[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.Fx1zP[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3,-1 \n')
                f.write(str(nsets.V011 + 1) + ',3, 1 \n')

            f.write('**** Front to Rear \n')
            # FrontToRear
            f.write('**** \n')
            f.write('**** X-DIR \n')
            for i in range(len(nsets.Fxy0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fxy0P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.Fxy1P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.V001 + 1) + ',1,-1 \n')
                f.write(str(nsets.V000 + 1) + ',1, 1 \n')

            f.write('**** \n')
            f.write('**** Y-DIR \n')
            for i in range(len(nsets.Fxy0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fxy0P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.Fxy1P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.V001 + 1) + ',2,-1 \n')
                f.write(str(nsets.V000 + 1) + ',2, 1 \n')

            f.write('**** \n')
            f.write('**** Z-DIR \n')
            for i in range(len(nsets.Fxy0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Fxy0P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.Fxy1P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.V001 + 1) + ',3,-1 \n')
                f.write(str(nsets.V000 + 1) + ',3, 1 \n')

            f.write('**** ======================================================== \n')
            f.write('**** Edges\n')
            f.write('**** \n')
            # Edges in x-y Plane
            # Right top edge to left top edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E11zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E11zP[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V101 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E11zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E11zP[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V101 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E11zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E11zP[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V101 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Right bottom edge to left bottom edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E10zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E10zP[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V101 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E10zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E10zP[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V101 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E10zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E10zP[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V101 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Left top edge to left bottom edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E01zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V011 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E01zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V011 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E01zP)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E01zP[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E00zP[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V011 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Edges in y-z Plane
            # Top back edge to top front edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.Ex10P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex10P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V000 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.Ex10P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex10P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V000 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.Ex10P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex10P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V000 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Bottom back edge to bottom front edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.Ex00P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex00P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V000 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.Ex00P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex00P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V000 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.Ex00P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex00P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V000 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # top front edge to bottom front edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.Ex11P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V011 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.Ex11P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V011 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.Ex11P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.Ex11P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.Ex01P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V011 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Edges in x-z Plane
            # Rear right edge to rear left edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E1y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y0P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V101 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E1y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y0P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V101 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E1y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y0P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V101 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Front right edge to front left edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E1y1P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y1P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V101 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E1y1P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y1P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V101 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E1y1P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E1y1P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V101 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # Top front edge to bottom front edge
            f.write('**** X-DIR \n')
            for i in range(len(nsets.E0y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',1, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',1,-1 \n')
                f.write(str(nsets.V000 + 1) + ',1,-1 \n')
                f.write(str(nsets.V001 + 1) + ',1, 1 \n')

            f.write('**** Y-DIR \n')
            for i in range(len(nsets.E0y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',2, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',2,-1 \n')
                f.write(str(nsets.V000 + 1) + ',2,-1 \n')
                f.write(str(nsets.V001 + 1) + ',2, 1 \n')

            f.write('**** Z-DIR \n')
            for i in range(len(nsets.E0y0P)):
                f.write('*Equation \n')
                f.write('4 \n')
                f.write(str(nsets.E0y0P[i] + 1) + ',3, 1 \n')
                f.write(str(nsets.E0y1P[i] + 1) + ',3,-1 \n')
                f.write(str(nsets.V000 + 1) + ',3,-1 \n')
                f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            f.write('**** ======================================================== \n')
            f.write('**** Corners \n')

            # V3 (V111) to V4 (V011)
            f.write('**** X-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V111 + 1) + ',1, 1 \n')
            f.write(str(nsets.V011 + 1) + ',1,-1 \n')
            f.write(str(nsets.V101 + 1) + ',1,-1 \n')
            f.write(str(nsets.V001 + 1) + ',1, 1 \n')
            f.write('**** y-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V111 + 1) + ',2, 1 \n')
            f.write(str(nsets.V011 + 1) + ',2,-1 \n')
            f.write(str(nsets.V101 + 1) + ',2,-1 \n')
            f.write(str(nsets.V001 + 1) + ',2, 1 \n')
            f.write('**** z-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V111 + 1) + ',3, 1 \n')
            f.write(str(nsets.V011 + 1) + ',3,-1 \n')
            f.write(str(nsets.V101 + 1) + ',3,-1 \n')
            f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # H4 (V010) to V4 (V011)
            f.write('**** X-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V010 + 1) + ',1, 1 \n')
            f.write(str(nsets.V011 + 1) + ',1,-1 \n')
            f.write(str(nsets.V000 + 1) + ',1,-1 \n')
            f.write(str(nsets.V001 + 1) + ',1, 1 \n')
            f.write('**** y-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V010 + 1) + ',2, 1 \n')
            f.write(str(nsets.V011 + 1) + ',2,-1 \n')
            f.write(str(nsets.V000 + 1) + ',2,-1 \n')
            f.write(str(nsets.V001 + 1) + ',2, 1 \n')
            f.write('**** z-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V010 + 1) + ',3, 1 \n')
            f.write(str(nsets.V011 + 1) + ',3,-1 \n')
            f.write(str(nsets.V000 + 1) + ',3,-1 \n')
            f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # H3 (V110) to V3 (V111)
            f.write('**** X-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V110 + 1) + ',1, 1 \n')
            f.write(str(nsets.V111 + 1) + ',1,-1 \n')
            f.write(str(nsets.V000 + 1) + ',1,-1 \n')
            f.write(str(nsets.V001 + 1) + ',1, 1 \n')
            f.write('**** y-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V110 + 1) + ',2, 1 \n')
            f.write(str(nsets.V111 + 1) + ',2,-1 \n')
            f.write(str(nsets.V000 + 1) + ',2,-1 \n')
            f.write(str(nsets.V001 + 1) + ',2, 1 \n')
            f.write('**** z-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V110 + 1) + ',3, 1 \n')
            f.write(str(nsets.V111 + 1) + ',3,-1 \n')
            f.write(str(nsets.V000 + 1) + ',3,-1 \n')
            f.write(str(nsets.V001 + 1) + ',3, 1 \n')

            # H2 (V100) to V2 (V101)
            f.write('**** X-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V100 + 1) + ',1, 1 \n')
            f.write(str(nsets.V101 + 1) + ',1,-1 \n')
            f.write(str(nsets.V000 + 1) + ',1,-1 \n')
            f.write(str(nsets.V001 + 1) + ',1, 1 \n')
            f.write('**** y-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V100 + 1) + ',2, 1 \n')
            f.write(str(nsets.V101 + 1) + ',2,-1 \n')
            f.write(str(nsets.V000 + 1) + ',2,-1 \n')
            f.write(str(nsets.V001 + 1) + ',2, 1 \n')
            f.write('**** z-DIR \n')
            f.write('*Equation \n')
            f.write('4 \n')
            f.write(str(nsets.V100 + 1) + ',3, 1 \n')
            f.write(str(nsets.V101 + 1) + ',3,-1 \n')
            f.write(str(nsets.V000 + 1) + ',3,-1 \n')
            f.write(str(nsets.V001 + 1) + ',3, 1 \n')

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
        f.write(f'{nsets.V000 + 1}\n')
        f.write(f'*Nset, nset=V001, instance=PART-1-1\n')
        f.write(f'{nsets.V001 + 1}\n')
        f.write(f'*Nset, nset=V010, instance=PART-1-1\n')
        f.write(f'{nsets.V010 + 1}\n')
        f.write(f'*Nset, nset=V100, instance=PART-1-1\n')
        f.write(f'{nsets.V100 + 1}\n')
        f.write(f'*Nset, nset=V011, instance=PART-1-1\n')
        f.write(f'{nsets.V011 + 1}\n')
        f.write(f'*Nset, nset=V101, instance=PART-1-1\n')
        f.write(f'{nsets.V101 + 1}\n')
        f.write(f'*Nset, nset=V110, instance=PART-1-1\n')
        f.write(f'{nsets.V110 + 1}\n')
        f.write(f'*Nset, nset=V111, instance=PART-1-1\n')
        f.write(f'{nsets.V111 + 1}\n')
        f.write('*Nset, nset=Vertices, instance=PART-1-1\n')
        f.write(f'{nsets.V000 + 1}, {nsets.V100 + 1}, {nsets.V010 + 1}, {nsets.V001 + 1}, {nsets.V011 + 1}, '
                f'{nsets.V101 + 1}, {nsets.V110 + 1}, {nsets.V111 + 1}\n')
        if periodicBC:
            f.write('*Nset, nset=VerticesPeriodic, instance=PART-1-1\n')
            f.write(f'{nsets.V001 + 1}, {nsets.V101 + 1}, {nsets.V011 + 1}, {nsets.V000 + 1} \n')

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
        ############################
        ### Creating Material
        ############################
        f.write('** MATERIALS\n')
        f.write('**\n')

        # track whether we ever did an include-file inside the loop
        did_include = False

        for pid in ph_set:
            f.write('*Material, name=PHASE{}_MAT\n'.format(pid))

            props = phase_props.get(pid)
            if props:
                # inline properties as before
                if 'damage_init' in props:
                    di = props['damage_init']
                    f.write('*Damage Initiation, criterion={}\n'.format(di['criterion']))
                    f.write(' {}\n'.format(', '.join(map(str, di['values']))))

                if 'damage_evol' in props:
                    de = props['damage_evol']
                    f.write('*Damage Evolution, type={}\n'.format(de['type']))
                    f.write(' {}\n'.format(', '.join(map(str, de['values']))))

                if 'elastic' in props:
                    f.write('*Elastic\n')
                    f.write(' {}\n'.format(', '.join(map(str, props['elastic']))))

                if 'plastic' in props:
                    f.write('*Plastic\n')
                    for sigma, eps in props['plastic']:
                        f.write(' {}, {}\n'.format(sigma, eps))

            else:
                # no inline props → include from file
                if dual_phase:
                    # per‐phase file for each pid
                    f.write('*Include, input=Material{}.inp\n'.format(pid))
                    did_include = True
                # else: defer the single‐file include until after the loop

            f.write('**\n')

        # if this wasn’t a dual‐phase run, do one global include once:
        if not dual_phase and not did_include:
            # strip off last 8 chars (e.g. “_mesh.inp”) and append “mat.inp”
            base = file[:-8]
            f.write('*Include, input={}mat.inp\n'.format(base))
            f.write('**\n')
        """
        Previous Material Section
        f.write('**__________________________________________________________________\n')
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
        """
        ##########################################
        ### Creating Periodic Boundary Conditions
        ##########################################
        if apply_bc:
            if not periodicBC and bc_type.lower() == 'uni-axial':
                write_boundary_conditions()
            if periodicBC:
                f.write('** BOUNDARY CONDITIONS\n')
                f.write('** \n')
                f.write('*Boundary \n')
                f.write('V001,1 \n')
                f.write('V001,2 \n')
                f.write('V001,3 \n')
                f.write('V101,2 \n')
                f.write('V000,1 \n')
                f.write('V011,3 \n')
                f.write('** \n')
                f.write('** \n')

        ############################
        ### Creating Step
        ############################
        if crystal_plasticity:  # Using crystal plasticity Umat
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
        else:  # Using build-in functions
            f.write('**\n')
            f.write('** STEP: Loading \n')
            f.write('** \n')
            f.write('*Step, name=Loading, nlgeom=YES, inc=500000 \n')
            f.write('*Static \n')
            f.write('0.001, 1., 1e-6, 0.02 \n')
            f.write('** \n')
        f.write('** \n')

        #################################
        ### Creating Load
        #################################
        if apply_bc:
            if not periodicBC:
                if bc_type.lower() == 'uni-axial':
                    if load_type == 'strain':
                        write_strain_load()
                    if load_type == 'stress':
                        write_stress_load()
                f.write('** \n')
            elif periodicBC:
                write_periodic_load()
        #################################
        ### Creating Fild Output
        #################################
        f.write('** OUTPUT REQUESTS \n')
        f.write('** \n')
        f.write('*Restart, write, frequency=0 \n')
        f.write('** \n')
        f.write('** FIELD OUTPUT: F-Output-1 \n')
        f.write('** \n')
        f.write('*Output, field \n')
        f.write('*Node Output \n')
        f.write('CF, COORD, RF, U \n')
        f.write('** \n')
        f.write('** FIELD OUTPUT: F-Output-2 \n')
        f.write('** \n')
        f.write('*Element Output, directions=YES \n')
        f.write('LE, MISES, PE, PEEQ, S, SDEG \n')
        f.write('*Output, history, frequency=0 \n')
        f.write('** \n')
        f.write('** HISTORY OUTPUT: H-Output-1 \n')
        f.write('** \n')
        f.write('*Output, history, variable=PRESELECT \n')
        f.write('*End Step \n')

    print('---->DONE! \n')
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
            gr_ori_dict[igr + 1] = ori
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
