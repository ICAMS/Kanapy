import os
import re

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
    output_dir = cwd + '/dump_files'    # output directory
    dump_outfile = output_dir + '/particle'	    # output dump file
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
        f.write('ITEM: ATOMS id x y z AsphericalShapeX AsphericalShapeY AsphericalShapeZ OrientationX OrientationY OrientationZ OrientationW\n')
        for ell in Ellipsoids:
            qw, qx, qy, qz = ell.quat
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(
                ell.id, ell.x, ell.y, ell.z, ell.a, ell.b, ell.c, qx, qy, qz, qw))


def read_dump(dump_file):
    """
    Reads the (.dump) file to extract information for voxelization (meshing) routine    

    :param dump_file: Contains information of ellipsoids generated in the packing routine.
    :type dump_file: document

    :returns: * Cuboid object that represents the RVE.
              * List of ellipsoid objects that represent the grains.
    :rtype: Tuple of python objects (:obj:`Cuboid`, :obj:`Ellipsoid`)
    """
    print('    Reading the .dump file for particle information')

    try:
        # Read the Simulation box dimensions
        with open(dump_file, 'r+') as fd:
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
        print('    .dump file not found, make sure "Packing" command is executed first!')
        raise FileNotFoundError
        
    # Create an instance of simulation box
    sim_box = Cuboid(RVE_minX, RVE_maxY, RVE_maxX, RVE_minY, RVE_maxZ, RVE_minZ)

    # Read the particle shape & position information
    # Create instances for ellipsoids & assign values from dump files
    Ellipsoids = []
    with open(dump_file, "r") as f:
        count = 0
        for num, lines in enumerate(f, 1):
            if num >= par_line_num:

                count += 1
                values = re.findall(r'\S+', lines)
                int_values = list(map(float, values[1:]))
                values = [values[0]] + int_values

                iden = count                                        # ellipsoid 'id'                
                a, b, c = values[4], values[5], values[6]           # Semi-major length, Semi-minor length 1 & 2
                x, y, z = values[1], values[2], values[3]
                qx, qy, qz, qw = values[7], values[8], values[9], values[10]
                quat = np.array([qw, qx, qy, qz])                
                ellipsoid = Ellipsoid(iden, x, y, z, a, b, c, quat) # instance of Ellipsoid class

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


def export2abaqus(nodes, fileName, simulation_data, elmtSetDict, elmtDict, grain_facesDict=None):
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
    print('')
    print('Writing ABAQUS (.inp) file', end="")

    # Factor used to generate nodal cordinates in 'mm' or 'um' scale
    if simulation_data['Output units'] == 'mm':
        scale = 'mm'
        divideBy = 1000
    elif simulation_data['Output units'] == 'um':
        scale = 'um'
        divideBy = 1

    with open(fileName, 'w') as f:
        f.write('** Input file generated by kanapy\n')
        f.write('** Nodal coordinates scale in {0}\n'.format(scale))
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
            f.write('{0}, {1}, {2}, {3}\n'.format(k+1, v[0]/divideBy, v[1]/divideBy, v[2]/divideBy))

        if grain_facesDict is None:
            # write voxelized structure with regular hex mesh
            # Create Elements
            f.write('*ELEMENT, TYPE=C3D8\n')
            for k, v in elmtDict.items():
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                    k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
            # Create element sets
            for k, v in elmtSetDict.items():
                f.write('*ELSET, ELSET=Grain{0}_set\n'.format(k))
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
            for k, v in elmtSetDict.items():
                f.write(
                    '*Solid Section, elset=Grain{0}_set, material=GRAIN{1}_mat\n'.format(k, k))
        else:
            # write smoothened structure with tetrahedral mesh
            f.write('*ELEMENT, TYPE=SFM3D4\n')
            fcList = {}
            fcNum = 0
            gr_fcs = defaultdict(list)
            for gid,ginfo in grain_facesDict.items():                
                for fc,conn in ginfo.items():
                    if fc not in fcList.keys():
                        fcNum += 1
                        fcList[fc] = fcNum
                        f.write('%d,%d,%d,%d,%d\n'%(fcNum,conn[0],conn[1],conn[2], conn[3]))            
                        gr_fcs[gid].append(fcNum)  
                    elif fc in fcList.keys():
                        f.write('%d,%d,%d,%d,%d\n'%(fcList[fc],conn[0],conn[1],conn[2], conn[3]))       
                        gr_fcs[gid].append(fcList[fc])

            for gid,fcs in gr_fcs.items():             
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

            for gid,fcs in gr_fcs.items():    
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
    print('---->DONE!\n')                                                                                    
    return


def writeAbaqusMat(ialloy, angles, filename=None, nsdv=200):
    '''
    angles : Euler angles with number of rows= number of grains and
            three columns phi1, Phi, phi2
    ialloy : alloy number in the umat, mod_alloys.f
    nsdv : number of state dependant variables default value is 200
    '''
    Ngr = len(angles)
    if filename is None:
        cwd = os.getcwd()
        filename = cwd + '/abq_px_{0}grains_materials.inp'.format(Ngr)
        if os.path.exists(filename):
            os.remove(filename)  # remove old file if it exists
    with open(filename, 'w') as f:
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(Ngr):
            f.write('*Material, name=GRAIN{}_mat\n'.format(i+1))
            f.write('*Depvar\n')
            f.write('    {}\n'.format(nsdv))
            f.write('*User Material, constants=4\n')
            f.write('{}, {}, {}, {}\n'.format(float(ialloy), angles[i,0],
                                              angles[i,1], angles[i,2]))
    return

def writeAbaqusPhase(grains, nsdv=200):
    with open('Material.inp', 'w') as f:
        f.write('** MATERIALS\n')
        f.write('**\n')
        for i in range(len(grains)):
            f.write('*Material, name=GRAIN{}_mat\n'.format(i+1))
            f.write('*Depvar\n')
            f.write('    {}\n'.format(nsdv))
            f.write('*User Material, constants=1\n')
            f.write('{}\n'.format(float(grains[i+1]['PhaseID'])))
    return
