""" Module defining class Microstructure that contains the necessary
methods and attributes to analyze experimental microstructures in form
of EBSD maps to generate statistical descriptors for 3D microstructures, and 
to create synthetic RVE that fulfill the requires statistical microstructure
descriptors.

The methods of the class Microstructure for an API that can be used to generate
Python workflows.

Authors: Alexander Hartmaier, Golsa Tolooei Eshlghi
Institution: ICAMS, Ruhr University Bochum

"""
import os
import json
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from kanapy.input_output import particleStatGenerator, RVEcreator,\
    write_output_stat, plot_output_stats, export2abaqus
from kanapy.packing import packingRoutine
from kanapy.voxelization import voxelizationRoutine
from kanapy.smoothingGB import smoothingRoutine
from kanapy.plotting import plot_voxels_3D, plot_ellipsoids_3D,\
    plot_polygons_3D

class Microstructure:
    '''Define class for synthetic microstructures'''
    def __init__(self, descriptor=None, file=None, name='Microstructure'):
        self.name = name
        self.particle_data = None
        self.RVE_data = None
        self.simulation_data = None
        self.particles = None
        self.simbox = None
        self.nodes_s = None
        self.nodes_v = None
        self.voxels = None
        self.grain_facesDict = None
        self.elmtSetDict = None
        self.res_data = None
        if descriptor is None:
            if file is None:
                raise ValueError('Please provide either a dictionary with statistics or an input file name')
                 
            # Open the user input statistics file and read the data
            try:
                with open(file) as json_file:  
                     self.descriptor = json.load(json_file)
            except:
                raise FileNotFoundError("File: '{}' does not exist in the current working directory!\n".format(file))
        else:
            self.descriptor = descriptor
            if file is not None:
                print('WARNING: Input parameter (descriptor) and file are given. Only descriptor will be used.')
    
    """
    --------        Routines for user interface        --------
    """
    def init_RVE(self, descriptor=None, save_files=False):    
        """ Creates RVE based on the data provided in the input file."""
        if descriptor is None:
            descriptor = self.descriptor  
        self.particle_data, self.RVE_data, self.simulation_data = \
            RVEcreator(descriptor, save_files=save_files)
        self.Ngr = self.particle_data['Number']
            
    def init_stats(self, descriptor=None, gs_data=None, ar_data=None,
                   save_files=False):    
        """ Generates particle statistics based on the data provided in the 
        input file."""
        if descriptor is None:
            descriptor = self.descriptor  
        particleStatGenerator(descriptor, gs_data=gs_data, ar_data=ar_data,
                              save_files=save_files)
        
    def pack(self, particle_data=None, RVE_data=None, simulation_data=None, 
             save_files=False):
        """ Packs the particles into a simulation box."""
        if particle_data is None:
            particle_data = self.particle_data
            if particle_data is None:
                raise ValueError('No particle_data in pack. Run create_RVE first.')
        if RVE_data is None:
            RVE_data = self.RVE_data
        if simulation_data is None:
            simulation_data = self.simulation_data
        self.particles, self.simbox = \
            packingRoutine(particle_data, RVE_data, simulation_data,
                           save_files=save_files)

    def voxelize(self, particle_data=None, RVE_data=None, particles=None, 
                 simbox=None, save_files=False):
        """ Generates the RVE by assigning voxels to grains."""   
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if particles is None:
            particles = self.particles
            if particles is None:
                raise ValueError('No particles in voxelize. Run pack first.')
        if simbox is None:
            simbox = self.simbox
        self.nodes_v, self.elmtDict, self.elmtSetDict,\
            self.vox_centerDict, self.voxels = \
            voxelizationRoutine(particle_data, RVE_data, particles, simbox,
                                save_files=save_files)

    def smoothen(self, nodes_v=None, elmtDict=None, elmtSetDict=None,
                 save_files=False):
        """ Generates smoothed grain boundary from a voxelated mesh."""
        if nodes_v is None:
            nodes_v = self.nodes_v
            if nodes_v is None:
                raise ValueError('No nodes_v in smoothen. Run voxelize first.')
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        self.nodes_s, self.grain_facesDict = \
            smoothingRoutine(nodes_v, elmtDict, elmtSetDict,
                             save_files=save_files)

    def analyze_RVE(self, nodes_v=None, elmtDict=None, elmtSetDict=None,
                     particle_data=None, RVE_data=None, simulation_data=None,
                     save_files=False):
        """ Writes out the particle- and grain diameter attributes for statistical comparison. Final RVE 
        grain volumes and shared grain boundary surface areas info are written out as well."""
        if nodes_v is None:
            nodes_v = self.nodes_v
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        if particle_data is None:
            particle_data = self.particle_data
        if RVE_data is None:
            RVE_data = self.RVE_data
        if simulation_data is None:
            simulation_data = self.simulation_data
            
        if nodes_v is None:
            raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        if particle_data is None:
            raise ValueError('No particles created yet. Run create_RVE, pack and voxelize first.')
            
        self.res_data = \
            write_output_stat(nodes_v, elmtDict, elmtSetDict,particle_data,
                              RVE_data, simulation_data, save_files=save_files)
        self.grain_facesDict, self.shared_area = self.calcPolygons()  # updates RVE_data 
        
    """
    --------     Plotting routines          --------
    """
    def plot_ellipsoids(self, cmap='prism'):
        """ Generates plot of particles"""
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        plot_ellipsoids_3D(self.particles, cmap=cmap)
        
    def plot_voxels(self, sliced=True, dual_phase=False, cmap='prism'):
        """ Generate 3D plot of grains in voxelized microstructure. """
        if self.voxels is None:
            raise ValueError('No voxels or elements to plot. Run voxelize first.')
        plot_voxels_3D(self.voxels, Ngr=self.particle_data['Number'],
                       sliced=sliced, dual_phase=dual_phase, cmap=cmap)

    def plot_polygons(self, grains=None, cmap='prism', alpha=0.4, 
                         ec=[0.5,0.5,0.5,0.1]):
        """ Plot polygonalized icrostructure"""
        if grains is None:
            if 'Grains' in self.RVE_data.keys():
                grains = self.RVE_data['Grains']
            else:
                raise ValueError('No polygons for grains defined. Run analyse first')
        plot_polygons_3D(grains, cmap=cmap, alpha=alpha, ec=ec)

        
    def plot_stats(self, data=None, gs_data=None, gs_param=None, 
                          ar_data=None, ar_param=None, save_files=False):
        """ Plots the particle- and grain diameter attributes for statistical 
        comparison."""   
        if data is None:
            data = self.res_data
        if data is None:
            raise ValueError('No microstructure data created yet. Run output_stats first.')
        plot_output_stats(data, gs_data=gs_data, gs_param=gs_param,
                          ar_data=ar_data, ar_param=ar_param,
                          save_files=save_files)

    """
    --------        Output/Export routines        --------
    """
    def output_abq(self, nodes=None, name=None, simulation_data=None,
                   elmtDict=None, elmtSetDict=None, faces=None):
        """ Writes out the Abaqus (.inp) file for the generated RVE."""    
        #write_abaqus_inp()
        if simulation_data is None:
            simulation_data = self.simulation_data
        if nodes is None:
            if self.nodes_s is not None and self.grain_facesDict is not None:
                print('\nWarning: No information about nodes is given, will write smoothened structure')
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            elif self.nodes_v is not None:
                print('\nWarning: No information about nodes is given, will write voxelized structure')
                nodes = self.nodes_v
                faces = None
                ntag = 'voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')
        elif nodes=='smooth' or nodes=='s':
            if self.nodes_s is not None and self.grain_facesDict is not None:
                nodes = self.nodes_s
                faces = self.grain_facesDict
                ntag = 'smooth'
            else:
                raise ValueError('No information about smoothed microstructure. Run smoothen first.')
        elif nodes=='voxels' or nodes=='v':
            if self.nodes_v is not None:
                nodes = self.nodes_v
                faces = None
                ntag = 'voxels'
            else:
                raise ValueError('No information about voxelized microstructure. Run voxelize first.')
            
        if elmtDict is None:
            elmtDict = self.elmtDict
        if elmtSetDict is None:
            elmtSetDict = self.elmtSetDict
        if simulation_data is None:
            raise ValueError('No simulation data exists. Run create_RVE, pack and voxelize first.')
        if name is None:
            cwd = os.getcwd()
            name = cwd + '/kanapy_{0}grains_{1}.inp'.format(len(elmtSetDict),ntag)
            if os.path.exists(name):
                os.remove(name)                  # remove old file if it exists
        export2abaqus(nodes, name, simulation_data, elmtSetDict, elmtDict, grain_facesDict=faces)

    #def output_neper(self, timestep=None):
    def output_neper(self):
        """ Writes out particle position and weights files required for
        tessellation in Neper."""
        #write_position_weights(timestep)
        if self.particles is None:
            raise ValueError('No particle to plot. Run pack first.')
        print('')
        print('Writing position and weights files for NEPER', end="")
        par_dict = dict()
        
        for pa in self.particles:
            x, y, z = pa.x, pa.y, pa.z
            a, b, c = pa.a, pa.b, pa.c
            par_dict[pa] = [x, y, z, a]
            
        with open('sphere_positions.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0} {1} {2}\n'.format(value[0], value[1], value[2]))
    
        with open('sphere_weights.txt', 'w') as fd:
            for key, value in par_dict.items():
                fd.write('{0}\n'.format(value[3]))
        print('---->DONE!\n')

    """
    --------        Supporting Routines         -------
    """
    def calcPolygons(self, tol=1.e-3):
        """
        Evaluates the grain volume and the grain boundary shared surface area 
        between neighbouring grains.

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

        """
        periodic = self.RVE_data['Periodic']
        print(periodic, type(periodic))
        RVE_min = np.amin(self.nodes_v, axis=0)
        RVE_max = np.amax(self.nodes_v, axis=0)
        voxel_size = self.RVE_data['Voxel_resolutionX']
        grain_facesDict = dict()      # {Grain: faces}
        Ng = len(self.elmtSetDict.keys())
        if not periodic:
            # create dicts for fake facets at surfaces
            for i in range(Ng+1,Ng+7):
                grain_facesDict[i]=dict()

        for gid, elset in self.elmtSetDict.items():               
            outer_faces = set()       # Stores only outer face IDs
            face_nodes = dict()       # {Faces: nodal connectivity} 
            nodeConn = [self.elmtDict[el] for el in elset]  # Nodal connectivity of a voxel

            # For each voxel, re-create its 6 faces
            for nc in nodeConn:
                faces = [[nc[0], nc[1], nc[2], nc[3]], [nc[4], nc[5], nc[6], nc[7]],
                         [nc[0], nc[1], nc[5], nc[4]], [nc[3], nc[2], nc[6], nc[7]],
                         [nc[0], nc[4], nc[7], nc[3]], [nc[1], nc[5], nc[6], nc[2]]]
                # Sort in ascending order
                sorted_faces = [sorted(fc) for fc in faces]  
                # create face ids by joining node id's                        
                face_ids = [int(''.join(str(c) for c in fc)) for fc in sorted_faces]        
                # Update {Faces: nodal connectivity} dictionary
                for enum, fid in enumerate(face_ids):
                    if fid not in face_nodes.keys():       
                        face_nodes[fid] = faces[enum]                
                # Identify outer faces that occur only once
                for fid in face_ids:        
                    if fid not in outer_faces:
                        outer_faces.add(fid)
                    else:
                        outer_faces.remove(fid)        
            
            # Update {Grain: faces} dictionary
            grain_facesDict[gid] = dict() 
            for of in outer_faces:
                # Treat faces belonging to RVE surface:
                # Create facets if not peridoic
                # Discard if periodic
                conn = face_nodes[of]
                n1 = self.nodes_v[conn[0]-1,:]
                n2 = self.nodes_v[conn[1]-1,:]
                n3 = self.nodes_v[conn[2]-1,:]
                n4 = self.nodes_v[conn[3]-1,:]
                h1 = np.abs(n1[0] - RVE_min[0]) < tol
                h2 = np.abs(n2[0] - RVE_min[0]) < tol
                h3 = np.abs(n3[0] - RVE_min[0]) < tol
                h4 = np.abs(n4[0] - RVE_min[0]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+1][of] = face_nodes[of]
                h1 = np.abs(n1[0] - RVE_max[0]) < tol
                h2 = np.abs(n2[0] - RVE_max[0]) < tol
                h3 = np.abs(n3[0] - RVE_max[0]) < tol
                h4 = np.abs(n4[0] - RVE_max[0]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+2][of] = face_nodes[of]
                h1 = np.abs(n1[1] - RVE_min[1]) < tol
                h2 = np.abs(n2[1] - RVE_min[1]) < tol
                h3 = np.abs(n3[1] - RVE_min[1]) < tol
                h4 = np.abs(n4[1] - RVE_min[1]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+3][of] = face_nodes[of]
                h1 = np.abs(n1[1] - RVE_max[1]) < tol
                h2 = np.abs(n2[1] - RVE_max[1]) < tol
                h3 = np.abs(n3[1] - RVE_max[1]) < tol
                h4 = np.abs(n4[1] - RVE_max[1]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+4][of] = face_nodes[of]
                h1 = np.abs(n1[2] - RVE_min[2]) < tol
                h2 = np.abs(n2[2] - RVE_min[2]) < tol
                h3 = np.abs(n3[2] - RVE_min[2]) < tol
                h4 = np.abs(n4[2] - RVE_min[2]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+5][of] = face_nodes[of]
                h1 = np.abs(n1[2] - RVE_max[2]) < tol
                h2 = np.abs(n2[2] - RVE_max[2]) < tol
                h3 = np.abs(n3[2] - RVE_max[2]) < tol
                h4 = np.abs(n4[2] - RVE_max[2]) < tol
                if (h1 and h2 and h3 and h4):
                    if periodic:
                        continue
                    else:
                        grain_facesDict[Ng+6][of] = face_nodes[of]
                grain_facesDict[gid][of] = face_nodes[of]  
                
        # Find all combination of grains to check for common area
        gbDict = dict()
        combis = list(itertools.combinations(sorted(grain_facesDict.keys()), 2))
        # Find the shared area
        shared_area = []
        for cb in combis:
            finter = set(grain_facesDict[cb[0]]).intersection(set(grain_facesDict[cb[1]]))
            if finter:
                ind = set()
                [ind.update(grain_facesDict[cb[0]][key]) for key in finter]
                key = 'f{}_{}'.format(cb[0], cb[1])
                gbDict[key] = ind
                try:
                    hull = ConvexHull(self.nodes_v[list(ind),:])
                    shared_area.append([cb[0], cb[1], hull.area])
                except:
                    sh_area = len(finter) * (voxel_size**2)
                    shared_area.append([cb[0], cb[1], sh_area])
            else:
                continue
        vert = dict()
        gbVert = dict()
        for i in grain_facesDict.keys():
            vert[i] = set()
        for key0 in gbDict.keys():
            gbVert[key0] = set()
        for key0 in gbDict.keys():
            klist = list(gbDict.keys())
            while key0 != klist.pop(0):
                pass
            for key1 in klist:
                finter = gbDict[key0].intersection(gbDict[key1])
                if finter:
                    if len(finter) == 1:
                        # only one node in intersection of GBs
                        elist = finter
                    else:
                        # mulitple nodes in intersection 
                        # identify end points of triple or quadruple line 
                        # = nodes in edge with largest mutual distance
                        edge = np.array(list(finter), dtype=int)
                        rn = self.nodes_v[edge-1]
                        dmax = 0.
                        for j, rp0 in enumerate(rn):
                            for k, rp1 in enumerate(rn[j+1:,:]):
                                d = np.sqrt(np.dot(rp0-rp1, rp0-rp1))
                                if d > dmax:
                                    elist = [edge[j], edge[k+j+1]]
                                    dmax = d
                    # add vertices to gbVert
                    gbVert[key0].update(elist)
                    gbVert[key1].update(elist)
                    # update all involved grains
                    grains = set()
                    j = key0.index('_')
                    grains.add(int(key0[1:j]))
                    grains.add(int(key0[j+1:]))
                    j = key1.index('_')
                    grains.add(int(key1[1:j]))
                    grains.add(int(key1[j+1:]))
                    for j in grains:
                        vert[j].update(elist)
                        
        grains = dict()
        vertices = set()
        for igr in self.elmtSetDict.keys():
            vertices.update(vert[igr])
            grains[igr] = dict()
            grains[igr]['Nodes'] = np.array(list(vert[igr]), dtype=int)-1
            points = self.nodes_v[grains[igr]['Nodes']]
            try:
                hull = ConvexHull(points, qhull_options='QJ Pp')
                grains[igr]['Points'] = hull.points
                grains[igr]['Vertices'] = hull.vertices
                grains[igr]['Simplices'] = hull.simplices
                grains[igr]['Volume'] = hull.volume
                grains[igr]['Area'] = hull.area
            except:
                grains[igr]['Points'] = points
                grains[igr]['Vertices'] = np.arange(len(points))
                grains[igr]['Simplices'] = []
                grains[igr]['Volume'] = 0.
                grains[igr]['Area'] = 0.
            grains[igr]['Center'] = np.mean(points, axis=0)
        self.RVE_data['Grains'] = grains
        self.RVE_data['Vertices'] = list(vertices)
        self.RVE_data['GBnodes'] = gbDict
        self.RVE_data['GBarea'] = shared_area
        
        return grain_facesDict, shared_area
