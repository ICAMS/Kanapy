structure = {
    "Info": {  # optional: provides meta data
        "Owner": str,  # name of owner of file
        "Institution": str,  # affiliation
        "Date": str,  # date at which file was created
        "Description": "Voxels of microstructure",  # free text description
        "Method": str,  # free text for description of method
        "System": {  # optional, sub-level class for hardware information
            "sysname": str,
            "nodename": str,
            "release": str,
            "version": str,
            "machine": str,
        },
    },
    "Model": {  # required: defines metadata for microstructure model
        "Creator": str,  # software name that created JSON file
        "Version": str,  # version of creator software
        "Repository": str,  # with link to public software repository
        "Script": str,  # script name that created JSON file
        "Input": str,  # additional input files for script 
        "Material": str,  # Material name for microstructure
        "Phase_names": list,  # list of phase names (str)
        "Size":tuple,  # 3-tuple of RVE size in length units
        "Periodicity": bool,  # indicator for model with periodic boundary conditions
        "Units": {  # sub-level class for model units
            'Length': str,  # RVE length units (‘um’ (microns) or ‘mm’)
        },
    },
    "Data": {  # required: provides grain or other order parameter for each voxel
        "Description": 'Grain numbers per voxel',  # free text description
        "Type": 'int',  # Python type definition
        "Shape": tuple,  # 3-tuple of RVE dimensions (voxel numbers in each direction)
        "Order": 'C',  # str indicating order used to flatten the 3D array of voxels
        "Values": list,  # flattened list of integer order parameters per voxel
    },
    "Grains": {  # optional, provides grain-level information, e.g. phase or orientation angle
        "Description": "Grain-related data",  # free text description
        "Phase": "Phase number"  # free text description for grain-level key
        "Orientation": "Euler-Bunge angle",  # free text description for grain-level key
        "{ID}" : int {  # key of sub-level class matching order parameter given in Data.Values
            "Phase": int,  # phase order parameter for grain #ID
            "Orientation": tuple,  # 3-tuple of Euler angles
        },
    },
    "Mesh": {  # optional, provides information of mesh, e.g. nodal positions
        "Nodes": {  # sub-level class containing nodal positions
            "Description": 'Nodal coordinates',  # free text description
            "Type": 'float',  # Python type
            "Shape": tuple,  # shape of nodal coordinates (number of nodes, 3)
            "Values": list,  # list of 3-tuples of nodal coordinates
        },
        "Voxels": {  # sub-level class containing node lists
            "Description": 'Node list per voxel',  # free text description
            "Type": 'int',  # Python type
            "Shape": tuple,  # tuple for voxel/element definition (number of voxels, nodes per voxel)
            "Values": list,  # lists of node numbers per voxel
        },
    },
}
