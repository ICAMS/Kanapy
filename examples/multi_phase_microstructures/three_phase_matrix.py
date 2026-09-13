"""Two grain-bearing phases in a grain-free matrix (GRAIN0 / PHASE0).

Run from any directory; the voxel JSON is written to the current directory.
Descriptor order below is [grains A, grains B, matrix]. Kanapy canonicalizes it
as [matrix, grains A, grains B], so all material lists use IDs [0, 1, 2].

Author: Alexander Hartmaier
ICAMS, Ruhr University Bochum, Germany

Kudos to Codex for assistance with code generation.

September 2026
"""
from copy import deepcopy
import random

import numpy as np
from kanapy import Microstructure


def descriptors():
    grain_a = {
        'Grain type': 'Equiaxed',
        'Equivalent diameter': {
            'sig': 0.15, 'loc': 0., 'scale': 3.,
            'cutoff_min': 2.4, 'cutoff_max': 3.6,
        },
        'RVE': {'sideX': 8, 'sideY': 8, 'sideZ': 8,
                'Nx': 16, 'Ny': 16, 'Nz': 16, 'ialloy': 4},
        'Simulation': {'periodicity': False, 'output_units': 'mm'},
        'Phase': {'Name': 'Grains A', 'Volume fraction': 0.2},
    }
    grain_b = deepcopy(grain_a)
    grain_b['RVE']['ialloy'] = 5
    grain_b['Phase'] = {'Name': 'Grains B', 'Volume fraction': 0.3}
    matrix = {'Grain type': 'Matrix',
              'Phase': {'Name': 'Matrix', 'Volume fraction': 0.5}}
    return [grain_a, grain_b, matrix]


def main():
    random.seed(12)
    np.random.seed(12)
    ms = Microstructure(descriptors(), name='three_phase_matrix')
    ms.init_RVE(nsteps=100)
    ms.pack(save_files=False, verbose=False)
    ms.voxelize()
    ms.generate_orientations('random', iphase=1)
    ms.generate_orientations('random', iphase=2)
    ms.write_voxels()
    print('Canonical phase order:', ms.rve.phase_names)
    print('Requested phase fractions:', ms.rve.phase_vf)
    print('Measured voxel fractions:', ms.vf_vox)
    # For Abaqus with CP in both grain-bearing phases, supply your material files:
    # ms.write_abq(ialloy=[0, 4, 5],
    #              props_file=['matrix_j2.inc', 'phase1_cp.inc', 'phase2_cp.inc'],
    #              crystal_plasticity=[False, True, True])
    # Use the same lists for ms.write_abq_ori(). GRAIN0 has no orientation.


if __name__ == '__main__':
    main()
