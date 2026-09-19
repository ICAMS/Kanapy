"""Voxelization by the anisotropic power diagram evaluated at voxel centers."""
import warnings

import numpy as np

from .power_diagram import AnisotropicPowerDiagram


def voxelizationRoutine(Ellipsoids, mesh, nphases, prec_vf=None, *,
                        periodic=None, fit_volumes=True, fit_options=None,
                        weights=None, chunk_size=8192):
    """Assign each voxel to the minimum APD cost at its center, in place.

    The original four positional arguments are retained. ``prec_vf`` is
    deprecated and ignored: APD partitions the entire domain. Use
    ``voxelization_legacy.voxelizationRoutine_legacy`` for a matrix/porosity
    fraction or the historical polygon/growth algorithm.

    ``periodic=None`` infers periodicity from particle duplicates; explicit
    booleans override this. Duplicate particles never form separate grains.
    By default additive weights are fitted to relative particle volumes using
    continuous Sobol quadrature (``fit_options`` forwards options to
    ``AnisotropicPowerDiagram.fit_volumes``). Voxel fractions are quantized and
    need not match these continuous targets exactly. Explicit finite ``weights``
    in original-particle order bypass fitting; ``fit_volumes=False`` uses zero
    weights. Costs are evaluated in batches of ``chunk_size`` voxel centers.
    Exact ties use original-particle order. Connectivity is not enforced.

    Returns the input mesh with grains, phases, grain_dict, grain_phase_dict,
    ngrains_phase and prec_vf_voxels populated. ``mesh.apd`` stores the diagram,
    whose coordinates are relative to the minimum mesh node coordinates.
    """
    if prec_vf is not None:
        warnings.warn('prec_vf is deprecated and ignored by APD voxelization; '
                      'use voxelizationRoutine_legacy for partial filling.',
                      DeprecationWarning, stacklevel=2)
    if not isinstance(nphases, (int, np.integer)) or nphases < 1:
        raise ValueError('nphases must be a positive integer')
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    particles = list(Ellipsoids)
    originals = [p for p in particles if p.duplicate is None]
    if not originals:
        raise ValueError('No original particles; run pack first')
    for p in originals:
        if not isinstance(p.id, (int, np.integer)) or p.id <= 0:
            raise ValueError('Original grain IDs must be positive integers')
        if not isinstance(p.phasenum, (int, np.integer)) or not 0 <= p.phasenum < nphases:
            raise ValueError('Particle phase numbers must be in [0, nphases)')
    ids = np.array(sorted(mesh.vox_center_dict))
    if (mesh.nvox < 1 or np.prod(mesh.dim) != mesh.nvox
            or not np.array_equal(ids, np.arange(1, mesh.nvox + 1))):
        raise ValueError('Mesh must have consecutive voxel IDs 1 through nvox')
    nodes = np.asarray(mesh.nodes, dtype=float)
    origin = nodes.min(axis=0)
    box_size = nodes.max(axis=0) - origin
    if periodic is None:
        periodic = any(p.duplicate is not None for p in particles)
    # Copies keep particle positions, dimensions and orientations untouched.
    from copy import copy
    local = [copy(p) for p in originals]
    for p, original in zip(local, originals):
        p.x, p.y, p.z = np.asarray(original.get_pos()) - origin
    diagram = AnisotropicPowerDiagram.from_particles(local, box_size, periodic=periodic)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != diagram.weights.shape or not np.all(np.isfinite(weights)):
            raise ValueError('weights must contain one finite value per original particle')
        diagram.weights[:] = weights
    elif fit_volumes:
        diagram.fit_volumes(**({} if fit_options is None else fit_options))
    centers = np.asarray([mesh.vox_center_dict[i] for i in ids], dtype=float) - origin
    labels = np.empty(mesh.nvox, dtype=int)
    for start in range(0, mesh.nvox, chunk_size):
        labels[start:start + chunk_size] = diagram.labels(centers[start:start + chunk_size])
    phase_by_id = {p.id: p.phasenum for p in originals}
    mesh.grain_dict = {p.id: ids[labels == p.id].tolist() for p in originals
                       if np.any(labels == p.id)}
    mesh.grain_phase_dict = {gid: phase_by_id[gid] for gid in mesh.grain_dict}
    mesh.ngrains_phase = np.bincount(list(mesh.grain_phase_dict.values()), minlength=nphases)
    mesh.grains = labels.reshape(mesh.dim, order='C')
    mesh.phases = np.array([phase_by_id[i] for i in labels]).reshape(mesh.dim, order='C')
    mesh.prec_vf_voxels = 1.0
    mesh.apd = diagram
    for p in particles:
        p.inside_voxels = list(mesh.grain_dict.get(p.id, [])) if p.duplicate is None else []
    return mesh
