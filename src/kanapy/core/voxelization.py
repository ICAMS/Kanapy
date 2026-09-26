"""Voxelization by the anisotropic power diagram evaluated at voxel centers."""
import warnings
from functools import lru_cache
from itertools import product

import numpy as np

from .power_diagram import AnisotropicPowerDiagram


def grain_boundary_voxels(grains, *, periodic=False):
    """Return a boolean mask of voxels with a differently labelled face neighbour.

    ``grains`` is a 3D array of grain IDs, e.g. ``mesh.grains``. Both sides
    of every interface are marked. Edge/corner contacts are not neighbours,
    and the exterior is not another grain. All labels (including zero) are
    treated as grain IDs. ``periodic`` enables wrapping on all three axes.

    Uses O(N) time and O(N) boolean storage, without copying the label array
    or looping over voxels. Obtain array coordinates with ``np.argwhere(mask)``
    or Kanapy's one-based voxel IDs with ``np.flatnonzero(mask) + 1``.
    """
    grains = np.asarray(grains)
    if grains.ndim != 3:
        raise ValueError('grains must be a 3D array of grain IDs')
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError('periodic must be a boolean')
    boundary = np.zeros(grains.shape, dtype=bool)
    for axis in range(3):
        if grains.shape[axis] < 2:
            continue
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        different = grains[left] != grains[right]
        boundary[left] |= different
        boundary[right] |= different
        if periodic:
            first = [slice(None)] * 3
            last = [slice(None)] * 3
            first[axis], last[axis] = 0, -1
            first, last = tuple(first), tuple(last)
            different = grains[first] != grains[last]
            boundary[first] |= different
            boundary[last] |= different
    return boundary


@lru_cache(maxsize=1)
def _nonmanifold_vertex_patterns():
    """Classify the boundary link on an octahedron around a grid vertex."""
    bad = np.zeros(256, dtype=bool)
    triangles = [tuple(2 * axis + bit for axis, bit in enumerate(bits))
                 for bits in product((0, 1), repeat=3)]
    for pattern in range(1, 255):
        edges = set()
        for octant, triangle in enumerate(triangles):
            if pattern & (1 << octant):
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    edge = (triangle[i], triangle[j])
                    edges.symmetric_difference_update((edge,))
        graph = {}
        for a, b in edges:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        # A manifold surface has one simple closed curve as its vertex link.
        seen, pending = set(), [next(iter(graph))]
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(graph[node] - seen)
        bad[pattern] = (len(seen) != len(graph) or
                        any(len(neighbours) != 2 for neighbours in graph.values()))
    bad.setflags(write=False)
    return bad


def nonmanifold_grain_boundary_voxels(grains, *, periodic=False, chunk_size=65536):
    """Mark boundary voxels incident to a non-manifold vertex of their grain.

    Each grain is treated as a union of closed voxel cubes. At each grid
    vertex, its surface must have a link consisting of one simple cycle.
    This detects edge and vertex self-contacts, including disconnected local
    surface sheets. Ordinary triple lines and junctions between distinct
    grains are valid when each individual grain surface is manifold.

    Input and output conventions match ``grain_boundary_voxels``. The result
    marks incident voxels belonging to the offending grain, not all neighbours.
    Nonperiodic grain shells are closed against the exterior for the topology
    check, but only grain-boundary voxels are returned. All labels, including
    zero, count as grains. Periodic mode identifies opposite domain faces.

    A 256-entry occupancy lookup table tests local 2x2x2 neighbourhoods.
    Work is O(N) after label encoding; temporary neighbourhood arrays are
    bounded by ``chunk_size`` vertices. Label encoding uses ``np.unique``.
    """
    grains = np.asarray(grains)
    boundary = grain_boundary_voxels(grains, periodic=periodic)
    if (isinstance(chunk_size, (bool, np.bool_)) or
            not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError('chunk_size must be a positive integer')
    if grains.size == 0:
        return boundary
    _, labels = np.unique(grains, return_inverse=True)
    labels = labels.reshape(grains.shape)
    if periodic:
        labels = np.pad(labels, ((1, 0),) * 3, mode='wrap')
        vertex_shape = grains.shape
    else:
        labels = np.pad(labels, 1, constant_values=-1)
        vertex_shape = tuple(n + 1 for n in grains.shape)
    flagged = np.zeros(labels.size, dtype=bool)
    strides = np.array([labels.shape[1] * labels.shape[2], labels.shape[2], 1])
    offsets = np.array(list(product((0, 1), repeat=3))) @ strides
    flat_labels = labels.ravel()
    lookup = _nonmanifold_vertex_patterns()
    weights = (1 << np.arange(8)).astype(np.uint8)
    for start in range(0, int(np.prod(vertex_shape)), chunk_size):
        vertices = np.arange(start, min(start + chunk_size, np.prod(vertex_shape)))
        base = np.array(np.unravel_index(vertices, vertex_shape)).T @ strides
        indices = base[:, None] + offsets
        local = flat_labels[indices]
        for octant in range(8):
            same = local == local[:, octant, None]
            pattern = np.sum(same * weights, axis=1)
            bad = lookup[pattern] & (local[:, octant] >= 0)
            flagged[indices[bad, octant]] = True
    flagged = flagged.reshape(labels.shape)
    if periodic:
        # Fold ghost voxels on the low sides back onto their physical copies.
        for axis in range(3):
            first = [slice(None)] * 3
            last = [slice(None)] * 3
            first[axis], last[axis] = 0, -1
            flagged[tuple(last)] |= flagged[tuple(first)]
        result = flagged[1:, 1:, 1:]
    else:
        result = flagged[1:-1, 1:-1, 1:-1]
    return result & boundary


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
