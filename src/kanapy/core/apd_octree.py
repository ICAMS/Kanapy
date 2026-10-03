"""Adaptive Cartesian APD sampling, before conforming tetrahedralization."""
from dataclasses import dataclass
from itertools import product

import numpy as np


_CHILDREN = np.array(list(product((0, 1), repeat=3)), dtype=int)
_SAMPLES = np.array(list(product((0., .5, 1.), repeat=3)))


@dataclass
class APDBackgroundOctree:
    """Leaf boxes of an octree forest covering the APD box without overlap.

    ``indices`` are integer cell indices at each leaf's ``levels``; physical
    bounds are derived from these dyadic coordinates. ``labels`` initially hold
    centre winners; clean_thin_grains() can edit these assignments independently
    of the continuous APD. ``boundary_cells``
    means a boundary cannot be excluded by cost bounds; ``sampled_boundary``
    means the 27 corner/edge/face/centre samples have different grain labels.
    These flags are sampling diagnostics, not non-manifold classifications.

    Leaves may have hanging nodes and are not yet 2:1 balanced or tetrahedralized.
    Periodic costs are used when requested by the diagram; opposite face leaf
    subdivisions are not yet explicitly paired. Arrays snapshot the build.
    """
    indices: np.ndarray
    levels: np.ndarray
    labels: np.ndarray
    boundary_cells: np.ndarray
    sampled_boundary: np.ndarray
    resolution: tuple
    max_depth: int
    box_size: np.ndarray
    periodic: bool

    def nonmanifold_grain_boundary_cells(self, *, chunk_size=65536, return_report=False):
        """Check current leaf labels, including coarse/fine and periodic contacts.

        Returns a boolean leaf mask, or a report with physical contact locations
        and offending grain IDs when return_report=True. See
        nonmanifold_octree_grain_boundary_cells for the topology definition.
        """
        return nonmanifold_octree_grain_boundary_cells(
            self, chunk_size=chunk_size, return_report=return_report)

    def thin_grain_cells(self, *, max_width=2):
        """Mask axial grain runs of <= max_width finest GB cells (1 or 2).

        Both ends must touch other grains; the box exterior does not count.
        Every cell in the run must be at max_depth and boundary_cells=True.
        This axis-based thickness diagnostic is not a manifoldness test.
        """
        _validate_width(max_width)
        neighbours = _finest_face_neighbours(self)
        eligible = self.boundary_cells & (self.levels == self.max_depth)
        return _thin_cells(self.labels, eligible, neighbours, max_width)

    def clean_thin_grains(self, diagram, *, max_width=2, max_passes=3,
                          batch_size=8192):
        """Reassign thin GB leaf labels in place, returning a cleanup report.

        Only initially detected thin cells can change, and each changes at most
        once. Revalidate thickness before each sequential change; choose the
        minimum centre APD cost among current face-neighbour grains other than
        the donor. Ties follow diagram grain order. Process lower cost penalties
        first, with geometric index tie-breaking. Coarse cells never change.

        Changes may remove small grains or shift junctions; they do not guarantee
        manifoldness or grain connectivity. APD functions and leaf geometry are
        unchanged. boundary_cells/sampled_boundary remain original APD diagnostics,
        not boundaries of the cleaned labels. All calculations complete before
        committing labels, so errors leave this octree unchanged.
        """
        _validate_width(max_width)
        for name, value in [('max_passes', max_passes), ('batch_size', batch_size)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (bool(diagram.periodic) != self.periodic or
                not np.array_equal(diagram.box_size, self.box_size)):
            raise ValueError('diagram box and periodicity must match the octree')
        grain_ids = np.asarray(diagram.grain_ids)
        if not np.all(np.isin(self.labels, grain_ids)):
            raise ValueError('All octree labels must occur in diagram.grain_ids')
        neighbours = _finest_face_neighbours(self)
        eligible = self.boundary_cells & (self.levels == self.max_depth)
        original = self.labels.copy()
        labels = original.copy()
        initial = _thin_cells(labels, eligible, neighbours, max_width)
        selected = np.flatnonzero(initial)
        costs = np.empty((len(selected), len(grain_ids)))
        centers = self.centers
        for start in range(0, len(selected), batch_size):
            costs[start:start + batch_size] = diagram.costs(centers[selected[start:start + batch_size]])
        if not np.all(np.isfinite(costs)):
            raise ValueError('APD costs must be finite')
        cost_rows = {int(cell): row for row, cell in enumerate(selected)}
        columns = {gid: column for column, gid in enumerate(grain_ids)}
        changed = np.zeros(len(labels), dtype=bool)

        def replacement(cell):
            adjacent = neighbours[cell]
            ids = set(labels[adjacent[adjacent >= 0]]) - {labels[cell]}
            choices = sorted(columns[gid] for gid in ids)
            if not choices:
                return None
            values = costs[cost_rows[cell]]
            column = min(choices, key=lambda c: (values[c], c))
            return grain_ids[column], float(values[column] - values[columns[labels[cell]]])

        changes, passes = [], 0
        for _ in range(max_passes):
            current = _thin_cells(labels, eligible, neighbours, max_width)
            queue = []
            for cell in np.flatnonzero(initial & current & ~changed):
                target = replacement(cell)
                if target is not None:
                    queue.append((target[1], tuple(self.indices[cell]), int(cell)))
            if not queue:
                break
            passes += 1
            for _, _, cell in sorted(queue):
                if not _is_thin(cell, labels, eligible, neighbours, max_width):
                    continue
                target = replacement(cell)
                if target is None:
                    continue
                new, penalty = target
                changes.append(dict(cell=cell, old_label=labels[cell].item(),
                                    new_label=new.item(), cost_increase=penalty))
                labels[cell] = new
                changed[cell] = True
        remaining = _thin_cells(labels, eligible, neighbours, max_width)
        volumes = np.prod(self.sizes, axis=1)
        before = {gid.item(): float(volumes[original == gid].sum()) for gid in grain_ids}
        after = {gid.item(): float(volumes[labels == gid].sum()) for gid in grain_ids}
        self.labels = labels
        return dict(initial_thin_mask=initial, changed_mask=changed,
                    remaining_thin_mask=remaining, changes=changes, passes=passes,
                    initial_thin_cells=int(initial.sum()), changed_cells=int(changed.sum()),
                    remaining_thin_cells=int(remaining.sum()),
                    volumes_before=before, volumes_after=after,
                    eliminated_grains=[g for g in before if before[g] > 0 and after[g] == 0])

    @property
    def sizes(self):
        return self.box_size / (np.asarray(self.resolution) *
                                np.exp2(self.levels[:, None]))

    @property
    def lower(self):
        return self.indices * self.sizes

    @property
    def upper(self):
        return (self.indices + 1) * self.sizes

    @property
    def centers(self):
        return (self.indices + .5) * self.sizes

    def summary(self):
        levels, counts = np.unique(self.levels, return_counts=True)
        return dict(leaves=len(self.levels),
                    leaves_by_level=dict(zip(levels.tolist(), counts.tolist())),
                    boundary_candidates=int(self.boundary_cells.sum()),
                    sampled_boundary_cells=int(self.sampled_boundary.sum()),
                    depth_limited_cells=int(np.count_nonzero(
                        self.boundary_cells & (self.levels == self.max_depth))),
                    total_volume=float(np.prod(self.sizes, axis=1).sum()),
                    uniform_finest_cells=int(np.prod(self.resolution)) * 8**self.max_depth,
                    periodic=self.periodic, balanced=False, tetrahedralized=False)

    def plot_slice(self, axis='z', position=None, *, color_by='level', ax=None):
        """Plot exact leaf rectangles intersecting a coordinate plane.

        Colour by ``level``, assigned ``grain`` label, ``nonmanifold`` mask,
        or original APD ``boundary`` status
        (0: bounded interior, 1: candidate, 2: sampled boundary). Half-open
        selection avoids drawing both cells at a coincident cell face.
        Returns the Matplotlib Axes; does not call show().
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import BoundaryNorm
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if color_by not in ('level', 'grain', 'boundary', 'nonmanifold'):
            raise ValueError("color_by must be 'level', 'grain', 'boundary', or 'nonmanifold'")
        normal = 'xyz'.index(axis)
        position = self.box_size[normal] / 2 if position is None else float(position)
        if not np.isfinite(position) or not 0 <= position <= self.box_size[normal]:
            raise ValueError('position must lie inside the box')
        lo, hi = self.lower, self.upper
        select = ((lo[:, normal] <= position) &
                  ((position < hi[:, normal]) |
                   ((position == self.box_size[normal]) &
                    np.isclose(hi[:, normal], position, rtol=0, atol=1e-14 * self.box_size[normal]))))
        axes = [i for i in range(3) if i != normal]
        a, b = lo[select][:, axes], hi[select][:, axes]
        polygons = np.stack([a, np.column_stack([b[:, 0], a[:, 1]]), b,
                             np.column_stack([a[:, 0], b[:, 1]])], axis=1)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if color_by == 'grain':
            ids, values = np.unique(self.labels, return_inverse=True)
            ticklabels = [str(g) for g in ids]
        elif color_by == 'boundary':
            values = self.boundary_cells.astype(int) + self.sampled_boundary.astype(int)
            ticklabels = ['interior', 'candidate', 'sampled GB']
        elif color_by == 'nonmanifold':
            values = self.nonmanifold_grain_boundary_cells().astype(int)
            ticklabels = ['unflagged', 'non-manifold']
        else:
            values = self.levels
            ticklabels = [str(i) for i in range(self.max_depth + 1)]
        cmap = plt.get_cmap('viridis', len(ticklabels))
        norm = BoundaryNorm(np.arange(len(ticklabels) + 1) - .5, cmap.N)
        collection = PolyCollection(polygons, array=values[select].astype(float),
                                    cmap=cmap, norm=norm, edgecolors='0.25', linewidths=.35)
        ax.add_collection(collection)
        colorbar = ax.figure.colorbar(collection, ax=ax, ticks=np.arange(len(ticklabels)))
        colorbar.ax.set_yticklabels(ticklabels)
        colorbar.set_label('assigned grain ID' if color_by == 'grain' else color_by)
        ax.set(xlim=(0, self.box_size[axes[0]]), ylim=(0, self.box_size[axes[1]]),
               xlabel='xyz'[axes[0]], ylabel='xyz'[axes[1]],
               title=f'APD octree: {axis} = {position:g}')
        ax.set_aspect('equal')
        return ax


def nonmanifold_octree_grain_boundary_cells(octree, *, chunk_size=65536,
                                           return_report=False):
    """Identify non-manifold per-grain surfaces in the current labelled octree.

    Treat leaves as closed boxes and test the eight local octants around every
    leaf corner, including hanging vertices. Coarse leaves can occupy multiple
    octants. Each grain's surface link must be a single simple cycle; edge and
    vertex self-contacts fail, while ordinary multi-grain junctions are valid.
    The returned boolean mask marks incident leaves of offending grains that
    also have a face neighbour of another grain. Thus it refers to the current
    GB zone, not the stored boundary_cells or sampled_boundary APD flags.

    Coordinates use the finest integer lattice only for queries. Ancestor
    lookup finds containing leaves without allocating a dense voxel grid.
    Nonperiodic exterior closes the shells but is not a grain. Periodic seams
    are identified, and report locations use [0, box_size) coordinates.

    ``return_report=True`` returns cell_mask, cell_indices (zero-based leaf
    indices), nonmanifold_cells, nonmanifold_vertices, and contacts. Each contact
    records a finest-lattice vertex_index, physical point, grain_id and incident
    offending cell_indices. Contacts are local observations, not connected
    components; an edge contact may be reported at both endpoints.

    Requires a valid, nonoverlapping octree forest covering the box. The lookup
    needs O(leaves) storage; neighbourhood arrays are bounded by chunk_size
    source corners, and detailed report storage scales with detected contacts.
    Nothing is modified. This tests the piecewise-constant leaf assignments,
    not the continuous APD, and does not test global grain connectivity.
    """
    from .voxelization import _nonmanifold_vertex_patterns

    if (isinstance(chunk_size, (bool, np.bool_)) or
            not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError('chunk_size must be a positive integer')
    if not isinstance(return_report, (bool, np.bool_)):
        raise ValueError('return_report must be a boolean')
    n = len(octree.labels)
    lookup = {(int(level), *map(int, index)): cell
              for cell, (level, index) in enumerate(zip(octree.levels, octree.indices))}
    levels = sorted(set(map(int, octree.levels)), reverse=True)
    shape = np.asarray(octree.resolution, dtype=np.int64) * 2**octree.max_depth
    strides = np.left_shift(np.int64(1), octree.max_depth - octree.levels)
    grain_ids, encoded = np.unique(octree.labels, return_inverse=True)
    # A spare exterior slot makes -1 leaf indices safe in vectorized lookups.
    encoded = np.append(encoded, -1)
    flagged, boundary = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    bad_patterns = _nonmanifold_vertex_patterns()
    weights = (1 << np.arange(8)).astype(np.uint8)
    contacts = {}

    for start in range(0, 8 * n, chunk_size):
        source = np.arange(start, min(start + chunk_size, 8 * n))
        cell = source // 8
        vertices = (octree.indices[cell] + _CHILDREN[source % 8]) * strides[cell, None]
        if octree.periodic:
            vertices %= shape
        vertices = np.unique(vertices, axis=0)
        # An infinitesimal displacement into each octant occupies the same leaf
        # as this finest voxel. Alignment makes the integer lookup exact.
        queries = vertices[:, None] + _CHILDREN - 1
        if octree.periodic:
            queries %= shape
        points, inverse = np.unique(queries.reshape(-1, 3), axis=0, return_inverse=True)
        owner = np.full(len(points), -1, dtype=np.int64)
        inside = np.all((points >= 0) & (points < shape), axis=1)
        for level in levels:
            pending = np.flatnonzero(inside & (owner < 0))
            if not len(pending):
                break
            indices = points[pending] // 2**(octree.max_depth - level)
            owner[pending] = [lookup.get((level, *map(int, index)), -1) for index in indices]
        if np.any(inside & (owner < 0)):
            raise ValueError('Octree leaves do not cover a vertex neighbourhood')
        incident = owner[inverse].reshape(-1, 8)
        local = encoded[incident]
        # Octants differing in one bit share a face. Mark actual GB leaves,
        # independently of the original APD sampling flags and refinement level.
        for bit in (1, 2, 4):
            for a in range(8):
                if a & bit:
                    continue
                b = a | bit
                different = (local[:, a] >= 0) & (local[:, b] >= 0) & (local[:, a] != local[:, b])
                boundary[incident[different, a]] = True
                boundary[incident[different, b]] = True
        for octant in range(8):
            pattern = np.sum((local == local[:, octant, None]) * weights, axis=1)
            bad = bad_patterns[pattern] & (local[:, octant] >= 0)
            flagged[incident[bad, octant]] = True
            if return_report:
                for row in np.flatnonzero(bad):
                    key = (*map(int, vertices[row]), int(local[row, octant]))
                    contacts.setdefault(key, set()).add(int(incident[row, octant]))
    flagged &= boundary
    if not return_report:
        return flagged
    records = []
    for key, cells in sorted(contacts.items()):
        cells = sorted(c for c in cells if flagged[c])
        if cells:
            vertex = np.array(key[:3], dtype=np.int64)
            records.append(dict(vertex_index=vertex,
                point=vertex / shape * octree.box_size,
                grain_id=grain_ids[key[3]].item(), cell_indices=np.array(cells, dtype=int)))
    return dict(cell_mask=flagged, cell_indices=np.flatnonzero(flagged),
                nonmanifold_cells=int(flagged.sum()),
                nonmanifold_vertices=len({tuple(c['vertex_index']) for c in records}),
                contacts=records)


def _validate_width(max_width):
    if isinstance(max_width, (bool, np.bool_)) or not isinstance(max_width, (int, np.integer)) or max_width not in (1, 2):
        raise ValueError('max_width must be 1 or 2')


def _finest_face_neighbours(tree):
    """Six neighbours of finest leaves, using ancestor lookup (no dense grid).

    A finest cell face touches exactly one same-size or coarser leaf. Coarse
    rows are unused, because they cannot participate in a short finest run.
    """
    lookup = {(int(level), *map(int, index)): cell
              for cell, (level, index) in enumerate(zip(tree.levels, tree.indices))}
    shape = np.asarray(tree.resolution, dtype=np.int64) * 2**tree.max_depth
    levels = sorted(set(map(int, tree.levels)), reverse=True)
    neighbours = np.full((len(tree.labels), 6), -1, dtype=np.int64)
    for cell in np.flatnonzero(tree.levels == tree.max_depth):
        for axis in range(3):
            for side, step in enumerate((-1, 1)):
                point = tree.indices[cell].copy()
                point[axis] += step
                if tree.periodic:
                    point %= shape
                elif np.any(point < 0) or np.any(point >= shape):
                    continue
                for level in levels:
                    index = point // 2**(tree.max_depth - level)
                    match = lookup.get((level, *map(int, index)))
                    if match is not None:
                        neighbours[cell, 2 * axis + side] = match
                        break
                else:
                    raise ValueError('Octree leaves do not cover a neighbour location')
    return neighbours


def _is_thin(cell, labels, eligible, neighbours, max_width):
    if not eligible[cell]:
        return False
    grain = labels[cell]
    for axis in range(3):
        count = 1
        bounded = True
        visited = {cell}
        for side in (0, 1):
            current = cell
            while True:
                other = int(neighbours[current, 2 * axis + side])
                if other < 0:
                    bounded = False
                    break
                if labels[other] != grain:
                    break
                if other in visited or not eligible[other]:
                    bounded = False
                    break
                count += 1
                if count > max_width:
                    bounded = False
                    break
                visited.add(other)
                current = other
            if not bounded:
                break
        if bounded:
            return True
    return False


def _thin_cells(labels, eligible, neighbours, max_width):
    result = np.zeros(len(labels), dtype=bool)
    for cell in np.flatnonzero(eligible):
        result[cell] = _is_thin(int(cell), labels, eligible, neighbours, max_width)
    return result


def _classify(diagram, centers, half_sizes):
    """Bound the cost gap across each box; never rely only on sampled labels."""
    costs = diagram.costs(centers)
    winner = np.argmin(costs, axis=1)
    row = np.arange(len(centers))
    if not diagram.periodic:
        # Quadratic cost differences cancel common terms, giving tighter bounds.
        gradients = 2 * np.einsum('gij,ngj->ngi', diagram.matrices,
                                  centers[:, None] - diagram.centers)
        delta_gradient = gradients - gradients[row, winner][:, None]
        delta_matrix = diagram.matrices[None] - diagram.matrices[winner][:, None]
        variation = np.einsum('ngi,ni->ng', np.abs(delta_gradient), half_sizes)
        variation += np.einsum('ngij,ni,nj->ng', np.abs(delta_matrix), half_sizes, half_sizes)
        margin = costs - costs[row, winner][:, None] - variation
    else:
        # Distance to the nearest lattice image is 1-Lipschitz in each grain's
        # metric, even where its nearest image changes within the box.
        corners = half_sizes[:, None] * (2 * _CHILDREN - 1)
        radius = np.sqrt(np.max(np.einsum('nki,gij,nkj->nkg', corners,
                                          diagram.matrices, corners), axis=1))
        distance = np.sqrt(np.maximum(costs + diagram.weights, 0))
        lower = np.maximum(distance - radius, 0)**2 - diagram.weights
        upper = (distance + radius)**2 - diagram.weights
        margin = lower - upper[row, winner][:, None]
    # Bias round-off towards refinement. Ignore the winner's self-comparison.
    tolerance = 128 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(costs), axis=1))
    margin[row, winner] = np.inf
    candidate = np.any(margin <= tolerance[:, None], axis=1)
    return diagram.grain_ids[winner], candidate


def build_background_octree(diagram, resolution=2, *, max_depth=3,
                            batch_size=8192, max_cells=1_000_000):
    """Refine a coarse Cartesian forest towards continuous APD interfaces.

    ``resolution`` counts root cells per axis (scalar or 3-tuple); each split
    creates eight children. A cell stops when its centre winner dominates
    throughout the box according to conservative floating-point cost bounds,
    or ``max_depth`` is reached. Bounds can over-refine, especially for periodic
    diagrams. Candidate cells at the depth limit are retained and flagged.
    Rectangular boxes yield eight-way rectangular subdivisions, not cubic cells.

    ``batch_size`` bounds cost-evaluation point counts; ``max_cells`` is a hard
    leaf budget (raises ValueError before exceeding it). No grain labels, weights,
    voxel data, or existing geometry are changed. This is step (1) only: no
    topology repair, balancing, or conversion to tetrahedra is performed.
    """
    raw = np.asarray(resolution)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or raw.dtype.kind not in 'iu' or np.any(raw < 1):
        raise ValueError('resolution must be a positive integer or three positive integers')
    for name, value, minimum in [('max_depth', max_depth, 0), ('batch_size', batch_size, 1),
                                  ('max_cells', max_cells, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if max_depth > 30:
        raise ValueError('max_depth must be <= 30 for dyadic integer coordinates')
    shape = tuple(int(v) for v in raw)
    root_count = shape[0] * shape[1] * shape[2]
    if root_count > max_cells:
        raise ValueError('Root cells exceed max_cells')
    if max(shape) * 2**max_depth > np.iinfo(np.int64).max:
        raise ValueError('resolution and max_depth exceed integer coordinate range')
    active = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), axis=-1).reshape(-1, 3)
    leaves, levels, labels, candidates = [], [], [], []
    leaf_count = root_count
    for depth in range(max_depth + 1):
        size = np.asarray(diagram.box_size) / (np.asarray(shape) * 2.**depth)
        children = []
        for start in range(0, len(active), batch_size):
            indices = active[start:start + batch_size]
            ids, candidate = _classify(diagram, (indices + .5) * size,
                                       np.broadcast_to(size / 2, indices.shape))
            split = candidate if depth < max_depth else np.zeros(len(indices), dtype=bool)
            leaf_count += 7 * int(split.sum())
            if leaf_count > max_cells:
                raise ValueError('Octree refinement exceeds max_cells; reduce max_depth or increase max_cells')
            keep = ~split
            leaves.append(indices[keep])
            levels.append(np.full(keep.sum(), depth, dtype=int))
            labels.append(ids[keep])
            candidates.append(candidate[keep])
            if split.any():
                children.append((2 * indices[split, None] + _CHILDREN).reshape(-1, 3))
        if not children:
            break
        active = np.concatenate(children)
    result = APDBackgroundOctree(np.concatenate(leaves), np.concatenate(levels),
        np.concatenate(labels), np.concatenate(candidates), np.zeros(leaf_count, dtype=bool),
        shape, int(max_depth), np.array(diagram.box_size, copy=True), bool(diagram.periodic))
    # Sampling is a plotting aid, not the refinement criterion. Batch even when
    # the caller requests fewer than 27 points per cost evaluation.
    candidate_ids = np.flatnonzero(result.boundary_cells)
    sizes, lower = result.sizes, result.lower
    for start in range(0, len(candidate_ids), max(1, batch_size // 27)):
        selected = candidate_ids[start:start + max(1, batch_size // 27)]
        points = (lower[selected, None] + sizes[selected, None] * _SAMPLES).reshape(-1, 3)
        ids = np.concatenate([diagram.labels(points[i:i + batch_size])
                              for i in range(0, len(points), batch_size)]).reshape(-1, 27)
        result.sampled_boundary[selected] = np.any(ids != ids[:, :1], axis=1)
    return result
