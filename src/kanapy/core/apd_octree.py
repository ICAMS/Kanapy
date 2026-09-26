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
    bounds are derived from these dyadic coordinates. ``labels`` are centre
    labels, not a piecewise-constant replacement for the APD. ``boundary_cells``
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

        Colour by ``level``, centre ``grain`` label, or ``boundary`` status
        (0: bounded interior, 1: candidate, 2: sampled boundary). Half-open
        selection avoids drawing both cells at a coincident cell face.
        Returns the Matplotlib Axes; does not call show().
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import BoundaryNorm
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if color_by not in ('level', 'grain', 'boundary'):
            raise ValueError("color_by must be 'level', 'grain', or 'boundary'")
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
        colorbar.set_label('centre grain ID' if color_by == 'grain' else color_by)
        ax.set(xlim=(0, self.box_size[axes[0]]), ylim=(0, self.box_size[axes[1]]),
               xlabel='xyz'[axes[0]], ylabel='xyz'[axes[1]],
               title=f'APD octree: {axis} = {position:g}')
        ax.set_aspect('equal')
        return ax


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
