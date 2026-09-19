"""Experimental continuous anisotropic power diagrams for packed ellipsoids.

The diagram is the lower envelope of quadratic costs, not a voxel assignment.
Volumes are fitted using Sobol quadrature. Junction extraction uses a conforming
background tetrahedralization and the lower envelope of *linearly interpolated
costs*. Its segments/vertices are approximations, not an exact CAD model or a
conformal grain volume mesh. Refinement is needed to resolve small features;
neither connectivity nor prescribed final aspect ratios are guaranteed.

Reference: Buze et al., Computational Materials Science 245 (2024), 113317,
https://doi.org/10.1016/j.commatsci.2024.113317.
"""

from dataclasses import dataclass
from itertools import combinations, permutations, product
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import qmc


@dataclass
class JunctionNetwork:
    """Piecewise-linear network clipped to the box (coordinates in input units).

    ``segments`` has shape (n, 2, 3); ``grain_ids`` contains its three grain IDs.
    ``vertices`` are junctions of at least four distinct grains, not ordinary
    segment endpoints. Box intersections of triple lines are stored separately.
    Periodic copies share grain IDs; opposite box faces are not welded for plots.
    """

    segments: np.ndarray
    grain_ids: np.ndarray
    vertices: np.ndarray
    vertex_grain_ids: tuple
    boundary_points: np.ndarray
    resolution: int


class AnisotropicPowerDiagram:
    """Partition [0, box_size] by min_i ((x-center_i)^T A_i (x-center_i)-w_i).

    Matrices must be symmetric positive definite; ``from_particles`` builds
    determinant-one matrices. Targets must be positive and sum to box volume.
    Weights have units of length squared. In periodic mode, the cost of a grain
    is the minimum over *all* lattice images, with a finite, proven search bound.
    """

    def __init__(self, centers, matrices, box_size, target_volumes=None,
                 grain_ids=None, periodic=False):
        self.centers = np.array(centers, dtype=float, copy=True)
        self.matrices = np.array(matrices, dtype=float, copy=True)
        self.box_size = np.array(box_size, dtype=float, copy=True)
        if self.centers.ndim != 2 or self.centers.shape[1:] != (3,) or not len(self.centers):
            raise ValueError("centers must have shape (n, 3), with n > 0")
        n = len(self.centers)
        if (self.box_size.shape != (3,) or not np.all(np.isfinite(self.box_size))
                or np.any(self.box_size <= 0)):
            raise ValueError("box_size must contain three finite positive lengths")
        if not np.all(np.isfinite(self.centers)):
            raise ValueError("centers must be finite")
        if (self.matrices.shape != (n, 3, 3)
                or not np.all(np.isfinite(self.matrices))
                or not np.allclose(self.matrices, self.matrices.transpose(0, 2, 1))
                or np.any(np.linalg.eigvalsh(self.matrices) <= 0)):
            raise ValueError("matrices must have shape (n, 3, 3) and be symmetric positive definite")
        self.periodic = bool(periodic)
        if self.periodic:
            self.centers %= self.box_size
        self.grain_ids = np.arange(1, n + 1) if grain_ids is None else np.array(grain_ids, copy=True)
        if self.grain_ids.shape != (n,) or len(np.unique(self.grain_ids)) != n:
            raise ValueError("grain_ids must contain one unique ID per grain")
        self.volume = float(np.prod(self.box_size))
        self.target_volumes = (np.full(n, self.volume / n) if target_volumes is None
                               else np.array(target_volumes, dtype=float, copy=True))
        if (self.target_volumes.shape != (n,)
                or not np.all(np.isfinite(self.target_volumes))
                or np.any(self.target_volumes <= 0)
                or not np.isclose(self.target_volumes.sum(), self.volume, rtol=1e-8, atol=0)):
            raise ValueError("positive target_volumes must sum to the box volume")
        self.weights = np.zeros(n)
        self._ellipsoids = []
        self._image_trees = []
        if self.periodic:
            corners = np.array(list(product((-0.5, 0.5), repeat=3))) * self.box_size
            for center, matrix in zip(self.centers, self.matrices):
                # A coordinatewise nearest image supplies a uniform upper bound
                # U on distance. Any better image has Euclidean distance <= R.
                upper = np.max(np.einsum('pi,ij,pj->p', corners, matrix, corners))
                radius = np.sqrt(upper / np.linalg.eigvalsh(matrix)[0])
                lo = np.ceil((-radius - center) / self.box_size).astype(int)
                hi = np.floor((self.box_size + radius - center) / self.box_size).astype(int)
                shifts = np.array(list(product(*(range(a, b + 1) for a, b in zip(lo, hi)))))
                images = center + shifts * self.box_size
                distance_to_box = np.maximum(np.maximum(-images, images - self.box_size), 0)
                images = images[np.linalg.norm(distance_to_box, axis=1) <= radius * (1 + 1e-12)]
                transform = np.linalg.cholesky(matrix)
                self._image_trees.append((cKDTree(images @ transform), transform))

    @classmethod
    def from_particles(cls, particles, box_size, periodic=False):
        """Use packed Kanapy particles, excluding their periodic duplicates.

        Current a*b*c ratios set target volumes, rescaled to fill the box.
        Kanapy surface points use ROW vectors ``local @ rotation_matrix``;
        consequently the matching metric is R.T @ diag(axes**-2) @ R.
        This is morphological orientation, not crystallographic orientation.
        """
        real = [p for p in particles if p.duplicate is None]
        if not real:
            raise ValueError("No original particles; run ms.pack() first")
        centers = np.array([p.get_pos() for p in real])
        axes = np.array([[p.a, p.b, p.c] for p in real])
        if not np.all(np.isfinite(axes)) or np.any(axes <= 0):
            raise ValueError("Particle semiaxes must be finite and positive")
        matrices = []
        for p, abc in zip(real, axes):
            diagonal = np.prod(abc) ** (2 / 3) / abc**2
            matrices.append(p.rotation_matrix.T @ np.diag(diagonal) @ p.rotation_matrix)
        volumes = np.prod(axes, axis=1)
        volumes = volumes / volumes.sum() * np.prod(box_size)
        diagram = cls(centers, matrices, box_size, volumes,
                      [p.id for p in real], periodic)
        diagram._ellipsoids = [(center.copy(), abc.copy(), p.rotation_matrix.copy())
                               for center, abc, p in zip(diagram.centers, axes, real)]
        return diagram

    def costs(self, points, weighted=True):
        """Evaluate continuous costs; output shape is (number of points, grains)."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite array of shape (m, 3)")
        values = np.empty((len(points), len(self.centers)))
        if self.periodic:
            points = points % self.box_size
            for i, (tree, transform) in enumerate(self._image_trees):
                values[:, i] = tree.query(points @ transform)[0] ** 2
        else:
            for i, (center, matrix) in enumerate(zip(self.centers, self.matrices)):
                d = points - center
                values[:, i] = np.einsum('pi,ij,pj->p', d, matrix, d)
        if weighted:
            values -= self.weights
        return values

    def labels(self, points):
        """Return original grain IDs; exact ties use the first grain."""
        return self.grain_ids[np.argmin(self.costs(points), axis=1)]

    def background_mesh(self, resolution=10, *, batch_size=8192):
        """Sample fixed APD costs on a conforming background tetrahedral mesh.

        Resolution counts Cartesian cells per direction (integer or 3-tuple).
        This is an intermediate reconstruction mesh, not a grain-conforming mesh.
        """
        from .apd_mesh import build_background_mesh
        return build_background_mesh(self, resolution, batch_size=batch_size)

    def check_topology(self, **kwargs):
        """Run practical continuous contact diagnostics (not a global certificate).

        See :func:`kanapy.core.power_diagram_diagnostics.check_topology` for
        controls, report contents, and limitations. Does not modify this diagram.
        """
        from .power_diagram_diagnostics import check_topology
        return check_topology(self, **kwargs)

    def _sample_points(self, n_samples, seed):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 2 or n_samples & (n_samples - 1):
            raise ValueError("n_samples must be a power of two, at least 2")
        return qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(n_samples).bit_length() - 1) * self.box_size

    def estimate_volumes(self, n_samples=32768, seed=1):
        """Independent Sobol estimate, not exact integration of the cells."""
        points = self._sample_points(n_samples, seed)
        counts = np.zeros(len(self.centers), dtype=int)
        for start in range(0, len(points), 8192):
            labels = np.argmin(self.costs(points[start:start + 8192]), axis=1)
            counts += np.bincount(labels, minlength=len(counts))
        return counts * (self.volume / n_samples)

    def fit_volumes(self, n_samples=65536, seed=0, tolerance=0.03, maxiter=500):
        """Fit weights by minimizing the sampled optimal-transport dual.

        Returns training-sample errors and optimizer status separately. Check
        ``estimate_volumes`` with another seed for quadrature error. Memory is
        O(n_samples * grains); fitting does not enforce topology or shape moments.
        """
        if not np.isfinite(tolerance) or tolerance <= 0 or maxiter < 1:
            raise ValueError("tolerance and maxiter must be positive")
        points = self._sample_points(n_samples, seed)
        scale = (self.volume / len(self.centers)) ** (2 / 3)
        base = self.costs(points, weighted=False) / scale
        target = self.target_volumes / self.volume

        def objective(free_weights):
            weights = np.r_[free_weights, 0.0]
            scores = base - weights
            owners = np.argmin(scores, axis=1)
            fractions = np.bincount(owners, minlength=len(target)) / n_samples
            value = -np.mean(scores[np.arange(n_samples), owners]) - target @ weights
            return value, (fractions - target)[:-1]

        if len(target) > 1:
            result = minimize(objective, (self.weights[:-1] - self.weights[-1]) / scale,
                              jac=True, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-8,
                                       'maxls': 40})
            self.weights = np.r_[result.x, 0.0] * scale
            self.weights -= self.weights.mean()
            optimizer_success, message = bool(result.success), str(result.message)
        else:
            optimizer_success, message = True, 'Single grain fills the box'
        owners = np.argmin(base - self.weights / scale, axis=1)
        fitted = np.bincount(owners, minlength=len(target)) * self.volume / n_samples
        errors = np.abs(fitted - self.target_volumes) / self.target_volumes
        converged = bool(np.max(errors) <= tolerance)
        if not converged:
            warnings.warn(f"APD sampled volume error {errors.max():.1%} exceeds {tolerance:.1%}; "
                          "increase n_samples/maxiter and inspect small grains.", RuntimeWarning, stacklevel=2)
        return dict(converged=converged, optimizer_success=optimizer_success,
                    message=message, volumes=fitted, relative_errors=errors,
                    max_relative_error=float(errors.max()), n_samples=n_samples)

    def extract_junctions(self, resolution=20):
        """Approximate triple lines and >=4-grain junctions without voxel labels.

        Split each of ``resolution**3`` background cubes into six conforming
        tetrahedra. Within each tetrahedron intersect three interpolated cost
        planes, clip their common line against the tetrahedron AND every competing
        cost. This also detects cells that win inside a tetrahedron but at none of
        its vertices. Only strictly dominated costs are pruned. For curved/periodic
        costs, sub-grid features can still be missed: compare finer resolutions.

        Complexity grows with resolution and the number of locally competing
        grains. Non-generic coincident interfaces are not resolved as isolated
        triple curves. No periodic self-image junctions are reported.
        """
        if not isinstance(resolution, (int, np.integer)) or resolution < 2:
            raise ValueError("resolution must be an integer >= 2")
        r = int(resolution)
        coords = [np.linspace(0, length, r + 1) for length in self.box_size]
        points = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1).reshape(-1, 3)
        scale = (self.volume / len(self.centers)) ** (2 / 3)
        scores = self.costs(points) / scale
        # Freudenthal triangulation: the same face diagonal in adjacent cubes.
        stride = np.array([(r + 1)**2, r + 1, 1])
        offsets = []
        for perm in permutations(range(3)):
            path = np.vstack([np.zeros(3, dtype=int), np.cumsum(np.eye(3, dtype=int)[list(perm)], axis=0)])
            offsets.append(path @ stride)
        origins = np.stack(np.meshgrid(*([np.arange(r)] * 3), indexing='ij'), axis=-1).reshape(-1, 3) @ stride
        segments, triples, vertices, vertex_ids, boundary = [], [], [], [], []
        seen = set()
        degenerate = False
        eps = 1e-10
        for offset in offsets:
            indices = origins[:, None] + offset
            values = scores[indices]  # tetrahedron, corner, grain
            owners = np.argmin(values, axis=2)
            mixed = np.any(owners != owners[:, :1], axis=1)
            # Uniform ownership implies uniform ownership of the linear envelope.
            for ids, corner_values, winners in zip(indices[mixed], values[mixed], owners[mixed]):
                f = corner_values.T
                local_winners = np.unique(winners)
                dominated = np.any(np.all(f[:, None, :] > f[local_winners][None, :, :] + eps, axis=2), axis=1)
                candidates = np.flatnonzero(~dominated)
                if len(candidates) < 3:
                    continue
                f = f[candidates]
                gradients = f[:, 1:] - f[:, :1]
                tetra = points[ids]
                for triple in combinations(range(len(candidates)), 3):
                    i, j, k = triple
                    equations = gradients[[j, k]] - gradients[i]
                    rhs = f[i, 0] - f[[j, k], 0]
                    direction = np.cross(*equations)
                    norm = np.linalg.norm(direction)
                    if norm <= 1e-12 * max(np.linalg.norm(equations[0]) * np.linalg.norm(equations[1]), 1e-20):
                        continue
                    direction /= norm
                    origin = np.linalg.lstsq(equations, rhs, rcond=None)[0]
                    # y=(lambda1,lambda2,lambda3), lambda0=1-sum(y).
                    g = np.vstack([-np.eye(3), np.ones(3), gradients[i] - gradients])
                    h = np.r_[np.zeros(3), 1., f[:, 0] - f[i, 0]]
                    slope, remaining = g @ direction, h - g @ origin
                    parallel = np.abs(slope) < 1e-12
                    if np.any(remaining[parallel] < -eps):
                        continue
                    positive, negative = slope > 1e-12, slope < -1e-12
                    low = np.max(remaining[negative] / slope[negative], initial=-np.inf)
                    high = np.min(remaining[positive] / slope[positive], initial=np.inf)
                    if high - low <= eps:
                        continue
                    y = origin + np.array([low, high])[:, None] * direction
                    mid_costs = f[:, 0] + gradients @ y.mean(axis=0)
                    if np.count_nonzero(mid_costs - mid_costs.min() < eps) > 3:
                        # Four grains meeting along a whole line is not an
                        # isolated vertex with incident ordinary triple lines.
                        degenerate = True
                        continue
                    ends = tetra[0] + y @ (tetra[1:] - tetra[0])
                    grain_tuple = tuple(sorted(self.grain_ids[candidates[list(triple)]].tolist()))
                    end_keys = [tuple(np.round(p / self.box_size, 9)) for p in ends]
                    key = (grain_tuple, tuple(sorted(end_keys)))
                    if key in seen:
                        continue
                    seen.add(key)
                    segments.append(ends)
                    triples.append(grain_tuple)
                    for point, bary in zip(ends, y):
                        costs = f[:, 0] + gradients @ bary
                        active = candidates[costs - costs.min() < 1e-8]
                        if len(active) >= 4:
                            vertices.append(point)
                            vertex_ids.append(set(self.grain_ids[active].tolist()))
                        elif np.any(np.isclose(point / self.box_size, 0, atol=1e-8)
                                    | np.isclose(point / self.box_size, 1, atol=1e-8, rtol=0)):
                            boundary.append(point)
        vertices, vertex_ids = _merge_points(vertices, self.box_size, vertex_ids)
        boundary, _ = _merge_points(boundary, self.box_size)
        if degenerate:
            warnings.warn("Non-generic lines shared by more than three grains were skipped; "
                          "they cannot be represented as ordinary triple lines.",
                          RuntimeWarning, stacklevel=2)
        return JunctionNetwork(np.array(segments).reshape(-1, 2, 3),
                               np.array(triples, dtype=self.grain_ids.dtype).reshape(-1, 3),
                               vertices, tuple(tuple(sorted(ids)) for ids in vertex_ids),
                               boundary, r)

    def plot_slice(self, axis=2, position=None, resolution=240, ax=None):
        """Plot a section of the continuous diagram; no voxel mesh is created."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm

        if axis not in (0, 1, 2) or resolution < 2:
            raise ValueError("axis must be 0, 1, or 2 and resolution >= 2")
        if position is None:
            position = self.box_size[axis] / 2
        other = [i for i in range(3) if i != axis]
        u, v = np.meshgrid(*(np.linspace(0, self.box_size[i], resolution) for i in other))
        points = np.empty((u.size, 3))
        points[:, axis] = position
        points[:, other] = np.column_stack([u.ravel(), v.ravel()])
        labels = np.argmin(self.costs(points), axis=1).reshape(u.shape)
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap('tab20', len(self.centers))
        ax.pcolormesh(u, v, labels, cmap=cmap,
                      norm=BoundaryNorm(np.arange(len(self.centers) + 1) - 0.5, cmap.N),
                      shading='nearest', rasterized=True)
        ax.set(xlabel='xyz'[other[0]], ylabel='xyz'[other[1]], aspect='equal',
               title=f"APD: {'xyz'[axis]} = {position:g}")
        return ax

    def plot_junctions(self, network, ax=None, ellipsoids=False):
        """Plot triple segments, >=4-grain vertices, and box crossings separately."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            ax = plt.figure(figsize=(9, 8)).add_subplot(projection='3d')
        if ellipsoids:
            u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 16), np.linspace(0, np.pi, 9))
            sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1)
            for center, axes, rotation in self._ellipsoids:
                surface = (sphere * axes) @ rotation + center
                surface[np.any((surface < 0) | (surface > self.box_size), axis=-1)] = np.nan
                ax.plot_wireframe(*surface.transpose(2, 0, 1), color='0.5', alpha=0.16,
                                  linewidth=0.4, rstride=2, cstride=2)
        if len(network.segments):
            ax.add_collection3d(Line3DCollection(network.segments, colors='steelblue',
                                                 linewidths=1.2, label='Triple lines (approx.)'))
        if len(network.vertices):
            ax.scatter(*network.vertices.T, color='crimson', s=18, depthshade=False,
                       label=f'≥4-grain junctions ({len(network.vertices)})')
        if len(network.boundary_points):
            ax.scatter(*network.boundary_points.T, color='darkorange', s=12, marker='x',
                       label='Box crossings')
        corners = np.array(list(product((0, 1), repeat=3)))
        edges = [corners[[i, j]] * self.box_size for i, j in combinations(range(8), 2)
                 if np.count_nonzero(corners[i] != corners[j]) == 1]
        ax.add_collection3d(Line3DCollection(edges, colors='0.35', linewidths=0.7))
        ax.set(xlim=(0, self.box_size[0]), ylim=(0, self.box_size[1]),
               zlim=(0, self.box_size[2]), xlabel='x', ylabel='y', zlabel='z',
               title=f'APD junction network (resolution {network.resolution})')
        # Some Matplotlib versions normalize their input array in place.
        ax.set_box_aspect(self.box_size.copy())
        if len(network.segments) or len(network.vertices) or len(network.boundary_points):
            ax.legend(loc='upper left', fontsize=8)
        return ax


def _merge_points(points, box_size, labels=None):
    """Merge numerically identical endpoints, retaining all incident grain IDs."""
    points = np.array(points).reshape(-1, 3)
    if not len(points):
        return points, []
    parents = list(range(len(points)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in cKDTree(points / box_size).query_pairs(1e-8):
        parents[root(j)] = root(i)
    groups = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    merged = np.array([points[indices].mean(axis=0) for indices in groups.values()])
    merged_labels = ([] if labels is None else
                     [set().union(*(labels[i] for i in indices)) for indices in groups.values()])
    return merged, merged_labels
