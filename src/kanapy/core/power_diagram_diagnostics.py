"""Numerical contact diagnostics for a continuous anisotropic power diagram.

Candidate discovery is finite and incomplete. Refinement, tracing, and spherical
link probes evaluate the original quadratic costs, including periodic image
branches. Findings are numerical evidence; an empty report is NOT a certificate
of manifoldness, connectivity, or absence of higher-order contacts.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


@dataclass
class ContactDiagnostic:
    """One contact observation, not necessarily one connected contact component.

    Residuals are divided by the mean-grain length squared. Branches record the
    grain ID and lattice shift; multiple images of one grain do not inflate the
    distinct grain count. ``trace`` is only a locally verified portion of a line.
    """

    point: np.ndarray
    grain_ids: tuple
    kind: str
    rank: int
    singular_values: np.ndarray
    residual: float
    branches: tuple
    trace: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    probe: dict = field(default_factory=dict)


@dataclass
class APDTopologyReport:
    contacts: list
    unresolved: list
    settings: dict
    discovery: dict
    certified: bool = False

    def summary(self):
        """Counts refer to observations; line components are not globally merged."""
        return {
            'certified': False,
            'contact_observations': len(self.contacts),
            'kinds': dict(Counter(c.kind for c in self.contacts)),
            'suspected_non_manifold': sum(c.probe.get('status') == 'suspected_non_manifold'
                                          for c in self.contacts),
            'unresolved_local_probes': sum(c.probe.get('status', '').startswith('unresolved')
                                            for c in self.contacts),
            'unprobed_contacts': sum(c.probe.get('status') == 'not_probed' for c in self.contacts),
            'unresolved_candidates': len(self.unresolved),
            'candidate_budget_exhausted': self.discovery['budget_exhausted'],
            'note': 'Finite numerical search; no global topology guarantee.',
        }

    def to_dict(self):
        """JSON-compatible report, including locations, grain IDs and settings."""
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        return plain(asdict(self))

    def plot(self, diagram, ax=None):
        """Plot observations and local line traces, without creating a grain mesh."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure(figsize=(9, 8)).add_subplot(projection='3d')
        colors = {'ordinary_vertex': '0.65', 'ordinary_triple_line': '0.65', 'ordinary_interface': '0.65',
                  'higher_order_vertex': 'purple', 'higher_order_line': 'darkorange',
                  'singular_contact': 'crimson', 'periodic_branch_contact': 'royalblue'}
        for kind in sorted({c.kind for c in self.contacts}):
            points = np.array([c.point for c in self.contacts if c.kind == kind])
            ax.scatter(*points.T, s=18, color=colors.get(kind, 'black'),
                       label=kind.replace('_', ' '), depthshade=False)
        for contact in self.contacts:
            if len(contact.trace):
                points = contact.trace
                if diagram.periodic:
                    # Break traces at box crossings rather than drawing across the RVE.
                    points = points % diagram.box_size
                for a, b in zip(points[:-1], points[1:]):
                    if not diagram.periodic or np.all(np.abs(b - a) < diagram.box_size / 2):
                        ax.plot(*np.array([a, b]).T, color='darkorange', linewidth=2)
            if contact.probe.get('status') == 'suspected_non_manifold':
                ax.scatter(*contact.point, s=85, facecolors='none', edgecolors='red')
        ax.set(xlim=(0, diagram.box_size[0]), ylim=(0, diagram.box_size[1]),
               zlim=(0, diagram.box_size[2]), xlabel='x', ylabel='y', zlabel='z',
               title='Direct APD contact diagnostics (numerical evidence)')
        ax.set_box_aspect(diagram.box_size.copy())
        if self.contacts:
            ax.legend(fontsize=8)
        return ax


class _Evaluator:
    def __init__(self, diagram, tolerance, rank_tolerance):
        self.apd = diagram
        self.length = (diagram.volume / len(diagram.centers)) ** (1 / 3)
        self.scale = self.length**2
        self.tolerance = tolerance
        self.rank_tolerance = rank_tolerance

    def values_gradients(self, point, indices=None):
        """Gradients of nearest branches, in physical coordinates."""
        d = self.apd
        x = np.asarray(point)
        indices = np.arange(len(d.centers)) if indices is None else np.asarray(indices, dtype=int)
        centers = d.centers[indices].copy()
        if d.periodic:
            x = x % d.box_size
            for column, i in enumerate(indices):
                tree, transform = d._image_trees[i]
                index = tree.query(x @ transform)[1]
                centers[column] = np.linalg.solve(transform.T, tree.data[index])
        delta = x - centers
        gradients = 2 * np.einsum('nij,nj->ni', d.matrices[indices], delta)
        values = .5 * np.einsum('ni,ni->n', delta, gradients) - d.weights[indices]
        return values / self.scale, gradients / self.scale

    def active(self, point):
        """All minimizing grain/image branches, not only one nearest image."""
        d = self.apd
        values, _ = self.values_gradients(point)
        grains = np.flatnonzero(values - values.min() <= self.tolerance)
        x = np.asarray(point) % d.box_size if d.periodic else np.asarray(point)
        branches, gradients = [], []
        for i in grains:
            images = [(d.centers[i], np.zeros(3, dtype=int))]
            if d.periodic:
                tree, transform = d._image_trees[i]
                # Include every image within the cost tolerance of the global minimum.
                radius = np.sqrt(max(0., (values.min() + self.tolerance) * self.scale + d.weights[i]))
                indices = tree.query_ball_point(x @ transform, radius * (1 + 1e-12) + 1e-12 * self.length)
                images = []
                for index in indices:
                    center = np.linalg.solve(transform.T, tree.data[index])
                    images.append((center, np.rint((center - d.centers[i]) / d.box_size).astype(int)))
            for center, shift in images:
                cost = ((x - center) @ d.matrices[i] @ (x - center) - d.weights[i]) / self.scale
                if cost - values.min() <= self.tolerance * 1.01:
                    branches.append((d.grain_ids[i].item(), tuple(shift.tolist())))
                    gradients.append(2 * d.matrices[i] @ (x - center) / self.length)
        gradients = np.array(gradients).reshape(-1, 3)
        jacobian = gradients[1:] - gradients[:1]
        _, s, vh = np.linalg.svd(jacobian, full_matrices=True)
        threshold = self.rank_tolerance * max(1., s[0] if len(s) else 0.)
        rank = int(np.count_nonzero(s > threshold))
        return grains, tuple(branches), rank, s, vh

    def refine(self, seed, grains, radius, tangent=None, prediction=None):
        """Bounded local solve; optional tangent plane is a continuation constraint."""
        seed = np.asarray(seed, dtype=float)
        grains = np.asarray(grains, dtype=int)
        low, high = (seed - radius) / self.length, (seed + radius) / self.length
        if not self.apd.periodic:
            low = np.maximum(low, 0.)
            high = np.minimum(high, self.apd.box_size / self.length)

        def function(y):
            values, gradients = self.values_gradients(y * self.length, grains)
            residual = values[1:] - values[0]
            jac = (gradients[1:] - gradients[0]) * self.length
            if tangent is not None:
                residual = np.r_[residual, ((y * self.length - prediction) @ tangent) / self.length]
                jac = np.vstack([jac, tangent])
            return residual, jac

        result = least_squares(lambda y: function(y)[0],
                               np.clip(seed / self.length, low, high),
                               jac=lambda y: function(y)[1], bounds=(low, high),
                               max_nfev=80, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        point = result.x * self.length
        values, _ = self.values_gradients(point)
        residual = float(np.max(np.abs(values[grains] - values[grains[0]])))
        dominance = float(np.max(values[grains]) - values.min())
        return point, residual, dominance, result.success

    def trace(self, point, grains, step, steps=6):
        """Short predictor/corrector continuation verified against all competitors."""
        paths = []
        for sign in (-1, 1):
            current = point.copy()
            previous = None
            path = []
            for _ in range(steps):
                active, branches, rank, _, vh = self.active(current)
                if rank != 2 or not set(grains).issubset(active):
                    break
                tangent = vh[-1] * sign
                if previous is not None and tangent @ previous < 0:
                    tangent = -tangent
                predicted = current + step * tangent
                if not self.apd.periodic and np.any((predicted <= 0) | (predicted >= self.apd.box_size)):
                    break
                new, residual, dominance, success = self.refine(
                    predicted, grains, step, tangent, predicted)
                if (not success or max(residual, dominance) > self.tolerance
                        or (new - current) @ tangent < .8 * step
                        or np.linalg.norm(new - current) > 1.5 * step):
                    break
                # Check the intervening point too: isolated rank loss is not a line.
                middle, er, dom, ok = self.refine((current + new) / 2, grains, step / 2,
                                                tangent, (current + new) / 2)
                if not ok or max(er, dom) > self.tolerance:
                    break
                if self.active(new)[2] != 2 or self.active(middle)[2] != 2:
                    break
                path.extend([middle, new])
                previous, current = tangent, new
            paths.append(path)
        return np.array(list(reversed(paths[0])) + [point] + paths[1]).reshape(-1, 3)

    def refine_branches(self, seed, indices, centers, radius):
        """Locate contacts with two images of one grain and another grain.

        Keep polynomial branches fixed during this solve, then verify them
        against the full periodic lower envelope, including all other images.
        """
        d = self.apd
        matrices = d.matrices[indices]

        def evaluate(y):
            delta = y * self.length - centers
            grads = 2 * np.einsum('nij,nj->ni', matrices, delta)
            costs = (.5 * np.einsum('ni,ni->n', delta, grads) - d.weights[indices]) / self.scale
            return costs, grads / self.length

        result = least_squares(lambda y: np.diff(evaluate(y)[0]), seed / self.length,
                               jac=lambda y: np.diff(evaluate(y)[1], axis=0),
                               bounds=((seed - radius) / self.length, (seed + radius) / self.length),
                               max_nfev=80, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        values, _ = evaluate(result.x)
        point = result.x * self.length
        minimum = d.costs([point]).min() / self.scale
        return point, float(np.ptp(values)), float(values.max() - minimum), result.success


def _periodic_candidates(e, points, order, h, per_tuple):
    """Discover image-branch contacts even when fewer than four grains meet."""
    d = e.apd
    candidates = []
    groups = {}
    for i, (tree, transform) in enumerate(d._image_trees):
        selected = np.flatnonzero(np.any(order[:, :2] == i, axis=1))
        x = points[selected] % d.box_size
        distances, images = tree.query(x @ transform, k=2)
        priority = np.argsort(distances[:, 1]**2 - distances[:, 0]**2)
        for row in priority:
            index = selected[row]
            j = next(int(g) for g in order[index, :2] if g != i)
            other_tree, other_transform = d._image_trees[j]
            other = other_tree.query(x[row] @ other_transform)[1]
            key = (i, j, *sorted(images[row].tolist()), int(other))
            group = groups.setdefault(key, [])
            if len(group) >= per_tuple or any(np.linalg.norm((x[row] - p) / h) < .5 for p in group):
                continue
            group.append(x[row])
            centers = np.vstack([np.linalg.solve(transform.T, tree.data[images[row]].T).T,
                                 np.linalg.solve(other_transform.T, other_tree.data[other])])
            # The job's branch payload is evaluated using exact quadratics.
            candidates.append((x[row], (i, j), ('branches', np.array([i, i, j]), centers)))
    return candidates


@lru_cache(maxsize=5)
def _sphere(level):
    """Closed triangular sphere with deterministic rotation to reduce exact ties."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
                        [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
                        [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    for _ in range(level):
        faces = ConvexHull(vertices).simplices
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                  faces[:, [0, 2]]]), axis=1), axis=0)
        mid = vertices[edges].mean(axis=1)
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        vertices = np.vstack([vertices, mid])
    rotation, _ = np.linalg.qr(np.array([[1., .37, .19], [.28, 1., .41], [.31, .23, 1.]]))
    vertices = vertices @ rotation
    faces = ConvexHull(vertices).simplices
    edges, inverse = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                      faces[:, [0, 2]]]), axis=1), axis=0, return_inverse=True)
    return vertices, edges, inverse.reshape(3, -1).T


def _link_signature(margin, edges, face_edges):
    """Count loops in the piecewise-linear zero contour on a triangular sphere."""
    ties = np.count_nonzero(np.abs(margin) <= 1e-12 * max(np.max(np.abs(margin)), 1e-30))
    inside = margin < 0
    if np.all(inside) or not np.any(inside):
        return dict(loops=0, irregular=0, sector='full' if np.all(inside) else 'absent', ties=int(ties))
    crossing = inside[edges[:, 0]] != inside[edges[:, 1]]
    adjacency = {int(i): [] for i in np.flatnonzero(crossing)}
    for triangle in face_edges:
        hit = triangle[crossing[triangle]]
        if len(hit) == 2:
            a, b = map(int, hit)
            adjacency[a].append(b)
            adjacency[b].append(a)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return dict(loops=components, irregular=sum(len(v) != 2 for v in adjacency.values()),
                sector='resolved', ties=int(ties))


def _probe(evaluator, point, grains, radius, level):
    d = evaluator.apd
    if not d.periodic:
        clearance = np.min(np.r_[point, d.box_size - point])
        if clearance <= 1e-7 * evaluator.length:
            return dict(status='unresolved_box_boundary', reason='A full probing sphere is outside the domain.')
        radius = min(radius, .8 * clearance)
    records = []
    for angular_level in (level, level + 1):
        directions, edges, face_edges = _sphere(angular_level)
        for r in (radius, radius / 2, radius / 4):
            scores = d.costs(point + r * directions)
            signatures = {}
            for grain in grains:
                others = np.min(np.delete(scores, grain, axis=1), axis=1)
                signatures[str(d.grain_ids[grain])] = _link_signature(scores[:, grain] - others, edges, face_edges)
            records.append(dict(radius=float(r), sphere_level=angular_level, grains=signatures))
    bad, unresolved = [], False
    for grain in grains:
        gid = str(d.grain_ids[grain])
        signatures = [row['grains'][gid] for row in records]
        loops = [s['loops'] for s in signatures]
        regular = all(s['sector'] == 'resolved' and not s['ties'] and not s['irregular'] for s in signatures)
        if regular and len(set(loops)) == 1 and loops[0] > 1:
            bad.append(d.grain_ids[grain].item())
        elif not regular or set(loops) != {1}:
            unresolved = True
    status = ('suspected_non_manifold' if bad else
              'unresolved_scale_or_angular_resolution' if unresolved else 'locally_regular_at_tested_scales')
    return dict(status=status, suspect_grain_ids=bad, observations=records,
                note='Spherical link sampling is local evidence, not a topological certificate.')


def check_topology(diagram, resolution=10, *, candidate_points=None, adaptive=True,
                   cost_tolerance=1e-8, rank_tolerance=1e-6, probe_radius=None,
                   sphere_level=2, max_candidates=1200, seeds_per_tuple=4,
                   probe_regular=12, trace_step=None, trace_steps=6):
    """Search and classify continuous APD contacts without voxelizing grains.

    ``resolution`` sets the candidate grid; ``adaptive=True`` adds finer samples
    around competitive coarse points. Exact stationary pair contacts are sought
    as well, including pinches invisible to ordinary junction extraction.
    ``candidate_points`` adds known suspect locations; it is never the only search.
    Rank-deficient contacts are traced before they are classified as higher-order
    lines. Singular and higher-order contacts receive local sphere/link tests at
    three radii and two angular resolutions (``sphere_level`` and +1).
    ``trace_step`` and ``probe_radius`` use physical length units. ``trace_steps``
    is the maximum number of continuation steps in each direction. Ordinary
    contacts are probed only up to ``probe_regular``; the remainder are reported
    as unprobed. A "higher_order_line" observation does not identify a whole line.

    Tolerances use mean grain length L=(box volume/grain count)**(1/3): cost
    differences are divided by L**2 and Jacobians use x/L coordinates. The rank
    threshold is rank_tolerance*max(1, largest singular value). Increase resolution,
    budgets and sphere_level, and vary tolerances/radii to assess robustness.

    Discovery, continuation and sphere sampling can miss small features. Failed
    solves, budget limits, ambiguous links, and unsupported singularities remain
    explicit in the report. This does not test global grain connectivity or genus.
    No modifications are made to diagram weights, geometry or particle data.
    """
    for name, value, minimum in [('resolution', resolution, 2), ('max_candidates', max_candidates, 1),
                                  ('seeds_per_tuple', seeds_per_tuple, 1), ('probe_regular', probe_regular, 0),
                                  ('trace_steps', trace_steps, 2)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(sphere_level, (int, np.integer)) or not 0 <= sphere_level <= 3:
        raise ValueError('sphere_level must be an integer from 0 through 3')
    if any(not np.isfinite(v) or v <= 0 for v in (cost_tolerance, rank_tolerance)):
        raise ValueError('cost_tolerance and rank_tolerance must be finite and positive')
    e = _Evaluator(diagram, cost_tolerance, rank_tolerance)
    if probe_radius is None:
        probe_radius = .04 * e.length
    if not np.isfinite(probe_radius) or probe_radius <= 0:
        raise ValueError('probe_radius must be finite and positive')
    if trace_step is None:
        trace_step = .025 * e.length
    if not np.isfinite(trace_step) or trace_step <= 0:
        raise ValueError('trace_step must be finite and positive')
    extra = np.empty((0, 3)) if candidate_points is None else np.asarray(candidate_points, dtype=float)
    if extra.ndim != 2 or extra.shape[1] != 3 or not np.all(np.isfinite(extra)):
        raise ValueError('candidate_points must have shape (m, 3) and be finite')
    if not diagram.periodic and np.any((extra < 0) | (extra > diagram.box_size)):
        raise ValueError('candidate_points must lie inside the nonperiodic box')
    settings = dict(resolution=resolution, adaptive=adaptive, cost_tolerance=cost_tolerance,
                    rank_tolerance=rank_tolerance, probe_radius=float(probe_radius), sphere_level=sphere_level,
                    max_candidates=max_candidates, seeds_per_tuple=seeds_per_tuple, probe_regular=probe_regular,
                    trace_step=float(trace_step), trace_steps=trace_steps)
    report = APDTopologyReport([], [], settings,
        dict(sample_points=0, attempted=0, omitted_by_budget=0, omitted_by_tuple_limit=0,
             rejected_nonminimal=0, budget_exhausted=False))
    if len(diagram.centers) < 2:
        return report

    h = diagram.box_size / resolution
    radius = np.linalg.norm(h)
    axes = [np.linspace(0, b, resolution + 1) for b in diagram.box_size]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    values = diagram.costs(points) / e.scale
    width = min(4, len(diagram.centers))
    order = np.argsort(values, axis=1)[:, :width]
    gap = np.take_along_axis(values, order[:, -1:], axis=1)[:, 0] - values.min(axis=1)
    if adaptive:
        # Refine the most competitive half of coarse samples, not the entire RVE.
        selected = points[gap <= np.median(gap)]
        offsets = np.array([[a, b, c] for a in (-.25, .25) for b in (-.25, .25) for c in (-.25, .25)]) * h
        fine = (selected[:, None] + offsets).reshape(-1, 3)
        fine = fine % diagram.box_size if diagram.periodic else fine[np.all((fine >= 0) & (fine <= diagram.box_size), axis=1)]
        points = np.vstack([points, fine])
    points = np.vstack([extra, points])
    values = diagram.costs(points) / e.scale
    order = np.argsort(values, axis=1)[:, :width]
    gap = np.take_along_axis(values, order[:, -1:], axis=1)[:, 0] - values.min(axis=1)
    report.discovery['sample_points'] = len(points)
    candidates = []
    groups = {}
    # Explicit user seeds take priority over the automatically selected samples.
    priority = np.r_[np.arange(len(extra)), np.argsort(gap[len(extra):]) + len(extra)]
    for index in priority:
        key = tuple(sorted(order[index].tolist()))
        group = groups.setdefault(key, [])
        if index >= len(extra) and len(group) >= seeds_per_tuple:
            report.discovery['omitted_by_tuple_limit'] += 1
            continue
        if any(np.linalg.norm((points[index] - p) / h) < .5 for p in group):
            continue
        group.append(points[index])
        candidates.append((points[index], key, False))

    # Stationary pair equations are linear, even though equality is quadratic.
    # Include projected stationary points near several seeds for singular families.
    pairs = set(pair for row in order for pair in combinations(sorted(row.tolist()), 2))
    stationary = []
    for i, j in sorted(pairs):
        near = points[np.any(order == i, axis=1) & np.any(order == j, axis=1)]
        for seed in near[np.linspace(0, len(near) - 1, min(3, len(near)), dtype=int)]:
            _, gradients = e.values_gradients(seed, [i, j])
            hessian = 2 * (diagram.matrices[j] - diagram.matrices[i]) / e.scale
            gradient = gradients[1] - gradients[0]
            displacement = np.linalg.lstsq(hessian, -gradient, rcond=None)[0]
            point = seed + displacement
            if not diagram.periodic and np.any((point < 0) | (point > diagram.box_size)):
                continue
            vals, grads = e.values_gradients(point)
            if (np.linalg.norm(grads[j] - grads[i]) * e.length <= rank_tolerance
                    and abs(vals[j] - vals[i]) <= cost_tolerance
                    and max(vals[i], vals[j]) - vals.min() <= cost_tolerance):
                stationary.append((point, (i, j), True))
    # Inspect supplied points themselves as well as refining their neighborhoods.
    branch_jobs = _periodic_candidates(e, points, order, h, seeds_per_tuple) if diagram.periodic else []
    report.discovery['periodic_branch_candidates'] = len(branch_jobs)
    # Interleave image jobs with ordinary grain jobs so neither search monopolizes
    # the budget. Known suspect points and analytic stationary contacts go first.
    interleaved = []
    for i in range(max(len(candidates), len(branch_jobs))):
        if i < len(candidates):
            interleaved.append(candidates[i])
        if i < len(branch_jobs):
            interleaved.append(branch_jobs[i])
    candidates = [(p, (), True) for p in extra] + stationary + interleaved
    report.discovery['omitted_by_budget'] = max(0, len(candidates) - max_candidates)
    report.discovery['budget_exhausted'] = len(candidates) > max_candidates
    regular_probes = 0
    for seed, grains, direct in candidates[:max_candidates]:
        report.discovery['attempted'] += 1
        if isinstance(direct, tuple):
            point, residual, dominance, success = e.refine_branches(seed, direct[1], direct[2], radius)
        else:
            point, residual, dominance, success = (seed, 0., 0., True) if direct else e.refine(seed, grains, radius)
        if not success or max(residual, dominance) > cost_tolerance:
            if residual <= cost_tolerance and dominance > cost_tolerance:
                report.discovery['rejected_nonminimal'] += 1
            else:
                report.unresolved.append(dict(point=point, grain_ids=diagram.grain_ids[list(grains)],
                                               residual=residual, minimum_gap=dominance, reason='local_refinement_failed'))
            continue
        if diagram.periodic:
            point = point % diagram.box_size
        active, branches, rank, singular_values, _ = e.active(point)
        if len(active) < 2:
            continue
        ids = tuple(diagram.grain_ids[active].tolist())
        # Keep nearby ordinary vertices distinct; deduplicate only within tolerance.
        merge_distance = 1e-5 * e.length
        duplicate = False
        for old in report.contacts:
            if old.grain_ids != ids:
                continue
            delta = old.point - point
            if diagram.periodic:
                delta -= np.round(delta / diagram.box_size) * diagram.box_size
            if np.linalg.norm(delta) < merge_distance:
                duplicate = True
                break
        if duplicate:
            continue
        trace = np.empty((0, 3))
        # Near a quadratic tangency, position error is O(sqrt(cost residual)).
        # Do not call its numerically perturbed root a regular vertex merely
        # because the nominal Jacobian threshold gives full rank.
        loose_threshold = max(rank_tolerance, 10 * np.sqrt(cost_tolerance)) * max(1., singular_values[0] if len(singular_values) else 0.)
        loose_rank = np.count_nonzero(singular_values > loose_threshold)
        if loose_rank != rank:
            kind = 'near_degenerate_contact'
        elif rank == 3 and len(active) == 4 and len(branches) == 4:
            kind = 'ordinary_vertex'
        elif rank == 3 and len(active) >= 5:
            kind = 'higher_order_vertex'
        elif rank == 2 and len(active) >= 4:
            trace = e.trace(point, active, trace_step, trace_steps)
            kind = 'higher_order_line' if len(trace) >= 5 else 'singular_contact'
        elif len(branches) > len(active):
            kind = 'periodic_branch_contact'
        elif rank == 2 and len(active) == 3:
            kind = 'ordinary_triple_line'
        elif rank == 1 and len(active) == 2:
            kind = 'ordinary_interface'
        else:
            kind = 'singular_contact'
        vals, _ = e.values_gradients(point)
        record = ContactDiagnostic(point, ids, kind, rank, singular_values,
                                   float(vals[active].max() - vals.min()), branches, trace)
        ordinary = kind in ('ordinary_vertex', 'ordinary_triple_line', 'ordinary_interface')
        if not ordinary or regular_probes < probe_regular:
            record.probe = _probe(e, point, active, probe_radius, sphere_level)
            if ordinary:
                regular_probes += 1
        else:
            record.probe = dict(status='not_probed', reason='Regular-contact probe budget reached.')
        if kind in ('singular_contact', 'near_degenerate_contact'):
            record.probe['classification_note'] = 'Rank loss alone does not establish contact dimension.'
        report.contacts.append(record)
    return report
