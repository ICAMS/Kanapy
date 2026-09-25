"""Shared boundary patches and junction curves of a polyhedral APD partition."""
from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class BoundaryPatch:
    """Edge-connected faces with the same ordered grain pair and box-plane ID.

    ``grains`` is (first, second), with None outside the box. Face normals point
    out of the first grain. Face indices refer to APDBoundaryComplex.faces.
    """
    grains: tuple
    boundary_id: int
    faces: np.ndarray


@dataclass
class JunctionCurve:
    """Ordered global vertex IDs; closed curves repeat the first vertex."""
    vertices: np.ndarray
    grains: tuple


@dataclass
class APDBoundaryComplex:
    """Boundary-only view, retaining the partition's global vertex numbering.

    Polygon faces are stored once. ``grain_shells`` maps each grain to a tuple
    of connected shells, each containing (face index, orientation sign) pairs.
    Disconnected patches/shells are preserved. Periodic faces remain separate.
    """
    points: np.ndarray
    faces: tuple
    source_faces: np.ndarray
    face_grains: tuple
    boundary_ids: np.ndarray
    patches: tuple
    grain_shells: dict
    junction_edges: np.ndarray
    junction_edge_grains: tuple
    junction_curves: tuple
    junction_vertices: np.ndarray
    vertex_grains: dict
    boundary_junction_vertices: np.ndarray
    boundary_trace_edges: np.ndarray

    def triangulate(self, *, include_exterior=False):
        """Triangulate each selected shared polygon once, independently of grains.

        Internal interfaces are selected by default; include_exterior adds box
        faces to close grain shells. Convex polygons use a centroid fan, retaining
        every perimeter edge, including collinear junction vertices. Existing
        triangles are retained. Normals point out of the first incident grain.
        No projection onto the continuous APD or volume meshing is performed.
        """
        points = list(self.points.copy())
        triangles, sources, pairs, boundaries = [], [], [], []
        seen = set()
        for fi, (face, pair) in enumerate(zip(self.faces, self.face_grains)):
            if pair[1] is None and not include_exterior:
                continue
            key = tuple(sorted(int(v) for v in face))
            if key in seen:
                raise ValueError('Duplicate shared polygon in boundary complex')
            seen.add(key)
            if len(face) == 3:
                local = [tuple(face)]
            else:
                center = len(points)
                points.append(self.points[face].mean(axis=0))
                local = [(int(a), int(b), center) for a, b in zip(face, np.roll(face, -1))]
            for tri in local:
                triangles.append(tri)
                sources.append(fi)
                pairs.append(pair)
                boundaries.append(self.boundary_ids[fi])
        result = APDBoundaryTriangles(np.asarray(points),
                    np.asarray(triangles, dtype=int).reshape(-1, 3),
                    np.asarray(sources, dtype=int), tuple(pairs),
                    np.asarray(boundaries, dtype=np.int8))
        if len(result.triangles) and np.any(result.areas <= 0):
            raise ValueError('Degenerate triangle in boundary triangulation')
        return result

    def summary(self):
        """Counts for inspection before surface or volume meshing."""
        return dict(faces=len(self.faces),
                    interface_faces=sum(pair[1] is not None for pair in self.face_grains),
                    exterior_faces=int(np.count_nonzero(self.boundary_ids)),
                    patches=len(self.patches), grains=len(self.grain_shells),
                    shells=sum(len(shells) for shells in self.grain_shells.values()),
                    junction_edges=len(self.junction_edges),
                    junction_curves=len(self.junction_curves),
                    junction_vertices=len(self.junction_vertices),
                    higher_order_vertices=sum(len(self.vertex_grains[v]) >= 4
                                              for v in self.junction_vertices))

    def residuals(self, diagram):
        """Continuous APD checks at interface vertices, edge midpoints and centers.

        Per-face maxima: equality spread of the two costs and positive excess
        of their maximum over the minimum of all grain costs. Units are cost
        (length squared); sampled checks are not global geometric certificates.
        Exterior faces have NaN residuals. The diagram must use these coordinates
        and the same fitted weights as the partition's source diagram.
        """
        columns = {gid: i for i, gid in enumerate(diagram.grain_ids)}
        equality = np.full(len(self.faces), np.nan)
        dominance = equality.copy()
        for i, (face, pair) in enumerate(zip(self.faces, self.face_grains)):
            if pair[1] is None:
                continue
            xyz = self.points[face]
            samples = np.vstack([xyz, (xyz+np.roll(xyz, -1, axis=0))/2, xyz.mean(axis=0)])
            values = diagram.costs(samples)
            a, b = (columns[g] for g in pair)
            equality[i] = np.max(np.abs(values[:, a]-values[:, b]))
            dominance[i] = np.max(np.maximum(values[:, a], values[:, b])-values.min(axis=1))
        return dict(equality=equality, dominance=dominance)

    def plot(self, diagram=None):
        """Return figure/axes showing interfaces, junctions and sampled residuals.

        Does not call show(). Matplotlib is imported only for visualization.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(15, 5))
        axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
        internal = [i for i, pair in enumerate(self.face_grains) if pair[1] is not None]
        polygons = [self.points[self.faces[i]] for i in internal]
        patch_ids = {int(fi): pi for pi, patch in enumerate(self.patches) for fi in patch.faces}
        colors = plt.get_cmap('tab20')([patch_ids[fi] % 20 for fi in internal])
        axes[0].add_collection3d(Poly3DCollection(polygons, facecolors=colors, alpha=.6))
        for curve in self.junction_curves:
            xyz = self.points[curve.vertices]
            axes[1].plot(*xyz.T)
        if len(self.junction_vertices):
            axes[1].scatter(*self.points[self.junction_vertices].T, color='black')
        if diagram is not None and internal:
            values = self.residuals(diagram)['equality'][internal]
            collection = Poly3DCollection(polygons, cmap='viridis')
            collection.set_array(values)
            axes[2].add_collection3d(collection)
            fig.colorbar(collection, ax=axes[2], shrink=.6, label='Cost equality residual')
        for ax, title in zip(axes, ['Grain interfaces', 'Junction curves', 'APD residuals']):
            ax.set_title(title)
            low, high = self.points.min(axis=0), self.points.max(axis=0)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
            ax.set_box_aspect(high-low)
        return fig, axes


def _components(items, adjacency):
    remaining = set(items)
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, group = [seed], []
        while stack:
            item = stack.pop()
            group.append(item)
            neighbors = remaining.intersection(adjacency[item])
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        groups.append(np.array(sorted(group), dtype=int))
    return groups


def extract_boundary_complex(partition):
    """Extract patches, closed grain shells and incidence-based junctions.

    Removes same-grain background faces. Keeps polygons intact; no projection,
    smoothing, independent triangulation or volume tetrahedralization occurs.
    Junction edges have at least three incident grains; exterior two-grain
    traces are stored separately. Degree changes, >=4-grain vertices and box
    contacts split junction curves. Components are connected through edges.
    """
    selected = sorted(set(partition.interface_faces).union(np.flatnonzero(partition.boundary_ids)))
    faces, pairs, boundaries = [], [], []
    for fi in selected:
        a, b = partition.face_regions[fi]
        ga = partition.region_grain_ids[a]
        gb = None if b < 0 else partition.region_grain_ids[b]
        loop = partition.faces[fi].copy()
        if gb is not None and gb < ga:
            ga, gb = gb, ga
            loop = loop[::-1]
        faces.append(loop)
        pairs.append((ga, gb))
        boundaries.append(partition.boundary_ids[fi])
    edges = defaultdict(list)
    vertices = defaultdict(set)
    exterior_vertices = set()
    for fi, (face, pair) in enumerate(zip(faces, pairs)):
        grains = set(g for g in pair if g is not None)
        for v in face:
            vertices[int(v)].update(grains)
            if pair[1] is None:
                exterior_vertices.add(int(v))
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((int(a), int(b))))].append(fi)
    adjacency = defaultdict(set)
    for incident in edges.values():
        for fi in incident:
            adjacency[fi].update(incident)
    by_patch = defaultdict(list)
    by_grain = defaultdict(dict)
    for fi, (pair, boundary) in enumerate(zip(pairs, boundaries)):
        by_patch[(pair, int(boundary))].append(fi)
        by_grain[pair[0]][fi] = 1
        if pair[1] is not None:
            by_grain[pair[1]][fi] = -1
    patches = tuple(BoundaryPatch(pair, boundary, group)
                    for (pair, boundary), ids in by_patch.items()
                    for group in _components(ids, adjacency))
    shells = {}
    for grain, signs in by_grain.items():
        shells[grain] = tuple(tuple((int(fi), signs[fi]) for fi in group)
                              for group in _components(signs, adjacency))
        # Check directed-edge cancellation separately on each shell.
        for shell in shells[grain]:
            balance = defaultdict(list)
            for fi, sign in shell:
                face = faces[fi] if sign == 1 else faces[fi][::-1]
                for a, b in zip(face, np.roll(face, -1)):
                    balance[tuple(sorted((a, b)))].append((a, b))
            if any(len(e) != 2 or e[0] != e[1][::-1] for e in balance.values()):
                raise ValueError(f'Grain {grain} has an open or nonmanifold boundary shell')
    junctions, signatures, traces = [], [], []
    for edge, incident in sorted(edges.items()):
        grains = tuple(sorted({g for fi in incident for g in pairs[fi] if g is not None}))
        if len(grains) >= 3:
            junctions.append(edge)
            signatures.append(grains)
        elif len(grains) == 2 and any(boundaries[fi] for fi in incident):
            traces.append(edge)
    vertex_edges = defaultdict(list)
    for ei, edge in enumerate(junctions):
        for v in edge:
            vertex_edges[v].append(ei)
    stops = {v for v, incident in vertex_edges.items()
             if len(incident) != 2 or len(vertices[v]) >= 4 or v in exterior_vertices
             or len({signatures[e] for e in incident}) != 1}
    curves, unused = [], set(range(len(junctions)))
    while unused:
        ei = min(unused)
        # Start at a terminal edge if one exists; otherwise this component is a loop.
        terminal = next(((e, v) for e in sorted(unused) for v in junctions[e] if v in stops), None)
        if terminal is not None:
            ei, start = terminal
        else:
            start = junctions[ei][0]
        signature = signatures[ei]
        path, current = [start], start
        while True:
            unused.remove(ei)
            a, b = junctions[ei]
            current = b if current == a else a
            path.append(current)
            if current in stops or current == start:
                break
            available = [e for e in vertex_edges[current] if e in unused]
            if not available:
                break
            ei = available[0]
        curves.append(JunctionCurve(np.array(path, dtype=int), signature))
    higher = {v for v, grains in vertices.items() if len(grains) >= 4}
    return APDBoundaryComplex(
        partition.points.copy(), tuple(faces), np.array(selected, dtype=int), tuple(pairs),
        np.array(boundaries, dtype=np.int8), patches, shells,
        np.array(junctions, dtype=int).reshape(-1, 2), tuple(signatures), tuple(curves),
        np.array(sorted(stops | higher), dtype=int),
        {v: tuple(sorted(g)) for v, g in vertices.items()},
        np.array(sorted((stops | higher) & exterior_vertices), dtype=int),
        np.array(traces, dtype=int).reshape(-1, 2))


@dataclass
class APDBoundaryTriangles:
    """Shared oriented surface triangles; source_faces indexes boundary polygons.

    Grain pairs are metadata, not separate copies of an interface. Points retain
    original global IDs and append polygon centroids when produced by
    ``APDBoundaryComplex.triangulate``. Gmsh remeshing instead creates compact new
    point IDs and uses source_faces for polygon classification (a remeshed
    triangle can cross its source polygon's edges). Exterior
    triangles have a second grain of None. STL cannot retain this metadata.
    """
    points: np.ndarray
    triangles: np.ndarray
    source_faces: np.ndarray
    face_grains: tuple
    boundary_ids: np.ndarray

    @property
    def area_vectors(self):
        xyz = self.points[self.triangles]
        return np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])/2

    @property
    def areas(self):
        return np.linalg.norm(self.area_vectors, axis=1)

    @property
    def normals(self):
        return self.area_vectors / self.areas[:, None]

    def write_stl(self, filename, *, name='APD_boundaries'):
        """Write each shared triangle once in ASCII STL with unit normals.

        Validate before opening the file. An interface network need not be a
        closed manifold solid; STL loses grain IDs, adjacency and shared nodes.
        """
        if not np.all(np.isfinite(self.points)) or np.any(self.areas <= 0):
            raise ValueError('STL requires finite nondegenerate triangles')
        if len({tuple(sorted(t)) for t in self.triangles}) != len(self.triangles):
            raise ValueError('Duplicate triangle in boundary triangulation')
        name = str(name).replace('\n', ' ').replace('\r', ' ')
        with open(filename, 'w', encoding='ascii') as stream:
            stream.write(f'solid {name.encode("ascii", "replace").decode()}\n')
            for normal, triangle in zip(self.normals, self.points[self.triangles]):
                stream.write('  facet normal ' + ' '.join(f'{x:.17g}' for x in normal) + '\n    outer loop\n')
                for vertex in triangle:
                    stream.write('      vertex ' + ' '.join(f'{x:.17g}' for x in vertex) + '\n')
                stream.write('    endloop\n  endfacet\n')
            stream.write('endsolid\n')
