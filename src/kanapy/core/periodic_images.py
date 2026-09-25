"""Reconstruct central APD image cells using a tiled background mesh.

Images compete separately before affine interpolation. A fundamental tile is
sufficient: all image-owned regions are translated into their central seed's
cell, removing computational tile seams rather than making them grain walls.
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .apd_boundary import APDBoundaryTriangles
from .periodic_grains import PeriodicGrainGeometry


def build_periodic_image_geometry(diagram, phases, resolution, *, batch_size,
                                  optimize, tolerance):
    from .surface_regularization import _validate
    images = diagram.periodic_image_diagram()
    background = images.background_mesh(resolution, batch_size=batch_size)
    if min(background.resolution) < 2:
        raise ValueError('Periodic image geometry requires resolution >= 2 on each axis')
    partition = background.assemble(tolerance=tolerance, optimize=optimize)
    geometry = periodic_geometry_from_partition(diagram, phases, images, background,
                                                partition, tolerance=tolerance)
    return geometry


def periodic_geometry_from_partition(diagram, phases, images, background, partition,
                                     *, tolerance=1e-10):
    """Lift an already assembled, image-labeled tile partition."""
    from .surface_regularization import _validate
    boundary = partition.boundary_complex()
    source = boundary.triangulate(include_exterior=True)
    box = diagram.box_size
    tol = tolerance*np.linalg.norm(box)
    wrapped = np.mod(source.points, box)
    wrapped[np.isclose(wrapped, box, atol=tol, rtol=0)] = 0
    wrapped[np.abs(wrapped) < tol] = 0
    roots = np.arange(len(wrapped))
    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return int(i)
    for a, b in sorted(cKDTree(wrapped).query_pairs(tol)):
        ra, rb = root(a), root(b)
        roots[max(ra, rb)] = min(ra, rb)
    unique, classes = np.unique([root(i) for i in range(len(roots))], return_inverse=True)
    canonical = wrapped[unique]
    lattice = np.rint((source.points-canonical[classes])/box).astype(int)
    if not np.allclose(source.points, canonical[classes]+lattice*box, atol=tol, rtol=0):
        raise ValueError('Ambiguous periodic image vertex welding')
    # Pair computational tile facets, including true interfaces on a tile plane.
    seams = defaultdict(list)
    interfaces = []
    for fi, (a, b) in enumerate(source.face_grains):
        if b is not None:
            interfaces.append(((fi, a, 1), (fi, b, -1)))
        else:
            axis = (int(source.boundary_ids[fi])-1)//2
            tangent = [k for k in range(3) if k != axis]
            key = tuple(sorted((int(classes[v]), *map(int, lattice[v, tangent]))
                               for v in source.triangles[fi]))
            seams[axis, key].append((fi, a, 1))
    for sides in seams.values():
        if len(sides) != 2:
            raise ValueError('Image-layer opposite tile triangulations do not match')
        if abs(int(source.boundary_ids[sides[0][0]])-int(source.boundary_ids[sides[1][0]])) != 1:
            raise ValueError('Invalid image-layer tile pairing')
        interfaces.append(tuple(sides))
    nodes, node_ids = [], {}
    triangles, pairs, sources, paired, translations = [], [], [], [], []
    for sides in interfaces:
        records = []
        for fi, image, sign in sides:
            keys = [(int(classes[v]), *map(int, lattice[v]-images.image_shifts[image]))
                    for v in source.triangles[fi][::sign]]
            records.append((images.image_parents[image], keys))
        (a, ka), (b, kb) = records
        if a == b and set(ka) == set(kb):
            continue  # internal computational tile seam of the same image cell
        mapped = []
        for keys in (ka, kb):
            tri = []
            for key in keys:
                if key not in node_ids:
                    node_ids[key] = len(nodes)
                    nodes.append(key)
                tri.append(node_ids[key])
            mapped.append(tri)
        ta, tb = mapped
        if set(ka) == set(kb):
            triangles.append(ta); pairs.append((a, b)); sources.append(sides[0][0])
        else:
            ca, cb = {k[0]: np.array(k[1:]) for k in ka}, {k[0]: np.array(k[1:]) for k in kb}
            if ca.keys() != cb.keys() or len(ca) != 3:
                raise ValueError('Refine background to resolve periodic image face correspondence')
            delta = np.array([cb[c]-ca[c] for c in ca])
            if not np.all(delta == delta[0]):
                raise ValueError('Image interface is not a lattice translation')
            paired.append((len(triangles), len(triangles)+1))
            translations.append(delta[0])
            triangles.extend([ta, tb]); pairs.extend([(a, None), (b, None)])
            sources.extend([sides[0][0], sides[1][0]])
    keys = np.array(nodes, dtype=int)
    points = canonical[keys[:, 0]]+keys[:, 1:]*box
    surface = APDBoundaryTriangles(points, np.array(triangles, dtype=int),
        np.full(len(triangles), -1, dtype=int), tuple(pairs), np.zeros(len(triangles), dtype=np.int8))
    stats = _validate(points, surface.triangles, pairs)
    expected = defaultdict(float)
    for image, volume in partition.grain_volumes.items():
        expected[images.image_parents[image]] += volume
    if set(expected) != set(map(int, diagram.grain_ids)):
        raise ValueError('Periodic image reconstruction lost a grain; refine resolution or refit weights')
    for gid, volume in expected.items():
        if not np.isclose(stats['grain_volumes'][gid], volume, rtol=1e-7, atol=1e-10*diagram.volume):
            raise ValueError(f'Image-cell shell volume disagrees with partition for grain {gid}')
    parents = {int(g): int(g) for g in diagram.grain_ids}
    report = dict(construction='explicit_periodic_images', whole_grains=True,
        artificial_box_faces=0, paired_faces=len(paired), image_seeds=len(images.centers),
        reference_box_volume=diagram.volume, grain_volumes=stats['grain_volumes'],
        parent_grain_volumes=stats['grain_volumes'].copy(), grain_parent_ids=parents,
        split_parent_grains=[], split_axes={}, split_planes=[], split_face_ids=[],
        source_triangles_reference='image partition surface')
    whole = PeriodicGrainGeometry(surface, box.copy(), keys[:, 0], keys[:, 1:],
        np.array(sources), np.array(paired, dtype=int).reshape(-1, 2),
        np.array(translations, dtype=int).reshape(-1, 3),
        {g: int(phases[g]) for g in parents}, report, parents)
    geometry = whole.as_geometry()
    shared = defaultdict(float)
    areas = surface.areas
    for pair, area in zip(surface.face_grains, areas):
        if pair[1] is not None:
            shared[tuple(sorted(pair))] += area
    for a, b in whole.paired_faces:
        ga, gb = surface.face_grains[a][0], surface.face_grains[b][0]
        if ga != gb:
            shared[tuple(sorted((ga, gb)))] += areas[a]
    geometry.update(Representation='APD', APD=diagram, ImageAPD=images,
        Background=background, Partition=partition, TileBoundary=boundary,
        Boundary=surface, WholeGrains=whole, PeriodicImageGeometry=True,
        GBarea=[[a, b, area] for (a, b), area in sorted(shared.items())])
    return geometry
