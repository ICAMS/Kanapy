"""Geometric regression checks for polygon reconstruction."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from kanapy.core.grains import calc_polygons
from kanapy.core.input_output import import_voxels


@pytest.fixture(scope="module")
def reconstructed_grains():
    path = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
    ms = import_voxels("demo_voxels.json", path=str(path))
    # Exercise unequal face areas as well as nonplanar shared interfaces.
    scale = np.array([1., 1.4, 1.8])
    ms.mesh.nodes *= scale
    ms.rve.size = np.asarray(ms.rve.size) * scale
    ms.mesh.vox_center_dict = {
        key: np.asarray(value) * scale
        for key, value in ms.mesh.vox_center_dict.items()
    }
    return ms, calc_polygons(ms.rve, ms.mesh)


def test_shared_area_matches_label_interfaces(reconstructed_grains):
    ms, geometry = reconstructed_grains
    labels = ms.mesh.grains
    spacing = np.asarray(ms.rve.size) / labels.shape
    expected = defaultdict(float)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        a, b = labels[tuple(left)], labels[tuple(right)]
        changed = a != b
        area = np.prod(np.delete(spacing, axis))
        for first, second in zip(a[changed], b[changed]):
            expected[tuple(sorted((first, second)))] += area
    actual = {(a, b): area for a, b, area in geometry['GBarea']}
    assert actual.keys() == expected.keys()
    for pair, area in actual.items():
        assert area == pytest.approx(expected[pair])


def test_shell_area_and_outward_signed_volume(reconstructed_grains):
    _, geometry = reconstructed_grains
    for grain in geometry['Grains'].values():
        points = geometry['Points'][grain['Simplices']]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        assert grain['Area'] == pytest.approx(np.linalg.norm(cross, axis=1).sum() / 2)
        volume = np.einsum('ij,ij->i', points[:, 0], cross).sum() / 6
        assert volume == pytest.approx(grain['Volume'])


def test_shared_winding_and_global_facets(reconstructed_grains):
    ms, geometry = reconstructed_grains
    owners = defaultdict(list)
    points = geometry['Points']
    for grain in geometry['Grains'].values():
        for face in grain['Simplices']:
            p = points[face]
            owners[tuple(sorted(face))].append(np.cross(p[1] - p[0], p[2] - p[0]))
    assert any(len(normals) == 2 for normals in owners.values())
    for normals in owners.values():
        assert len(normals) in (1, 2)
        if len(normals) == 2:
            np.testing.assert_allclose(normals[0], -normals[1], atol=1e-10)
    assert len(geometry['Facets']) == len(owners)
    assert {tuple(sorted(face)) for face in geometry['Facets']} == owners.keys()
    for face in geometry['Facets']:
        p = points[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        normals = owners[tuple(sorted(face))]
        assert any(np.allclose(normal, candidate) for candidate in normals)
        if len(normals) == 1:
            assert np.dot(normal, p.mean(axis=0) - np.asarray(ms.rve.size) / 2) > 0
