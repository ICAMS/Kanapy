#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from kanapy.core.smoothingGB import (
    Node,
    initalizeSystem,
    readGrainFaces,
    smoothingRoutine,
)


@pytest.fixture
def two_grain_mesh():
    nodes_v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    elmt_dict = {
        1: [1, 2, 3, 4, 5, 6, 7, 8],
        2: [2, 9, 10, 3, 6, 11, 12, 7],
    }
    elmt_set_dict = {
        1: [1],
        2: [2],
    }

    return nodes_v, elmt_dict, elmt_set_dict


def test_node_updates_velocity_and_position_without_changing_original_position():
    node = Node(1, 1.0, 2.0, 3.0)

    node.compute_acc(3.0, 6.0, 9.0, mass=3.0)
    node.update_vel(dt=0.5)
    node.update_pos(dt=2.0)

    assert np.allclose(node.get_vel(), [0.5, 1.0, 1.5])
    assert np.allclose(node.get_pos(), [2.0, 4.0, 6.0])
    assert np.allclose(node.get_Oripos(), [1.0, 2.0, 3.0])


def test_read_grain_faces_keeps_internal_grain_boundary_faces(two_grain_mesh):
    nodes_v, elmt_dict, elmt_set_dict = two_grain_mesh

    grain_faces = readGrainFaces(nodes_v, elmt_dict, elmt_set_dict)

    assert set(grain_faces) == {1, 2}
    assert len(grain_faces[1]) == 1
    assert len(grain_faces[2]) == 1
    assert sorted(next(iter(grain_faces[1].values()))) == [2, 3, 6, 7]
    assert sorted(next(iter(grain_faces[2].values()))) == [2, 3, 6, 7]


def test_initialize_system_creates_shared_anchor_for_grain_boundary(
    two_grain_mesh,
):
    nodes_v, elmt_dict, elmt_set_dict = two_grain_mesh
    grain_faces = readGrainFaces(nodes_v, elmt_dict, elmt_set_dict)

    all_nodes, anchor_dict = initalizeSystem(nodes_v, grain_faces)

    assert len(all_nodes) == len(nodes_v)
    assert len(anchor_dict) == 1
    assert np.allclose(next(iter(anchor_dict.values())), [1.0, 0.5, 0.5])
    for node_id in [2, 3, 6, 7]:
        assert len(all_nodes[node_id - 1].anchors) == 1


def test_smoothing_routine_returns_smoothed_nodes_and_grain_faces(
    two_grain_mesh,
):
    nodes_v, elmt_dict, elmt_set_dict = two_grain_mesh

    nodes_smooth, grain_faces = smoothingRoutine(
        nodes_v,
        elmt_dict,
        elmt_set_dict,
    )

    assert nodes_smooth.shape == nodes_v.shape
    assert set(grain_faces) == {1, 2}
    assert np.allclose(nodes_smooth, nodes_v)
