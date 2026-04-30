#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull

from kanapy.core import plotting


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class FakeParticle:
    def __init__(self, particle_id=1, duplicate=False, phasenum=0):
        self.id = particle_id
        self.duplicate = duplicate
        self.phasenum = phasenum
        self.inner = None

    def surfacePointsGen(self, nang=100):
        u = np.linspace(0, 2 * np.pi, nang)
        v = np.linspace(0, np.pi, nang)
        x = np.outer(np.cos(u), np.sin(v)).ravel()
        y = np.outer(np.sin(u), np.sin(v)).ravel()
        z = np.outer(np.ones_like(u), np.cos(v)).ravel()
        return np.column_stack([x, y, z])

    def get_pos(self):
        return np.array([0.0, 0.0, 0.0])

    def sync_poly(self):
        return


class InnerPolyhedron:
    def __init__(self):
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.convex_hull = ConvexHull(self.points).simplices


@pytest.fixture
def grain_statistics():
    return {
        "eqd": np.array([2.0, 3.0, 4.0, 5.0]),
        "eqd_scale": 3.0,
        "eqd_sig": 0.25,
        "ar": np.array([1.1, 1.3, 1.5, 1.8]),
        "ar_scale": 1.4,
        "ar_sig": 0.2,
    }


@pytest.fixture
def overlay_statistics():
    return {
        "semi_axes": {
            "initial": {
                "a": [2.0, 2.2],
                "b": [1.2, 1.3],
                "c": [1.0, 1.1],
                "a_sig": 0.2,
                "a_scale": 2.0,
                "b_sig": 0.2,
                "b_scale": 1.2,
                "c_sig": 0.2,
                "c_scale": 1.0,
                "xmin_gl": 0.5,
                "xmax_gl": 3.0,
            },
            "regridded": {
                "a": [2.4, 2.6],
                "b": [1.4, 1.5],
                "c": [1.1, 1.2],
                "a_sig": 0.25,
                "a_scale": 2.4,
                "b_sig": 0.25,
                "b_scale": 1.4,
                "c_sig": 0.25,
                "c_scale": 1.1,
                "xmin_gl": 0.5,
                "xmax_gl": 3.0,
            },
        },
        "aspect": {
            "initial": {
                "ar_sig": 0.2,
                "ar_scale": 1.4,
                "xmin_ar_i": 1.0,
                "xmax_ar_i": 2.0,
            },
            "regridded": {
                "ar_sig": 0.25,
                "ar_scale": 1.6,
                "xmin_ar_r": 1.0,
                "xmax_ar_r": 2.2,
            },
        },
        "eq_diam": {
            "initial": {
                "eqd_sig": 0.2,
                "eqd_scale": 3.0,
                "xmin_eq_i": 2.0,
                "xmax_eq_i": 5.0,
            },
            "regridded": {
                "eqd_sig": 0.25,
                "eqd_scale": 3.5,
                "xmin_eq_r": 2.0,
                "xmax_eq_r": 5.5,
            },
        },
    }


def test_plot_voxels_3d_returns_figure_in_silent_mode():
    data = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 2], [2, 1]],
        ]
    )

    fig = plotting.plot_voxels_3D(
        data,
        silent=True,
        alpha=0.5,
        asp_arr=[1, 1, 1],
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Voxelated microstructure"
    assert fig.axes[0].get_xlabel() == "x"
    assert fig.axes[0].get_ylabel() == "y"
    assert fig.axes[0].get_zlabel() == "z"


def test_plot_polygons_3d_returns_figure_for_minimal_geometry():
    geometry = {
        "Points": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "Grains": {
            1: {
                "Simplices": [[0, 1, 2], [0, 1, 3]],
                "Phase": 0,
            },
        },
    }

    fig = plotting.plot_polygons_3D(geometry, silent=True)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Polygonized microstructure"
    assert fig.axes[0].get_xlabel() == "x"
    assert fig.axes[0].get_ylabel() == "y"
    assert fig.axes[0].get_zlabel() == "z"


def test_plot_output_stats_rejects_labels_without_grains_or_voxels():
    with pytest.raises(ValueError, match="Either grains or voxels"):
        plotting.plot_output_stats([{}], ["Partcls"], silent=True)


def test_plot_ellipsoids_3d_returns_figure_and_skips_duplicates():
    particles = [
        FakeParticle(particle_id=1, duplicate=False, phasenum=0),
        FakeParticle(particle_id=2, duplicate=True, phasenum=1),
    ]

    fig = plotting.plot_ellipsoids_3D(
        particles,
        silent=True,
        phases=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "x"


def test_plot_particles_3d_returns_figure_for_inner_polyhedron():
    particle = FakeParticle(particle_id=1, duplicate=None, phasenum=1)
    particle.inner = InnerPolyhedron()

    fig = plotting.plot_particles_3D(
        [particle],
        silent=True,
        dual_phase=True,
        plot_hull=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "y"


def test_plot_output_stats_enhanced_returns_figure(grain_statistics):
    fig = plotting.plot_output_stats(
        [grain_statistics, grain_statistics],
        ["Grains", "Partcls"],
        iphase=0,
        gs_data=np.array([2.5, 3.5]),
        gs_param=(0.2, 0.0, 3.0),
        ar_data=np.array([1.2, 1.7]),
        ar_param=(0.2, 0.0, 1.4),
        silent=True,
        enhanced_plot=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Equivalent Diameter Distribution - Phase 0"
    assert fig.axes[1].get_title() == "Aspect Ratio Distribution - Phase 0"


def test_plot_init_stats_returns_figure_for_elongated_descriptor():
    stats_dict = {
        "Grain type": "Elongated",
        "Equivalent diameter": {
            "sig": 0.2,
            "scale": 3.0,
            "loc": 0.0,
            "cutoff_min": 2.0,
            "cutoff_max": 5.0,
        },
        "Aspect ratio": {
            "sig": 0.2,
            "scale": 1.5,
            "loc": 0.0,
            "cutoff_min": 1.0,
            "cutoff_max": 2.0,
        },
        "Phase": {
            "Number": 0,
            "Name": "P0",
        },
    }

    fig = plotting.plot_init_stats(
        stats_dict,
        gs_data=np.array([2.5, 3.5]),
        ar_data=np.array([1.2, 1.7]),
        gs_param=(0.2, 0.0, 3.0, 2.0, 5.0),
        ar_param=(0.2, 0.0, 1.5, 1.0, 2.0),
        silent=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Microstructure statistics of phase 0 (P0)"


def test_plot_stats_returns_six_panel_figure(overlay_statistics):
    fig = plotting.plot_stats(
        overlay_statistics,
        show=False,
        n_points_sa=20,
        n_points_other=20,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 6


def test_plot_stats_overlay_returns_three_panel_figure(overlay_statistics):
    fig = plotting.plot_stats_overlay(
        overlay_statistics,
        show=False,
        n_points_sa=20,
        n_points_other=20,
        fill_sa=False,
        fill_other=False,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_mean_ellipsoids_from_stats_returns_two_3d_axes(overlay_statistics):
    fig = plotting.plot_mean_ellipsoids_from_stats(
        overlay_statistics,
        out_png=None,
        show=False,
        nu=12,
        nv=8,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Initial mean ellipsoid"
    assert fig.axes[1].get_title() == "Regridded mean ellipsoid"
