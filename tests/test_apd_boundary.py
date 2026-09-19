"""Verification of shared boundaries, junctions and continuous APD residuals."""
import numpy as np
import pytest
from kanapy.core.power_diagram import AnisotropicPowerDiagram


def build(centers, resolution=2):
    apd = AnisotropicPowerDiagram(centers, np.tile(np.eye(3), (len(centers), 1, 1)), [1,1,1])
    partition = apd.background_mesh(resolution).assemble()
    return apd, partition, partition.boundary_complex()


def test_single_grain_box():
    _, p, c = build([[.5,.5,.5]])
    assert len(c.patches) == 6
    assert len(c.grain_shells[1]) == 1
    assert len(c.junction_edges) == 0
    assert len(c.junction_vertices) == 0
    assert len(c.boundary_trace_edges) == 0
    assert len(c.faces) == np.count_nonzero(p.boundary_ids)


def test_planar_interface_shared_shells_and_residuals():
    apd, p, c = build([[.25,.5,.5],[.75,.5,.5]])
    internal = [patch for patch in c.patches if patch.grains[1] is not None]
    assert len(internal) == 1
    assert internal[0].grains == (1,2)
    assert len(c.junction_edges) == 0
    assert len(c.boundary_trace_edges) > 0
    for fi in internal[0].faces:
        np.testing.assert_allclose(c.points[c.faces[fi],0], .5)
        assert (fi, 1) in c.grain_shells[1][0]
        assert (fi, -1) in c.grain_shells[2][0]
    residuals = c.residuals(apd)
    assert np.nanmax(residuals['equality']) < 1e-12
    assert np.nanmax(residuals['dominance']) < 1e-12
    assert set(c.source_faces) == set(p.interface_faces).union(np.flatnonzero(p.boundary_ids))


def test_triple_line_and_four_grain_vertex():
    _, _, triple = build([[.2,.2,.5],[.8,.2,.5],[.5,.8,.5]])
    assert len(triple.junction_curves) == 1
    curve = triple.junction_curves[0]
    assert curve.grains == (1,2,3)
    xyz = triple.points[curve.vertices]
    np.testing.assert_allclose(xyz[:,0], .5)
    np.testing.assert_allclose(xyz[:,1], .425)
    assert sorted(xyz[[0,-1],2]) == pytest.approx([0,1])
    assert len(triple.boundary_junction_vertices) == 2
    apd, _, four = build([[.2,.2,.2],[.8,.2,.2],[.5,.8,.2],[.5,.5,.8]])
    high = [v for v in four.junction_vertices if len(four.vertex_grains[v]) == 4]
    assert len(high) == 1
    np.testing.assert_allclose(four.points[high[0]], [.5,.425,.3875])
    assert len(four.junction_curves) == 4
    assert np.nanmax(four.residuals(apd)['equality']) < 1e-12
    network = apd.extract_junctions(2)
    assert np.min(np.linalg.norm(network.vertices-four.points[high[0]],axis=1)) < 1e-10
    # Every junction edge occurs exactly once in the extracted curves.
    traced = [tuple(sorted((a,b))) for curve in four.junction_curves
              for a,b in zip(curve.vertices[:-1],curve.vertices[1:])]
    assert sorted(traced) == sorted(map(tuple,four.junction_edges))


def test_disconnected_patches_and_shells():
    # Grain 1 occupies x<.25 and x>.75; grain 2 is a slab.
    apd = AnisotropicPowerDiagram([[.5,.5,.5]]*2,
                                 [np.eye(3),np.diag([2.,1,1])], [1,1,1])
    apd.weights[:] = [0,.0625]
    c = apd.background_mesh((4,1,1)).assemble().boundary_complex()
    assert len([p for p in c.patches if p.grains == (1,2)]) == 2
    assert len(c.grain_shells[1]) == 2
    assert len(c.grain_shells[2]) == 1


def test_curved_residuals_and_plot():
    import matplotlib.pyplot as plt
    apd = AnisotropicPowerDiagram([[0,0,0]]*2, [np.diag([2.,1,1]),np.eye(3)], [1,1,1])
    apd.weights[:] = [.2,0]
    residuals = []
    for n in (2,4):
        c = apd.background_mesh((n,1,1)).assemble().boundary_complex()
        residuals.append(np.nanmax(c.residuals(apd)['equality']))
    assert 0 < residuals[1] < residuals[0]
    fig, axes = c.plot(apd)
    fig.canvas.draw()
    assert len(axes) == 3
    plt.close(fig)
