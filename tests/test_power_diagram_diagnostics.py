"""Analytical contacts exercise the direct APD checker independently of meshing."""

import json

import numpy as np
import pytest

from kanapy.core.power_diagram import AnisotropicPowerDiagram


def spheres(centers, periodic=False, scale=1.):
    return AnisotropicPowerDiagram(np.array(centers) * scale,
                                    np.tile(np.eye(3), (len(centers), 1, 1)),
                                    np.ones(3) * scale, periodic=periodic)


def check(apd, **kwargs):
    return apd.check_topology(resolution=4, sphere_level=1, max_candidates=60, **kwargs)


def test_regular_quadruple_vertex_and_immutable_inputs():
    apd = spheres([[.2, .2, .2], [.8, .2, .2], [.5, .8, .2], [.5, .5, .8]])
    before = [a.copy() for a in (apd.centers, apd.matrices, apd.box_size, apd.weights)]
    report = check(apd)
    assert not report.certified
    assert len(report.contacts) == 1
    contact = report.contacts[0]
    assert contact.kind == 'ordinary_vertex'
    np.testing.assert_allclose(contact.point, [.5, .425, .3875], atol=1e-8)
    assert contact.rank == 3
    assert contact.grain_ids == (1, 2, 3, 4)
    assert contact.residual < 1e-8
    assert contact.probe['status'] == 'locally_regular_at_tested_scales'
    for old, current in zip(before, (apd.centers, apd.matrices, apd.box_size, apd.weights)):
        np.testing.assert_array_equal(current, old)
    assert json.loads(json.dumps(report.to_dict()))['certified'] is False


@pytest.mark.parametrize('n', [4, 5])
def test_higher_order_lines_are_traced_and_not_assumed_nonmanifold(n):
    angles = np.arange(n) * (2 * np.pi / n)
    centers = np.column_stack([.5 + .25 * np.cos(angles), .5 + .25 * np.sin(angles), np.full(n, .5)])
    apd = spheres(centers)
    report = check(apd, candidate_points=[[.5, .5, .5]])
    lines = [c for c in report.contacts if c.kind == 'higher_order_line']
    assert lines
    contact = min(lines, key=lambda c: np.linalg.norm(c.point - .5))
    assert len(contact.grain_ids) == n
    assert contact.rank == 2
    assert len(contact.trace) >= 5
    np.testing.assert_allclose(contact.trace[:, :2], .5, atol=1e-7)
    values = apd.costs(contact.trace)
    assert np.max(np.ptp(values, axis=1)) < 1e-8
    assert np.ptp(contact.trace[:, 2]) > .1
    assert contact.probe['status'] == 'locally_regular_at_tested_scales'
    assert not contact.probe['suspect_grain_ids']


def test_five_grain_vertex_is_not_a_quadruple_line():
    centers = .5 + .25 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                   [0, 0, 1], [0, -.6, -.8]])
    report = check(spheres(centers))
    assert len(report.contacts) == 1
    contact = report.contacts[0]
    assert contact.kind == 'higher_order_vertex'
    assert contact.rank == 3
    np.testing.assert_allclose(contact.point, .5, atol=1e-8)
    assert len(contact.trace) == 0


def test_pinched_two_grain_boundary_is_found_without_junction_network():
    # q1-q0 = (x-.5)^2 + (y-.5)^2 - .5*(z-.5)^2: double-cone pinch.
    apd = AnisotropicPowerDiagram([[.5, .5, .5]] * 2,
                                  [np.eye(3), np.diag([2., 2., .5])], [1, 1, 1])
    assert len(apd.extract_junctions(4).vertices) == 0
    report = check(apd)
    pinches = [c for c in report.contacts if c.probe['status'] == 'suspected_non_manifold']
    assert len(pinches) == 1
    contact = pinches[0]
    np.testing.assert_allclose(contact.point, .5, atol=1e-10)
    assert contact.kind == 'singular_contact'
    assert contact.rank == 0
    assert contact.probe['suspect_grain_ids'] == [1, 2]
    for observation in contact.probe['observations']:
        assert all(g['loops'] == 2 for g in observation['grains'].values())


def test_isolated_rank_loss_is_not_reported_as_a_line():
    # q1-q0=x-.5, q2-q0=y-.5, q3-q0=(z-.5)^2.
    apd = AnisotropicPowerDiagram([[.5, .5, .5], [.25, .5, .5], [.5, .25, .5], [.5, .5, .5]],
                                  [2 * np.eye(3)] * 3 + [np.diag([2., 2., 3.])], [1, 1, 1])
    apd.weights[:] = [0, .125, .125, 0]
    report = check(apd, candidate_points=[[.5, .5, .5]])
    contact = report.contacts[0]
    assert contact.kind == 'singular_contact'
    assert contact.rank == 2
    assert len(contact.trace) < 5
    assert not any(c.kind == 'higher_order_line' for c in report.contacts)
    assert contact.probe['status'].startswith('unresolved')


def test_nonminimal_equalities_are_not_contacts():
    centers = [[.2, .2, .2], [.8, .2, .2], [.5, .8, .2], [.5, .5, .8], [.5, .5, .5]]
    apd = spheres(centers)
    apd.weights[-1] = 100.
    report = check(apd, candidate_points=[[.5, .425, .3875]])
    assert not report.contacts
    assert not report.certified  # Missing contacts cannot certify the entire diagram.


def test_periodic_branches_do_not_inflate_distinct_grain_count():
    apd = spheres([[.25, .5, .5], [.75, .5, .5]], periodic=True)
    report = check(apd, candidate_points=[[.5, 0., .5], [1.5, -1., .5]])
    contacts = [c for c in report.contacts if np.linalg.norm(c.point - [.5, 0, .5]) < 1e-7]
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.grain_ids == (1, 2)
    assert len(contact.branches) == 4
    assert contact.kind == 'periodic_branch_contact'
    assert contact.probe['status'] == 'locally_regular_at_tested_scales'


def test_periodic_image_contacts_are_discovered_without_supplied_seeds():
    apd = spheres([[.25, .5, .5], [.75, .5, .5]], periodic=True)
    report = apd.check_topology(resolution=4, sphere_level=1, max_candidates=200)
    assert report.discovery['periodic_branch_candidates'] > 0
    assert any(c.kind == 'periodic_branch_contact' for c in report.contacts)
    assert all(len(c.grain_ids) == 2 for c in report.contacts)


def test_regular_two_grain_interface_is_not_singular():
    report = check(spheres([[.25, .5, .5], [.75, .5, .5]]))
    assert report.contacts
    assert all(c.kind == 'ordinary_interface' for c in report.contacts)
    assert report.summary()['suspected_non_manifold'] == 0


def test_tolerances_and_locations_scale_with_physical_units():
    centers = [[.2, .2, .2], [.8, .2, .2], [.5, .8, .2], [.5, .5, .8]]
    small = check(spheres(centers))
    big = check(spheres(centers, scale=1000.))
    assert [c.kind for c in small.contacts] == [c.kind for c in big.contacts]
    np.testing.assert_allclose(big.contacts[0].point / 1000, small.contacts[0].point, atol=1e-8)


def test_budget_boundary_and_plot_reporting(tmp_path):
    apd = spheres([[.2, .2, .5], [.8, .2, .5], [.2, .8, .5], [.8, .8, .5]])
    report = apd.check_topology(resolution=4, candidate_points=[[.5, .5, 0]],
                                 sphere_level=1, max_candidates=1)
    assert report.discovery['budget_exhausted']
    assert report.discovery['omitted_by_budget'] > 0
    assert report.contacts[0].probe['status'] == 'unresolved_box_boundary'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original_box = apd.box_size.copy()
    ax = report.plot(apd)
    ax.figure.savefig(tmp_path / 'diagnostics.png')
    plt.close(ax.figure)
    np.testing.assert_array_equal(apd.box_size, original_box)


def test_validation_and_single_grain():
    apd = spheres([[.5, .5, .5]])
    assert check(apd).summary()['contact_observations'] == 0
    for kwargs in [dict(resolution=1), dict(cost_tolerance=0), dict(sphere_level=4),
                   dict(max_candidates=0), dict(candidate_points=[[2., 0., 0.]]),
                   dict(candidate_points=[[.5, .5]]), dict(probe_radius=np.nan),
                   dict(trace_step=0), dict(trace_steps=1)]:
        with pytest.raises(ValueError):
            apd.check_topology(**kwargs)
