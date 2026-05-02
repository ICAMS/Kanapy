import json
from copy import deepcopy

import numpy as np
import pytest

import kanapy as kanapy


def equiaxed_descriptor(side_length, nvox, diameter_scale, cutoff_min, cutoff_max):
    return {
        "Grain type": "Equiaxed",
        "Equivalent diameter": {
            "sig": 0.15,
            "scale": diameter_scale,
            "loc": 0.0,
            "cutoff_min": cutoff_min,
            "cutoff_max": cutoff_max,
        },
        "RVE": {
            "sideX": side_length,
            "sideY": side_length,
            "sideZ": side_length,
            "Nx": nvox,
            "Ny": nvox,
            "Nz": nvox,
        },
        "Simulation": {
            "periodicity": False,
            "output_units": "um",
        },
    }


def elongated_descriptor():
    descriptor = equiaxed_descriptor(
        side_length=8,
        nvox=8,
        diameter_scale=4.0,
        cutoff_min=3.0,
        cutoff_max=5.0,
    )
    descriptor["Grain type"] = "Elongated"
    descriptor["Aspect ratio"] = {
        "sig": 0.2,
        "scale": 1.5,
        "loc": 0.0,
        "cutoff_min": 1.0,
        "cutoff_max": 2.0,
    }
    descriptor["Tilt angle"] = {
        "kappa": 1.0,
        "loc": np.pi / 2,
        "cutoff_min": 0.0,
        "cutoff_max": np.pi,
    }
    return descriptor


@pytest.mark.parametrize(
    "case_name, side_length, nvox, diameter_scale, cutoff_min, cutoff_max",
    [
        # Smallest practical descriptor: checks the workflow on a tiny RVE.
        ("tiny", 6, 6, 2.8, 2.2, 3.4),
        # Standard descriptor: checks the regular RVE generation path.
        ("normal", 8, 8, 4.0, 3.0, 5.0),
        # Larger descriptor: still fast enough for pytest/CI, but less trivial.
        ("larger_but_ci_safe", 10, 10, 4.5, 3.4, 5.6),
    ],
)
def test_create_rve_workflow_for_representative_sizes(
    tmp_path,
    case_name,
    side_length,
    nvox,
    diameter_scale,
    cutoff_min,
    cutoff_max,
):
    np.random.seed(0)
    descriptor = equiaxed_descriptor(
        side_length=side_length,
        nvox=nvox,
        diameter_scale=diameter_scale,
        cutoff_min=cutoff_min,
        cutoff_max=cutoff_max,
    )
    microstructure = kanapy.Microstructure(
        descriptor=descriptor,
        name=f"integration_{case_name}",
    )

    microstructure.init_RVE(nsteps=2)
    assert microstructure.rve.dim == (nvox, nvox, nvox)
    assert microstructure.nparticles[0] > 0

    microstructure.pack(save_files=False, verbose=False)
    assert len(microstructure.particles) == microstructure.nparticles[0]

    microstructure.voxelize()
    assert microstructure.mesh.grains.shape == (nvox, nvox, nvox)
    assert microstructure.Ngr > 0
    assert len(microstructure.mesh.grain_dict) == microstructure.Ngr

    microstructure.generate_grains()
    assert microstructure.geometry is not None
    assert len(microstructure.geometry["Grains"]) > 0

    microstructure.generate_orientations("random", Nbase=50)
    assert len(microstructure.mesh.grain_ori_dict) == microstructure.Ngr

    output_file = tmp_path / f"{case_name}_voxels.json"
    microstructure.write_voxels(
        file=output_file.name,
        path=tmp_path,
        mesh=False,
        system=False,
    )

    assert output_file.is_file()
    with output_file.open() as json_file:
        voxel_data = json.load(json_file)

    assert voxel_data["Model"]["Material"] == f"integration_{case_name}"
    assert voxel_data["Data"]["Shape"] == [nvox, nvox, nvox]
    assert len(voxel_data["Data"]["Values"]) == nvox**3
    assert len(voxel_data["Grains"]) - 3 == microstructure.Ngr




def test_create_rve_workflow_rejects_too_coarse_voxel_grid():
    # Edge case descriptor: voxel size is larger than the smallest grain.
    descriptor = equiaxed_descriptor(
        side_length=8,
        nvox=2,
        diameter_scale=1.5,
        cutoff_min=1.0,
        cutoff_max=2.0,
    )
    microstructure = kanapy.Microstructure(
        descriptor=descriptor,
        name="integration_invalid_voxel_size",
    )

    with pytest.raises(ValueError, match="Voxel size larger than minimum grain size"):
        microstructure.init_RVE(nsteps=2)


@pytest.mark.parametrize(
    "case_name, descriptor, error_message",
    [
        (
            "missing_rve_settings",
            # Edge case descriptor: required RVE block is absent.
            lambda: {
                key: value
                for key, value in equiaxed_descriptor(8, 8, 4.0, 3.0, 5.0).items()
                if key != "RVE"
            },
            "RVE properties must be specified",
        ),
        (
            "missing_simulation_settings",
            # Edge case descriptor: required Simulation block is absent.
            lambda: {
                key: value
                for key, value in equiaxed_descriptor(8, 8, 4.0, 3.0, 5.0).items()
                if key != "Simulation"
            },
            "Simulation attributes must be specified",
        ),
        (
            "unsupported_output_units",
            # Edge case descriptor: output units must be "mm" or "um".
            lambda: _with_nested_value(
                equiaxed_descriptor(8, 8, 4.0, 3.0, 5.0),
                ["Simulation", "output_units"],
                "cm",
            ),
            'Output units can only be "mm" or "um"',
        ),
        (
            "non_cubic_voxels",
            # Edge case descriptor: voxel spacings differ between directions.
            lambda: _with_nested_value(
                equiaxed_descriptor(8, 4, 4.0, 3.0, 5.0),
                ["RVE", "Ny"],
                8,
            ),
            "Voxels are not cubic",
        ),
        (
            "unsupported_grain_type",
            # Edge case descriptor: grain type is not supported by Kanapy.
            lambda: _with_nested_value(
                equiaxed_descriptor(8, 8, 4.0, 3.0, 5.0),
                ["Grain type"],
                "Needle",
            ),
            'must be either "Equiaxed" or "Elongated"',
        ),
        (
            "diameter_cutoff_range_too_narrow",
            # Edge case descriptor: equivalent diameter cutoff min/max are too close.
            lambda: equiaxed_descriptor(8, 8, 4.0, 3.9, 4.0),
            "cutoffs of equiavalent diameter are too close",
        ),
        (
            "phase_volume_fraction_above_one",
            # Edge case descriptor: phase volume fractions cannot exceed 1.
            lambda: _with_nested_value(
                equiaxed_descriptor(8, 8, 4.0, 3.0, 5.0),
                ["Phase"],
                {"Name": "TooMuchPhase", "Number": 0, "Volume fraction": 1.2},
            ),
            "Sum of all phase fractions exceeds 1",
        ),
    ],
)
def test_create_rve_workflow_rejects_invalid_descriptor_inputs(
    case_name,
    descriptor,
    error_message,
):
    microstructure = kanapy.Microstructure(
        descriptor=descriptor(),
        name=f"integration_invalid_{case_name}",
    )

    with pytest.raises(ValueError, match=error_message):
        microstructure.init_RVE(nsteps=2)


@pytest.mark.parametrize(
    "case_name, mutation_path, mutation_value, error_message",
    [
        (
            "aspect_ratio_cutoff_range_too_narrow",
            ["Aspect ratio", "cutoff_min"],
            1.9,
            "cutoffs of aspect ratio are too close",
        ),
        (
            "tilt_angle_cutoff_range_too_narrow",
            ["Tilt angle", "cutoff_min"],
            3.0,
            "cutoffs of orientation of tilt axis are too close",
        ),
    ],
)
def test_create_rve_workflow_rejects_invalid_elongated_descriptor_inputs(
    case_name,
    mutation_path,
    mutation_value,
    error_message,
):
    # Edge case descriptor: elongated grains add aspect ratio and tilt angle constraints.
    descriptor = _with_nested_value(
        elongated_descriptor(),
        mutation_path,
        mutation_value,
    )
    microstructure = kanapy.Microstructure(
        descriptor=descriptor,
        name=f"integration_invalid_{case_name}",
    )

    with pytest.raises(ValueError, match=error_message):
        microstructure.init_RVE(nsteps=2)


def test_create_rve_workflow_rejects_unimodal_orientation_without_parameters():
    np.random.seed(0)
    # Valid descriptor up to voxelization; the edge case is the orientation call.
    descriptor = equiaxed_descriptor(
        side_length=6,
        nvox=6,
        diameter_scale=2.8,
        cutoff_min=2.2,
        cutoff_max=3.4,
    )
    microstructure = kanapy.Microstructure(
        descriptor=descriptor,
        name="integration_invalid_unimodal_orientation",
    )
    microstructure.init_RVE(nsteps=2)
    microstructure.pack(save_files=False, verbose=False)
    microstructure.voxelize()

    with pytest.raises(ValueError, match="angle .* and kernel"):
        microstructure.generate_orientations("unimodal")


def _with_nested_value(descriptor, path, value):
    updated = deepcopy(descriptor)
    target = updated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return updated
