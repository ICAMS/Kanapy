"""
Generate an RVE from EBSD data using graph based grain reconstruction.

This complete workflow includes EBSD analysis and grain statistics, followed
by RVE initialization, packing, voxelization, orientation assignment, and export.

Edit the parameter block below, then run `python graph_to_rve.py` in the knpy
environment. The workflow follows convert_ang2rve.py and uses Kanapy installed
in that environment. Input and default output paths resolve from this file.

The grain graph is the first figure, both when saved and when previewed.
Figures are saved and closed automatically. Matching filenames are overwritten;
unrelated files remain in the output folder. The manifest lists this run's files.
Command line options such as --output-dir, --min-size and --show override the
corresponding defaults below. Run with --help for the available options.

The optional extra outputs include polyhedral figures and Abaqus files.
With multiple retained phases, the EBSD IPF and graph figures show them together.
Gray graph edges record contact between different phases, which cannot merge.
RVE generation continues to use the retained phase with the largest fraction.

Author: Jun Xue
ICAMS, Ruhr University Bochum, Germany

October 2026
"""

import argparse
from pathlib import Path
import random
import shutil

import kanapy as knpy
from kanapy.core.plotting import capture_figures
import matplotlib.pyplot as plt
import numpy as np
from orix.quaternion import Orientation

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]

# input and RVE settings
fname = REPO_DIR / "examples/fixtures/ebsd_316L_500x500.ang"
nvox = 30
box_length = 50  # micrometers
periodic = False
seed = 2026

# EBSD grain reconstruction
vf_min = 0.03
max_angle = 5.0  # degrees
min_size = 10.0  # pixels
connectivity = 4
boundary_artifact_size_factor = 2.0
boundary_artifact_fraction_min = 0.85
convex_hull_min_points = 3
merge_angle = None  # use max_angle for similar grain merges
hex_grid_target_spacing = None

# RVE distribution limits, following convert_ang2rve.py
deq_min, deq_max = 8.0, 19.0  # micrometers
asp_min, asp_max = 0.95, 3.5
omega_min, omega_max = 0.0, np.pi  # radians

# crystallographic orientation fitting
res_low, res_high = 7, 7
lim = 0
hw_init = np.radians(7)

# output choices
output_dir = BASE_DIR / "outputs"
show_figures = False
write_graph_pickle = True
write_graph_summary = True
extra_outputs = False

# optional Abaqus export
material_file = REPO_DIR / "examples/materials/austenite_inp_22p.inc"
ialloy = 4
abaqus_units = "mm"
applied_strain = [0.05, "*", "*", "*", "*", "*"]


def run_workflow(args):
    """
    Run graph analysis followed by the Kanapy RVE stages.

    The sequence follows convert_ang2rve.py, with graph reconstruction supplying
    the experimental statistics and the EBSD instance.

    Parameters
    ----------
    args : argparse.Namespace
        Example settings after any command line overrides.

    Returns
    -------
    pathlib.Path
        Manifest listing the files written by this run.
    """
    if not args.show:
        plt.switch_backend("Agg")
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    files = []
    config = knpy.EBSDGraphConfig(
        min_phase_pixel_fraction=args.vf_min, max_misorientation_degrees=args.max_angle,
        min_grain_size_pixels=args.min_size, connectivity=args.connectivity,
        boundary_artifact_size_factor=boundary_artifact_size_factor,
        boundary_artifact_fraction_min=boundary_artifact_fraction_min,
        convex_hull_min_points=convex_hull_min_points,
        similar_orientation_merge_degrees=(args.max_angle if merge_angle is None else merge_angle),
        hex_grid_target_spacing=hex_grid_target_spacing,
    )
    output_options = knpy.EBSDGraphOutputOptions(
        output_dir=output, clear_output_dir=False,
        write_pickle=write_graph_pickle, write_summary_json=write_graph_summary,
    )
    result = knpy.build_ebsd_graph(
        args.input, config=config, include_statistics=True,
        output_options=output_options,
    )
    files.extend(Path(path) for path in result.written_files.values())
    ebsd, phase = result.ebsd, result.phase_data
    selected_index = result.selected_phase_position
    print(f"Selected EBSD phase #{phase['index']} ({phase['name']}), {phase['ngrains']} grains")

    with capture_figures(output, "ebsd_graph_overlay", files, args.show):
        ebsd.plot_graph_overlay(iphase=None)
    with capture_figures(output, "ebsd_histograms", files, args.show):
        ebsd.plot_histograms(iphase=selected_index)
    for stage, plot in [
        ("ebsd_misorientation", ebsd.plot_mo_map),
        ("ebsd_ipf", ebsd.plot_ipf_map),
        ("ebsd_segmentation", ebsd.plot_segmentation),
    ]:
        with capture_figures(output, stage, files, args.show):
            plot()
    with capture_figures(output, "ebsd_pole_figures", files, args.show):
        ebsd.plot_pf()
    with capture_figures(output, "ebsd_ipf_projection", files, args.show):
        ebsd.plot_pf_proj()
    descriptor = knpy.set_stats(
        phase["gs_param"], phase["ar_param"], phase["om_param"],
        deq_min=deq_min, deq_max=deq_max, asp_min=asp_min, asp_max=asp_max,
        omega_min=omega_min, omega_max=omega_max, voxels=args.voxels, size=args.size,
        periodicity=args.periodic, VF=1.0, phasename=phase["name"], phasenum=0,
    )
    statistics_path = output / "statistics.json"
    write_statistics(statistics_path, result, descriptor)
    files.append(statistics_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ms = knpy.Microstructure(descriptor=descriptor, name="from_ebsd")
    ms.init_RVE()
    with capture_figures(output, "rve_initial_statistics", files, args.show):
        ms.plot_stats_init(gs_data=phase["gs_data"], ar_data=phase["ar_data"], save_files=False)
    ms.pack()
    with capture_figures(output, "rve_packed_ellipsoids", files, args.show):
        ms.plot_ellipsoids()
    ms.voxelize()
    with capture_figures(output, "rve_voxels", files, args.show):
        ms.plot_voxels(sliced=False)
    with capture_figures(output, "rve_voxel_statistics", files, args.show):
        _, comparison = ms.plot_stats_init(
            show_res=True, gs_data=phase["gs_data"], ar_data=phase["ar_data"],
            save_files=False, return_descriptors=True,
        )
    write_statistics(statistics_path, result, descriptor, comparison)

    with capture_figures(output, "rve_orientation_fit", files, args.show):
        ms.generate_orientations(
            ebsd, ebsd_phase_map={0: selected_index}, verbose=True,
            res_low=res_low, res_high=res_high, lim=lim, hw_init=hw_init,
        )
    orientations = mesh_orientations(ms.mesh, phase["cs"])
    with capture_figures(output, "rve_orientation_ipf", files, args.show):
        # match the voxel coloring used in convert_ang2rve.py
        ms.plot_voxels(ori=True)
    crystal_phase = ebsd.emap.phases[phase["index"]]
    with capture_figures(output, "rve_pole_figures", files, args.show):
        knpy.plot_pole_figure(orientations, crystal_phase)
    with capture_figures(output, "rve_ipf_projection", files, args.show):
        knpy.plot_pole_figure_proj(orientations, crystal_phase)

    if args.extra_outputs:
        ms.generate_grains()
        with capture_figures(output, "rve_polyhedral_grains", files, args.show):
            ms.plot_grains()
        with capture_figures(output, "rve_statistics_comparison", files, args.show):
            ms.plot_stats(show_all=True, gs_data=phase["gs_data"], ar_data=phase["ar_data"], save_files=False)
        material = output / material_file.name
        shutil.copyfile(material_file, material)
        files.append(material)
        geometry = Path(ms.write_abq(
            nodes="v", file="from_ebsd_geom.inp", path=output, units=abaqus_units,
            ialloy=ialloy, props_file=material.name, crystal_plasticity=True,
            boundary_conditions={"apply_bc": True, "periodic_bc": args.periodic,
                                 "type_bc": "strain", "components_bc": applied_strain},
        ))
        files.extend([geometry, output / "from_ebsd_mat.inp"])
    voxel_path = output / "from_ebsd_voxels.json"
    ms.write_voxels(file=voxel_path.name, path=output, script_name=__file__,
                    source=str(args.input), mesh=False, system=False)
    files.append(voxel_path)
    restored = knpy.import_voxels(file=voxel_path.name, path=output)
    np.testing.assert_array_equal(restored.mesh.grains, ms.mesh.grains)
    np.testing.assert_allclose(restored.rve.size, ms.rve.size)
    if restored.mesh.grain_phase_dict != ms.mesh.grain_phase_dict:
        raise RuntimeError("Voxel import changed grain phases.")
    if restored.mesh.grain_ori_dict.keys() != ms.mesh.grain_ori_dict.keys():
        raise RuntimeError("Voxel import changed oriented grain identifiers.")
    for grain, angles in ms.mesh.grain_ori_dict.items():
        np.testing.assert_allclose(restored.mesh.grain_ori_dict[grain], angles)
    with capture_figures(output, "rve_imported_orientation_ipf", files, args.show):
        restored.plot_voxels(ori=True)

    manifest = output / "manifest.json"
    write_manifest(manifest, {
        "input_file": args.input, "kanapy_version": knpy.__version__,
        "graph_config": config.to_dict(), "selected_phase_id": phase["index"],
        "selected_phase_name": phase["name"], "selected_phase_position": selected_index,
        "rve_phase_id": 0, "voxels_per_side": args.voxels, "side_length": args.size,
        "periodic": args.periodic, "seed": args.seed,
        "grain_method": "graph",
        "geometry_convention": result.graph.graph.get("geometry_convention"),
        "extra_outputs": args.extra_outputs,
        "units": {"rve_length": "um", "abaqus_length": abaqus_units, "angles": "rad"},
        "voxel_roundtrip_verified": True, "rve_grain_count": ms.Ngr,
    }, files)
    print(f"Completed EBSD to RVE workflow. Manifest: {manifest}")
    return manifest


def mesh_orientations(mesh, crystal_symmetry):
    """
    Convert stored grain Euler angles while preserving their crystal symmetry.

    The resulting object supplies the RVE pole figures.

    Parameters
    ----------
    mesh : object
        Generated or imported mesh with a grain orientation dictionary.
    crystal_symmetry : orix.quaternion.Symmetry
        Symmetry of the experimental phase used for orientation generation.

    Returns
    -------
    orix.quaternion.Orientation
        Grain orientations in their stored dictionary order.
    """
    return Orientation.from_euler(
        np.array(list(mesh.grain_ori_dict.values())), symmetry=crystal_symmetry,
    )


def write_statistics(path, result, descriptor, rve_statistics=None):
    """
    Save experimental statistics and the descriptor used to generate the RVE.

    Arrays and fitted parameters remain available for subsequent quantitative
    analysis, with lengths in micrometers and tilt angles in radians.

    Parameters
    ----------
    path : pathlib.Path
        Destination JSON file.
    result : kanapy.EBSDGraphResult
        Graph analysis result containing the experimental statistics.
    descriptor : dict
        Statistical input passed to Microstructure.
    rve_statistics : list of dict, optional
        Comparison statistics from the generated microstructure.
    """
    path = Path(path)
    payload = result.to_statistics()
    payload["rve_descriptor"] = descriptor
    payload["rve_statistics"] = rve_statistics
    knpy.write_stats(payload, path.name, path=path.parent, indent=2)


def write_manifest(path, metadata, files):
    """
    Write provenance and only the artifacts produced during this run.

    Existing output folders are preserved. Explicit file tracking prevents
    figures left by earlier runs from being claimed in the current manifest.

    Parameters
    ----------
    path : pathlib.Path
        Destination manifest, beside the tracked artifacts.
    metadata : dict
        Input path, configuration, phase identity and verification information.
    files : list of pathlib.Path
        Artifacts written during the current run.
    """
    path = Path(path)
    payload = dict(metadata)
    payload["files"] = sorted({str(Path(file).relative_to(path.parent)) for file in files})
    knpy.write_stats(payload, path.name, path=path.parent, indent=2)


def parse_args(argv=None):
    """
    Apply optional command line overrides to the settings above.

    Editing the example variables changes the defaults for a normal run.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments, or None to read them from the process.

    Returns
    -------
    argparse.Namespace
        Effective example settings.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=fname)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--voxels", type=int, default=nvox, help="Voxels per side")
    parser.add_argument("--size", type=int, default=box_length, help="Side length in um")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--min-size", type=float, default=min_size, help="Minimum grain size in pixels")
    parser.add_argument("--max-angle", type=float, default=max_angle, help="Grain misorientation threshold in degrees")
    parser.add_argument("--vf-min", type=float, default=vf_min, help="Minimum retained phase fraction")
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=connectivity,
                        help="Pixel connectivity")
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=periodic,
                        help="Use a periodic RVE")
    parser.add_argument("--extra-outputs", action=argparse.BooleanOptionalAction,
                        default=extra_outputs,
                        help="Also generate polyhedral grains and Abaqus files")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction,
                        default=show_figures,
                        help="Preview each saved figure interactively")
    args = parser.parse_args(argv)
    if args.voxels < 2 or args.size <= 0:
        parser.error("--voxels must be at least 2 and --size must be positive")
    if args.min_size <= 0 or args.max_angle <= 0 or not 0 <= args.vf_min < 1:
        parser.error("Grain size and angle must be positive; phase fraction must be in [0, 1)")
    args.input = args.input.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


if __name__ == "__main__":
    run_workflow(parse_args())
