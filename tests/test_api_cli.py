"""Focused unit tests for Microstructure API guards and CLI commands."""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import kanapy
from kanapy.core import api, cli


def test_microstructure_requires_descriptor_or_file():
    with pytest.raises(ValueError, match="provide either"):
        kanapy.Microstructure()


def test_microstructure_supports_from_voxels_mode():
    microstructure = kanapy.Microstructure(descriptor="from_voxels")
    assert microstructure.from_voxels is True
    assert microstructure.nphases is None


def test_microstructure_rejects_missing_input_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        kanapy.Microstructure(file=tmp_path / "missing.json")


def test_voxelize_rejects_non_tuple_dimensions():
    microstructure = SimpleNamespace(
        particles=[],
        rve=SimpleNamespace(dim=(2, 2, 2)),
    )
    with pytest.raises(ValueError, match="3-tuple"):
        api.Microstructure.voxelize(microstructure, particles=[], dim=[2, 2, 2])


@pytest.mark.parametrize('feature', ['matrix', 'inclusions', 'inner', 'ordinary'])
@pytest.mark.parametrize('method', [None, 'apd', 'legacy'])
def test_voxelize_automatic_legacy_dispatch(monkeypatch, caplog, feature, method):
    from kanapy.core.entities import Simulation_Box

    particles = [SimpleNamespace(inner=None),
                 SimpleNamespace(inner=object() if feature == 'inner' else None)]
    ms = SimpleNamespace(
        particles=[], rve=SimpleNamespace(dim=(2, 2, 2), periodic=False,
                                         matrix_phase=0 if feature == 'matrix' else None),
        simbox=Simulation_Box((1, 1, 1)), nphases=1,
        precipit=0.4 if feature == 'inclusions' else None,
        nparticles=[1], geometry=None,
    )
    calls = []

    def legacy(actual_particles, mesh, nphases, prec_vf=None):
        calls.append(('legacy', actual_particles, prec_vf, {}))
        mesh.ngrains_phase = [1]
        return mesh

    def apd(actual_particles, mesh, nphases, prec_vf=None, **options):
        calls.append(('apd', actual_particles, prec_vf, options))
        mesh.ngrains_phase = [1]
        return mesh

    monkeypatch.setattr(api, 'voxelizationRoutine', apd)
    monkeypatch.setattr(api, 'voxelizationRoutine_legacy', legacy)
    options = {} if method == 'legacy' else dict(
        fit_volumes=False, fit_options={}, weights=None, chunk_size=16, periodic=True)
    if method is not None:
        options['method'] = method
    api.Microstructure.voxelize(ms, particles=particles, **options)

    expected = 'legacy' if method == 'legacy' or feature != 'ordinary' else 'apd'
    assert len(calls) == 1
    assert calls[0][:3] == (expected, particles, ms.precipit)
    if expected == 'apd':
        assert calls[0][3]['periodic'] is True
        assert calls[0][3]['fit_volumes'] is False
    assert ('switching to legacy' in caplog.text) == (
        method != 'legacy' and feature != 'ordinary')


def test_identifier_is_deterministic_and_content_sensitive():
    microstructure = kanapy.Microstructure(descriptor="from_voxels")
    snapshot = {
        "grid": {"grid_size": [1.0, 1.0, 1.0]},
        "grains": [{"grain_id": 1, "phase_id": 0, "grain_volume": 1.0}],
        "voxels": [],
    }
    identifier = microstructure.create_microstructure_identifier(snapshot)
    assert identifier == microstructure.create_microstructure_identifier(snapshot)
    changed = {**snapshot, "grains": [{**snapshot["grains"][0], "grain_volume": 2.0}]}
    assert identifier != microstructure.create_microstructure_identifier(changed)


def test_identifier_rejects_non_mapping_input():
    microstructure = kanapy.Microstructure(descriptor="from_voxels")
    with pytest.raises(TypeError, match="mapping"):
        microstructure.create_microstructure_identifier([])


def test_cli_help_and_version():
    runner = CliRunner()

    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "runTests" in help_result.output
    assert "setupMTEX" in help_result.output

    version_result = runner.invoke(cli.main, ["--version"])
    assert version_result.exit_code == 0
    assert kanapy.__version__ in version_result.output


def test_cli_run_tests_invokes_full_suite(monkeypatch, tmp_path):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cli.shutil, "rmtree", lambda path: calls.append(("rmtree", path)))

    result = CliRunner().invoke(cli.main, ["runTests"])

    assert result.exit_code == 0
    args, kwargs = calls[0]
    command = args[0]
    assert command == [
        cli.sys.executable,
        "-m",
        "pytest",
        str(tmp_path / "tests"),
        "-v",
    ]
    assert kwargs == {"check": False}
    assert calls[1] == ("rmtree", str(tmp_path / "dump_files"))


def test_cli_docs_opens_documentation(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    result = CliRunner().invoke(cli.main, ["readDocs"])
    assert result.exit_code == 0
    assert opened == ["https://icams.github.io/Kanapy/"]


def test_cli_setup_mtex_requires_optional_dependency():
    with pytest.raises(ModuleNotFoundError, match="kanapy-mtex"):
        cli.setPaths()


def test_cli_matlab_version_parser(capsys):
    assert cli.chkVersion("MATLAB Version: 9.14.0.2206163 (R2023a)") == 2023
    assert cli.chkVersion("MATLAB version unavailable") is None
    assert "Detected Matlab version R2023" in capsys.readouterr().out
