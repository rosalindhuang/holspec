from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from holspec.cli import app, main
from holspec.point_data import PointData, PointDataEnsemble
from holspec.utilities import save_h5


runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run holspec workflows" in result.output
    assert "run" in result.output


def test_cli_run_help():
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "CONFIG" in result.output
    assert "--project-root" in result.output


def test_cli_run_missing_config():
    result = runner.invoke(app, ["run", "missing.yml"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_python_module_entrypoint_delegates_to_cli_main():
    import holspec.__main__ as module

    assert module.main is main


def test_cli_run_tiny_pipeline(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "cli" / "triangle.h5"
    config_path = project_root / "configs" / "pipeline_cli.yml"

    _save_triangle_input(input_path)
    _write_tiny_pipeline_config(config_path)

    result = runner.invoke(
        app,
        ["run", str(config_path), "--project-root", str(project_root)],
    )

    assert result.exit_code == 0
    assert "holspec pipeline run complete." in result.output
    assert (
        "stages:       topology_simplicial, geometry_metric, "
        "hodge_laplacian, spectra"
    ) in result.output
    assert "output files: 4" in result.output

    output_root = project_root / "data" / "interim" / "cli_test"
    assert (
        output_root / "topology_simplicial" / "triangle" / "delaunay.h5"
    ).exists()
    assert (
        output_root
        / "geometry_metric"
        / "triangle"
        / "delaunay__combinatorial.h5"
    ).exists()
    assert (
        output_root
        / "hodge_laplacian"
        / "triangle"
        / "delaunay__combinatorial.h5"
    ).exists()
    assert (
        output_root / "spectra" / "triangle" / "delaunay__combinatorial.h5"
    ).exists()


def _save_triangle_input(input_path: Path) -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    ensemble = PointDataEnsemble(
        members=[PointData(positions=positions)],
        metadata={"source": "test_cli"},
    )
    ensemble.save(input_path)
    save_h5(
        input_path,
        attributes={
            "created_by": "tests/test_cli.py",
            "creation_time": "2026-05-11T00:00:00",
            "stage_name": "point_data",
            "stage_config": {"source": "inline test fixture"},
        },
        group=None,
        mode="update",
    )


def _write_tiny_pipeline_config(config_path: Path) -> None:
    config = {
        "summary": {
            "created_by": "tests/test_cli.py",
            "creation_time": "2026-05-11T00:00:00",
        },
        "inputs": {
            "data_dir": "data/raw",
            "filepaths": {
                "triangle": "data/raw/cli/triangle.h5",
            },
        },
        "stages": {
            "topology_simplicial": {
                "configs": {
                    "simplicial_constructions": {
                        "delaunay": {
                            "method": "delaunay",
                            "params": {},
                        },
                    },
                },
                "runtime": {
                    "cache_incidence": False,
                    "validate_boundary_property": False,
                    "verbose": False,
                },
            },
            "geometry_metric": {
                "configs": {
                    "metric_models": {
                        "combinatorial": {
                            "model": "combinatorial",
                            "params": {},
                        },
                    },
                },
                "runtime": {
                    "validate_metric": True,
                    "verbose": False,
                },
            },
            "hodge_laplacian": {
                "configs": {},
                "runtime": {
                    "cache_laplacians": False,
                    "validate_laplacians": False,
                    "verbose": False,
                },
            },
            "spectra": {
                "configs": {
                    "compute_spectra": True,
                    "solver": "dense",
                    "compute_eigenvectors": False,
                },
                "runtime": {
                    "verbose": False,
                },
            },
        },
        "runtime": {
            "verbose": False,
            "save_stage_configs": False,
            "error_handling": "raise",
        },
        "outputs": {
            "data_dir": "data/interim/cli_test",
            "configs_dir": "configs/pipeline_cli_stages",
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
