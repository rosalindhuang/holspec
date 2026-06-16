"""
CLI smoke tests for the public holspec command-line interface.

These tests exercise help output, option handling, tiny data import/generation,
pipeline execution, and HDF5 inspection through Typer. They verify the public
CLI path, not deep mathematical correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from holspec.cli import app, main
from holspec.point_data import PointData, PointDataEnsemble
from holspec.utilities import save_h5


# Test client
runner = CliRunner()


# Help and option handling

def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run holspec workflows" in result.output
    assert "import-data" in result.output
    assert "generate-data" in result.output
    assert result.output.index("import-data") < result.output.index("generate-data")
    assert "run" in result.output


def test_cli_run_help():
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "CONFIG" in result.output
    assert "Run the full holspec pipeline" in result.output
    assert "Project root" in result.output
    assert "Suppress pipeline progress" in result.output


def test_cli_import_data_help():
    result = runner.invoke(app, ["import-data", "--help"])

    assert result.exit_code == 0
    assert "CONFIG" in result.output
    assert "Import external point data" in result.output
    assert "Dataset glob" in result.output
    assert "Suppress import progress" in result.output


def test_cli_generate_data_help():
    result = runner.invoke(app, ["generate-data", "--help"])

    assert result.exit_code == 0
    assert "CONFIG" in result.output
    assert "Generate synthetic point cloud data" in result.output
    assert "Dataset glob" in result.output
    assert "Suppress generation progress" in result.output


def test_cli_inspect_help():
    result = runner.invoke(app, ["inspect", "--help"])

    assert result.exit_code == 0
    assert "PATH" in result.output
    assert "Inspect the structure" in result.output
    assert "Project root" in result.output


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
    assert "config:       configs/pipeline_cli.yml" in result.output
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


def test_cli_run_quiet_suppresses_pipeline_progress(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "cli" / "triangle.h5"
    config_path = project_root / "configs" / "pipeline_cli.yml"

    _save_triangle_input(input_path)
    _write_tiny_pipeline_config(config_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(config_path),
            "--project-root",
            str(project_root),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert "holspec pipeline run complete." in result.output
    assert "Constructed" not in result.output
    assert "Computed" not in result.output


def test_cli_run_rejects_conflicting_output_options(tmp_path: Path):
    config_path = tmp_path / "configs" / "pipeline_cli.yml"
    _write_tiny_pipeline_config(config_path)

    result = runner.invoke(app, ["run", str(config_path), "--verbose", "--quiet"])

    assert result.exit_code != 0
    assert "Use either --verbose or --quiet" in result.output


# Tiny workflow smoke tests

def test_cli_import_data_with_dataset_filter(tmp_path: Path):
    project_root = tmp_path
    config_path = project_root / "configs" / "data_import.yml"

    _write_tiny_data_import_files(project_root)
    _write_tiny_data_import_config(config_path)

    result = runner.invoke(
        app,
        [
            "import-data",
            str(config_path),
            "--project-root",
            str(project_root),
            "--dataset",
            "cli",
        ],
    )

    assert result.exit_code == 0
    assert "holspec data import complete." in result.output
    assert "config:               configs/data_import.yml" in result.output
    assert "selected datasets:    cli" in result.output
    assert "imported datasets:    cli" in result.output
    assert "output files:         1" in result.output

    output_path = project_root / "data" / "imported" / "cli" / "positions_csv.h5"
    assert output_path.exists()
    assert not (project_root / "data" / "imported" / "skip").exists()

    ensemble = PointDataEnsemble.load(output_path)
    assert ensemble.size == 1
    assert ensemble.num_points == 3
    assert ensemble.dimension == 2
    np.testing.assert_allclose(
        ensemble[0].get_positions(),
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.8],
            ],
        ),
    )


def test_cli_import_data_quiet_suppresses_import_progress(tmp_path: Path):
    project_root = tmp_path
    config_path = project_root / "configs" / "data_import.yml"

    _write_tiny_data_import_files(project_root)
    _write_tiny_data_import_config(config_path)

    result = runner.invoke(
        app,
        [
            "import-data",
            str(config_path),
            "--project-root",
            str(project_root),
            "--dataset",
            "cli",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert "holspec data import complete." in result.output
    assert "Imported 1 members" not in result.output


def test_cli_generate_data_with_dataset_filter(tmp_path: Path):
    project_root = tmp_path
    config_path = project_root / "configs" / "data_generation.yml"

    _write_tiny_data_generation_config(config_path)

    result = runner.invoke(
        app,
        [
            "generate-data",
            str(config_path),
            "--project-root",
            str(project_root),
            "--dataset",
            "cli",
        ],
    )

    assert result.exit_code == 0
    assert "holspec data generation complete." in result.output
    assert "config:               configs/data_generation.yml" in result.output
    assert "selected datasets:    cli" in result.output
    assert "generated datasets:   cli" in result.output
    assert "output files:         1" in result.output
    assert (project_root / "data" / "raw" / "cli" / "regpoly_ns3__BASE.h5").exists()
    assert not (project_root / "data" / "raw" / "skip").exists()


def test_cli_generate_data_quiet_suppresses_generation_progress(tmp_path: Path):
    project_root = tmp_path
    config_path = project_root / "configs" / "data_generation.yml"

    _write_tiny_data_generation_config(config_path)

    result = runner.invoke(
        app,
        [
            "generate-data",
            str(config_path),
            "--project-root",
            str(project_root),
            "--dataset",
            "cli",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert "holspec data generation complete." in result.output
    assert "Generated 1 members" not in result.output


def test_cli_generate_data_rejects_category_option(tmp_path: Path):
    project_root = tmp_path
    config_path = project_root / "configs" / "data_generation.yml"

    _write_tiny_data_generation_config(config_path)

    result = runner.invoke(
        app,
        [
            "generate-data",
            str(config_path),
            "--project-root",
            str(project_root),
            "--category",
            "cli",
        ],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_cli_inspect_hdf5_file(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "cli" / "triangle.h5"

    _save_triangle_input(input_path)

    result = runner.invoke(
        app,
        ["inspect", "data/raw/cli/triangle.h5", "--project-root", str(project_root)],
    )

    assert result.exit_code == 0
    assert "data/raw/cli/triangle.h5/" in result.output
    assert "member_0000/" in result.output
    assert "positions (type=Dataset" in result.output


def test_cli_inspect_max_depth(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "cli" / "triangle.h5"

    _save_triangle_input(input_path)

    result = runner.invoke(
        app,
        [
            "inspect",
            "data/raw/cli/triangle.h5",
            "--project-root",
            str(project_root),
            "--max-depth",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "member_0000/" in result.output
    assert "positions (type=Dataset" not in result.output


# Local fixtures and config writers

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


def _write_tiny_data_import_files(project_root: Path) -> None:
    input_dir = project_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "positions.csv").write_text(
        "x,y\n"
        "0.0,0.0\n"
        "1.0,0.0\n"
        "0.5,0.8\n"
    )


def _write_tiny_data_import_config(config_path: Path) -> None:
    config = {
        "summary": {
            "created_by": "tests/test_cli.py",
            "creation_time": "2026-05-11T00:00:00",
        },
        "configs": {
            "cli": {
                "positions_csv": {
                    "import_mode": "files",
                    "file_configs": [
                        {
                            "data_type": "positions",
                            "filepath": "inputs/positions.csv",
                            "params": {
                                "columns": ["x", "y"],
                            },
                        },
                    ],
                },
            },
            "skip": {
                "positions_csv": {
                    "import_mode": "files",
                    "file_configs": [
                        {
                            "data_type": "positions",
                            "filepath": "inputs/positions.csv",
                            "params": {
                                "columns": ["x", "y"],
                            },
                        },
                    ],
                },
            },
        },
        "runtime": {
            "verbose": False,
        },
        "outputs": {
            "stage_name": "point_data",
            "data_dir": "data/imported",
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _write_tiny_data_generation_config(config_path: Path) -> None:
    config = {
        "summary": {
            "created_by": "tests/test_cli.py",
            "creation_time": "2026-05-11T00:00:00",
        },
        "configs": {
            "cli": {
                "regpoly_ns3__BASE": {
                    "base_config": {
                        "generator": "regpoly",
                        "params": {
                            "n_sides": 3,
                        },
                    },
                    "noise_config": {
                        "scale": 0,
                        "distribution": "normal",
                    },
                    "num_realizations": 1,
                    "base_seed": 42,
                },
            },
            "skip": {
                "tetrahedron__BASE": {
                    "base_config": {
                        "generator": "tetrahedron",
                        "params": {},
                    },
                    "noise_config": {
                        "scale": 0,
                        "distribution": "normal",
                    },
                    "num_realizations": 1,
                    "base_seed": 42,
                },
            },
        },
        "runtime": {
            "verbose": False,
        },
        "outputs": {
            "stage_name": "point_data",
            "data_dir": "data/raw",
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


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
