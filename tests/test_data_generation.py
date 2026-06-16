"""
Direct API smoke tests for synthetic data generation.

These tests exercise ``run_data_generation`` as a public Python API, separate
from YAML loading and Typer command-line behavior covered by CLI tests.
"""

from pathlib import Path

import pytest

from holspec.point_data import PointDataEnsemble, run_data_generation


def test_run_data_generation_with_dataset_filter(tmp_path: Path):
    config = _tiny_data_generation_config()

    results = run_data_generation(
        config=config,
        project_root=tmp_path,
        select_datasets=["api"],
    )

    expected_path = tmp_path / "data" / "raw" / "api" / "regpoly_ns3__BASE.h5"

    assert list(results) == ["api"]
    assert results["api"] == {"regpoly_ns3__BASE": expected_path}
    assert expected_path.exists()
    assert not (tmp_path / "data" / "raw" / "skip").exists()

    ensemble = PointDataEnsemble.load(expected_path)

    assert ensemble.size == 1
    assert ensemble.num_points == 3
    assert ensemble.dimension == 2
    assert ensemble.members[0].has_positions


def test_run_data_generation_without_dataset_subdirs(tmp_path: Path):
    config = _tiny_data_generation_config()
    config["outputs"]["dataset_subdirs"] = False

    results = run_data_generation(
        config=config,
        project_root=tmp_path,
        select_datasets=["api"],
    )

    expected_path = tmp_path / "data" / "raw" / "regpoly_ns3__BASE.h5"

    assert results["api"] == {"regpoly_ns3__BASE": expected_path}
    assert expected_path.exists()
    assert not (tmp_path / "data" / "raw" / "api").exists()


def test_run_data_generation_rejects_category_subdirs(tmp_path: Path):
    config = _tiny_data_generation_config()
    config["outputs"]["category_subdirs"] = False

    with pytest.raises(ValueError, match="outputs.dataset_subdirs"):
        run_data_generation(config=config, project_root=tmp_path)


def _tiny_data_generation_config() -> dict:
    return {
        "summary": {
            "created_by": "tests/test_data_generation.py",
        },
        "configs": {
            "api": {
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
