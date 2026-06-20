"""
Direct API smoke tests for external point data import.

These tests exercise ``run_data_import`` as a public Python API, separate from
lower-level array loaders and object constructors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holspec.point_data import PointDataEnsemble, run_data_import
from holspec.utilities import read_h5


POSITIONS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
    ],
)
VARIABLE_POSITIONS_SHORT = POSITIONS[:2]
VARIABLE_POSITIONS_LONG = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.2, 0.4],
    ],
)


def test_run_data_import_files_with_dataset_filter(tmp_path: Path):
    _write_import_files(tmp_path)
    config = _tiny_data_import_config()

    results = run_data_import(
        config=config,
        project_root=tmp_path,
        select_datasets=["api"],
    )

    expected_path = tmp_path / "data" / "imported" / "api" / "positions_csv.h5"

    assert list(results) == ["api"]
    assert results["api"] == {"positions_csv": expected_path}
    assert expected_path.exists()
    assert not (tmp_path / "data" / "imported" / "skip").exists()

    ensemble = PointDataEnsemble.load(expected_path)
    assert ensemble.size == 1
    assert ensemble.num_points == 3
    assert ensemble.dimension == 2
    assert ensemble.metadata["import_mode"] == "files"
    assert ensemble.metadata["data_type"] == "positions"
    np.testing.assert_allclose(ensemble[0].get_positions(), POSITIONS)

    _, attributes = read_h5(expected_path)
    assert attributes["created_by"] == "tests/test_data_import.py"
    assert attributes["stage_name"] == "point_data"
    assert attributes["stage_config"] == config["configs"]["api"]["positions_csv"]
    assert "creation_time" in attributes


def test_run_data_import_files_accepts_variable_size_members(tmp_path: Path):
    _write_import_files(tmp_path)
    config = _tiny_data_import_config()
    config["configs"]["variable"] = {
        "positions_csv": {
            "import_mode": "files",
            "file_configs": [
                {
                    "data_type": "positions",
                    "filepath": "inputs/positions_short.csv",
                    "params": {
                        "columns": ["x", "y"],
                    },
                },
                {
                    "data_type": "positions",
                    "filepath": "inputs/positions_long.csv",
                    "params": {
                        "columns": ["x", "y"],
                    },
                },
            ],
        },
    }

    results = run_data_import(
        config=config,
        project_root=tmp_path,
        select_datasets=["variable"],
    )

    expected_path = (
        tmp_path / "data" / "imported" / "variable" / "positions_csv.h5"
    )

    assert results["variable"] == {"positions_csv": expected_path}
    assert expected_path.exists()

    ensemble = PointDataEnsemble.load(expected_path)
    assert ensemble.size == 2
    assert ensemble.num_points_per_member == (2, 4)
    assert not ensemble.has_uniform_num_points
    assert ensemble.dimension == 2
    np.testing.assert_allclose(
        ensemble[0].get_positions(),
        VARIABLE_POSITIONS_SHORT,
    )
    np.testing.assert_allclose(
        ensemble[1].get_positions(),
        VARIABLE_POSITIONS_LONG,
    )
    with pytest.raises(ValueError, match="num_points_per_member"):
        ensemble.num_points


def test_run_data_import_file_with_noise_without_dataset_subdirs(
    tmp_path: Path,
):
    _write_import_files(tmp_path)
    config = _tiny_data_import_config()
    config.pop("runtime")
    config["outputs"]["dataset_subdirs"] = False

    results = run_data_import(
        config=config,
        project_root=tmp_path,
        select_datasets=["api_noise"],
    )

    expected_path = tmp_path / "data" / "imported" / "positions_noise.h5"

    assert results["api_noise"] == {"positions_noise": expected_path}
    assert expected_path.exists()
    assert not (tmp_path / "data" / "imported" / "api_noise").exists()

    ensemble = PointDataEnsemble.load(expected_path)
    assert ensemble.size == 3
    assert ensemble.metadata["import_mode"] == "file_with_noise"
    assert ensemble.metadata["num_realizations"] == 2
    assert ensemble.metadata["include_base"] is True
    np.testing.assert_allclose(ensemble[0].get_positions(), POSITIONS)
    assert ensemble[1].metadata["seed"] == 11
    assert ensemble[2].metadata["seed"] == 12


def test_run_data_import_rejects_unknown_import_mode(tmp_path: Path):
    config = _tiny_data_import_config()
    config["configs"]["api"]["positions_csv"]["import_mode"] = "mystery"

    with pytest.raises(ValueError, match="Unsupported import_mode.*files"):
        run_data_import(
            config=config,
            project_root=tmp_path,
            select_datasets=["api"],
        )


def _write_import_files(path: Path) -> None:
    data_dir = path / "inputs"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "x": POSITIONS[:, 0],
            "y": POSITIONS[:, 1],
        },
    ).to_csv(data_dir / "positions.csv", index=False)
    pd.DataFrame(
        {
            "x": VARIABLE_POSITIONS_SHORT[:, 0],
            "y": VARIABLE_POSITIONS_SHORT[:, 1],
        },
    ).to_csv(data_dir / "positions_short.csv", index=False)
    pd.DataFrame(
        {
            "x": VARIABLE_POSITIONS_LONG[:, 0],
            "y": VARIABLE_POSITIONS_LONG[:, 1],
        },
    ).to_csv(data_dir / "positions_long.csv", index=False)
    np.save(data_dir / "positions.npy", POSITIONS)


def _tiny_data_import_config() -> dict:
    return {
        "summary": {
            "created_by": "tests/test_data_import.py",
        },
        "configs": {
            "api": {
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
            "api_noise": {
                "positions_noise": {
                    "import_mode": "file_with_noise",
                    "base_config": {
                        "data_type": "positions",
                        "filepath": "inputs/positions.csv",
                        "params": {
                            "columns": ["x", "y"],
                        },
                    },
                    "noise_config": {
                        "scale": 0.05,
                        "distribution": "normal",
                    },
                    "num_realizations": 2,
                    "base_seed": 11,
                    "include_base": True,
                },
            },
            "skip": {
                "positions_npy": {
                    "import_mode": "files",
                    "file_configs": [
                        {
                            "data_type": "positions",
                            "filepath": "inputs/positions.npy",
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
