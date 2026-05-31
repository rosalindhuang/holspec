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


def test_run_data_import_files_with_category_filter(tmp_path: Path):
    _write_import_files(tmp_path)
    config = _tiny_data_import_config()

    results = run_data_import(
        config=config,
        project_root=tmp_path,
        select_categories=["api"],
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


def test_run_data_import_file_with_noise_without_category_subdirs(
    tmp_path: Path,
):
    _write_import_files(tmp_path)
    config = _tiny_data_import_config()
    config.pop("runtime")
    config["outputs"]["category_subdirs"] = False

    results = run_data_import(
        config=config,
        project_root=tmp_path,
        select_categories=["api_noise"],
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
            select_categories=["api"],
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
