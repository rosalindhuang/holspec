"""
Tests for external point data array loaders.

These tests cover file-to-array behavior only. ``PointData`` owns validation
and framework object construction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holspec.point_data import (
    SUPPORTED_FILE_FORMATS,
    load_array_from_config,
    load_distances_array,
    load_positions_array,
)


POSITIONS = np.array(
    [
        [0.0, 0.0],
        [1.5, 0.0],
        [0.5, 1.25],
    ],
)

DISTANCES = np.array(
    [
        [0.0, 1.5, 1.3462912],
        [1.5, 0.0, 1.6007811],
        [1.3462912, 1.6007811, 0.0],
    ],
)


@pytest.mark.parametrize(
    ("filename", "kwargs"),
    [
        ("positions.csv", {"columns": ["x", "y"]}),
        ("positions.tsv", {"columns": ["x", "y"]}),
        ("positions.txt", {}),
        ("positions.npy", {}),
    ],
)
def test_load_positions_array_supported_formats(
    tmp_path: Path,
    filename: str,
    kwargs: dict,
):
    _write_position_files(tmp_path)

    loaded = load_positions_array(tmp_path / filename, **kwargs)

    np.testing.assert_allclose(loaded, POSITIONS)


@pytest.mark.parametrize(
    "filename",
    [
        "distances.csv",
        "distances.tsv",
        "distances.txt",
        "distances.npy",
    ],
)
def test_load_distances_array_supported_formats(
    tmp_path: Path,
    filename: str,
):
    _write_distance_files(tmp_path)

    loaded = load_distances_array(tmp_path / filename)

    np.testing.assert_allclose(loaded, DISTANCES)


def test_load_array_from_config_loads_positions_with_base_dir(tmp_path: Path):
    _write_position_files(tmp_path)
    config = {
        "data_type": "positions",
        "filepath": "positions.csv",
        "params": {
            "columns": ["x", "y"],
        },
    }

    loaded = load_array_from_config(config, base_dir=tmp_path)

    np.testing.assert_allclose(loaded, POSITIONS)


def test_load_array_from_config_loads_distances_with_explicit_format(
    tmp_path: Path,
):
    _write_distance_files(tmp_path)
    config = {
        "data_type": "distances",
        "filepath": "distances.npy",
        "file_format": "npy",
    }

    loaded = load_array_from_config(config, base_dir=tmp_path)

    np.testing.assert_allclose(loaded, DISTANCES)


def test_supported_file_formats_are_public():
    assert SUPPORTED_FILE_FORMATS == {"csv", "tsv", "txt", "npy"}


def test_load_array_from_config_rejects_missing_keys(tmp_path: Path):
    with pytest.raises(ValueError, match="data_type"):
        load_array_from_config({"filepath": "positions.csv"}, base_dir=tmp_path)

    with pytest.raises(ValueError, match="filepath"):
        load_array_from_config({"data_type": "positions"}, base_dir=tmp_path)


def test_load_array_from_config_rejects_unknown_data_type(tmp_path: Path):
    _write_position_files(tmp_path)
    config = {
        "data_type": "weights",
        "filepath": "positions.csv",
    }

    with pytest.raises(ValueError, match="positions.*distances"):
        load_array_from_config(config, base_dir=tmp_path)


def test_load_array_rejects_unsupported_file_format(tmp_path: Path):
    _write_position_files(tmp_path)

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_positions_array(tmp_path / "positions.csv", file_format="json")


def test_load_array_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_positions_array(tmp_path / "missing.csv")


def _write_position_files(path: Path) -> None:
    positions_df = pd.DataFrame(
        {
            "x": POSITIONS[:, 0],
            "y": POSITIONS[:, 1],
            "label": [10, 11, 12],
        },
    )
    positions_df.to_csv(path / "positions.csv", index=False)
    positions_df.to_csv(path / "positions.tsv", sep="\t", index=False)
    np.savetxt(path / "positions.txt", POSITIONS, fmt="%.8f")
    np.save(path / "positions.npy", POSITIONS)


def _write_distance_files(path: Path) -> None:
    distances_df = pd.DataFrame(DISTANCES)
    distances_df.to_csv(path / "distances.csv", index=False, header=False)
    distances_df.to_csv(path / "distances.tsv", sep="\t", index=False, header=False)
    np.savetxt(path / "distances.txt", DISTANCES, fmt="%.8f")
    np.save(path / "distances.npy", DISTANCES)
