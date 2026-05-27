"""
Contract tests for point-data containers.

These tests verify public constructor and access-method behavior for point-data
objects without covering generated-data workflows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holspec.point_data import PointData, PointDataEnsemble


# PointData construction and access

def test_point_data_accepts_positions_and_computes_distances():
    positions = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
    )

    point_data = PointData(positions=positions)

    assert point_data.num_points == 2
    assert point_data.dimension == 2
    assert point_data.has_positions
    assert point_data.data_type == "positions"
    np.testing.assert_allclose(point_data.get_positions(), positions)
    np.testing.assert_allclose(
        point_data.get_distances(),
        [
            [0.0, 5.0],
            [5.0, 0.0],
        ],
    )


def test_point_data_accepts_distances_without_positions():
    distances = np.array(
        [
            [0.0, 1.5],
            [1.5, 0.0],
        ],
    )

    point_data = PointData(distances=distances)

    assert point_data.num_points == 2
    assert point_data.dimension is None
    assert not point_data.has_positions
    assert point_data.data_type == "distances"
    np.testing.assert_allclose(point_data.get_distances(), distances)
    with pytest.raises(ValueError, match="Position data not available"):
        point_data.get_positions()


def test_point_data_rejects_neither_positions_nor_distances():
    with pytest.raises(ValueError, match="exactly one"):
        PointData()


def test_point_data_rejects_both_positions_and_distances():
    positions = np.zeros((2, 2))
    distances = np.zeros((2, 2))

    with pytest.raises(ValueError, match="exactly one"):
        PointData(positions=positions, distances=distances)


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (np.array([0.0, 1.0]), "2D array"),
        (np.array([[0.0, np.nan]]), "NaN or inf"),
        (np.empty((0, 2)), "Insufficient points"),
    ],
)
def test_point_data_rejects_invalid_positions(
    positions: np.ndarray,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        PointData(positions=positions)


@pytest.mark.parametrize(
    ("distances", "match"),
    [
        (np.zeros((2, 3)), "square array"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "non-negative"),
        (np.array([[1.0, 1.0], [1.0, 0.0]]), "diagonal"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (np.array([[0.0, np.inf], [np.inf, 0.0]]), "NaN or inf"),
    ],
)
def test_point_data_rejects_invalid_distances(
    distances: np.ndarray,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        PointData(distances=distances)


# PointData file construction

def test_point_data_from_file_loads_positions(tmp_path: Path):
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "label": ["a", "b", "c"],
        },
    ).to_csv(tmp_path / "positions.csv", index=False)
    config = {
        "data_type": "positions",
        "filepath": "positions.csv",
        "params": {
            "columns": ["x", "y"],
        },
    }

    point_data = PointData.from_file(config, base_dir=tmp_path)

    assert point_data.data_type == "positions"
    assert point_data.has_positions
    assert point_data.num_points == 3
    assert point_data.dimension == 2
    np.testing.assert_allclose(point_data.get_positions(), positions)
    assert point_data.metadata["config"] == config
    assert point_data.metadata["base_dir"] == str(tmp_path)


def test_point_data_from_file_loads_distances(tmp_path: Path):
    distances = np.array(
        [
            [0.0, 1.0, 1.2],
            [1.0, 0.0, 0.8],
            [1.2, 0.8, 0.0],
        ],
    )
    np.savetxt(tmp_path / "distances.csv", distances, delimiter=",")
    config = {
        "data_type": "distances",
        "filepath": "distances.csv",
    }

    point_data = PointData.from_file(config, base_dir=tmp_path)

    assert point_data.data_type == "distances"
    assert not point_data.has_positions
    assert point_data.num_points == 3
    assert point_data.dimension is None
    np.testing.assert_allclose(point_data.get_distances(), distances)
    assert point_data.metadata["config"] == config


def test_point_data_from_file_preserves_custom_metadata(tmp_path: Path):
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    np.save(tmp_path / "positions.npy", positions)
    config = {
        "data_type": "positions",
        "filepath": Path("positions.npy"),
        "file_format": "npy",
    }

    point_data = PointData.from_file(
        config,
        base_dir=tmp_path,
        metadata={"source": "unit test"},
    )

    assert point_data.metadata["source"] == "unit test"
    assert point_data.metadata["config"]["filepath"] == Path("positions.npy")


def test_point_data_from_file_uses_point_data_validation(tmp_path: Path):
    bad_positions = np.array([0.0, 1.0])
    np.save(tmp_path / "bad_positions.npy", bad_positions)
    config = {
        "data_type": "positions",
        "filepath": "bad_positions.npy",
    }

    with pytest.raises(ValueError, match="2D array"):
        PointData.from_file(config, base_dir=tmp_path)


# PointDataEnsemble construction

def test_point_data_ensemble_accepts_point_data_members():
    members = [
        PointData(positions=np.array([[0.0, 0.0], [1.0, 0.0]])),
        PointData(positions=np.array([[0.0, 1.0], [1.0, 1.0]])),
    ]

    ensemble = PointDataEnsemble(members)

    assert ensemble.size == 2
    assert ensemble.num_points == 2
    assert ensemble.dimension == 2
    assert ensemble.members == members


def test_point_data_ensemble_rejects_empty_members():
    with pytest.raises(ValueError, match="at least one member"):
        PointDataEnsemble([])


def test_point_data_ensemble_rejects_non_point_data_members():
    with pytest.raises(TypeError, match="PointData objects"):
        PointDataEnsemble([object()])


# PointDataEnsemble file construction

def test_point_data_ensemble_from_files_loads_positions(tmp_path: Path):
    positions_0 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    positions_1 = positions_0 + 0.1
    _write_positions_csv(tmp_path / "positions_0.csv", positions_0)
    _write_positions_csv(tmp_path / "positions_1.csv", positions_1)
    configs = [
        _positions_config("positions_0.csv"),
        _positions_config("positions_1.csv"),
    ]

    ensemble = PointDataEnsemble.from_files(
        configs,
        base_dir=tmp_path,
        metadata={"label": "imported positions"},
        member_metadata={"source": "unit test"},
    )

    assert ensemble.size == 2
    assert ensemble.num_points == 3
    assert ensemble.dimension == 2
    assert ensemble.metadata["label"] == "imported positions"
    assert ensemble.metadata["import_mode"] == "files"
    assert ensemble.metadata["data_type"] == "positions"
    assert ensemble.metadata["source_files"] == [
        "positions_0.csv",
        "positions_1.csv",
    ]
    assert ensemble.metadata["base_dir"] == str(tmp_path)
    assert all(member.data_type == "positions" for member in ensemble)
    np.testing.assert_allclose(ensemble[0].get_positions(), positions_0)
    np.testing.assert_allclose(ensemble[1].get_positions(), positions_1)
    assert ensemble[0].metadata["member_index"] == 0
    assert ensemble[1].metadata["member_index"] == 1
    assert ensemble[0].metadata["source"] == "unit test"
    assert ensemble[0].metadata["config"] == configs[0]


def test_point_data_ensemble_from_files_loads_distances(tmp_path: Path):
    distances_0 = np.array(
        [
            [0.0, 1.0, 1.2],
            [1.0, 0.0, 0.8],
            [1.2, 0.8, 0.0],
        ],
    )
    distances_1 = np.array(
        [
            [0.0, 1.1, 1.3],
            [1.1, 0.0, 0.9],
            [1.3, 0.9, 0.0],
        ],
    )
    np.savetxt(tmp_path / "distances_0.csv", distances_0, delimiter=",")
    np.savetxt(tmp_path / "distances_1.csv", distances_1, delimiter=",")
    configs = [
        {"data_type": "distances", "filepath": "distances_0.csv"},
        {"data_type": "distances", "filepath": "distances_1.csv"},
    ]

    ensemble = PointDataEnsemble.from_files(configs, base_dir=tmp_path)

    assert ensemble.size == 2
    assert ensemble.num_points == 3
    assert ensemble.dimension is None
    assert ensemble.metadata["import_mode"] == "files"
    assert ensemble.metadata["data_type"] == "distances"
    assert ensemble.metadata["source_files"] == [
        "distances_0.csv",
        "distances_1.csv",
    ]
    assert all(member.data_type == "distances" for member in ensemble)
    np.testing.assert_allclose(ensemble[0].get_distances(), distances_0)
    np.testing.assert_allclose(ensemble[1].get_distances(), distances_1)


def test_point_data_ensemble_from_files_rejects_empty_configs():
    with pytest.raises(ValueError, match="at least one file config"):
        PointDataEnsemble.from_files([])


def test_point_data_ensemble_from_files_rejects_mixed_data_types(tmp_path: Path):
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    _write_positions_csv(tmp_path / "positions.csv", positions)
    np.savetxt(tmp_path / "distances.csv", distances, delimiter=",")
    configs = [
        _positions_config("positions.csv"),
        {"data_type": "distances", "filepath": "distances.csv"},
    ]

    with pytest.raises(ValueError, match="same data_type"):
        PointDataEnsemble.from_files(configs, base_dir=tmp_path)


def test_point_data_ensemble_from_files_rejects_position_shape_mismatch(
    tmp_path: Path,
):
    positions_0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    positions_1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    _write_positions_csv(tmp_path / "positions_0.csv", positions_0)
    _write_positions_csv(tmp_path / "positions_1.csv", positions_1)
    configs = [
        _positions_config("positions_0.csv"),
        _positions_config("positions_1.csv"),
    ]

    with pytest.raises(ValueError, match="same array shape"):
        PointDataEnsemble.from_files(configs, base_dir=tmp_path)


def test_point_data_ensemble_from_file_with_noise_is_reproducible(
    tmp_path: Path,
):
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    _write_positions_csv(tmp_path / "positions.csv", positions)
    config = _positions_config("positions.csv")
    noise_config = {"scale": 0.05, "distribution": "normal"}

    ensemble_0 = PointDataEnsemble.from_file_with_noise(
        config,
        noise_config,
        num_realizations=3,
        base_dir=tmp_path,
        base_seed=7,
        metadata={"label": "imported noisy positions"},
        member_metadata={"source": "unit test"},
    )
    ensemble_1 = PointDataEnsemble.from_file_with_noise(
        config,
        noise_config,
        num_realizations=3,
        base_dir=tmp_path,
        base_seed=7,
        metadata={"label": "imported noisy positions"},
        member_metadata={"source": "unit test"},
    )

    assert ensemble_0.size == 3
    assert ensemble_0.num_points == 3
    assert ensemble_0.dimension == 2
    assert ensemble_0.base_config == config
    assert ensemble_0.noise_config == noise_config
    assert ensemble_0.metadata["label"] == "imported noisy positions"
    assert ensemble_0.metadata["import_mode"] == "file_with_noise"
    assert ensemble_0.metadata["num_realizations"] == 3
    assert ensemble_0.metadata["base_seed"] == 7
    assert ensemble_0.metadata["include_base"] is False
    assert ensemble_0.metadata["base_dir"] == str(tmp_path)

    for i, (member_0, member_1) in enumerate(
        zip(ensemble_0, ensemble_1, strict=True),
    ):
        np.testing.assert_allclose(
            member_0.get_positions(),
            member_1.get_positions(),
        )
        assert member_0.metadata["base_config"] == config
        assert member_0.metadata["noise_config"] == noise_config
        assert member_0.metadata["member_index"] == i
        assert member_0.metadata["noise_index"] == i
        assert member_0.metadata["seed"] == 7 + i
        assert member_0.metadata["is_base"] is False
        assert member_0.metadata["source"] == "unit test"


def test_point_data_ensemble_from_file_with_noise_can_include_base(
    tmp_path: Path,
):
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    _write_positions_csv(tmp_path / "positions.csv", positions)
    config = _positions_config("positions.csv")
    noise_config = {"scale": 0.05, "distribution": "uniform"}

    ensemble = PointDataEnsemble.from_file_with_noise(
        config,
        noise_config,
        num_realizations=2,
        base_dir=tmp_path,
        base_seed=11,
        include_base=True,
    )

    assert ensemble.size == 3
    np.testing.assert_allclose(ensemble[0].get_positions(), positions)
    assert ensemble[0].metadata["member_index"] == 0
    assert ensemble[0].metadata["noise_index"] is None
    assert "seed" not in ensemble[0].metadata
    assert ensemble[0].metadata["is_base"] is True

    assert ensemble[1].metadata["member_index"] == 1
    assert ensemble[1].metadata["noise_index"] == 0
    assert ensemble[1].metadata["seed"] == 11
    assert ensemble[1].metadata["is_base"] is False
    assert ensemble[2].metadata["member_index"] == 2
    assert ensemble[2].metadata["noise_index"] == 1
    assert ensemble[2].metadata["seed"] == 12


def test_point_data_ensemble_from_file_with_noise_rejects_distances(
    tmp_path: Path,
):
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.savetxt(tmp_path / "distances.csv", distances, delimiter=",")
    config = {
        "data_type": "distances",
        "filepath": "distances.csv",
    }

    with pytest.raises(ValueError, match="only supports position data"):
        PointDataEnsemble.from_file_with_noise(
            config,
            {"scale": 0.05},
            num_realizations=2,
            base_dir=tmp_path,
        )


def test_point_data_ensemble_from_file_with_noise_rejects_invalid_count(
    tmp_path: Path,
):
    _write_positions_csv(
        tmp_path / "positions.csv",
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="at least 1"):
        PointDataEnsemble.from_file_with_noise(
            _positions_config("positions.csv"),
            {"scale": 0.05},
            num_realizations=0,
            base_dir=tmp_path,
        )


def test_point_data_ensemble_from_file_with_noise_rejects_missing_scale(
    tmp_path: Path,
):
    _write_positions_csv(
        tmp_path / "positions.csv",
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="scale"):
        PointDataEnsemble.from_file_with_noise(
            _positions_config("positions.csv"),
            {"distribution": "uniform"},
            num_realizations=2,
            base_dir=tmp_path,
        )


def _write_positions_csv(path: Path, positions: np.ndarray) -> None:
    pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
        },
    ).to_csv(path, index=False)


def _positions_config(filepath: str) -> dict:
    return {
        "data_type": "positions",
        "filepath": filepath,
        "params": {
            "columns": ["x", "y"],
        },
    }
