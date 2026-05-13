"""
Contract tests for point-data containers.

These tests verify public constructor and access-method behavior for point-data
objects without covering HDF5 persistence or generated-data workflows.
"""

import numpy as np
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
