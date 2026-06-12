"""
Focused tests for visualization helper behavior.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from holspec.visualization import plot_point_data


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_point_data_accepts_scalar_radius_2d():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )

    _, ax = plot_point_data(positions, radius=0.2)

    assert len(ax.patches) == len(positions)
    np.testing.assert_allclose(
        [patch.radius for patch in ax.patches],
        np.full(len(positions), 0.2),
    )


def test_plot_point_data_accepts_per_point_radius_2d():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    radii = np.array([0.1, 0.2, 0.3])

    _, ax = plot_point_data(positions, radius=radii)

    assert len(ax.patches) == len(positions)
    np.testing.assert_allclose(
        [patch.radius for patch in ax.patches],
        radii,
    )


def test_plot_point_data_rejects_radius_length_mismatch():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )

    with pytest.raises(ValueError, match="Length of radius array"):
        plot_point_data(positions, radius=np.array([0.1, 0.2]))


def test_plot_point_data_accepts_per_point_radius_3d():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8, 0.2],
        ],
    )
    radii = np.array([0.1, 0.2, 0.3])

    _, ax = plot_point_data(positions, radius=radii)

    assert len(ax.collections) == len(positions)
