"""
Focused tests for visualization helper behavior.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from holspec.analysis import EnsembleSpectraAnalysis
from holspec.spectra import Spectrum
from holspec.visualization import (
    format_axis,
    plot_distribution_heatmap,
    plot_eigval_distribution,
    plot_observable_vs_parameter,
    plot_point_data,
)


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


def test_plot_distribution_heatmap_uses_nonuniform_parameter_edges():
    series = {
        "exp_param": "size",
        "exp_values": np.array([1.0, 2.0, 5.0]),
        "analysis_keys": [(0, "full")],
        "distribution_series": {
            (0, "full"): {
                "x": np.array([0.5, 1.5]),
                "density_stack": np.ones((3, 2)),
            },
        },
    }

    _, axes = plot_distribution_heatmap(series)

    coordinates = axes[0].collections[0].get_coordinates()
    np.testing.assert_allclose(coordinates[:, 0, 1], [0.5, 1.5, 3.5, 6.5])


def test_plot_observable_uses_sqrt_coordinate_label():
    series = {
        "exp_param": "noise",
        "exp_values": np.array([0.1, 0.2]),
        "eigenvalue_transform": "sqrt",
        "analysis_keys": [(0, "full")],
        "observable_series": {
            (0, "full"): {
                "eigval_mean_nz": {
                    "mean": np.array([1.0, 2.0]),
                    "std": np.array([0.1, 0.2]),
                },
            },
        },
        "transition_points": {},
    }

    _, axes = plot_observable_vs_parameter(series, "eigval_mean_nz")

    assert axes[0].get_ylabel() == r"$\langle\sqrt{\lambda}\rangle_\mathrm{nonzero}$"


def test_plot_eigval_distribution_annotates_sqrt_coordinate():
    analysis = EnsembleSpectraAnalysis(
        {
            (0, "full"): [
                Spectrum(np.array([0.0, 4.0, 9.0]), dimension=3),
            ],
        },
        eigenvalue_transform="sqrt",
    )
    analysis.eigenvalue_distribution(
        0,
        "full",
        method_params={"bins": 2, "range": (0.0, 3.0)},
    )

    _, axes = plot_eigval_distribution(
        analysis,
        {
            "degrees": [0],
            "components": ("full",),
            "nonzero": True,
            "distribution_params": {0: {"range": (0.0, 3.0)}},
        },
    )

    assert r"\sqrt{\lambda}_{\min}" in axes[0].texts[0].get_text()
    assert r"\sqrt{\lambda}_{\max}" in axes[0].texts[0].get_text()


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


def test_format_axis_sets_ticks_and_ticklabels():
    _, ax = plt.subplots()

    format_axis(
        ax,
        xticks=[0, 1],
        yticks=[2, 3],
        xticklabels=["zero", "one"],
        yticklabels=["two", "three"],
    )

    np.testing.assert_allclose(ax.get_xticks(), [0, 1])
    np.testing.assert_allclose(ax.get_yticks(), [2, 3])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["zero", "one"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["two", "three"]


def test_format_axis_removes_ticks_with_empty_lists():
    _, ax = plt.subplots()

    format_axis(ax, xticks=[], yticks=[])

    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0


def test_format_axis_controls_axis_visibility():
    _, ax = plt.subplots()

    format_axis(ax, axis_visible=False)

    assert not ax.axison


def test_format_axis_controls_2d_axis_lines_visibility():
    _, ax = plt.subplots()

    format_axis(ax, axis_lines_visible=False)

    assert all(not spine.get_visible() for spine in ax.spines.values())


def test_format_axis_controls_3d_ticks_and_axis_lines_visibility():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    format_axis(
        ax,
        xticks=[],
        yticks=[],
        zticks=[],
        axis_lines_visible=False,
    )

    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    assert len(ax.get_zticks()) == 0
    assert not ax.xaxis.line.get_visible()
    assert not ax.yaxis.line.get_visible()
    assert not ax.zaxis.line.get_visible()
