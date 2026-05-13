"""
Core tests for dense Hodge Laplacian spectra.

These tests verify dense full-spectrum computation on tiny combinatorial
examples with known eigenvalues and kernel dimensions, plus basic Spectrum
observable methods on a hand-written spectrum.
"""

import numpy as np
import pytest

from holspec.spectra import HodgeLaplacianSpectra, Spectrum

from tests.helpers import assert_eigenvalues_allclose


# Dense spectra contract

def test_dense_full_spectra_have_expected_contract(
    filled_triangle_spectra: HodgeLaplacianSpectra,
):
    for k in filled_triangle_spectra.degrees:
        spectrum = filled_triangle_spectra.spectrum(k, "full")
        eigenvalues = spectrum.eigenvalues

        assert spectrum.dimension == filled_triangle_spectra.dimensions[k]
        assert spectrum.num_eigenvalues == spectrum.dimension
        assert spectrum.is_complete
        assert spectrum.eigenvectors is None
        assert np.all(np.diff(eigenvalues) >= 0)
        assert np.all(eigenvalues >= 0)


def test_sparse_spectrum_computes_partial_eigenvalues(
    filled_triangle_hodge_laplacian,
):
    spectra = HodgeLaplacianSpectra(
        filled_triangle_hodge_laplacian,
        solver="sparse",
        solver_params={"num_eigenvalues": 1, "which": "SM"},
    )

    spectrum = spectra.spectrum(0, "full")

    assert spectra.solver_params == {
        "num_eigenvalues": 1,
        "which": "SM",
        "sigma": None,
    }
    assert spectrum.dimension == 3
    assert spectrum.num_eigenvalues == 1
    assert not spectrum.is_complete
    assert spectrum.eigenvectors is None
    assert np.all(np.diff(spectrum.eigenvalues) >= 0)
    assert np.all(spectrum.eigenvalues >= 0)
    assert spectrum.eigenvalues[0] < 1e-8


# Known tiny-example spectra

def test_edge_full_spectrum_eigenvalues(edge_spectra: HodgeLaplacianSpectra):
    # Expected spectra are for full combinatorial Hodge Laplacians on the
    # hand-built examples from conftest.py.
    expected = {
        0: [0.0, 2.0],
        1: [2.0],
    }

    for k, expected_eigenvalues in expected.items():
        assert_eigenvalues_allclose(
            edge_spectra.eigenvalues(k, "full"),
            expected_eigenvalues,
        )


def test_triangle_boundary_full_spectrum_eigenvalues(
    triangle_boundary_spectra: HodgeLaplacianSpectra,
):
    expected = {
        0: [0.0, 3.0, 3.0],
        1: [0.0, 3.0, 3.0],
    }

    for k, expected_eigenvalues in expected.items():
        assert_eigenvalues_allclose(
            triangle_boundary_spectra.eigenvalues(k, "full"),
            expected_eigenvalues,
        )


def test_filled_triangle_full_spectrum_eigenvalues(
    filled_triangle_spectra: HodgeLaplacianSpectra,
):
    expected = {
        0: [0.0, 3.0, 3.0],
        1: [3.0, 3.0, 3.0],
        2: [3.0],
    }

    for k, expected_eigenvalues in expected.items():
        assert_eigenvalues_allclose(
            filled_triangle_spectra.eigenvalues(k, "full"),
            expected_eigenvalues,
        )


# Kernel dimensions

def test_full_laplacian_kernel_dimensions_match_tiny_topology(
    edge_spectra: HodgeLaplacianSpectra,
    triangle_boundary_spectra: HodgeLaplacianSpectra,
    filled_triangle_spectra: HodgeLaplacianSpectra,
):
    # Kernel dimensions are checked only for full Laplacians, where they match
    # Betti-style expectations.
    examples = [
        (edge_spectra, {0: 1, 1: 0}),
        (triangle_boundary_spectra, {0: 1, 1: 1}),
        (filled_triangle_spectra, {0: 1, 1: 0, 2: 0}),
    ]

    for spectra, expected_kernel_dimensions in examples:
        for k, expected_dim_ker in expected_kernel_dimensions.items():
            assert spectra.spectrum(k, "full").dim_ker() == expected_dim_ker


# Spectrum observables

def test_spectrum_observables_on_known_values():
    spectrum = Spectrum(np.array([0.0, 2.0, 4.0]), dimension=3)

    assert spectrum.dim_ker() == 1
    assert_eigenvalues_allclose(spectrum.nonzero_eigenvalues(), [2.0, 4.0])
    assert spectrum.moment(1, normalized=True) == 2.0
    assert spectrum.moment(1, normalized=False) == 6.0
    assert spectrum.moment(1, normalized=True, nonzero=True) == 3.0
    assert spectrum.heat_trace(0.0) == 3.0
    np.testing.assert_allclose(
        spectrum.heat_trace(np.array([0.0, 1.0])),
        [3.0, 1.0 + np.exp(-2.0) + np.exp(-4.0)],
    )


# Spectrum contracts

@pytest.mark.parametrize(
    ("eigenvalues", "dimension", "eigenvectors", "exception", "match"),
    [
        (np.array([[0.0, 1.0]]), 2, None, ValueError, "1D"),
        (np.array([0.0]), 1.5, None, TypeError, "integer"),
        (np.array([]), -1, None, ValueError, "non-negative"),
        (np.array([0.0, 1.0]), 1, None, ValueError, "More eigenvalues"),
        (np.array([1.0, 0.0]), 2, None, ValueError, "sorted"),
        (np.array([-1.0]), 1, None, ValueError, "below -tol"),
        (np.array([0.0, 1.0]), 2, np.eye(3), ValueError, "eigenvectors shape"),
    ],
)
def test_spectrum_rejects_invalid_constructor_inputs(
    eigenvalues: np.ndarray,
    dimension: int,
    eigenvectors: np.ndarray | None,
    exception: type[Exception],
    match: str,
):
    with pytest.raises(exception, match=match):
        Spectrum(eigenvalues, dimension=dimension, eigenvectors=eigenvectors)


def test_spectrum_clamps_near_zero_eigenvalues():
    spectrum = Spectrum(np.array([-1e-12, 1e-12, 1.0]), dimension=3)

    assert_eigenvalues_allclose(spectrum.eigenvalues, [0.0, 0.0, 1.0])


def test_spectrum_rejects_nonzero_eigenvectors_when_missing():
    spectrum = Spectrum(np.array([0.0, 1.0]), dimension=2)

    with pytest.raises(ValueError, match="Eigenvectors were not provided"):
        spectrum.nonzero_eigenvectors()


# Hodge Laplacian spectra contracts

def test_hodge_laplacian_spectra_rejects_unknown_solver(
    filled_triangle_hodge_laplacian,
):
    with pytest.raises(ValueError, match="Unknown solver"):
        HodgeLaplacianSpectra(filled_triangle_hodge_laplacian, solver="other")


@pytest.mark.parametrize(
    ("solver_params", "match"),
    [
        (None, "solver_params is required"),
        ({}, "solver_params is required"),
        ({"unknown": 1}, "Unknown solver_params"),
        ({"which": "SM"}, "num_eigenvalues"),
        ({"num_eigenvalues": 0}, "positive integer"),
        ({"num_eigenvalues": 1, "which": "bad"}, "Invalid which"),
        ({"num_eigenvalues": 1, "sigma": "zero"}, "sigma must be"),
    ],
)
def test_hodge_laplacian_spectra_rejects_invalid_sparse_solver_params(
    filled_triangle_hodge_laplacian,
    solver_params: dict | None,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        HodgeLaplacianSpectra(
            filled_triangle_hodge_laplacian,
            solver="sparse",
            solver_params=solver_params,
        )


def test_hodge_laplacian_spectra_rejects_invalid_eigenvector_degree(
    filled_triangle_hodge_laplacian,
):
    with pytest.raises(ValueError, match="Invalid degree"):
        HodgeLaplacianSpectra(
            filled_triangle_hodge_laplacian,
            compute_eigenvectors=[(3, "full")],
        )


def test_hodge_laplacian_spectra_rejects_invalid_eigenvector_component(
    filled_triangle_hodge_laplacian,
):
    with pytest.raises(ValueError, match="Unknown component"):
        HodgeLaplacianSpectra(
            filled_triangle_hodge_laplacian,
            compute_eigenvectors=[(0, "diagonal")],
        )


@pytest.mark.parametrize("degree", [-1, 3])
def test_hodge_laplacian_spectra_rejects_invalid_spectrum_degree(
    filled_triangle_spectra: HodgeLaplacianSpectra,
    degree: int,
):
    with pytest.raises(ValueError, match="out of range"):
        filled_triangle_spectra.spectrum(degree)


def test_hodge_laplacian_spectra_rejects_invalid_spectrum_component(
    filled_triangle_spectra: HodgeLaplacianSpectra,
):
    with pytest.raises(ValueError, match="Unknown component"):
        filled_triangle_spectra.spectrum(0, "diagonal")
