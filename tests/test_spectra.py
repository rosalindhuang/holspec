import numpy as np

from holspec.spectra import HodgeLaplacianSpectra, Spectrum

from tests.helpers import assert_eigenvalues_allclose


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


def test_edge_full_spectrum_eigenvalues(edge_spectra: HodgeLaplacianSpectra):
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


def test_full_laplacian_kernel_dimensions_match_tiny_topology(
    edge_spectra: HodgeLaplacianSpectra,
    triangle_boundary_spectra: HodgeLaplacianSpectra,
    filled_triangle_spectra: HodgeLaplacianSpectra,
):
    examples = [
        (edge_spectra, {0: 1, 1: 0}),
        (triangle_boundary_spectra, {0: 1, 1: 1}),
        (filled_triangle_spectra, {0: 1, 1: 0, 2: 0}),
    ]

    for spectra, expected_kernel_dimensions in examples:
        for k, expected_dim_ker in expected_kernel_dimensions.items():
            assert spectra.spectrum(k, "full").dim_ker() == expected_dim_ker


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
