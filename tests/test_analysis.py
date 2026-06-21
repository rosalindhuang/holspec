"""
Contract tests for ensemble spectra analysis objects.

These tests use hand-written Spectrum objects to verify public construction and
access-method behavior without covering HDF5 persistence.
"""

from pathlib import Path

import numpy as np
import pytest

from holspec.analysis import EnsembleSpectraAnalysis, extract_exp_params
from holspec.spectra import Spectrum
from holspec.utilities import save_h5


def _member_spectra() -> dict[tuple[int, str], list[Spectrum]]:
    return {
        (0, "full"): [
            Spectrum(np.array([0.0, 2.0]), dimension=2),
            Spectrum(np.array([0.0, 3.0]), dimension=2),
        ],
        (1, "full"): [
            Spectrum(np.array([1.0]), dimension=1),
            Spectrum(np.array([2.0]), dimension=1),
        ],
    }


def _analysis() -> EnsembleSpectraAnalysis:
    return EnsembleSpectraAnalysis(_member_spectra())


# Construction and access

def test_ensemble_spectra_analysis_accepts_member_spectra():
    member_spectra = _member_spectra()
    analysis = EnsembleSpectraAnalysis(member_spectra)

    assert analysis.num_members == 2
    assert analysis.degrees == [0, 1]
    assert analysis.components == ("full",)
    assert analysis.max_dim == 1
    assert analysis.dimensions == {0: 2, 1: 1}
    assert analysis.keys == [(0, "full"), (1, "full")]
    assert not analysis.has_eigenvectors
    assert len(analysis.member_spectra(0, "full")) == 2

    eigenvalues = analysis.member_eigenvalues(0, "full")
    np.testing.assert_allclose(eigenvalues[0], [0.0, 2.0])
    np.testing.assert_allclose(eigenvalues[1], [0.0, 3.0])

    observables = analysis.member_observables(0, "full")
    assert "dim_ker" in observables
    np.testing.assert_allclose(observables["dim_ker"], [1.0, 1.0])


def test_ensemble_spectra_analysis_rejects_empty_member_spectra():
    with pytest.raises(ValueError, match="non-empty"):
        EnsembleSpectraAnalysis({})


def test_ensemble_spectra_analysis_rejects_inconsistent_list_lengths():
    member_spectra = {
        (0, "full"): [Spectrum(np.array([0.0]), dimension=1)],
        (1, "full"): [
            Spectrum(np.array([1.0]), dimension=1),
            Spectrum(np.array([2.0]), dimension=1),
        ],
    }

    with pytest.raises(ValueError, match="same length"):
        EnsembleSpectraAnalysis(member_spectra)


@pytest.mark.parametrize(
    "key",
    [
        (-1, "full"),
        ("0", "full"),
        (0, "diagonal"),
    ],
)
def test_ensemble_spectra_analysis_rejects_invalid_member_spectrum_keys(
    key: tuple[int, str],
):
    with pytest.raises(ValueError):
        EnsembleSpectraAnalysis({key: [Spectrum(np.array([0.0]), dimension=1)]})


def test_ensemble_spectra_analysis_rejects_invalid_public_key_access():
    analysis = _analysis()

    with pytest.raises(KeyError, match="not present"):
        analysis.member_spectra(2, "full")


@pytest.mark.parametrize("ci", [0.0, 1.0])
def test_ensemble_spectra_analysis_rejects_invalid_confidence_interval(
    ci: float,
):
    analysis = _analysis()

    with pytest.raises(ValueError, match="ci must be"):
        analysis.observables_summary(0, "full", ci=ci)


def test_ensemble_spectra_analysis_rejects_duplicate_observable_cache_name():
    analysis = _analysis()

    with pytest.raises(ValueError, match="already exists"):
        analysis.compute_member_observable(
            0,
            "full",
            func=lambda spectrum: float(spectrum.dimension),
            cache_name="dim_ker",
        )


# Eigenvector spectra

def test_ensemble_spectra_analysis_accepts_eigenvector_spectra():
    analysis = _analysis()
    eigenvector_spectra = {
        (0, "full"): [
            Spectrum(
                np.array([0.0]),
                dimension=2,
                eigenvectors=np.array([[1.0], [0.0]]),
            ),
            Spectrum(
                np.array([0.0]),
                dimension=2,
                eigenvectors=np.array([[0.0], [1.0]]),
            ),
        ],
    }

    analysis.set_eigenvector_spectra(eigenvector_spectra)

    assert analysis.has_eigenvectors
    assert analysis.eigenvector_keys == [(0, "full")]
    assert analysis.eigenvector_spectra(0, "full") == eigenvector_spectra[(0, "full")]
    assert analysis.eigenvector_spectra(1, "full") is None


@pytest.mark.parametrize(
    ("eigenvector_spectra", "match"),
    [
        ({(-1, "full"): [None, None]}, "non-negative"),
        ({(0, "diagonal"): [None, None]}, "Unknown component"),
        ({(0, "full"): [None]}, "length"),
    ],
)
def test_ensemble_spectra_analysis_rejects_invalid_eigenvector_spectra(
    eigenvector_spectra: dict[tuple[int, str], list[Spectrum | None]],
    match: str,
):
    analysis = _analysis()

    with pytest.raises(ValueError, match=match):
        analysis.set_eigenvector_spectra(eigenvector_spectra)


def test_extract_exp_params_reads_generic_point_data_metadata(tmp_path: Path):
    point_data_path = tmp_path / "point_data.h5"
    save_h5(
        point_data_path,
        attributes={
            "metadata": {
                "exp_param": "area_fraction",
                "exp_value": 0.74,
            },
        },
    )

    exp_params = extract_exp_params(
        {"point_data": point_data_path},
        ["area_fraction", "gamma"],
    )

    assert exp_params["area_fraction"] == 0.74
    assert exp_params["gamma"] is None
