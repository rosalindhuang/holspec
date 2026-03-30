"""Tests for EnsembleSpectraAnalysis observables and custom observable API."""
from __future__ import annotations

import numpy as np
import pytest

from holspec.analysis import EnsembleSpectraAnalysis, STANDARD_OBSERVABLES
from holspec.spectra import Spectrum


# =============================================================================
# Helpers
# =============================================================================

def _make_esa(
    eigenvalue_lists: list[np.ndarray],
    dimension: int | None = None,
    k: int = 0,
    component: str = 'full',
) -> EnsembleSpectraAnalysis:
    """Build a minimal EnsembleSpectraAnalysis from raw eigenvalue arrays."""
    spectra = []
    for evals in eigenvalue_lists:
        evals = np.sort(np.asarray(evals, dtype=np.float64))
        dim = dimension if dimension is not None else len(evals)
        spectra.append(Spectrum(evals, dim))
    return EnsembleSpectraAnalysis({(k, component): spectra})


# =============================================================================
# Standard observable names
# =============================================================================

class TestStandardObservableNames:
    """Verify the observable set has the expected names."""

    def test_standard_names(self):
        expected = {
            'dim_ker', 'eigval_min_nz', 'eigval_max',
            'eigval_mean', 'eigval_mean_nz',
            'eigval_var', 'eigval_var_nz',
            'eigval_sum', 'num_nonzero',
        }
        assert set(STANDARD_OBSERVABLES) == expected

    def test_member_observables_keys_match(self):
        esa = _make_esa([np.array([0.0, 1.0, 2.0, 3.0])])
        obs = esa.member_observables(0)
        assert set(obs.keys()) == set(STANDARD_OBSERVABLES)


# =============================================================================
# Observable values
# =============================================================================

class TestObservableValues:
    """Verify computed values for a known spectrum."""

    @pytest.fixture()
    def esa_known(self):
        """Two members with known eigenvalues."""
        # Member 0: [0, 0, 1, 4]  →  nz = [1, 4]
        # Member 1: [0, 2, 3, 9]  →  nz = [2, 3, 9]
        return _make_esa([
            np.array([0.0, 0.0, 1.0, 4.0]),
            np.array([0.0, 2.0, 3.0, 9.0]),
        ])

    def test_dim_ker(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_array_equal(obs['dim_ker'], [2.0, 1.0])

    def test_eigval_min_nz(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_array_equal(obs['eigval_min_nz'], [1.0, 2.0])

    def test_eigval_max(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_array_equal(obs['eigval_max'], [4.0, 9.0])

    def test_eigval_mean(self, esa_known):
        obs = esa_known.member_observables(0)
        # Member 0: mean([0,0,1,4]) = 5/4 = 1.25
        # Member 1: mean([0,2,3,9]) = 14/4 = 3.5
        np.testing.assert_allclose(obs['eigval_mean'], [1.25, 3.5])

    def test_eigval_mean_nz(self, esa_known):
        obs = esa_known.member_observables(0)
        # Member 0: mean([1,4]) = 2.5
        # Member 1: mean([2,3,9]) = 14/3
        np.testing.assert_allclose(obs['eigval_mean_nz'], [2.5, 14.0 / 3.0])

    def test_eigval_var(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_allclose(
            obs['eigval_var'],
            [np.var([0, 0, 1, 4]), np.var([0, 2, 3, 9])],
        )

    def test_eigval_var_nz(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_allclose(
            obs['eigval_var_nz'],
            [np.var([1, 4]), np.var([2, 3, 9])],
        )

    def test_eigval_sum(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_allclose(obs['eigval_sum'], [5.0, 14.0])

    def test_num_nonzero(self, esa_known):
        obs = esa_known.member_observables(0)
        np.testing.assert_array_equal(obs['num_nonzero'], [2.0, 3.0])


# =============================================================================
# Empty nonzero spectrum
# =============================================================================

class TestEmptyNonzeroSpectrum:
    """When all eigenvalues are zero, _nz observables should be nan."""

    @pytest.fixture()
    def esa_all_zero(self):
        return _make_esa([np.array([0.0, 0.0, 0.0])])

    def test_eigval_min_nz_nan(self, esa_all_zero):
        obs = esa_all_zero.member_observables(0)
        assert np.isnan(obs['eigval_min_nz'][0])

    def test_eigval_mean_nz_nan(self, esa_all_zero):
        obs = esa_all_zero.member_observables(0)
        assert np.isnan(obs['eigval_mean_nz'][0])

    def test_eigval_var_nz_nan(self, esa_all_zero):
        obs = esa_all_zero.member_observables(0)
        assert np.isnan(obs['eigval_var_nz'][0])


# =============================================================================
# compute_member_observable
# =============================================================================

class TestComputeMemberObservable:
    """Tests for the custom observable API."""

    @pytest.fixture()
    def esa(self):
        return _make_esa([
            np.array([0.0, 1.0, 4.0]),
            np.array([0.0, 2.0, 6.0]),
        ])

    def test_returns_correct_shape_and_values(self, esa):
        result = esa.compute_member_observable(
            0, func=lambda spc: float(spc.eigenvalues[-1]),
        )
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_cache_name_stores_result(self, esa):
        esa.compute_member_observable(
            0,
            func=lambda spc: float(np.median(spc.eigenvalues)),
            cache_name='eigval_median',
        )
        obs = esa.member_observables(0)
        assert 'eigval_median' in obs
        np.testing.assert_array_equal(obs['eigval_median'], [1.0, 2.0])

    def test_cache_name_collision_raises(self, esa):
        with pytest.raises(ValueError, match="already exists"):
            esa.compute_member_observable(
                0,
                func=lambda spc: 0.0,
                cache_name='dim_ker',
            )

    def test_no_cache_name_does_not_store(self, esa):
        before_keys = set(esa.member_observables(0).keys())
        esa.compute_member_observable(
            0, func=lambda spc: 0.0,
        )
        after_keys = set(esa.member_observables(0).keys())
        assert before_keys == after_keys

    def test_func_required(self, esa):
        with pytest.raises(TypeError):
            esa.compute_member_observable(0)


# =============================================================================
# save() metadata
# =============================================================================

class TestSaveObservableNames:
    """Verify save() writes observable_names from cache, not the static tuple."""

    def test_save_includes_custom_observable(self, tmp_path):
        esa = _make_esa([np.array([0.0, 1.0, 2.0])])
        esa.compute_member_observable(
            0,
            func=lambda spc: float(spc.eigenvalues[-1]),
            cache_name='custom_max',
        )
        filepath = tmp_path / 'test_analysis.h5'
        esa.save(filepath)

        from holspec.utilities import read_h5
        _, attrs = read_h5(filepath, dataset_names=[])
        saved_names = list(attrs['observable_names'])
        assert 'custom_max' in saved_names
        # Standard ones should also be there
        for name in STANDARD_OBSERVABLES:
            assert name in saved_names
