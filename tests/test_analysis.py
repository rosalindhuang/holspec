"""
Contract tests for ensemble spectra analysis objects.

These tests use hand-written Spectrum objects and compact HDF5 fixtures to verify
public construction, access behavior, and analysis persistence.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from holspec.analysis import (
    EnsembleSpectraAnalysis,
    extract_exp_params,
    get_observable_label,
    include_in_exp_comparisons,
    load_experiment_series,
    load_spectra_analyses,
    save_exp_series,
    spectra_analysis_cache_path,
)
from holspec.spectra import Spectrum, ZERO_EIGENVALUE_TOL
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


def _save_pipeline_spectra(filepath: Path, num_members: int) -> None:
    """Write the minimal stage-4 layout needed by ESA.from_file()."""
    for member_index in range(num_members):
        member_group = f"member_{member_index:04d}"
        save_h5(
            filepath,
            attributes={"cached_keys": ["0_full"]},
            group=member_group,
            mode="update",
        )
        save_h5(
            filepath,
            datasets={"eigenvalues": np.array([0.0, member_index + 1.0])},
            attributes={"dimension": 2},
            group=f"{member_group}/degree_0/component_full",
            mode="update",
        )


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


def test_ensemble_spectra_analysis_applies_sqrt_transform():
    eigenvectors = np.eye(4)
    member_spectra = {
        (0, "full"): [
            Spectrum(
                np.array([0.0, ZERO_EIGENVALUE_TOL, 4.0, 9.0]),
                dimension=4,
                eigenvectors=eigenvectors,
            ),
            Spectrum(
                np.array([0.0, ZERO_EIGENVALUE_TOL, 16.0, 25.0]),
                dimension=4,
            ),
        ],
    }

    analysis = EnsembleSpectraAnalysis(
        member_spectra,
        eigenvalue_transform="sqrt",
    )

    assert analysis.eigenvalue_transform == "sqrt"
    np.testing.assert_allclose(
        analysis.member_eigenvalues(0, "full")[0],
        [0.0, 0.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        analysis.member_eigenvectors(0, "full")[0],
        eigenvectors,
    )
    assert analysis.member_spectra(0, "full")[0].dimension == 4
    assert np.all(np.diff(analysis.member_eigenvalues(0, "full")[0]) >= 0)
    assert "eigenvalue_transform: sqrt" in analysis.summary()
    assert "eigenvalue_transform='sqrt'" in repr(analysis)

    observables = analysis.member_observables(0, "full")
    np.testing.assert_allclose(observables["dim_ker"], [2.0, 2.0])
    np.testing.assert_allclose(observables["num_nonzero"], [2.0, 2.0])
    np.testing.assert_allclose(observables["eigval_mean_nz"], [2.5, 4.5])
    np.testing.assert_allclose(observables["eigval_var_nz"], [0.25, 0.25])
    np.testing.assert_allclose(observables["eigval_max"], [3.0, 5.0])


def test_ensemble_spectra_analysis_rejects_unknown_eigenvalue_transform():
    with pytest.raises(ValueError, match="Unknown eigenvalue_transform"):
        EnsembleSpectraAnalysis(
            _member_spectra(),
            eigenvalue_transform="log",
        )


def test_get_observable_label_uses_sqrt_coordinate():
    assert get_observable_label("eigval_mean_nz") == (
        r"$\langle\lambda\rangle_\mathrm{nonzero}$"
    )
    assert get_observable_label("eigval_mean_nz", "sqrt") == (
        r"$\langle\sqrt{\lambda}\rangle_\mathrm{nonzero}$"
    )
    assert get_observable_label("dim_ker", "sqrt") == r"$\dim\ker$"


def test_spectra_analysis_cache_path_namespaces_dataset(tmp_path: Path):
    cache_root = tmp_path / "spectra_analysis"
    source_a = tmp_path / "experiment_a" / "spectra" / "shared" / "output.h5"
    source_b = tmp_path / "experiment_b" / "spectra" / "shared" / "output.h5"

    path_a = spectra_analysis_cache_path(cache_root, "dataset_a", source_a)
    path_b = spectra_analysis_cache_path(cache_root, "dataset_b", source_b)

    assert path_a == cache_root / "dataset_a" / "shared" / "output.h5"
    assert path_b == cache_root / "dataset_b" / "shared" / "output.h5"
    assert path_a != path_b


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


def test_ensemble_spectra_analysis_cache_validates_eigenvalue_transform(
    tmp_path: Path,
):
    cache_path = tmp_path / "sqrt_analysis.h5"
    sqrt_analysis = EnsembleSpectraAnalysis(
        _member_spectra(),
        eigenvalue_transform="sqrt",
    )
    sqrt_analysis.eigenvalue_distribution(0, "full")
    sqrt_analysis.save(cache_path)

    loaded = EnsembleSpectraAnalysis(
        _member_spectra(),
        eigenvalue_transform="sqrt",
    )
    loaded.load_cache(cache_path)
    assert loaded.eigenvalue_transform == "sqrt"
    np.testing.assert_allclose(
        loaded.member_observables(0, "full")["eigval_max"],
        [np.sqrt(2.0), np.sqrt(3.0)],
    )

    identity_analysis = _analysis()
    with pytest.raises(ValueError, match="eigenvalue_transform mismatch"):
        identity_analysis.load_cache(cache_path)


def test_ensemble_spectra_analysis_loads_legacy_cache_as_identity(tmp_path: Path):
    cache_path = tmp_path / "legacy_analysis.h5"
    analysis = _analysis()
    analysis.save(cache_path)

    with h5py.File(cache_path, "a") as h5file:
        del h5file.attrs["eigenvalue_transform"]

    loaded = _analysis()
    loaded.load_cache(cache_path)
    assert loaded.eigenvalue_transform == "identity"


@pytest.mark.parametrize(
    ("stored_transform", "expected_transform"),
    [("sqrt", "sqrt"), (None, "identity")],
)
def test_load_spectra_analyses_reconstructs_cached_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_transform: str | None,
    expected_transform: str,
):
    pipeline_path = (
        tmp_path / "data" / "interim" / "dataset" / "spectra" / "input" / "output.h5"
    )
    cache_root = tmp_path / "processed"
    cache_path = spectra_analysis_cache_path(
        cache_root,
        "dataset",
        pipeline_path,
    )
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.touch()
    cache_path.parent.mkdir(parents=True)
    attributes = {"degrees": [0], "components": ["full"], "metadata": {}}
    if stored_transform is not None:
        attributes["eigenvalue_transform"] = stored_transform
    save_h5(cache_path, attributes=attributes)

    monkeypatch.setattr(
        "holspec.analysis.helpers.select_stage_outputs",
        lambda **kwargs: {"input": {"output": pipeline_path}},
    )
    seen = {}

    class FakeAnalysis:
        def load_cache(self, filepath):
            seen["cache_path"] = filepath

    def fake_from_file(cls, filepath, project_root, **kwargs):
        seen["transform"] = kwargs["eigenvalue_transform"]
        return FakeAnalysis()

    monkeypatch.setattr(
        EnsembleSpectraAnalysis,
        "from_file",
        classmethod(fake_from_file),
    )

    loaded = load_spectra_analyses(
        {"dataset": {"base_dir": "unused", "exp_param": None}},
        cache_root,
        tmp_path,
    )

    assert seen == {
        "transform": expected_transform,
        "cache_path": cache_path,
    }
    assert loaded["dataset"]["input/output"]["esa"].__class__ is FakeAnalysis


def test_load_spectra_analyses_isolates_colliding_dataset_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache_root = tmp_path / "processed" / "spectra_analysis"
    sources = {
        "noise": tmp_path / "data/interim/noise/spectra/shared/output.h5",
        "size": tmp_path / "data/interim/size/spectra/shared/output.h5",
    }
    member_counts = {"noise": 5, "size": 1}

    for dataset_name, pipeline_path in sources.items():
        _save_pipeline_spectra(pipeline_path, member_counts[dataset_name])
        spectra = {
            (0, "full"): [
                Spectrum(np.array([0.0, member_index + 1.0]), dimension=2)
                for member_index in range(member_counts[dataset_name])
            ],
        }
        analysis = EnsembleSpectraAnalysis(
            spectra,
            metadata={
                "source_file": str(pipeline_path),
                "dataset_name": dataset_name,
            },
        )
        analysis.save(
            spectra_analysis_cache_path(
                cache_root,
                dataset_name,
                pipeline_path,
            ),
        )

    monkeypatch.setattr(
        "holspec.analysis.helpers.select_stage_outputs",
        lambda **kwargs: {
            "shared": {"output": sources[kwargs["base_dir"]]},
        },
    )
    datasets = {
        name: {"base_dir": name, "exp_param": None} for name in ("noise", "size")
    }

    loaded = load_spectra_analyses(datasets, cache_root, tmp_path)

    assert loaded["noise"]["shared/output"]["esa"].num_members == 5
    assert loaded["size"]["shared/output"]["esa"].num_members == 1
    assert (
        loaded["noise"]["shared/output"]["esa_path"]
        != loaded["size"]["shared/output"]["esa_path"]
    )


def test_experiment_series_roundtrip_preserves_eigenvalue_transform(
    tmp_path: Path,
):
    filepath = tmp_path / "dataset__group.h5"
    analysis_key = (0, "full")
    series = {
        "exp_param": "noise",
        "exp_values": np.array([0.1, 0.2]),
        "eigenvalue_transform": "sqrt",
        "observable_series": {
            analysis_key: {
                "eigval_mean_nz": {
                    "mean": np.array([1.0, 2.0]),
                    "std": np.array([0.1, 0.2]),
                },
            },
        },
        "distribution_series": {
            analysis_key: {
                "x": np.array([0.5, 1.5]),
                "density_stack": np.ones((2, 2)),
            },
        },
        "distance_series": {analysis_key: np.array([0.5])},
        "transition_points": {analysis_key: 0.15},
    }
    analysis_config = {
        "degrees": [0],
        "components": ("full",),
        "analysis_keys": [analysis_key],
        "observable_names": ("eigval_mean_nz",),
        "eigenvalue_transform": "sqrt",
    }

    save_exp_series(
        filepath,
        "dataset",
        ("group",),
        series,
        analysis_config,
    )
    loaded = load_experiment_series(tmp_path)

    assert loaded[("dataset", ("group",))]["eigenvalue_transform"] == "sqrt"


def test_load_experiment_series_treats_legacy_file_as_identity(tmp_path: Path):
    filepath = tmp_path / "dataset__group.h5"
    analysis_key = (0, "full")
    series = {
        "exp_param": "noise",
        "exp_values": np.array([0.1, 0.2]),
        "observable_series": {
            analysis_key: {
                "eigval_mean_nz": {
                    "mean": np.array([1.0, 2.0]),
                    "std": np.array([0.1, 0.2]),
                },
            },
        },
        "distribution_series": {
            analysis_key: {
                "x": np.array([0.5, 1.5]),
                "density_stack": np.ones((2, 2)),
            },
        },
        "distance_series": {analysis_key: np.array([0.5])},
        "transition_points": {analysis_key: 0.15},
    }
    analysis_config = {
        "degrees": [0],
        "components": ("full",),
        "analysis_keys": [analysis_key],
        "observable_names": ("eigval_mean_nz",),
    }
    save_exp_series(
        filepath,
        "dataset",
        ("group",),
        series,
        analysis_config,
    )
    with h5py.File(filepath, "a") as h5file:
        del h5file.attrs["eigenvalue_transform"]

    loaded = load_experiment_series(tmp_path)

    assert loaded[("dataset", ("group",))]["eigenvalue_transform"] == "identity"


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


@pytest.mark.parametrize(
    ("generator", "params", "expected"),
    [
        ("trilatthex", {"n_rings": 8}, 8),
        ("trilattrect", {"nx": 10, "ny": 10}, 10),
        ("bcclatt", {"nx": 4, "ny": 4, "nz": 4}, 4),
        ("trilattrect", {"nx": 10, "ny": 12}, None),
    ],
)
def test_extract_exp_params_reads_generated_lattice_size(
    tmp_path: Path,
    generator: str,
    params: dict,
    expected: int | None,
):
    point_data_path = tmp_path / "point_data.h5"
    save_h5(
        point_data_path,
        attributes={
            "base_config": {
                "generator": generator,
                "params": params,
            },
        },
    )

    exp_params = extract_exp_params({"point_data": point_data_path}, ["size"])

    assert exp_params["size"] == expected


def test_extract_exp_params_prefers_generic_size_metadata(tmp_path: Path):
    point_data_path = tmp_path / "point_data.h5"
    save_h5(
        point_data_path,
        attributes={
            "metadata": {"exp_param": "size", "exp_value": 99},
            "base_config": {
                "generator": "trilatthex",
                "params": {"n_rings": 8},
            },
        },
    )

    exp_params = extract_exp_params({"point_data": point_data_path}, ["size"])

    assert exp_params["size"] == 99


@pytest.mark.parametrize(
    ("exp_param", "exp_value", "analysis_config", "expected"),
    [
        ("noise", 0.0, {"noise_exp_include_zero": False}, False),
        ("noise", 0.0, {"noise_exp_include_zero": True}, True),
        ("noise", 0.1, {"noise_exp_include_zero": False}, True),
        ("noise", 0.0, {}, True),
        ("size", 1, {"size_exp_min_size": 2}, False),
        ("size", 2, {"size_exp_min_size": 2}, True),
        ("size", 3, {"size_exp_min_size": 2}, True),
        ("size", 1, {"size_exp_min_size": None}, True),
        ("size", 1, {}, True),
        ("alpha", 0.0, {}, True),
        ("size", None, {"size_exp_min_size": 2}, True),
        (None, None, {}, True),
    ],
)
def test_include_in_exp_comparisons(
    exp_param: str | None,
    exp_value: float | int | None,
    analysis_config: dict,
    expected: bool,
):
    assert include_in_exp_comparisons(exp_param, exp_value, analysis_config) is expected
