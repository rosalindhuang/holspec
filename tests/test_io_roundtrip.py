"""
I/O round-trip tests for core pipeline objects.

These tests verify HDF5 persistence for standalone objects and lazy-cache
objects using tiny in-memory examples and temporary files only.
"""

from pathlib import Path

import numpy as np
import pytest

from holspec.cochain_metric import (
    CochainMetric,
    construct_diagonal_metric,
)
from holspec.hodge_laplacian import HodgeLaplacian
from holspec.point_data import PointData, PointDataEnsemble
from holspec.simplicial import SimplicialComplex
from holspec.spectra import HodgeLaplacianSpectra

from tests.helpers import assert_eigenvalues_allclose, assert_sparse_allclose


def _scaled_metric(sc: SimplicialComplex, scale: float = 2.0) -> CochainMetric:
    return CochainMetric(
        {
            k: construct_diagonal_metric(np.full(n_k, scale))
            for k, n_k in sc.num_simplices.items()
        },
    )


# Standalone object round trips

def test_point_data_positions_round_trips(tmp_path):
    positions = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
    )
    point_data = PointData(positions=positions, metadata={"label": "positions"})
    path = tmp_path / "point_positions.h5"

    saved_hash = point_data.save(path)
    loaded = PointData.load(path)

    assert saved_hash == point_data.content_hash == loaded.content_hash
    assert loaded.metadata["label"] == "positions"
    assert loaded.has_positions
    assert loaded.data_type == "positions"
    np.testing.assert_allclose(loaded.get_positions(), positions)
    np.testing.assert_allclose(
        loaded.get_distances(),
        [
            [0.0, 5.0],
            [5.0, 0.0],
        ],
    )


def test_point_data_distances_round_trips(tmp_path):
    distances = np.array(
        [
            [0.0, 1.5],
            [1.5, 0.0],
        ],
    )
    point_data = PointData(distances=distances, metadata={"label": "distances"})
    path = tmp_path / "point_distances.h5"

    saved_hash = point_data.save(path)
    loaded = PointData.load(path)

    assert saved_hash == point_data.content_hash == loaded.content_hash
    assert loaded.metadata["label"] == "distances"
    assert not loaded.has_positions
    assert loaded.data_type == "distances"
    np.testing.assert_allclose(loaded.get_distances(), distances)
    with pytest.raises(ValueError, match="Position data not available"):
        loaded.get_positions()


def test_point_data_metadata_with_paths_round_trips(tmp_path):
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    point_data = PointData(
        positions=positions,
        metadata={
            "config": {
                "filepath": Path("positions.npy"),
                "nested": {
                    "paths": [Path("a.csv"), Path("b.csv")],
                },
            },
        },
    )
    path = tmp_path / "point_metadata_paths.h5"

    point_data.save(path)
    loaded = PointData.load(path)

    assert loaded.metadata["config"]["filepath"] == "positions.npy"
    assert loaded.metadata["config"]["nested"]["paths"] == ["a.csv", "b.csv"]


def test_point_data_ensemble_round_trips(tmp_path):
    members = [
        PointData(
            positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            metadata={"member": 0},
        ),
        PointData(
            positions=np.array([[0.0, 1.0], [1.0, 1.0]]),
            metadata={"member": 1},
        ),
    ]
    ensemble = PointDataEnsemble(
        members,
        base_config={"generator": "manual"},
        noise_config={"scale": 0.0},
        metadata={"label": "ensemble"},
    )
    path = tmp_path / "ensemble.h5"

    ensemble.save(path)
    loaded = PointDataEnsemble.load(path)

    assert loaded.size == 2
    assert loaded.num_points == 2
    assert loaded.dimension == 2
    assert loaded.base_config == {"generator": "manual"}
    assert loaded.noise_config == {"scale": 0.0}
    assert loaded.metadata["label"] == "ensemble"
    for original_member, loaded_member in zip(members, loaded.members, strict=True):
        assert loaded_member.content_hash == original_member.content_hash
        assert loaded_member.metadata["member"] == original_member.metadata["member"]
        np.testing.assert_allclose(
            loaded_member.get_positions(),
            original_member.get_positions(),
        )


def test_imported_noisy_point_data_ensemble_round_trips(tmp_path):
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    np.save(tmp_path / "positions.npy", positions)
    config = {
        "data_type": "positions",
        "filepath": "positions.npy",
    }
    noise_config = {"scale": 0.05, "distribution": "normal"}
    ensemble = PointDataEnsemble.from_file_with_noise(
        config,
        noise_config,
        num_realizations=2,
        base_dir=tmp_path,
        base_seed=5,
        include_base=True,
    )
    path = tmp_path / "imported_noisy_ensemble.h5"

    ensemble.save(path)
    loaded = PointDataEnsemble.load(path)

    assert loaded.size == 3
    assert loaded.base_config == config
    assert loaded.noise_config == noise_config
    assert loaded.metadata["import_mode"] == "file_with_noise"
    assert loaded.metadata["num_realizations"] == 2
    assert loaded.metadata["base_seed"] == 5
    assert loaded.metadata["include_base"] is True
    np.testing.assert_allclose(loaded[0].get_positions(), positions)
    assert loaded[0].metadata["is_base"] is True
    assert loaded[1].metadata["seed"] == 5
    assert loaded[2].metadata["seed"] == 6


def test_simplicial_complex_round_trips_with_incidence_cache(
    tmp_path,
    filled_triangle_complex: SimplicialComplex,
):
    filled_triangle_complex = SimplicialComplex(
        filled_triangle_complex.simplices,
        metadata={"label": "filled_triangle"},
    )
    _ = filled_triangle_complex.incidence_matrix(1)
    path = tmp_path / "simplicial_complex.h5"

    saved_hash = filled_triangle_complex.save(path)
    loaded = SimplicialComplex.load(path)

    assert saved_hash == filled_triangle_complex.content_hash == loaded.content_hash
    assert loaded.metadata["label"] == "filled_triangle"
    assert loaded.simplices == filled_triangle_complex.simplices
    assert loaded.f_vector == filled_triangle_complex.f_vector
    assert_sparse_allclose(
        loaded.incidence_matrix(1),
        filled_triangle_complex.incidence_matrix(1).toarray(),
    )


def test_cochain_metric_round_trips(
    tmp_path,
    filled_triangle_metric: CochainMetric,
):
    filled_triangle_metric.metadata["label"] = "combinatorial"
    path = tmp_path / "cochain_metric.h5"

    saved_hash = filled_triangle_metric.save(path)
    loaded = CochainMetric.load(path)

    assert saved_hash == filled_triangle_metric.content_hash == loaded.content_hash
    assert loaded.metadata["label"] == "combinatorial"
    assert loaded.degrees == filled_triangle_metric.degrees
    assert loaded.dimensions == filled_triangle_metric.dimensions
    for k in filled_triangle_metric.degrees:
        assert_sparse_allclose(
            loaded[k].to_matrix(),
            filled_triangle_metric[k].to_matrix().toarray(),
        )


# Grouped HDF5 round trip

def test_point_data_round_trips_from_hdf5_group(tmp_path):
    point_data = PointData(positions=np.array([[0.0], [1.0]]))
    path = tmp_path / "grouped.h5"

    point_data.save(path, group="member_0000")
    loaded = PointData.load(path, group="member_0000")

    assert loaded.content_hash == point_data.content_hash
    np.testing.assert_allclose(loaded.get_positions(), point_data.get_positions())


# Derived cache round trips

def test_hodge_laplacian_cache_round_trips(
    tmp_path,
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_metric: CochainMetric,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    path = tmp_path / "hodge_laplacian.h5"
    original_full_0 = filled_triangle_hodge_laplacian.to_matrix(0, "full")
    original_lower_1 = filled_triangle_hodge_laplacian.to_matrix(1, "lower")

    saved_hash = filled_triangle_hodge_laplacian.save(path)
    loaded = HodgeLaplacian(filled_triangle_complex, filled_triangle_metric)
    loaded.load_cache(path)

    assert saved_hash == filled_triangle_hodge_laplacian.content_hash
    assert loaded.content_hash == filled_triangle_hodge_laplacian.content_hash
    assert_sparse_allclose(loaded.to_matrix(0, "full"), original_full_0.toarray())
    assert_sparse_allclose(loaded.to_matrix(1, "lower"), original_lower_1.toarray())


def test_hodge_laplacian_cache_rejects_hash_mismatch(
    tmp_path,
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    path = tmp_path / "hodge_laplacian_mismatch.h5"
    _ = filled_triangle_hodge_laplacian.to_matrix(0, "full")
    filled_triangle_hodge_laplacian.save(path)

    mismatched_metric = _scaled_metric(filled_triangle_complex)
    mismatched_hodge_laplacian = HodgeLaplacian(
        filled_triangle_complex,
        mismatched_metric,
    )

    with pytest.raises(ValueError, match="CM content hash mismatch"):
        mismatched_hodge_laplacian.load_cache(path)


def test_hodge_laplacian_spectra_cache_round_trips(
    tmp_path,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    path = tmp_path / "hodge_laplacian_spectra.h5"
    spectra = HodgeLaplacianSpectra(
        filled_triangle_hodge_laplacian,
        compute_eigenvectors=[(0, "full")],
    )
    original_full_0 = spectra.spectrum(0, "full")
    original_upper_1 = spectra.spectrum(1, "upper")

    saved_hash = spectra.save(path)
    loaded = HodgeLaplacianSpectra(filled_triangle_hodge_laplacian)
    loaded.load_cache(path)
    loaded_full_0 = loaded.spectrum(0, "full")
    loaded_upper_1 = loaded.spectrum(1, "upper")

    assert saved_hash == spectra.content_hash == loaded.content_hash
    assert loaded_full_0.dimension == original_full_0.dimension
    assert_eigenvalues_allclose(loaded_full_0.eigenvalues, original_full_0.eigenvalues)
    assert loaded_full_0.eigenvectors is not None
    assert loaded_full_0.eigenvectors.shape == original_full_0.eigenvectors.shape
    assert loaded_upper_1.dimension == original_upper_1.dimension
    assert_eigenvalues_allclose(
        loaded_upper_1.eigenvalues,
        original_upper_1.eigenvalues,
    )
    assert loaded_upper_1.eigenvectors is None


def test_hodge_laplacian_spectra_cache_rejects_hash_mismatch(
    tmp_path,
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    path = tmp_path / "hodge_laplacian_spectra_mismatch.h5"
    spectra = HodgeLaplacianSpectra(filled_triangle_hodge_laplacian)
    _ = spectra.spectrum(0, "full")
    spectra.save(path)

    mismatched_metric = _scaled_metric(filled_triangle_complex)
    mismatched_hodge_laplacian = HodgeLaplacian(
        filled_triangle_complex,
        mismatched_metric,
    )
    mismatched_spectra = HodgeLaplacianSpectra(mismatched_hodge_laplacian)

    with pytest.raises(ValueError, match="HL content hash mismatch"):
        mismatched_spectra.load_cache(path)
