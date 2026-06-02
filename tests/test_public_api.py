"""
Tests for the supported public import surface.

These tests keep the top-level holspec API and core subpackage exports aligned
with the intended public interface used by examples, documentation, and users.
"""

import holspec
from holspec import (
    analysis,
    cochain_metric,
    hodge_laplacian,
    pipeline,
    point_data,
    simplicial,
    spectra,
)


def test_top_level_public_api_imports():
    expected_names = {
        "__version__",
        "PointData",
        "PointDataEnsemble",
        "run_data_import",
        "run_data_generation",
        "SimplicialComplex",
        "MetricTensor",
        "CochainMetric",
        "HodgeLaplacian",
        "Spectrum",
        "HodgeLaplacianSpectra",
        "run_pipeline",
        "run_pipeline_stages",
        "trace_provenance",
        "load_hodge_laplacian",
        "load_spectra",
        "EnsembleSpectraAnalysis",
    }

    assert set(holspec.__all__) == expected_names
    for name in expected_names:
        assert hasattr(holspec, name)


def test_core_subpackages_define_public_api():
    for module in (
        analysis,
        cochain_metric,
        hodge_laplacian,
        pipeline,
        point_data,
        simplicial,
        spectra,
    ):
        assert module.__all__
        for name in module.__all__:
            assert hasattr(module, name)
