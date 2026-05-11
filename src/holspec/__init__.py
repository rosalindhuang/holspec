"""
holspec: Hodge Laplacian spectral analysis for point-cloud data.

The public API exposes the core framework objects, pipeline entry points,
and analysis utilities. Lower-level construction, validation, visualization,
and I/O helpers are available from their subpackages.
"""

from importlib.metadata import PackageNotFoundError, version

from holspec.point_data import PointData, PointDataEnsemble, run_data_generation
from holspec.simplicial import SimplicialComplex
from holspec.cochain_metric import CochainMetric, MetricTensor
from holspec.hodge_laplacian import HodgeLaplacian
from holspec.spectra import HodgeLaplacianSpectra, Spectrum
from holspec.pipeline import (
    load_hodge_laplacian,
    load_spectra,
    run_pipeline,
    run_pipeline_stages,
    trace_provenance,
)
from holspec.analysis import EnsembleSpectraAnalysis

try:
    __version__ = version("holspec")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",

    # Point data
    "PointData",
    "PointDataEnsemble",
    "run_data_generation",

    # Core framework objects
    "SimplicialComplex",
    "MetricTensor",
    "CochainMetric",
    "HodgeLaplacian",
    "Spectrum",
    "HodgeLaplacianSpectra",

    # Pipeline execution and loading
    "run_pipeline",
    "run_pipeline_stages",
    "trace_provenance",
    "load_hodge_laplacian",
    "load_spectra",

    # Analysis
    "EnsembleSpectraAnalysis",
]
