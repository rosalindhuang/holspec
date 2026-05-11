"""
Config-driven pipeline orchestration and persisted-result loading.

This subpackage runs the staged holspec pipeline, manages provenance across
saved artifacts, selects pipeline inputs and outputs, and reconstructs derived
objects from HDF5 results.
"""

from .stages import (
    PIPELINE_STAGE_NAMES,
    run_topology_simplicial,
    run_geometry_metric,
    run_hodge_laplacian,
    run_spectra,
)
from .provenance import (
    trace_provenance,
    load_hodge_laplacian,
    load_spectra,
)
from .orchestration import (
    run_pipeline,
    run_pipeline_stages,
)
from .helpers import (
    select_pipeline_inputs,
    select_stage_outputs,
    split_pipeline_config,
)

__all__ = [
    "PIPELINE_STAGE_NAMES",
    "run_topology_simplicial",
    "run_geometry_metric",
    "run_hodge_laplacian",
    "run_spectra",
    "trace_provenance",
    "load_hodge_laplacian",
    "load_spectra",
    "run_pipeline",
    "run_pipeline_stages",
    "select_pipeline_inputs",
    "select_stage_outputs",
    "split_pipeline_config",
]
