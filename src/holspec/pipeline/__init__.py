"""
Pipeline orchestration for holspec.

Provides stage metadata, provenance tracing, loading functions for
objects that require live references to upstream objects.

Provides pipeline stage functions ``run_{stage}`` that formalize the
pipeline computations into callable functions.
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
