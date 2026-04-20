"""
Spectral analysis utilities for holspec.

Provides standalone distribution estimation and comparison functions,
the EnsembleSpectraAnalysis class for per-dataset spectral analysis,
and helpers for indexing, persisting, and loading analyses and
experiment series across datasets.
"""

from .distributions import (
    empirical_distribution,
    distribution_distance,
)
from .spectra_analysis import (
    EnsembleSpectraAnalysis,
    STANDARD_OBSERVABLES,
    OBSERVABLE_LABELS,
)
from .helpers import (
    entry_group_key,
    entry_fixed_varying,
    extract_exp_params,
    build_spectra_file_records,
    compute_transition_points,
    save_exp_series,
    load_spectra_analyses,
    load_experiment_series,
)
