"""
Analysis of Hodge Laplacian spectra across ensembles and experiments.

This subpackage summarizes computed spectra into observables, distributions,
experiment-series comparisons, transition estimates, and persisted analysis
results.
"""

from .distributions import (
    empirical_distribution,
    distribution_distance,
)
from .spectra_analysis import (
    EnsembleSpectraAnalysis,
    STANDARD_OBSERVABLES,
    OBSERVABLE_LABELS,
    get_observable_label,
)
from .helpers import (
    spectra_analysis_cache_path,
    entry_group_key,
    entry_fixed_varying,
    include_in_exp_comparisons,
    extract_exp_params,
    build_spectra_file_records,
    compute_transition_points,
    save_exp_series,
    load_spectra_analyses,
    load_experiment_series,
)

__all__ = [
    "empirical_distribution",
    "distribution_distance",
    "EnsembleSpectraAnalysis",
    "STANDARD_OBSERVABLES",
    "OBSERVABLE_LABELS",
    "get_observable_label",
    "spectra_analysis_cache_path",
    "entry_group_key",
    "entry_fixed_varying",
    "include_in_exp_comparisons",
    "extract_exp_params",
    "build_spectra_file_records",
    "compute_transition_points",
    "save_exp_series",
    "load_spectra_analyses",
    "load_experiment_series",
]
