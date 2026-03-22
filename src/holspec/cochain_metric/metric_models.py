"""
Cochain metric model construction.

Factory functions for constructing MetricTensor instances and collections
from geometric data, with config-driven dispatch via the metric model registry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from .metric_tensor import MetricTensor
from .validation import METRIC_POSITIVITY_TOL
from holspec.utilities import format_float_str

if TYPE_CHECKING:
    from holspec.simplicial import SimplicialComplex
    from holspec.point_data import PointData


# =============================================================================
# Per-Degree Construction Functions
# =============================================================================

def construct_diagonal_metric(
    diagonal_elements: np.ndarray,
    tol: float = METRIC_POSITIVITY_TOL,
) -> MetricTensor:
    """
    Construct a diagonal metric tensor from explicit weights.

    Parameters
    ----------
    diagonal_elements : ndarray, shape (N,)
        Strictly positive weights for each basis element. Validated against
        the SPD contract by MetricTensor at construction. An empty array
        (N=0) is permitted and returns the unique metric on the zero vector space.
    tol : float, default=METRIC_POSITIVITY_TOL
        Positivity tolerance passed through to MetricTensor validation.

    Returns
    -------
    MetricTensor
        Diagonal metric with the specified weights.
    """
    diagonal_elements = np.asarray(diagonal_elements, dtype=float)
    if diagonal_elements.ndim != 1:
        raise ValueError(
            f"diagonal_elements must be a 1D array, "
            f"got shape {diagonal_elements.shape}"
        )
    return MetricTensor(
        sparse.diags(diagonal_elements, format='csr'), is_diagonal=True, tol=tol
    )


# =============================================================================
# Collection-Level Construction Functions
# =============================================================================

def construct_combinatorial_cochain_metric(
    sc: SimplicialComplex,
    ptd: PointData | None = None,
    **params,
) -> tuple[dict[int, MetricTensor], dict]:
    """
    Construct the combinatorial cochain metric for a simplicial complex.

    Assigns the identity metric G^k = I on each cochain space C^k, giving
    equal weight to every k-simplex. This is the canonical topology-only
    baseline: the resulting Hodge Laplacian depends purely on combinatorial
    structure, with no geometric information from point positions.

    Parameters
    ----------
    sc : SimplicialComplex
        Source simplicial complex. Only `num_simplices` is accessed.
    ptd : PointData, optional
        Accepted for interface consistency with other metric model construction
        functions; not used in the combinatorial special case.
    **params
        Accepted for interface consistency; no parameters are defined for
        the combinatorial model.

    Returns
    -------
    tuple[dict[int, MetricTensor], dict]
        Metric tensors (identity at each degree k = 0, ..., max_dim) and
        diagnostics dict (empty for the combinatorial model).
    """
    metric_tensors = {
        k: construct_diagonal_metric(np.ones(n_k))
        for k, n_k in sc.num_simplices.items()
    }
    return metric_tensors, {}


def construct_hodge_star_cochain_metric(
    sc: SimplicialComplex,
    ptd: PointData,
    **params,
) -> tuple[dict[int, MetricTensor], dict]:
    """Construct the Hodge star cochain metric. Not yet implemented."""
    raise NotImplementedError(
        "Hodge star cochain metric construction is not yet implemented."
    )

# =============================================================================
# Registry and Config-Driven Utilities
# =============================================================================

COCHAIN_METRIC_MODEL_REGISTRY: dict[str, callable] = {
    'combinatorial': construct_combinatorial_cochain_metric,
    'hodge_star': construct_hodge_star_cochain_metric,
}


def construct_cochain_metric_from_config(
    config: dict,
    sc: SimplicialComplex,
    ptd: PointData,
) -> tuple[dict[int, MetricTensor], dict]:
    """
    Construct cochain metric tensors G^k : C^k -> C^k for k = 0, ..., n
    on the cochain spaces of a simplicial complex.

    Parameters
    ----------
    config : dict
        Must contain 'model' (str) and 'params' (dict). For example:
            {'model': 'combinatorial', 'params': {}}

    sc : SimplicialComplex
        Source simplicial complex.
    ptd : PointData
        Point cloud data. Required by geometric metric models; unused in
        the combinatorial special case.

    Returns
    -------
    tuple[dict[int, MetricTensor], dict]
        Metric tensors at each degree k = 0, ..., max_dim, and a diagnostics
        dict (empty for models with no diagnostics).

    Raises
    ------
    ValueError
        If 'model' key is missing or the model name is not in the registry.
    """
    if 'model' not in config:
        raise ValueError("config must contain a 'model' key")

    model = config['model']
    if model not in COCHAIN_METRIC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown metric model '{model}'. "
            f"Available models: {list(COCHAIN_METRIC_MODEL_REGISTRY)}"
        )

    params = config.get('params', {})
    construct_fn = COCHAIN_METRIC_MODEL_REGISTRY[model]
    return construct_fn(sc, ptd, **params)


def create_metric_model_label(
    config: dict,
    float_fmt: str | None = 'g',
    strip_zeros: bool = True,
) -> str:
    """
    Create a descriptive label from a metric model config.

    Label format: ``model_param1_param2...``
    Parameter names are abbreviated to their first three non-underscore characters.
    Floats are formatted with ``float_fmt``. None values are omitted.

    Parameters
    ----------
    config : dict
        Must contain 'model' (str) and 'params' (dict).
    float_fmt : str or None, default='g'
        Format specifier for float values (e.g. 'g', '.2f').
    strip_zeros : bool, default=True
        If True, trailing zeros after the decimal point are removed from floats.

    Returns
    -------
    label : str

    Raises
    ------
    ValueError
        If 'model' key is missing or the model name is not in the registry.

    Examples
    --------
    >>> create_metric_model_label({'model': 'combinatorial', 'params': {}})
    'combinatorial'
    """
    if 'model' not in config:
        raise ValueError("config must contain a 'model' key")

    model = config['model']
    if model not in COCHAIN_METRIC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown metric model '{model}'. "
            f"Available models: {list(COCHAIN_METRIC_MODEL_REGISTRY)}"
        )

    params = config.get('params', {})
    parts = [model]

    for key, value in params.items():
        # Skip None values — they represent 'use default / no constraint'
        if value is None:
            continue

        param_abbr = key.replace('_', '')[:3]

        if isinstance(value, bool):
            value_str = '1' if value else '0'
        elif isinstance(value, float):
            value_str = format_float_str(value, float_fmt, strip_zeros=strip_zeros)
        elif isinstance(value, str):
            value_str = value[:4].lower().replace('_', '')
        else:
            value_str = str(value).replace('.', 'p')

        parts.append(f"{param_abbr}{value_str}")

    return '_'.join(parts)