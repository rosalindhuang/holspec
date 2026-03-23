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
from .dec_geometry import (
    VOLUME_DEGENERACY_TOL,
    compute_dual_volumes,
    compute_simplex_volumes,
)
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
    positivity: str = 'strict',
    degeneracy_tol: float = VOLUME_DEGENERACY_TOL,
    positivity_tol: float = METRIC_POSITIVITY_TOL,
) -> tuple[dict[int, MetricTensor], dict]:
    """
    Construct the Hodge star cochain metric for a simplicial complex.

    Assigns the circumcentric discrete Hodge star as the metric tensor at
    each cochain degree: G^k = diag([*^k]), where the diagonal entries are
    the ratio of signed dual cell volume to unsigned primal simplex volume.

    Parameters
    ----------
    sc : SimplicialComplex
        Source simplicial complex providing simplex structure.
    ptd : PointData
        Point data providing vertex positions.
    positivity : {'strict', 'abs'}, default='strict'
        How to handle negative raw Hodge star diagonal entries.
        - 'strict': raise ValueError if any raw entry is negative
        - 'abs': replace raw entries with their absolute values before
          constructing the metric tensors
    degeneracy_tol : float, default=VOLUME_DEGENERACY_TOL
        Minimum acceptable volume magnitude for primal and dual volume
        degeneracy checks.
    positivity_tol : float, default=METRIC_POSITIVITY_TOL
        Positivity tolerance passed to construct_diagonal_metric and
        MetricTensor validation.

    Returns
    -------
    tuple[dict[int, MetricTensor], dict]
        Per-degree diagonal metric tensors and a diagnostics dict describing
        the raw Hodge star computation before any positivity handling.

    Raises
    ------
    ValueError
        If positions are unavailable, the simplicial complex is not
        full-dimensional, the ambient dimension is outside the current
        supported scope, positivity is invalid, or raw Hodge star entries are
        negative under positivity='strict'.
    
    References
    ----------
    .. [1] A. N. Hirani, K. Kalyanaraman, and E. B. VanderZee, "Delaunay
    Hodge star," Computer-Aided Design, vol. 45, no. 2, pp. 540-544, Feb.
    2013, doi: 10.1016/j.cad.2012.10.038.
    """
    # --- Validate preconditions ---

    if not ptd.has_positions:
        raise ValueError(
            "Hodge star metric requires position data: "
            "ptd.has_positions is False"
        )

    if sc.max_dim != ptd.dimension:
        raise ValueError(
            f"Hodge star metric requires a full-dimensional complex: "
            f"sc.max_dim={sc.max_dim} != ptd.dimension={ptd.dimension}"
        )

    if ptd.dimension not in (2, 3):
        raise ValueError(
            f"Hodge star metric is currently supported for 2D and 3D "
            f"complexes: ptd.dimension={ptd.dimension}"
        )

    if positivity not in ('strict', 'abs'):
        raise ValueError(
            f"positivity must be 'strict' or 'abs', got '{positivity}'"
        )
    

    # --- Compute Hodge star from primal and dual volumes ---

    simplices = sc.simplices
    positions = ptd.get_positions()

    # Compute primal volumes and check for degeneracy
    primal_volumes = compute_simplex_volumes(simplices, positions)

    for k, primal_vols_k in sorted(primal_volumes.items()):
        degenerate_mask = primal_vols_k < degeneracy_tol
        if np.any(degenerate_mask):
            n_degenerate = int(np.sum(degenerate_mask))
            raise ValueError(
                f"Degenerate primal volumes at degree k={k}: "
                f"{n_degenerate} of {len(primal_vols_k)} simplices have "
                f"volume below degeneracy_tol={degeneracy_tol}"
            )

    # Compute signed dual volumes and check for degeneracy
    dual_volumes = compute_dual_volumes(simplices, positions)

    for k, dual_vols_k in sorted(dual_volumes.items()):
        degenerate_mask = np.abs(dual_vols_k) < degeneracy_tol
        if np.any(degenerate_mask):
            n_degenerate = int(np.sum(degenerate_mask))
            raise ValueError(
                f"Degenerate dual volumes at degree k={k}: "
                f"{n_degenerate} of {len(dual_vols_k)} simplices have "
                f"absolute dual volume below degeneracy_tol={degeneracy_tol}"
            )

    # Assemble raw Hodge star diagonals
    raw_diagonals = {
        k: dual_volumes[k] / primal_volumes[k]
        for k in sorted(primal_volumes)
    }

    # --- Compute diagnostics from raw values ---

    per_degree = {}
    total_negative = 0

    for k in sorted(simplices):
        primal_vols_k = primal_volumes[k]
        dual_vols_k = dual_volumes[k]
        raw_diags_k = raw_diagonals[k]

        num_negative = int(np.sum(raw_diags_k < 0.0))
        total_negative += num_negative

        per_degree[k] = {
            'num_simplices': len(simplices[k]),
            'num_negative': num_negative,
            'primal_volume_range': [
                float(primal_vols_k.min()),
                float(primal_vols_k.max()),
            ],
            'dual_volume_range': [
                float(dual_vols_k.min()),
                float(dual_vols_k.max()),
            ],
            'hodge_star_range': [
                float(raw_diags_k.min()),
                float(raw_diags_k.max()),
            ],
        }

    diagnostics = {
        'positivity': positivity,
        'total_negative': total_negative,
        'per_degree': per_degree,
    }

    # --- Apply positivity handling ---

    if positivity == 'strict':
        if total_negative > 0:
            breakdown = ", ".join(
                f"k={k}: {per_degree[k]['num_negative']}/{per_degree[k]['num_simplices']}"
                for k in sorted(per_degree)
                if per_degree[k]['num_negative'] > 0
            )
            raise ValueError(
                f"Hodge star has {total_negative} negative "
                f"{'entry' if total_negative == 1 else 'entries'} "
                f"(positivity='strict'). Per-degree: {breakdown}"
            )
        diagonals = raw_diagonals

    elif positivity == 'abs':
        diagonals = {
            k: np.abs(raw_diagonals[k])
            for k in sorted(raw_diagonals)
        }

    # --- Construct diagonal metric tensors ---

    metric_tensors = {
        k: construct_diagonal_metric(diagonals[k], tol=positivity_tol)
        for k in sorted(diagonals)
    }

    return metric_tensors, diagnostics


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
            value_str = value[:3].lower().replace('_', '')
        else:
            value_str = str(value).replace('.', 'p')

        parts.append(f"{param_abbr}{value_str}")

    return '_'.join(parts)