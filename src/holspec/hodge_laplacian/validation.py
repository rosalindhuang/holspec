"""
Hodge Laplacian validation.

Standalone functions for validating Hodge Laplacian inputs (cross-compatibility
of SimplicialComplex and CochainMetric) and mathematical properties of computed
Laplacian matrices.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from holspec.simplicial import SimplicialComplex
    from holspec.cochain_metric import CochainMetric, MetricTensor


# =============================================================================
# Validation Functions
# =============================================================================

def validate_hodge_laplacian_inputs(
    sc: SimplicialComplex,
    cm: CochainMetric,
) -> None:
    """
    Validate cross-compatibility of a SimplicialComplex and CochainMetric.

    Called by the HodgeLaplacian constructor to ensure its two inputs are
    consistent before storing references for lazy computation.

    Parameters
    ----------
    sc : SimplicialComplex
        Source simplicial complex.
    cm : CochainMetric
        Cochain metric to validate against sc.

    Raises
    ------
    ValueError
        If max_dim disagrees or cochain dimensions do not match simplex
        counts at any degree.

    Notes
    -----
    Checks performed:

    - Maximum dimension: sc.max_dim == cm.max_dim.
    - Cochain dimensions: cm.dimensions[k] == sc.num_simplices[k] for
      all k = 0, ..., n. The metric tensor G^k must have size N_k equal
      to the number of k-simplices, since it acts on the cochain space
      C^k whose dimension is N_k.

    Each object is assumed to be internally valid (SimplicialComplex
    validates its simplicial structure; CochainMetric validates consecutive
    degrees and per-degree SPD contracts). This function checks only
    cross-compatibility.
    """
    # --- Maximum dimension ---
    if sc.max_dim != cm.max_dim:
        raise ValueError(
            f"max_dim mismatch: SimplicialComplex has max_dim={sc.max_dim}, "
            f"CochainMetric has max_dim={cm.max_dim}"
        )

    # --- Cochain dimensions ---
    sc_dimensions = sc.num_simplices
    cm_dimensions = cm.dimensions

    mismatches = []
    for k in range(sc.max_dim + 1):
        sc_nk = sc_dimensions[k]
        cm_nk = cm_dimensions[k]
        if sc_nk != cm_nk:
            mismatches.append((k, cm_nk, sc_nk))

    if mismatches:
        detail = "\n  ".join(
            f"k={k}: CochainMetric has size {cm_nk} "
            f"(expected {sc_nk} from SimplicialComplex)"
            for k, cm_nk, sc_nk in mismatches
        )
        raise ValueError(
            f"Cochain dimension mismatch at "
            f"{'degree' if len(mismatches) == 1 else 'degrees'}:\n  {detail}"
        )


def validate_laplacian_properties(
    laplacian_matrix: sparse.csr_matrix,
    metric_k: MetricTensor,
    tol: float = 1e-10,
) -> dict[str, bool]:
    pass