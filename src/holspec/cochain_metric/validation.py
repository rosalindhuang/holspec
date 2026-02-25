"""
Cochain metric validation.

Standalone functions for validating metric tensors and cochain metric
collections against their mathematical contracts (SPD, size consistency).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from .metric_tensor import MetricTensor


# =============================================================================
# Validation Functions
# =============================================================================

def validate_metric_tensor(
    matrix: sparse.spmatrix,
    is_diagonal: bool,
    check_positive_definite: bool = True,
) -> None:
    """
    Validate a sparse matrix against the mathematical contract of a metric tensor.

    Parameters
    ----------
    matrix : sparse.spmatrix
        Candidate metric tensor matrix.
    is_diagonal : bool
        Whether the matrix is diagonal. When True, performs a diagonal
        positivity check. When False, raises NotImplementedError (non-diagonal
        validation is not yet implemented).
    check_positive_definite : bool, default=True
        Whether to check positive definiteness. For the diagonal case, the
        diagonal positivity check is always performed regardless of this flag.
        Reserved for the non-diagonal case (Cholesky-based check), not yet
        implemented.

    Raises
    ------
    TypeError
        If matrix is not a scipy sparse matrix.
    ValueError
        If any structural or mathematical invariant is violated.
    NotImplementedError
        If is_diagonal is False (non-diagonal validation not yet implemented).

    Notes
    -----
    Current checks (diagonal case only):

    - Type: must be a scipy sparse matrix.
    - Shape: must be square and non-empty.
    - Values: all stored entries must be finite (NaN and Inf not allowed).
    - Positive definiteness: all diagonal entries must be strictly positive.
      For a diagonal matrix, this is equivalent to the full SPD contract.

    Non-diagonal case raises NotImplementedError. Extension path: full symmetry
    check and Cholesky-based positive definiteness check (attempting the
    factorization both validates PD and produces the factor needed for
    apply_inverse).
    """
    # --- Type ---
    if not sparse.issparse(matrix):
        raise TypeError(
            f"matrix must be a scipy sparse matrix, got {type(matrix).__name__}"
        )

    # --- Shape ---
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Metric tensor must be a square matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise ValueError("Metric tensor must be non-empty (size > 0)")

    # --- Finite values ---
    # Only stored (non-zero) entries are checked; implicit zeros are always finite.
    if matrix.nnz > 0 and not np.all(np.isfinite(matrix.data)):
        raise ValueError(
            "Metric tensor contains non-finite values (NaN or Inf)"
        )

    # --- Positive definiteness ---
    if is_diagonal:
        diag = matrix.diagonal()
        non_positive = np.sum(diag <= 0)
        if non_positive > 0:
            raise ValueError(
                f"Diagonal metric tensor must have strictly positive diagonal entries: "
                f"{non_positive} non-positive {'entry' if non_positive == 1 else 'entries'} found"
            )
    else:
        raise NotImplementedError(
            "Validation for non-diagonal metric tensors is not yet implemented. "
            "Extension path: full symmetry check and Cholesky-based positive "
            "definiteness check (check_positive_definite=True)."
        )


def validate_cochain_metric(
    cochain_metrics: dict[int, MetricTensor],
    cochain_dimensions: dict[int, int],
) -> None:
    """
    Validate a collection of metric tensors against expected cochain space dimensions.

    Parameters
    ----------
    cochain_metrics : dict[int, MetricTensor]
        Dictionary mapping degree k to MetricTensor at that degree.
    cochain_dimensions : dict[int, int]
        Dictionary mapping degree k to the expected dimension N_k of C^k.

    Raises
    ------
    ValueError
        If any structural invariant is violated.

    Notes
    -----
    Performs the following checks:

    - Degrees are non-empty and form a consecutive sequence 0, 1, ..., n.
    - The set of degrees in cochain_metrics matches that in cochain_dimensions.
    - For each degree k, MetricTensor.size equals cochain_dimensions[k].
    """
    if not cochain_metrics:
        raise ValueError("cochain_metrics cannot be empty")

    # --- Consecutive degrees starting at 0 ---
    degrees = sorted(cochain_metrics.keys())
    if degrees[0] != 0:
        raise ValueError(
            f"Cochain metric degrees must start at 0, got min degree {degrees[0]}"
        )
    if degrees != list(range(degrees[-1] + 1)):
        missing = sorted(set(range(degrees[-1] + 1)) - set(degrees))
        raise ValueError(
            f"Cochain metric degrees must be consecutive integers 0, ..., n. "
            f"Missing degrees: {missing}"
        )

    # --- Degree sets match ---
    dim_degrees = sorted(cochain_dimensions.keys())
    if degrees != dim_degrees:
        raise ValueError(
            f"Degree mismatch between cochain_metrics {degrees} "
            f"and provided cochain_dimensions {dim_degrees}"
        )

    # --- Size agreement at each degree ---
    mismatches = []
    for k in degrees:
        actual = cochain_metrics[k].size
        expected = cochain_dimensions[k]
        if actual != expected:
            mismatches.append((k, actual, expected))

    if mismatches:
        detail = ", ".join(
            f"k={k}: size={actual} (expected {expected})"
            for k, actual, expected in mismatches
        )
        raise ValueError(
            f"Metric tensor size mismatch at "
            f"{'degree' if len(mismatches) == 1 else 'degrees'}: {detail}"
        )