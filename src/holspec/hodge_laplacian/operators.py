"""
Discrete differential operator matrix computations.

Pure functions for computing coboundary, dual coboundary, and Hodge Laplacian
matrices from incidence matrices and metric tensors. Decoupled from the 
HodgeLaplacian container class for independent testability.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from holspec.cochain_metric import MetricTensor


# =============================================================================
# Boundary and Dual Coboundary
# =============================================================================

def compute_coboundary_matrix(
    incidence_kp1: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Compute the coboundary matrix d^k: C^k -> C^{k+1}.

    The coboundary (discrete exterior derivative) is purely topological:

        d^k = D_{k+1}^T

    It is determined entirely by the oriented incidence structure with 
    no metric dependence.

    Parameters
    ----------
    incidence_kp1 : sparse.spmatrix, shape (N_k, N_{k+1})
        Incidence (boundary operator) matrix D_{k+1}.

    Returns
    -------
    sparse.csr_matrix, shape (N_{k+1}, N_k)
    """
    if not sparse.issparse(incidence_kp1):
        raise TypeError(
            f"incidence_kp1 must be a sparse matrix, "
            f"got {type(incidence_kp1).__name__}"
        )
    return incidence_kp1.T.tocsr()


def compute_dual_coboundary_matrix(
    incidence_k: sparse.spmatrix,
    metric_km1: MetricTensor,
    metric_k: MetricTensor,
) -> sparse.csr_matrix:
    """
    Compute the dual coboundary (codifferential) matrix δ^k: C^k -> C^{k-1}.

    The dual coboundary is the formal adjoint of d^{k-1} with respect to
    the G^{k-1} and G^k inner products:

        δ^k = (G^{k-1})^{-1} D_k G^k

    Parameters
    ----------
    incidence_k : sparse.spmatrix, shape (N_{k-1}, N_k)
        Incidence (boundary operator) matrix D_k.
    metric_km1 : MetricTensor, size N_{k-1}
        Metric tensor G^{k-1} on C^{k-1}.
    metric_k : MetricTensor, size N_k
        Metric tensor G^k on C^k.

    Returns
    -------
    sparse.csr_matrix, shape (N_{k-1}, N_k)

    Raises
    ------
    TypeError
        If incidence_k is not a sparse matrix.
    ValueError
        If matrix dimensions are inconsistent.
    NotImplementedError
        If either metric tensor is non-diagonal.
    """
    if not sparse.issparse(incidence_k):
        raise TypeError(
            f"incidence_k must be a sparse matrix, "
            f"got {type(incidence_k).__name__}"
        )

    # --- Shape validation ---
    n_km1, n_k = incidence_k.shape
    if metric_km1.size != n_km1:
        raise ValueError(
            f"metric_km1 size {metric_km1.size} does not match "
            f"incidence_k row count {n_km1}"
        )
    if metric_k.size != n_k:
        raise ValueError(
            f"metric_k size {metric_k.size} does not match "
            f"incidence_k column count {n_k}"
        )

    # --- Compute δ^k = (G^{k-1})^{-1} D_k G^k ---
    result = metric_km1.to_inverse_matrix() @ incidence_k @ metric_k.to_matrix()

    return result.tocsr()


# =============================================================================
# Hodge Laplacians
# =============================================================================

def compute_laplacian_lower_matrix(
    incidence_k: sparse.spmatrix,
    metric_km1: MetricTensor,
    metric_k: MetricTensor,
) -> sparse.csr_matrix:
    """
    Compute the lower Hodge Laplacian matrix L^{k,low}: C^k -> C^k.

    The lower component captures the coupling between k-cochains and
    (k-1)-cochains, composed from the coboundary and dual coboundary:

        L^{k,low} = d^{k-1} δ^k = D_k^T (G^{k-1})^{-1} D_k G^k

    Parameters
    ----------
    incidence_k : sparse.spmatrix, shape (N_{k-1}, N_k)
        Incidence (boundary operator) matrix D_k.
    metric_km1 : MetricTensor, size N_{k-1}
        Metric tensor G^{k-1} on C^{k-1}.
    metric_k : MetricTensor, size N_k
        Metric tensor G^k on C^k.

    Returns
    -------
    sparse.csr_matrix, shape (N_k, N_k)

    Raises
    ------
    TypeError
        If incidence_k is not a sparse matrix.
    ValueError
        If matrix dimensions are inconsistent.
    NotImplementedError
        If either metric tensor is non-diagonal.
    """
    # L^{k,low} = d^{k-1} delta^k = D_k^T (G^{k-1})^{-1} D_k G^k
    d_km1 = compute_coboundary_matrix(incidence_k)           # D_k^T
    delta_k = compute_dual_coboundary_matrix(                # (G^{k-1})^{-1} D_k G^k
        incidence_k, metric_km1, metric_k
    )
    return (d_km1 @ delta_k).tocsr()


def compute_laplacian_upper_matrix(
    incidence_kp1: sparse.spmatrix,
    metric_k: MetricTensor,
    metric_kp1: MetricTensor,
) -> sparse.csr_matrix:
    """
    Compute the upper Hodge Laplacian matrix L^{k,upp}: C^k -> C^k.

    The upper component captures the coupling between k-cochains and
    (k+1)-cochains, composed from the dual coboundary and coboundary:

        L^{k,upp} = δ^{k+1} d^k = (G^k)^{-1} D_{k+1} G^{k+1} D_{k+1}^T

    Parameters
    ----------
    incidence_kp1 : sparse.spmatrix, shape (N_k, N_{k+1})
        Incidence (boundary operator) matrix D_{k+1}.
    metric_k : MetricTensor, size N_k
        Metric tensor G^k on C^k.
    metric_kp1 : MetricTensor, size N_{k+1}
        Metric tensor G^{k+1} on C^{k+1}.

    Returns
    -------
    sparse.csr_matrix, shape (N_k, N_k)

    Raises
    ------
    TypeError
        If incidence_kp1 is not a sparse matrix.
    ValueError
        If matrix dimensions are inconsistent.
    NotImplementedError
        If either metric tensor is non-diagonal.
    """
    # L^{k,upp} = delta^{k+1} d^k = (G^k)^{-1} D_{k+1} G^{k+1} D_{k+1}^T
    delta_kp1 = compute_dual_coboundary_matrix(               # (G^k)^{-1} D_{k+1} G^{k+1}
        incidence_kp1, metric_k, metric_kp1
    )
    d_k = compute_coboundary_matrix(incidence_kp1)            # D_{k+1}^T
    return (delta_kp1 @ d_k).tocsr()