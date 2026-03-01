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
# Differential Operator Matrices
# =============================================================================

def compute_coboundary_matrix(
    incidence_kp1: sparse.csr_matrix,
) -> sparse.csr_matrix:
    pass


def compute_dual_coboundary_matrix(
    incidence_k: sparse.csr_matrix,
    metric_km1: MetricTensor,
    metric_k: MetricTensor,
) -> sparse.csr_matrix:
    pass


# =============================================================================
# Laplacian Matrices
# =============================================================================

def compute_laplacian_lower_matrix(
    incidence_k: sparse.csr_matrix,
    metric_km1: MetricTensor,
    metric_k: MetricTensor,
) -> sparse.csr_matrix:
    pass


def compute_laplacian_upper_matrix(
    incidence_kp1: sparse.csr_matrix,
    metric_k: MetricTensor,
    metric_kp1: MetricTensor,
) -> sparse.csr_matrix:
    pass
