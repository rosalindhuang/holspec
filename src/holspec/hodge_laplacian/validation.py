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
# Constants
# =============================================================================

# Numerical tolerance for Laplacian property checks (self-adjointness of GL
# and positive semidefiniteness via eigenvalue threshold).
LAPLACIAN_PROPERTY_TOL = 1e-10


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
    laplacian_matrix: sparse.spmatrix,
    metric: MetricTensor,
    tol: float = LAPLACIAN_PROPERTY_TOL,
) -> None:
    """
    Validate mathematical properties of a Hodge Laplacian matrix.

    Checks that a computed Laplacian matrix satisfies the expected
    mathematical properties: square, self-adjoint w.r.t. the metric,
    and positive semidefinite.

    Parameters
    ----------
    laplacian_matrix : sparse.spmatrix, shape (N, N)
        A Laplacian matrix (full, lower, or upper component).
    metric : MetricTensor, size N
        Metric tensor G defining the inner product with respect to which
        the Laplacian should be self-adjoint.
    tol : float, default=LAPLACIAN_PROPERTY_TOL
        Numerical tolerance for approximate checks.

    Raises
    ------
    ValueError
        If any property is violated. The error message identifies all
        failing properties and their diagnostic values.

    Notes
    -----
    The positive semidefiniteness check uses dense eigendecomposition,
    requiring O(N^2) memory. This is acceptable for the diagnostic use
    case but impractical for very large matrices.

    For N=0 (boundary-degree Laplacians), all properties are vacuously
    satisfied.
    """
    N = laplacian_matrix.shape[0]

    # --- Vacuous case (N=0) ---
    if N == 0:
        return

    # --- Square ---
    if laplacian_matrix.shape[0] != laplacian_matrix.shape[1]:
        raise ValueError(
            f"Laplacian property violation: is_square: shape={laplacian_matrix.shape}"
        )

    if N != metric.size:
        raise ValueError(
            f"Laplacian property violation: "
            f"matrix size N={N} does not match metric size={metric.size}"
        )

    failures = []

    # --- Self-adjoint w.r.t. G: G L should be symmetric ---
    GL = metric.to_matrix() @ laplacian_matrix
    GL_dense = GL.toarray()
    if not np.allclose(GL_dense, GL_dense.T, atol=tol, rtol=0):
        max_deviation = np.max(np.abs(GL_dense - GL_dense.T))
        failures.append(
            f"is_self_adjoint: max |G L - (G L)^T| = {max_deviation:.2e} (tol={tol})"
        )

    # --- Positive semidefinite via symmetrized form ---
    L_sym = (
        metric.to_matrix_power(0.5) @ laplacian_matrix @ metric.to_matrix_power(-0.5)
    )
    eigenvalues = np.linalg.eigvalsh(L_sym.toarray())
    if not np.all(eigenvalues >= -tol):
        min_eigenvalue = eigenvalues.min()
        failures.append(
            f"is_positive_semidefinite: min eigenvalue = {min_eigenvalue:.2e} "
            f"(tol={tol})"
        )

    if failures:
        if len(failures) == 1:
            raise ValueError(f"Laplacian property violation: {failures[0]}")
        detail = "\n  ".join(failures)
        raise ValueError(f"Laplacian property violations:\n  {detail}")
