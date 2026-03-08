"""
Eigendecomposition of self-adjoint operators.

Provides pure eigensolver methods for computing eigenvalues and
eigenvectors of self-adjoint and PSD operators with respect to a 
metric. Handles symmetrization, solver dispatch, eigenvalue clamping, 
and eigenvector back-transformation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from holspec.hodge_laplacian.operators import symmetrize_matrix

from .validation import ZERO_EIGENVALUE_TOL

if TYPE_CHECKING:
    from holspec.cochain_metric import MetricTensor


# =============================================================================
# Constants
# =============================================================================

# Recognized solver names for eigendecomposition dispatch.
VALID_SOLVER_NAMES = ('dense', 'sparse')


# =============================================================================
# Eigendecomposition
# =============================================================================

def compute_eigendecomposition(
    matrix: sparse.spmatrix,
    metric: MetricTensor,
    solver: str = 'dense',
    compute_eigenvectors: bool = False,
    tol: float = ZERO_EIGENVALUE_TOL,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute eigendecomposition of a metric-self-adjoint and -PSD operator.

    Given a sparse matrix M that is self-adjoint with respect to the inner
    product defined by metric tensor G, computes the eigendecomposition by
    first transforming to the standard symmetric problem via

        M_sym = G^{1/2} M G^{-1/2}

    and then applying a standard symmetric eigensolver. Eigenvectors are back-
    transformed via phi_i = G^{-1/2} phi_sym_i before returning, if requested.

    Parameters
    ----------
    matrix : sparse.spmatrix, shape (N, N)
        Self-adjoint positive semidefinite operator matrix (with respect to
        the metric).
    metric : MetricTensor, size N
        Metric tensor defining the inner product with respect to which
        the matrix is self-adjoint.
    solver : {'dense', 'sparse'}, default='dense'
        Eigensolver backend. 'dense' converts to a dense matrix and uses
        numpy.linalg.eigvalsh or numpy.linalg.eigh. 'sparse' is
        accepted but raises NotImplementedError.
    compute_eigenvectors : bool, default=False
        Whether to compute eigenvectors in addition to eigenvalues.
    tol : float, default=ZERO_EIGENVALUE_TOL
        Tolerance for clamping near-zero eigenvalues. Values with
        abs(value) < tol are set to 0.0.

    Returns
    -------
    eigenvalues : ndarray, shape (N,)
        Eigenvalues sorted ascending, non-negative after clamping.
    eigenvectors : ndarray or None
        If compute_eigenvectors is True, an (N, N) array whose columns
        are eigenvectors in the original basis, ordered to match the
        eigenvalues. None otherwise.

    Raises
    ------
    ValueError
        If solver is not a recognized name.
    NotImplementedError
        If solver is 'sparse'.

    Notes
    -----
    - Symmetrization is always performed regardless of metric. For the
      identity metric, it is a no-op. This avoids branching and avoids
      the non-symmetric eigensolver (numpy.linalg.eig), which would
      require additional post-processing for sorting and complex-part
      tolerance checks.
    - Clamping at this level is the primary numerical cleanup. The
      Spectrum constructor applies a second round of clamping as a
      safety net for data from external sources; for output of this
      function, the constructor clamp is effectively a no-op.
    - For N=0, returns an empty eigenvalue array and either None or an
      empty (0, 0) eigenvector array depending on compute_eigenvectors.
    """
    # --- Validate solver ---
    if solver not in VALID_SOLVER_NAMES:
        raise ValueError(
            f"Unknown solver '{solver}'. "
            f"Valid solvers: {VALID_SOLVER_NAMES}."
        )

    N = matrix.shape[0]

    # --- N=0 early return ---
    if N == 0:
        eigenvalues = np.array([], dtype=np.float64)
        eigenvectors = np.empty((0, 0), dtype=np.float64) if compute_eigenvectors else None
        return eigenvalues, eigenvectors

    # --- Symmetrize: M_sym = G^{1/2} M G^{-1/2} ---
    M_sym = symmetrize_matrix(matrix, metric)

    # --- Solver dispatch ---
    if solver == 'dense':
        M_dense = M_sym.toarray()
        if compute_eigenvectors:
            eigenvalues, eigenvectors_sym = np.linalg.eigh(M_dense)
        else:
            eigenvalues = np.linalg.eigvalsh(M_dense)
            eigenvectors_sym = None

    elif solver == 'sparse':
        raise NotImplementedError(
            "Sparse eigendecomposition is not yet implemented."
        )

    # --- Clamp near-zero eigenvalues ---
    eigenvalues[np.abs(eigenvalues) < tol] = 0.0

    # --- Back-transform eigenvectors: phi_i = G^{-1/2} phi_sym_i ---
    if compute_eigenvectors:
        G_inv_sqrt = metric.to_matrix_power(-0.5)
        eigenvectors = G_inv_sqrt @ eigenvectors_sym
        # Ensure dense ndarray (sparse @ dense returns ndarray, but be explicit)
        eigenvectors = np.asarray(eigenvectors)
    else:
        eigenvectors = None

    return eigenvalues, eigenvectors