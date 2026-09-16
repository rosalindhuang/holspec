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
from scipy.sparse.linalg import eigsh

from holspec.hodge_laplacian.operators import symmetrize_matrix

from .validation import ZERO_EIGENVALUE_TOL

if TYPE_CHECKING:
    from holspec.cochain_metric import MetricTensor


# =============================================================================
# Constants
# =============================================================================

# Recognized solver names for eigendecomposition dispatch.
VALID_SOLVER_NAMES = ("dense", "sparse")

# Recognized keys for sparse solver parameters.
VALID_SPARSE_SOLVER_PARAMS_KEYS = ("num_eigenvalues", "which", "sigma")

# Recognized values for the eigsh 'which' parameter.
VALID_EIGSH_WHICH = ("LM", "SM", "LA", "SA", "BE")


# =============================================================================
# Solver Parameter Validation
# =============================================================================


def validate_solver_params(
    solver_params: dict | None,
) -> dict:
    """
    Validate and normalize sparse solver parameters.

    Checks that all keys are recognized, required keys are present, and
    values have correct types. Returns a new dict with defaults filled
    in for optional keys.

    Parameters
    ----------
    solver_params : dict or None
        Sparse solver parameters. Must contain 'num_eigenvalues' (int > 0).
        Optional keys: 'which' (default 'SM'), 'sigma' (default None).

    Returns
    -------
    dict
        Normalized parameters with defaults applied:
        ``{'num_eigenvalues': int, 'which': str, 'sigma': float or None}``.

    Raises
    ------
    ValueError
        If solver_params is None or empty, contains unknown keys,
        or has invalid values.
    """
    if not solver_params:
        raise ValueError(
            "solver_params is required when solver='sparse'. "
            "Must contain at least 'num_eigenvalues'."
        )

    # Check for unknown keys
    unknown_keys = set(solver_params) - set(VALID_SPARSE_SOLVER_PARAMS_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown solver_params keys: {sorted(unknown_keys)}. "
            f"Valid keys: {VALID_SPARSE_SOLVER_PARAMS_KEYS}."
        )

    # num_eigenvalues: required, int > 0
    if "num_eigenvalues" not in solver_params:
        raise ValueError("solver_params must contain 'num_eigenvalues'.")
    num_eigenvalues = solver_params["num_eigenvalues"]
    if not isinstance(num_eigenvalues, (int, np.integer)) or num_eigenvalues <= 0:
        raise ValueError(
            f"num_eigenvalues must be a positive integer, got {num_eigenvalues!r}."
        )

    # which: optional, default 'SM'
    which = solver_params.get("which", "SM")
    if which not in VALID_EIGSH_WHICH:
        raise ValueError(f"Invalid which='{which}'. Valid values: {VALID_EIGSH_WHICH}.")

    # sigma: optional, default None
    sigma = solver_params.get("sigma", None)
    if sigma is not None and not isinstance(
        sigma, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"sigma must be a number or None, got {type(sigma).__name__}.")

    return {
        "num_eigenvalues": int(num_eigenvalues),
        "which": which,
        "sigma": float(sigma) if sigma is not None else None,
    }


# =============================================================================
# Eigendecomposition
# =============================================================================


def compute_eigendecomposition(
    matrix: sparse.spmatrix,
    metric: MetricTensor,
    solver: str = "dense",
    solver_params: dict | None = None,
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
        numpy.linalg.eigvalsh or numpy.linalg.eigh. 'sparse' uses
        scipy.sparse.linalg.eigsh to compute a subset of eigenvalues.
    solver_params : dict, optional
        Parameters for the sparse eigensolver. Required when solver='sparse',
        ignored when solver='dense'. Supported keys:

        - ``num_eigenvalues`` (int, required): number of eigenvalues to
          compute. Must be positive. If >= N, falls back to the dense
          solver automatically.
        - ``which`` (str, default 'SM'): which eigenvalues to find.
          One of 'LM', 'SM', 'LA', 'SA', 'BE'.
        - ``sigma`` (float, optional): shift for shift-invert mode.
          When set, ``which`` selects eigenvalues nearest to sigma.
    compute_eigenvectors : bool, default=False
        Whether to compute eigenvectors in addition to eigenvalues.
    tol : float, default=ZERO_EIGENVALUE_TOL
        Tolerance for clamping near-zero eigenvalues. Values with
        abs(value) < tol are set to 0.0.

    Returns
    -------
    eigenvalues : ndarray, shape (num_eigenvalues,)
        Eigenvalues sorted ascending, non-negative after clamping.
        num_eigenvalues equals N for the dense solver, or the requested
        count (possibly N via dense fallback) for the sparse solver.
    eigenvectors : ndarray or None
        If compute_eigenvectors is True, an (N, num_eigenvalues) array
        whose columns are eigenvectors in the original basis, ordered to
        match the eigenvalues. None otherwise.

    Raises
    ------
    ValueError
        If solver is not a recognized name, or if solver_params is
        invalid for the sparse solver.

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
    - For the sparse solver, eigsh does not guarantee sorted output,
      so eigenvalues (and eigenvectors) are sorted ascending after
      computation. When num_eigenvalues >= N, the function falls back
      to the dense solver because eigsh requires k < N.
    """
    # --- Validate solver ---
    if solver not in VALID_SOLVER_NAMES:
        raise ValueError(
            f"Unknown solver '{solver}'. Valid solvers: {VALID_SOLVER_NAMES}."
        )

    N = matrix.shape[0]

    # --- N=0 early return ---
    if N == 0:
        eigenvalues = np.array([], dtype=np.float64)
        eigenvectors = (
            np.empty((0, 0), dtype=np.float64) if compute_eigenvectors else None
        )
        return eigenvalues, eigenvectors

    # --- Symmetrize: M_sym = G^{1/2} M G^{-1/2} ---
    M_sym = symmetrize_matrix(matrix, metric)

    # --- Solver dispatch ---
    if solver == "dense":
        M_dense = M_sym.toarray()
        if compute_eigenvectors:
            eigenvalues, eigenvectors_sym = np.linalg.eigh(M_dense)
        else:
            eigenvalues = np.linalg.eigvalsh(M_dense)
            eigenvectors_sym = None

    elif solver == "sparse":
        solver_params = validate_solver_params(solver_params)
        num_eigenvalues = solver_params["num_eigenvalues"]
        which = solver_params["which"]
        sigma = solver_params["sigma"]

        # Dense fallback when num_eigenvalues >= N (eigsh requires k < N)
        if num_eigenvalues >= N:
            return compute_eigendecomposition(
                matrix,
                metric,
                solver="dense",
                compute_eigenvectors=compute_eigenvectors,
                tol=tol,
            )

        if compute_eigenvectors:
            eigenvalues, eigenvectors_sym = eigsh(
                M_sym,
                k=num_eigenvalues,
                which=which,
                sigma=sigma,
                return_eigenvectors=True,
            )
        else:
            eigenvalues = eigsh(
                M_sym,
                k=num_eigenvalues,
                which=which,
                sigma=sigma,
                return_eigenvectors=False,
            )
            eigenvectors_sym = None

        # eigsh does NOT guarantee sorted output — sort ascending
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        if eigenvectors_sym is not None:
            eigenvectors_sym = eigenvectors_sym[:, sort_idx]

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
