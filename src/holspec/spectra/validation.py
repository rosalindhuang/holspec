"""
Spectrum validation and numerical tolerances.

Provides input validation for Spectrum construction and the module-level
zero-eigenvalue tolerance constant used throughout the spectra subpackage.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Numerical tolerance for zero-eigenvalue detection. Values within this
# distance of zero (by absolute magnitude) are treated as numerical noise.
ZERO_EIGENVALUE_TOL = 1e-10


# =============================================================================
# Validation Functions
# =============================================================================


def validate_spectrum(
    eigenvalues: np.ndarray,
    dimension: int,
    eigenvectors: np.ndarray | None = None,
    tol: float = ZERO_EIGENVALUE_TOL,
) -> None:
    """
    Validate inputs for Spectrum construction.

    Called by the Spectrum constructor before clamping. Checks structural
    and numerical invariants on the raw input arrays.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalue array to validate. Must be 1D, sorted ascending, and
        contain no values below -tol.
    dimension : int
        Total dimension of the vector space. Must be a non-negative
        integer, at least as large as len(eigenvalues).
    eigenvectors : ndarray, optional
        Eigenvector matrix to validate. If provided, must have shape
        (dimension, len(eigenvalues)).
    tol : float, default=ZERO_EIGENVALUE_TOL
        Tolerance for rejecting negative eigenvalues. Eigenvalues
        below -tol are considered genuinely out of range.

    Raises
    ------
    TypeError
        If dimension is not an integer.
    ValueError
        If any structural or numerical check fails.

    Notes
    -----
    Checks performed (in order):

    - eigenvalues is 1D.
    - dimension is a non-negative integer.
    - len(eigenvalues) <= dimension.
    - eigenvalues are sorted in ascending order.
    - No eigenvalue is below -tol.
    - If eigenvectors is not None: shape is (dimension, len(eigenvalues)).

    Size-0 spectra (dimension=0, empty eigenvalues) pass all checks.

    Validation runs on the raw input before the constructor applies
    clamping. If the caller passes unsorted data, that is a genuine
    error regardless of whether clamping would fix the ordering.
    """
    eigenvalues = np.asarray(eigenvalues)
    n = len(eigenvalues)

    # --- eigenvalues is 1D ---
    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must be 1D, got shape {eigenvalues.shape}")

    # --- dimension is a non-negative integer ---
    if not isinstance(dimension, (int, np.integer)):
        raise TypeError(f"dimension must be an integer, got {type(dimension).__name__}")
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")

    # --- len(eigenvalues) <= dimension ---
    if n > dimension:
        raise ValueError(
            f"More eigenvalues ({n}) than vector space dimension ({dimension})"
        )

    # --- eigenvalues are sorted ascending ---
    if n > 1 and not np.all(np.diff(eigenvalues) >= 0):
        raise ValueError("eigenvalues must be sorted in ascending order")

    # --- No eigenvalue below -tol ---
    if n > 0 and np.any(eigenvalues < -tol):
        min_val = float(eigenvalues[0])
        raise ValueError(
            f"Eigenvalue {min_val:.2e} is below -tol ({-tol:.2e}). "
            f"This indicates a genuine error, not numerical noise."
        )

    # --- eigenvectors shape ---
    if eigenvectors is not None:
        eigenvectors = np.asarray(eigenvectors)
        expected_shape = (dimension, n)
        if eigenvectors.shape != expected_shape:
            raise ValueError(
                f"eigenvectors shape {eigenvectors.shape} does not match "
                f"expected shape {expected_shape} = (dimension, num_eigenvalues)"
            )
