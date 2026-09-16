"""
Spectrum of a single operator.

Provides the Spectrum class, a lightweight container enforcing the sorted
non-negative eigenvalue contract and providing per-spectrum observable
methods.
"""

from __future__ import annotations

import numpy as np

from .validation import ZERO_EIGENVALUE_TOL, validate_spectrum


# =============================================================================
# Spectrum
# =============================================================================


class Spectrum:
    """
    Spectrum of a linear operator.

    A lightweight container storing eigenvalues (and optionally eigenvectors)
    of a self-adjoint positive semidefinite operator. Enforces a sorted
    non-negative eigenvalue contract at construction and provides methods for
    computing derived spectral quantities (observables).

    Parameters
    ----------
    eigenvalues : ndarray
        1D array of eigenvalues, sorted ascending. Values with absolute magnitude
        below tol are clamped to zero; values below -tol are rejected.
    dimension : int
        Total dimension of the vector space. Required because partial
        spectra (from sparse eigensolvers) cannot infer this from the
        eigenvalue array length.
    eigenvectors : ndarray, optional
        Eigenvector matrix of shape (dimension, num_eigenvalues), where
        columns are eigenvectors corresponding to the eigenvalues. If
        None, eigenvector-dependent methods are unavailable.
    tol : float, default=ZERO_EIGENVALUE_TOL
        Tolerance for eigenvalue validation and clamping. Eigenvalues
        below -tol are rejected; values with abs(value) < tol are clamped
        to zero.

    Notes
    -----
    - Calls validate_spectrum at construction; raises on invalid input.
    - Input arrays are copied to float64 to prevent external mutation.
    - The tol parameter controls construction-time processing only
      and is not stored on the instance. The stored eigenvalue array is
      the source of truth for all subsequent operations.
    - Size-0 spectra (dimension=0, empty eigenvalues) are permitted.
    - In the Hodge Laplacian context, stores the eigendecomposition of a
      single Laplacian component at one degree, where eigenvalues are real
      and non-negative by the self-adjointness and PSD properties.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        eigenvalues: np.ndarray,
        dimension: int,
        eigenvectors: np.ndarray | None = None,
        tol: float = ZERO_EIGENVALUE_TOL,
    ):
        eigenvalues = np.asarray(eigenvalues)
        validate_spectrum(eigenvalues, dimension, eigenvectors, tol)

        # Copy to float64 and clamp near-zero values
        self._eigenvalues = np.array(eigenvalues, dtype=np.float64)
        self._eigenvalues[np.abs(self._eigenvalues) < tol] = 0.0

        self._dimension = int(dimension)

        self._eigenvectors: np.ndarray | None = (
            np.array(eigenvectors, dtype=np.float64)
            if eigenvectors is not None
            else None
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues, shape (num_eigenvalues,), sorted ascending, non-negative."""
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray | None:
        """Eigenvectors, shape (dimension, num_eigenvalues), or None."""
        return self._eigenvectors

    @property
    def dimension(self) -> int:
        """Total dimension of the vector space."""
        return self._dimension

    @property
    def num_eigenvalues(self) -> int:
        """Number of stored eigenvalues."""
        return len(self._eigenvalues)

    @property
    def is_complete(self) -> bool:
        """Whether all eigenvalues are stored (num_eigenvalues == dimension)."""
        return self.num_eigenvalues == self._dimension

    # =========================================================================
    # Observable Methods
    # =========================================================================

    def dim_ker(self, tol: float = ZERO_EIGENVALUE_TOL) -> int:
        """
        Dimension of the kernel (number of zero eigenvalues).

        Parameters
        ----------
        tol : float, default=ZERO_EIGENVALUE_TOL
            Eigenvalues <= tol are counted as zero.

        Returns
        -------
        int
        """
        # Eigenvalues are sorted ascending, so searchsorted gives the
        # count of values <= tol in O(log n).
        return int(np.searchsorted(self._eigenvalues, tol, side="right"))

    def nonzero_eigenvalues(self, tol: float = ZERO_EIGENVALUE_TOL) -> np.ndarray:
        """
        Return eigenvalues above the zero threshold.

        Parameters
        ----------
        tol : float, default=ZERO_EIGENVALUE_TOL
            Eigenvalues > tol are considered nonzero.

        Returns
        -------
        ndarray
            Nonzero eigenvalues, sorted ascending. Empty array if all
            eigenvalues are at or below tol.
        """
        return self._eigenvalues[self._eigenvalues > tol]

    def nonzero_eigenvectors(self, tol: float = ZERO_EIGENVALUE_TOL) -> np.ndarray:
        """
        Return eigenvectors corresponding to nonzero eigenvalues.

        Applies the same mask as nonzero_eigenvalues: columns of the
        eigenvector matrix whose eigenvalue exceeds tol.

        Parameters
        ----------
        tol : float, default=ZERO_EIGENVALUE_TOL
            Eigenvalues > tol are considered nonzero.

        Returns
        -------
        ndarray, shape (dimension, num_nonzero)
            Eigenvector columns for nonzero eigenvalues.

        Raises
        ------
        ValueError
            If eigenvectors were not provided at construction.
        """
        if self._eigenvectors is None:
            raise ValueError("Eigenvectors were not provided at construction.")
        mask = self._eigenvalues > tol
        return self._eigenvectors[:, mask]

    def moment(
        self,
        p: float,
        normalized: bool = True,
        nonzero: bool = False,
        tol: float = ZERO_EIGENVALUE_TOL,
    ) -> float:
        """
        Spectral moment of the eigenvalue distribution.

        Evaluates the sum of eigenvalue powers sum(lambda_i^p), optionally
        restricted to nonzero eigenvalues and optionally normalized by count.

        Parameters
        ----------
        p : float
            Exponent applied to each eigenvalue.
        normalized : bool, default=True
            If True, divide the sum by the number of eigenvalues in the
            selected set.
        nonzero : bool, default=False
            If True, restrict to eigenvalues > tol. When normalized, the
            divisor is the nonzero count rather than the total count.
        tol : float, default=ZERO_EIGENVALUE_TOL
            Threshold for nonzero selection when nonzero=True.

        Returns
        -------
        float
            The spectral moment. Returns 0.0 if the selected eigenvalue
            set is empty.

        Notes
        -----
        When nonzero=False and p < 0, zero eigenvalues produce inf terms
        (0^p = inf for p < 0). Use nonzero=True to restrict to the
        positive part of the spectrum for negative exponents.
        """
        selected = self.nonzero_eigenvalues(tol) if nonzero else self._eigenvalues
        n = len(selected)
        if n == 0:
            return 0.0

        total = float(np.sum(selected**p))
        return total / n if normalized else total

    def heat_trace(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the heat trace over all stored eigenvalues.

        Evaluates Z(t) = sum_i exp(-t * lambda_i), where the sum runs
        over all stored eigenvalues (including zeros, which contribute
        exp(0) = 1 each).

        Parameters
        ----------
        t : float or ndarray
            Time parameter(s). Must be non-negative.

        Returns
        -------
        float or ndarray
            Scalar t returns float; array t returns ndarray of matching
            shape.

        Notes
        -----
        To obtain the heat trace restricted to the nonzero spectrum,
        subtract dim_ker: ``spectrum.heat_trace(t) - spectrum.dim_ker()``.
        """
        t = np.asarray(t, dtype=np.float64)
        scalar_input = t.ndim == 0

        # t[..., newaxis] broadcasts against eigenvalues: (..., num_eigenvalues)
        # sum over last axis collapses to (...)
        result = np.exp(-t[..., np.newaxis] * self._eigenvalues).sum(axis=-1)

        if scalar_input:
            return float(result)
        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"Spectrum(dimension={self._dimension}, "
            f"num_eigenvalues={self.num_eigenvalues}, "
            f"is_complete={self.is_complete}, "
            f"has_eigenvectors={self._eigenvectors is not None})"
        )
