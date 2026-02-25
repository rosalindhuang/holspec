"""
Metric tensor class.

Provides the MetricTensor class representing a metric tensor (inner product
matrix) on a finite-dimensional real vector space.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .validation import validate_metric_tensor


# =============================================================================
# MetricTensor
# =============================================================================

class MetricTensor:
    """
    Metric tensor (inner product matrix) on a finite-dimensional real vector space.

    A lightweight container enforcing the SPD mathematical contract on a sparse
    matrix. Provides linear algebra operations for computing the action of the
    metric and its inverse. 

    Parameters
    ----------
    matrix : sparse.spmatrix
        Square SPD sparse matrix of shape (N, N).
    is_diagonal : bool, default=False
        Whether the matrix is diagonal. Set by construction functions, which
        have the context to know the structure of what they built. When True,
        enables efficient elementwise operations via a cached diagonal array.
        Non-diagonal case raises NotImplementedError in apply and apply_inverse.

    Notes
    -----
    - Calls validate_metric_tensor at construction; raises on invalid input.
    - The matrix is stored internally as CSR for format consistency.
    - For diagonal metrics, the diagonal is extracted and cached at construction
      for use in apply and apply_inverse (O(N) cost, paid once).
    - Non-diagonal support is deferred. Extension path: lazy sparse Cholesky
      factorization for apply_inverse; full symmetry and Cholesky-based PD check
      in validate_metric_tensor.
    - In the cochain metric context, N = N_k = dim(C^k) and the metric tensor
      represents the inner product G^k on the k-cochain space C^k.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self, 
        matrix: sparse.spmatrix, 
        is_diagonal: bool = False
    ):  
        # Validate provided matrix
        validate_metric_tensor(matrix, is_diagonal)

        self._matrix = matrix.tocsr()
        self._is_diagonal = is_diagonal
        
        # Cache diagonal array eagerly
        self._diagonal: np.ndarray | None = (
            np.asarray(self._matrix.diagonal()) if is_diagonal else None
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Dimension N of the vector space."""
        return self._matrix.shape[0]

    @property
    def is_diagonal(self) -> bool:
        """Whether the metric tensor is diagonal."""
        return self._is_diagonal

    @property
    def matrix(self) -> sparse.csr_matrix:
        """Underlying sparse matrix (read-only; do not modify in place)."""
        return self._matrix

    # =========================================================================
    # Linear Algebra Methods
    # =========================================================================

    def apply(self, v: np.ndarray) -> np.ndarray:
        """
        Compute G v.

        Parameters
        ----------
        v : ndarray, shape (N,) or (N, m)
            Vector or matrix to multiply.

        Returns
        -------
        ndarray, same shape as v
        """
        v = np.asarray(v)
        if v.shape[0] != self.size:
            raise ValueError(
                f"Vector length {v.shape[0]} does not match metric tensor size {self.size}"
            )
        if self._is_diagonal:
            # Reshape diagonal to (N, 1, 1, ...) to broadcast correctly
            # against v of any number of trailing dimensions.
            d = self._diagonal.reshape(self.size, *([1] * (v.ndim - 1)))
            return d * v
        raise NotImplementedError(
            "apply is not yet implemented for non-diagonal metric tensors."
        )

    def apply_inverse(self, v: np.ndarray) -> np.ndarray:
        """
        Compute G^{-1} v.

        Parameters
        ----------
        v : ndarray, shape (N,) or (N, m)
            Vector or matrix to solve against.

        Returns
        -------
        ndarray, same shape as v
        """
        v = np.asarray(v)
        if v.shape[0] != self.size:
            raise ValueError(
                f"Vector length {v.shape[0]} does not match metric tensor size {self.size}"
            )
        if self._is_diagonal:
            d = self._diagonal.reshape(self.size, *([1] * (v.ndim - 1)))
            return v / d
        raise NotImplementedError(
            "apply_inverse is not yet implemented for non-diagonal metric tensors. "
            "Extension path: lazy sparse Cholesky factorization."
        )

    def to_matrix(self) -> sparse.csr_matrix:
        """Return the metric tensor as an explicit sparse matrix."""
        return self._matrix

    # =========================================================================
    # Utilities
    # =========================================================================

    def __repr__(self) -> str:
        return f"MetricTensor(size={self.size}, is_diagonal={self._is_diagonal})"

