"""
Small assertion helpers for numerical tests.

These helpers keep sparse-matrix and eigenvalue comparisons concise in the
mathematical core tests without introducing a large custom testing layer.
"""

import numpy as np


def assert_sparse_allclose(matrix, expected, **kwargs) -> None:
    """Compare a sparse matrix to expected dense values."""
    np.testing.assert_allclose(matrix.toarray(), expected, **kwargs)


def assert_eigenvalues_allclose(actual, expected, **kwargs) -> None:
    """Compare eigenvalue arrays with NumPy allclose semantics."""
    np.testing.assert_allclose(actual, expected, **kwargs)
