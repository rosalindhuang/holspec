import numpy as np


def assert_sparse_allclose(matrix, expected, **kwargs) -> None:
    np.testing.assert_allclose(matrix.toarray(), expected, **kwargs)


def assert_eigenvalues_allclose(actual, expected, **kwargs) -> None:
    np.testing.assert_allclose(actual, expected, **kwargs)
