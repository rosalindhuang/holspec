"""
Core tests for combinatorial cochain metrics.

These tests verify that the topology-only metric model assigns identity
metrics of the correct size to each cochain degree, including the
zero-dimensional boundary metrics used by Hodge-operator edge cases.
"""

import numpy as np
import pytest
from scipy import sparse

from holspec.cochain_metric import (
    CochainMetric,
    MetricTensor,
    construct_diagonal_metric,
)
from holspec.simplicial import SimplicialComplex

from tests.helpers import assert_sparse_allclose


# Metric structure

def test_combinatorial_metric_dimensions_match_simplices(
    edge_complex: SimplicialComplex,
    edge_metric: CochainMetric,
    triangle_boundary_complex: SimplicialComplex,
    triangle_boundary_metric: CochainMetric,
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_metric: CochainMetric,
):
    examples = [
        (edge_complex, edge_metric),
        (triangle_boundary_complex, triangle_boundary_metric),
        (filled_triangle_complex, filled_triangle_metric),
    ]

    for sc, cm in examples:
        assert cm.degrees == list(range(sc.max_dim + 1))
        assert cm.max_dim == sc.max_dim
        assert cm.dimensions == sc.num_simplices
        assert len(cm) == sc.max_dim + 1
        assert list(cm) == cm.degrees
        assert cm.all_diagonal


def test_combinatorial_metric_tensors_are_identity(
    filled_triangle_metric: CochainMetric,
):
    for k, n_k in filled_triangle_metric.dimensions.items():
        G_k = filled_triangle_metric[k]

        assert G_k.size == n_k
        assert G_k.is_diagonal
        assert_sparse_allclose(G_k.to_matrix(), np.eye(n_k))
        assert_sparse_allclose(G_k.to_matrix_inverse(), np.eye(n_k))


# Boundary-degree behavior

def test_cochain_metric_boundary_degree_metrics_are_zero_dimensional(
    filled_triangle_metric: CochainMetric,
):
    # Boundary-degree lookups provide zero-dimensional metrics for operator formulas.
    lower_boundary_metric = filled_triangle_metric[-1]
    upper_boundary_metric = filled_triangle_metric[filled_triangle_metric.max_dim + 1]

    for G_boundary in (lower_boundary_metric, upper_boundary_metric):
        assert G_boundary.size == 0
        assert G_boundary.is_diagonal
        assert G_boundary.to_matrix().shape == (0, 0)
        assert G_boundary.to_matrix().nnz == 0


# Validation

def test_combinatorial_metrics_validate_against_simplex_counts(
    edge_complex: SimplicialComplex,
    edge_metric: CochainMetric,
    triangle_boundary_complex: SimplicialComplex,
    triangle_boundary_metric: CochainMetric,
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_metric: CochainMetric,
):
    edge_metric.validate(edge_complex.num_simplices)
    triangle_boundary_metric.validate(triangle_boundary_complex.num_simplices)
    filled_triangle_metric.validate(filled_triangle_complex.num_simplices)


# Metric tensor contracts

def test_metric_tensor_accepts_positive_diagonal_metric():
    metric = construct_diagonal_metric(np.array([2.0, 4.0]))

    assert metric.size == 2
    assert metric.is_diagonal
    np.testing.assert_allclose(metric.apply(np.array([3.0, 5.0])), [6.0, 20.0])
    np.testing.assert_allclose(
        metric.apply_inverse(np.array([3.0, 5.0])),
        [1.5, 1.25],
    )
    assert_sparse_allclose(metric.to_matrix_inverse(), np.diag([0.5, 0.25]))


def test_metric_tensor_accepts_zero_dimensional_metric():
    metric = construct_diagonal_metric(np.array([]))

    assert metric.size == 0
    assert metric.is_diagonal
    assert metric.to_matrix().shape == (0, 0)
    assert metric.to_matrix().nnz == 0
    np.testing.assert_allclose(metric.apply(np.array([])), np.array([]))


def test_metric_tensor_rejects_non_sparse_matrix():
    with pytest.raises(TypeError, match="scipy sparse"):
        MetricTensor(np.eye(2), is_diagonal=True)


def test_metric_tensor_rejects_non_square_sparse_matrix():
    with pytest.raises(ValueError, match="square matrix"):
        MetricTensor(sparse.csr_matrix((2, 3)), is_diagonal=True)


def test_metric_tensor_rejects_non_finite_entries():
    matrix = sparse.diags([1.0, np.inf], format="csr")

    with pytest.raises(ValueError, match="non-finite"):
        MetricTensor(matrix, is_diagonal=True)


@pytest.mark.parametrize("diagonal", [np.array([1.0, 0.0]), np.array([1.0, -1.0])])
def test_metric_tensor_rejects_non_positive_diagonal_entries(
    diagonal: np.ndarray,
):
    with pytest.raises(ValueError, match="positive diagonal"):
        construct_diagonal_metric(diagonal)


def test_metric_tensor_rejects_non_diagonal_path_until_implemented():
    with pytest.raises(NotImplementedError, match="non-diagonal metric tensors"):
        MetricTensor(sparse.eye(2, format="csr"), is_diagonal=False)


# Cochain metric contracts

def test_cochain_metric_rejects_empty_metric_collection():
    with pytest.raises(ValueError, match="cannot be empty"):
        CochainMetric({})


def test_cochain_metric_rejects_degrees_that_do_not_start_at_zero():
    metric_tensors = {1: construct_diagonal_metric(np.ones(1))}

    with pytest.raises(ValueError, match="start at 0"):
        CochainMetric(metric_tensors)


def test_cochain_metric_rejects_degree_gaps():
    metric_tensors = {
        0: construct_diagonal_metric(np.ones(1)),
        2: construct_diagonal_metric(np.ones(1)),
    }

    with pytest.raises(ValueError, match="consecutive"):
        CochainMetric(metric_tensors)


def test_cochain_metric_validate_rejects_dimension_mismatch():
    metric = CochainMetric({0: construct_diagonal_metric(np.ones(2))})

    with pytest.raises(ValueError, match="size mismatch"):
        metric.validate({0: 3})


@pytest.mark.parametrize("degree", [-2, 4])
def test_cochain_metric_rejects_invalid_degree_lookup(
    filled_triangle_metric: CochainMetric,
    degree: int,
):
    with pytest.raises(KeyError, match="No metric tensor"):
        filled_triangle_metric[degree]
