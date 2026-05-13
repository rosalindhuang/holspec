"""
Core tests for combinatorial cochain metrics.

These tests verify that the topology-only metric model assigns identity
metrics of the correct size to each cochain degree, including the
zero-dimensional boundary metrics used by Hodge-operator edge cases.
"""

import numpy as np

from holspec.cochain_metric import CochainMetric
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
