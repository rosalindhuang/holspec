"""
Core tests for Hodge Laplacian operators.

These tests verify that Hodge Laplacians built from tiny combinatorial examples
respect the expected operator shapes, coboundary/incidence relationship,
lower-plus-upper decomposition, symmetrization behavior, and Laplacian
validation checks.
"""

import numpy as np

from holspec.hodge_laplacian import HodgeLaplacian
from holspec.simplicial import SimplicialComplex

from tests.helpers import assert_sparse_allclose


# Operator dimensions

def test_hodge_laplacian_dimensions_match_metric(
    edge_hodge_laplacian: HodgeLaplacian,
    triangle_boundary_hodge_laplacian: HodgeLaplacian,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    for hl in (
        edge_hodge_laplacian,
        triangle_boundary_hodge_laplacian,
        filled_triangle_hodge_laplacian,
    ):
        assert hl.degrees == list(range(hl.max_dim + 1))
        assert hl.dimensions == hl.cm.dimensions
        assert len(hl) == hl.max_dim + 1
        assert list(hl) == hl.degrees

        for k, n_k in hl.dimensions.items():
            for component in ("lower", "upper", "full"):
                assert hl.to_matrix(k, component).shape == (n_k, n_k)


# Differential-operator identities

def test_coboundary_is_transposed_incidence(
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    for k in filled_triangle_hodge_laplacian.degrees:
        # Cochains use the dual convention d^k = D_{k+1}.T.
        expected = filled_triangle_complex.incidence_matrix(k + 1).T.toarray()
        assert_sparse_allclose(filled_triangle_hodge_laplacian.coboundary(k), expected)


def test_dual_coboundary_zero_degree_boundary_shape(
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    delta_0 = filled_triangle_hodge_laplacian.dual_coboundary(0)

    assert delta_0.shape == (0, filled_triangle_hodge_laplacian.dimensions[0])
    assert delta_0.nnz == 0


def test_full_laplacian_decomposes_into_lower_plus_upper(
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    for k in filled_triangle_hodge_laplacian.degrees:
        # The full Laplacian is defined as the sum of lower and upper components.
        lower = filled_triangle_hodge_laplacian.to_matrix(k, "lower")
        upper = filled_triangle_hodge_laplacian.to_matrix(k, "upper")
        full = filled_triangle_hodge_laplacian.to_matrix(k, "full")

        assert_sparse_allclose(full, (lower + upper).toarray())


# Symmetrization and validation

def test_symmetrized_full_laplacians_are_symmetric(
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    for k in filled_triangle_hodge_laplacian.degrees:
        symmetric = filled_triangle_hodge_laplacian.to_symmetric_matrix(k, "full")

        np.testing.assert_allclose(symmetric.toarray(), symmetric.toarray().T)


def test_laplacian_validation_passes_for_tiny_examples(
    edge_hodge_laplacian: HodgeLaplacian,
    triangle_boundary_hodge_laplacian: HodgeLaplacian,
    filled_triangle_hodge_laplacian: HodgeLaplacian,
):
    edge_hodge_laplacian.validate_laplacians()
    triangle_boundary_hodge_laplacian.validate_laplacians()
    filled_triangle_hodge_laplacian.validate_laplacians()
