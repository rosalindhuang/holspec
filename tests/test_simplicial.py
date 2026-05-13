"""
Core tests for simplicial-complex structure and incidence conventions.

These tests use hand-built edge, triangle-boundary, and filled-triangle
examples to verify simplex counts, orientation-dependent incidence matrices,
boundary-degree behavior, and the boundary-of-boundary property.
"""

import numpy as np

from holspec.simplicial import (
    SimplicialComplex,
    compute_simplicial_closure,
    get_boundary,
    get_faces,
)


# Local helpers

def assert_sparse_array_equal(matrix, expected: np.ndarray) -> None:
    """Compare a sparse matrix to expected dense values exactly."""
    np.testing.assert_array_equal(matrix.toarray(), expected)


# Basic complex invariants

def test_edge_complex_basic_invariants(edge_complex: SimplicialComplex):
    assert edge_complex.max_dim == 1
    assert edge_complex.num_vertices == 2
    assert edge_complex.num_simplices == {0: 2, 1: 1}
    assert edge_complex.f_vector == [2, 1]
    assert edge_complex.euler_characteristic == 1


def test_triangle_boundary_complex_basic_invariants(
    triangle_boundary_complex: SimplicialComplex,
):
    assert triangle_boundary_complex.max_dim == 1
    assert triangle_boundary_complex.num_vertices == 3
    assert triangle_boundary_complex.num_simplices == {0: 3, 1: 3}
    assert triangle_boundary_complex.f_vector == [3, 3]
    assert triangle_boundary_complex.euler_characteristic == 0


def test_filled_triangle_complex_basic_invariants(
    filled_triangle_complex: SimplicialComplex,
):
    assert filled_triangle_complex.max_dim == 2
    assert filled_triangle_complex.num_vertices == 3
    assert filled_triangle_complex.num_simplices == {0: 3, 1: 3, 2: 1}
    assert filled_triangle_complex.f_vector == [3, 3, 1]
    assert filled_triangle_complex.euler_characteristic == 1


# Incidence and boundary conventions

def test_boundary_degree_incidence_shapes(
    edge_complex: SimplicialComplex,
    triangle_boundary_complex: SimplicialComplex,
    filled_triangle_complex: SimplicialComplex,
):
    for complex_ in (
        edge_complex,
        triangle_boundary_complex,
        filled_triangle_complex,
    ):
        D0 = complex_.incidence_matrix(0)
        D_top = complex_.incidence_matrix(complex_.max_dim + 1)

        assert D0.shape == (0, complex_.num_vertices)
        assert D0.nnz == 0
        assert D_top.shape == (
            complex_.num_simplices[complex_.max_dim],
            0,
        )
        assert D_top.nnz == 0


def test_triangle_boundary_incidence_orientation(
    triangle_boundary_complex: SimplicialComplex,
):
    # Columns are oriented edges [(0, 1), (0, 2), (1, 2)];
    # rows are vertices [(0,), (1,), (2,)].
    expected_D1 = np.array(
        [
            [-1, -1, 0],
            [1, 0, -1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )

    assert_sparse_array_equal(
        triangle_boundary_complex.incidence_matrix(1),
        expected_D1,
    )


def test_filled_triangle_top_incidence_orientation(
    filled_triangle_complex: SimplicialComplex,
):
    # Boundary of oriented face (0, 1, 2):
    # +(1, 2) - (0, 2) + (0, 1), ordered as stored edges.
    expected_D2 = np.array(
        [
            [1],
            [-1],
            [1],
        ],
        dtype=np.int8,
    )

    assert_sparse_array_equal(filled_triangle_complex.incidence_matrix(2), expected_D2)


def test_boundary_matrix_alias_matches_incidence_matrix(
    filled_triangle_complex: SimplicialComplex,
):
    for k in range(filled_triangle_complex.max_dim + 2):
        assert_sparse_array_equal(
            filled_triangle_complex.boundary_matrix(k),
            filled_triangle_complex.incidence_matrix(k).toarray(),
        )


# Chain-complex identities

def test_boundary_property_passes_for_tiny_complexes(
    edge_complex: SimplicialComplex,
    triangle_boundary_complex: SimplicialComplex,
    filled_triangle_complex: SimplicialComplex,
):
    edge_complex.validate_boundary_property()
    triangle_boundary_complex.validate_boundary_property()
    filled_triangle_complex.validate_boundary_property()


def test_filled_triangle_boundary_of_boundary_is_zero(
    filled_triangle_complex: SimplicialComplex,
):
    D1 = filled_triangle_complex.incidence_matrix(1)
    D2 = filled_triangle_complex.incidence_matrix(2)

    product = D1 @ D2

    assert product.shape == (3, 1)
    assert product.nnz == 0


# Simplex utility conventions

def test_simplicial_closure_of_triangle(
    filled_triangle_simplices: dict[int, list[tuple[int, ...]]],
):
    assert compute_simplicial_closure([(0, 1, 2)]) == filled_triangle_simplices


def test_triangle_faces_and_boundary_conventions():
    assert get_faces((0, 1, 2), 1) == [(0, 1), (0, 2), (1, 2)]
    # Boundary signs follow the alternating omitted-vertex convention.
    assert get_boundary((0, 1, 2)) == [
        ((1, 2), 1),
        ((0, 2), -1),
        ((0, 1), 1),
    ]
