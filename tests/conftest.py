import pytest

from holspec.simplicial import SimplicialComplex


@pytest.fixture
def edge_simplices() -> dict[int, list[tuple[int, ...]]]:
    return {
        0: [(0,), (1,)],
        1: [(0, 1)],
    }


@pytest.fixture
def triangle_boundary_simplices() -> dict[int, list[tuple[int, ...]]]:
    return {
        0: [(0,), (1,), (2,)],
        1: [(0, 1), (0, 2), (1, 2)],
    }


@pytest.fixture
def filled_triangle_simplices() -> dict[int, list[tuple[int, ...]]]:
    return {
        0: [(0,), (1,), (2,)],
        1: [(0, 1), (0, 2), (1, 2)],
        2: [(0, 1, 2)],
    }


@pytest.fixture
def edge_complex(edge_simplices: dict[int, list[tuple[int, ...]]]) -> SimplicialComplex:
    return SimplicialComplex(edge_simplices)


@pytest.fixture
def triangle_boundary_complex(
    triangle_boundary_simplices: dict[int, list[tuple[int, ...]]],
) -> SimplicialComplex:
    return SimplicialComplex(triangle_boundary_simplices)


@pytest.fixture
def filled_triangle_complex(
    filled_triangle_simplices: dict[int, list[tuple[int, ...]]],
) -> SimplicialComplex:
    return SimplicialComplex(filled_triangle_simplices)
