import pytest

from holspec.cochain_metric import (
    CochainMetric,
    construct_combinatorial_cochain_metric,
)
from holspec.hodge_laplacian import HodgeLaplacian
from holspec.simplicial import SimplicialComplex
from holspec.spectra import HodgeLaplacianSpectra


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


def _combinatorial_metric(sc: SimplicialComplex) -> CochainMetric:
    metric_tensors, _ = construct_combinatorial_cochain_metric(sc)
    return CochainMetric(metric_tensors)


@pytest.fixture
def edge_metric(edge_complex: SimplicialComplex) -> CochainMetric:
    return _combinatorial_metric(edge_complex)


@pytest.fixture
def triangle_boundary_metric(
    triangle_boundary_complex: SimplicialComplex,
) -> CochainMetric:
    return _combinatorial_metric(triangle_boundary_complex)


@pytest.fixture
def filled_triangle_metric(filled_triangle_complex: SimplicialComplex) -> CochainMetric:
    return _combinatorial_metric(filled_triangle_complex)


@pytest.fixture
def edge_hodge_laplacian(
    edge_complex: SimplicialComplex,
    edge_metric: CochainMetric,
) -> HodgeLaplacian:
    return HodgeLaplacian(edge_complex, edge_metric)


@pytest.fixture
def triangle_boundary_hodge_laplacian(
    triangle_boundary_complex: SimplicialComplex,
    triangle_boundary_metric: CochainMetric,
) -> HodgeLaplacian:
    return HodgeLaplacian(triangle_boundary_complex, triangle_boundary_metric)


@pytest.fixture
def filled_triangle_hodge_laplacian(
    filled_triangle_complex: SimplicialComplex,
    filled_triangle_metric: CochainMetric,
) -> HodgeLaplacian:
    return HodgeLaplacian(filled_triangle_complex, filled_triangle_metric)


@pytest.fixture
def edge_spectra(edge_hodge_laplacian: HodgeLaplacian) -> HodgeLaplacianSpectra:
    return HodgeLaplacianSpectra(edge_hodge_laplacian)


@pytest.fixture
def triangle_boundary_spectra(
    triangle_boundary_hodge_laplacian: HodgeLaplacian,
) -> HodgeLaplacianSpectra:
    return HodgeLaplacianSpectra(triangle_boundary_hodge_laplacian)


@pytest.fixture
def filled_triangle_spectra(
    filled_triangle_hodge_laplacian: HodgeLaplacian,
) -> HodgeLaplacianSpectra:
    return HodgeLaplacianSpectra(filled_triangle_hodge_laplacian)
