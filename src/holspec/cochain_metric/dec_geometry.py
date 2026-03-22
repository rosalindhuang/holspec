"""
Discrete exterior calculus geometry computations.

Pure mathematical routines for computing circumcenters, simplex volumes,
signed dual volumes, and Hodge star diagonals on simplicial complexes.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Tolerance for detecting degenerate simplices (near-zero primal or dual
# volumes). Simplex volumes below this threshold indicate ill-conditioned
# geometry where circumcenter computations and Hodge star entries are
# numerically unreliable.
VOLUME_DEGENERACY_TOL = 1e-10


# =============================================================================
# Per-Simplex Primitives
# =============================================================================

def _validate_simplex_vertices(
    vertices: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """
    Validate and normalize the vertex array for a single simplex.

    Parameters
    ----------
    vertices : ndarray, shape (k + 1, d)
        Vertex coordinates of a single k-simplex embedded in R^d.

    Returns
    -------
    vertices : ndarray, shape (k + 1, d)
        Input vertices converted to float ndarray.
    num_vertices : int
        Number of simplex vertices.
    k : int
        Simplex degree.
    d : int
        Ambient dimension.

    Raises
    ------
    ValueError
        If vertices is not a 2D array, contains no vertices, contains
        non-finite values, or has simplex degree exceeding ambient
        dimension.
    """
    vertices = np.asarray(vertices, dtype=float)

    # Check array shape
    if vertices.ndim != 2:
        raise ValueError(
            f"vertices must be a 2D array of shape (k+1, d), "
            f"got shape {vertices.shape}"
        )

    num_vertices, d = vertices.shape
    k = num_vertices - 1

    # Check nonempty input
    if num_vertices == 0:
        raise ValueError("vertices must contain at least one vertex")

    # Check finite coordinates
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values")

    # Check dimension compatibility
    if k > d:
        raise ValueError(
            f"Simplex degree k={k} cannot exceed ambient dimension d={d}"
        )

    return vertices, num_vertices, k, d


def compute_circumcenter(vertices: np.ndarray) -> np.ndarray:
    """
    Compute the circumcenter of a simplex from its vertex coordinates.
 
    Parameters
    ----------
    vertices : ndarray, shape (k+1, d)
        Vertex coordinates of a k-simplex in R^d. Requires k >= 0 and
        d >= k (the vertices must be affinely independent).
 
    Returns
    -------
    ndarray, shape (d,)
        Cartesian coordinates of the simplex circumcenter.
 
    Notes
    -----
    This function is dimension-general: it applies to simplices of any degree
    ``k >= 0`` satisfying ``k <= d``, including lower-dimensional
    simplices embedded in a higher-dimensional ambient space. 

    The circumcenter is computed from the barycentric-coordinate linear system,
    then returned in Cartesian coordinates. Following [1]_, solve the 
    (k+2) x (k+2) system
 
        | 2 V V^T    1 |   | b |   | diag(V V^T) |
        |              | * |   | = |             |
        |   1^T      0 |   | Q |   |      1      |
 
    for barycentric coordinates b then convert to Cartesian coordinates 
    via vertices^T @ b. For low dimensions (k <= 3), the cost of the direct 
    solve is negligible. For k=0, the circumcenter is the vertex itself.

    References
    ----------
    .. [1] N. Bell and A. N. Hirani, “PyDEC: Software and Algorithms for 
    Discretization of Exterior Calculus,” ACM Trans. Math. Softw., vol. 39, 
    no. 1, pp. 1–41, Nov. 2012, doi: 10.1145/2382585.2382588.

    """
    vertices, num_vertices, k, d = _validate_simplex_vertices(vertices)
    
    # Handle 0-simplex case
    if k == 0:
        return vertices[0].copy()

    # Gram matrix of position vectors: G_ij = v_i . v_j
    gram = vertices @ vertices.T

    # Assemble the (k+2) x (k+2) linear system
    A = np.zeros((num_vertices + 1, num_vertices + 1))
    A[:num_vertices, :num_vertices] = 2.0 * gram
    A[:num_vertices, -1] = 1.0
    A[-1, :num_vertices] = 1.0

    rhs = np.zeros(num_vertices + 1)
    rhs[:num_vertices] = np.diag(gram)
    rhs[-1] = 1.0

    # Solve for barycentric coordinates
    try:
        barycentric_coords = np.linalg.solve(A, rhs)[:-1]
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Simplex geometry is degenerate and does not determine a unique "
            "circumcenter"
        ) from exc

    # Convert barycentric to Cartesian coordinates
    return barycentric_coords @ vertices
 
 
def compute_simplex_volume(vertices: np.ndarray) -> float:
    """
    Compute the unsigned Euclidean volume of a simplex.

    Parameters
    ----------
    vertices : ndarray, shape (k + 1, d)
        Vertex coordinates of a single k-simplex embedded in Euclidean space.

    Returns
    -------
    float
        Unsigned k-dimensional volume of the simplex.

    Notes
    -----
    A 0-simplex has volume 1 by convention. For ``k >= 1``, the volume is computed 
    from the Gram determinant of the edge matrix, giving a single dimension-general
    implementation for edges, triangles, tetrahedra, and higher-dimensional
    simplices.
    """
    vertices, num_vertices, k, d = _validate_simplex_vertices(vertices)

    # Handle 0-simplex case
    if k == 0:
        return 1.0

    # Edge matrix: rows are v_i - v_0 for i = 1, ..., k
    edge_matrix = vertices[1:] - vertices[0]

    # Build Gram matrix
    gram = edge_matrix @ edge_matrix.T

    # Guard against small negative roundoff
    det_gram = float(np.linalg.det(gram))
    det_gram = max(det_gram, 0.0)

    # Convert Gram determinant to simplex volume
    return float(np.sqrt(det_gram) / math.factorial(k))
 
 
# =============================================================================
# Batch Functions for Simplicial Complexes
# =============================================================================
 
def compute_circumcenters(
    simplices: dict[int, list[tuple]],
    positions: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute circumcenters for all simplices at all degrees.
 
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Mapping from degree k to list of k-simplices, each a sorted tuple
        of vertex indices.
    positions : ndarray, shape (num_vertices, d)
        Vertex coordinates in R^d.
 
    Returns
    -------
    dict[int, ndarray]
        Mapping from degree k to array of shape (N_k, d) containing
        the circumcenter of each k-simplex. 
    """
    circumcenters = {}
    for k, k_simplices in sorted(simplices.items()):
        if len(k_simplices) == 0:
            circumcenters[k] = np.empty((0, positions.shape[1]), dtype=float)
            continue

        circumcenters[k] = np.array([
            compute_circumcenter(positions[list(s)]) for s in k_simplices
        ])
    return circumcenters
 
 
def compute_simplex_volumes(
    simplices: dict[int, list[tuple]],
    positions: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute unsigned volumes for all simplices at all degrees.
 
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Mapping from degree k to list of k-simplices, each a sorted tuple
        of vertex indices.
    positions : ndarray, shape (num_vertices, d)
        Vertex positions in R^d.
 
    Returns
    -------
    dict[int, ndarray]
        Mapping from degree k to array of shape (N_k,) containing the
        unsigned k-volume of each k-simplex.
    """
    volumes = {}
    for k, k_simplices in sorted(simplices.items()):
        volumes[k] = np.array([
            compute_simplex_volume(positions[list(s)]) for s in k_simplices
        ])
    return volumes
 
 
# =============================================================================
# Dual Volume Computation
# =============================================================================
 
def _build_coface_lookup(
    simplices: dict[int, list[tuple]],
    k: int,
) -> dict[int, list[int]]:
    """
    Build a mapping from k-simplex indices to their (k+1)-coface indices.
 
    For each (k+1)-simplex, enumerates its k-faces via combinatorial
    selection and records the reverse mapping. This provides the coface
    relation needed for flag enumeration without depending on incidence
    matrices.
 
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Mapping from degree to list of simplices. Must contain keys k
        and k+1.
    k : int
        Degree of the simplices whose cofaces are sought.
 
    Returns
    -------
    dict[int, list[int]]
        Mapping from each k-simplex index i to the list of (k+1)-simplex
        indices j such that simplices[k][i] is a face of simplices[k+1][j].
    """
    pass
 
 
def _compute_halfspace_sign(
    point: np.ndarray,
    test_point: np.ndarray,
    hyperplane_points: np.ndarray,
) -> int:
    """
    Determine whether two points lie on the same side of a hyperplane.
 
    Given a hyperplane defined by the affine hull of hyperplane_points,
    determines whether point and test_point are on the same side (+1) or
    opposite sides (-1) of the hyperplane, working within the affine
    subspace spanned by hyperplane_points and test_point.
 
    Parameters
    ----------
    point : ndarray, shape (d,)
        Point to classify (typically a circumcenter).
    test_point : ndarray, shape (d,)
        Reference point of known sidedness (typically the opposite vertex
        not in the face).
    hyperplane_points : ndarray, shape (m, d)
        Points defining the hyperplane (typically vertices of the face
        sigma^{(i)}).
 
    Returns
    -------
    int
        +1 if point and test_point are on the same side, -1 if on
        opposite sides.
    """
    pass
 
 
def compute_dual_volumes(
    simplices: dict[int, list[tuple]],
    positions: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute signed circumcentric dual volumes for all simplices at all degrees.

 
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Mapping from degree k to list of k-simplices, each a sorted tuple
        of vertex indices.
    positions : ndarray, shape (num_vertices, d)
        Vertex positions in R^d.
 
    Returns
    -------
    dict[int, ndarray]
        Mapping from degree k to array of shape (N_k,) containing the
        signed (n-k)-volume of the dual cell for each k-simplex.
        Top-degree simplices (k=n) have dual volume 1 by convention.
 
    Notes
    -----
    For each k-simplex sigma^{(k)}, the circumcentric dual cell volume is
    the sum of signed elementary dual simplex volumes over all flags
    (maximal chains) containing sigma:
 
        |*sigma^{(k)}| = sum over flags  s(flag) * vol(dual simplex)
 
    where each flag is a chain sigma^{(k)} < sigma^{(k+1)} < ... <
    sigma^{(n)} through the coface structure, and the elementary dual
    simplex has vertices given by the circumcenter sequence
    [c(sigma^{(k)}), c(sigma^{(k+1)}), ..., c(sigma^{(n)})]. The sign
    s(flag) is the product of halfspace signs s_k * s_{k+1} * ... *
    s_{n-1} following [1]_ with the corrigendum correction.

    The algorithm is dimension-general with no branching on n. The depth
    of flag enumeration and elementary dual simplex dimension adapt to the
    complex dimension.

    References
    ----------
    .. [1] A. N. Hirani, K. Kalyanaraman, and E. B. VanderZee, “Delaunay 
    Hodge star,” Computer-Aided Design, vol. 45, no. 2, pp. 540–544, Feb. 
    2013, doi: 10.1016/j.cad.2012.10.038.

    """
    pass
 
 
# =============================================================================
# Hodge Star Assembly
# =============================================================================
 
def compute_hodge_star_diagonals(
    simplices: dict[int, list[tuple]],
    positions: np.ndarray,
    degeneracy_tol: float = VOLUME_DEGENERACY_TOL,
) -> dict[int, np.ndarray]:
    """
    Compute diagonal entries of the discrete Hodge star at all degrees.

    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Mapping from degree k to list of k-simplices, each a sorted tuple
        of vertex indices.
    positions : ndarray, shape (num_vertices, d)
        Vertex positions in R^d.
    degeneracy_tol : float, default=VOLUME_DEGENERACY_TOL
        Minimum acceptable volume magnitude. Primal volumes below this
        threshold and dual volumes with absolute value below this threshold
        raise ValueError.
 
    Returns
    -------
    dict[int, ndarray]
        Mapping from degree k to array of shape (N_k,) containing the
        Hodge star diagonal entries. Entries may be negative when the dual
        volume is negative (e.g. for non-Delaunay meshes).
 
    Raises
    ------
    ValueError
        If any primal volume is below degeneracy_tol, or if any absolute
        dual volume is below degeneracy_tol. 
 
    Notes
    -----
    Assembles the diagonal entries of the circumcentric Hodge star matrix:
 
        [*^k]_{ii} = |*sigma_i^{(k)}| / |sigma_i^{(k)}|
 
    where |sigma| is the unsigned primal volume and |*sigma| is the signed
    dual cell volume. Performs degeneracy checks on both primal and dual
    volumes before division.
    """
    pass