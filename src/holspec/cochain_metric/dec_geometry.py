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
    # Map each k-simplex to its local index
    simplex_to_index = {
        simplex: i for i, simplex in enumerate(simplices[k])
    }

    # Initialize lookup for all k-simplices
    coface_lookup = {
        i: [] for i in range(len(simplices[k]))
    }

    # Record reverse face-to-coface relation
    for j, coface in enumerate(simplices[k + 1]):
        for face in itertools.combinations(coface, k + 1):
            i = simplex_to_index[face]
            coface_lookup[i].append(j)

    return coface_lookup
 
 
def _compute_halfspace_sign(
    point: np.ndarray,
    test_point: np.ndarray,
    hyperplane_points: np.ndarray,
) -> int:
    """
    Determine whether two points lie on the same side of a hyperplane.
 
    Given a hyperplane defined by the affine hull of hyperplane_points,
    determines whether point and test_point are on the same side (+1) or
    opposite sides (-1) of the hyperplane.
 
    Parameters
    ----------
    point : ndarray, shape (d,)
        Point whose side of the hyperplane is to be classified.
    test_point : ndarray, shape (d,)
        Reference point used to determine the positive side for the local sign
        convention.
    hyperplane_points : ndarray, shape (m, d)
        Points defining the hyperplane.
 
    Returns
    -------
    int
        +1 if point and test_point are on the same side, -1 if on
        opposite sides.
    """
    point = np.asarray(point, dtype=float)
    test_point = np.asarray(test_point, dtype=float)
    hyperplane_points = np.asarray(hyperplane_points, dtype=float)
    
    # Translate so the affine hull passes through the origin
    origin = hyperplane_points[0]
    point_vec = point - origin
    test_vec = test_point - origin

    # Basis for the hyperplane direction space
    basis = (hyperplane_points[1:] - origin).T

    # Remove the component tangential to the hyperplane to get the normal
    if basis.shape[1] == 0:
        # Hyperplane is a single point; normal direction is the test vector
        test_vec_normal = test_vec
    else:
        # Solve least squares to find the component of test_vec in the hyperplane
        coeffs = np.linalg.lstsq(basis, test_vec, rcond=None)[0]
        test_vec_normal = test_vec - basis @ coeffs

    # Classify point by the sign of its projection onto the test point normal
    projection = np.dot(point_vec, test_vec_normal)
    return 1 if projection >= 0.0 else -1
 
 
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
    n = max(simplices)

    # Precompute circumcenters and local coface relations
    circumcenters = compute_circumcenters(simplices, positions)
    coface_lookups = {
        k: _build_coface_lookup(simplices, k) for k in range(n)
    }

    # Precompute simplex vertex sets for opposite-vertex lookup
    simplex_sets = {
        k: [set(simplex) for simplex in simplices[k]]
        for k in simplices
    }

    # Accumulate signed elementary dual volumes over all flags
    def accumulate_elementary_duals(
        curr_degree: int,
        curr_index: int,
        dual_vertices: list[np.ndarray],
        sign_product: int,
    ) -> float:
        """
        Recursively accumulate signed elementary dual volumes over all flags
        of the current simplex, where each flag is a nested sequence of
        incident cofaces sigma^(k) < sigma^(k+1) < ... < sigma^(n) from the 
        current simplex up to a top-dimensional simplex.

        Each recursive step extends the flag by one coface sigma^(j+1),
        appends its circumcenter c(sigma^(j+1)), and updates the sign
        product by the corresponding local halfspace sign s_j.

        At a top-dimensional simplex sigma^(n), return the signed volume
        of the resulting elementary dual simplex determined by the
        accumulated circumcenter sequence.
        """
        # Reached a full chain, so the current circumcenter sequence 
        # forms an elementary dual simplex; return its signed volume
        if curr_degree == n:
            return (
                sign_product
                * compute_simplex_volume(
                    np.array(dual_vertices, dtype=float)
                )
            )

        sigma_curr = simplices[curr_degree][curr_index]
        sigma_curr_set = simplex_sets[curr_degree][curr_index]
        hyperplane_points = positions[list(sigma_curr)]

        total = 0.0
        # Iterate over each coface and accumulate its signed contribution
        for next_index in coface_lookups[curr_degree][curr_index]:
            tau = simplices[curr_degree + 1][next_index]
            tau_cc = circumcenters[curr_degree + 1][next_index]

            # Vertex added when extending sigma_curr to tau
            opposite_vertex = next(
                v for v in tau if v not in sigma_curr_set
            )

            # Compute the local halfspace sign for this step
            step_sign = _compute_halfspace_sign(
                point=tau_cc,
                test_point=positions[opposite_vertex],
                hyperplane_points=hyperplane_points,
            )

            total += accumulate_elementary_duals(
                curr_degree=curr_degree + 1,
                curr_index=next_index,
                dual_vertices=dual_vertices + [tau_cc],
                sign_product=sign_product * step_sign,
            )

        return total

    dual_volumes = {}

    for k, k_simplices in sorted(simplices.items()):
        # Top-degree dual cells are 0-cells with volume 1 by convention
        if k == n:
            dual_volumes[k] = np.ones(len(k_simplices), dtype=float)
            continue

        dual_volumes_k = np.zeros(len(k_simplices), dtype=float)

        for i in range(len(k_simplices)):
            # Accumulate all flag contributions rooted at this simplex
            dual_volumes_k[i] = accumulate_elementary_duals(
                curr_degree=k,
                curr_index=i,
                dual_vertices=[circumcenters[k][i]],
                sign_product=1,
            )

        dual_volumes[k] = dual_volumes_k

    return dual_volumes


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
        Minimum acceptable volume magnitude. Primal volumes and dual volumes
        with absolute value below this threshold raise ValueError.
 
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
    # Compute primal simplex volumes
    primal_volumes = compute_simplex_volumes(simplices, positions)

    # Degeneracy check: near-zero primal volumes
    for k, primal_vols_k in primal_volumes.items():
        degenerate_mask = primal_vols_k < degeneracy_tol
        if np.any(degenerate_mask):
            n_degenerate = int(np.sum(degenerate_mask))
            raise ValueError(
                f"Degenerate primal volumes at degree k={k}: "
                f"{n_degenerate} of {len(primal_vols_k)} simplices have "
                f"volume below degeneracy_tol={degeneracy_tol}"
            )

    # Compute signed dual volumes
    dual_volumes = compute_dual_volumes(simplices, positions)

    # Degeneracy check: near-zero dual volumes
    for k, dual_vols_k in dual_volumes.items():
        degenerate_mask = np.abs(dual_vols_k) < degeneracy_tol
        if np.any(degenerate_mask):
            n_degenerate = int(np.sum(degenerate_mask))
            raise ValueError(
                f"Degenerate dual volumes at degree k={k}: "
                f"{n_degenerate} of {len(dual_vols_k)} simplices have "
                f"absolute dual volume below degeneracy_tol={degeneracy_tol}"
            )

    # Assemble Hodge star diagonals
    hodge_star = {
        k: dual_volumes[k] / primal_volumes[k]
        for k in sorted(primal_volumes)
    }

    return hodge_star