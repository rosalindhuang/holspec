"""
Simplicial complex construction from point data.

Factory functions for building simplicial complexes from geometric data
using various construction methods (Delaunay, alpha, Vietoris-Rips).
"""

import numpy as np
from scipy.spatial import Delaunay
import gudhi

from .base import SimplicialComplex
from .simplex import compute_simplicial_closure
from holspec.utilities.validation import validate_positions, validate_distances
from holspec.utilities.numerical import compute_content_hash


# =============================================================================
# Main Construction Functions
# =============================================================================

def build_delaunay_complex(
    positions: np.ndarray,
    max_dim: int | None = None,
    metadata: dict | None = None,
    validate: bool = True
) -> SimplicialComplex:
    """
    Construct Delaunay triangulation as a simplicial complex.
    
    Computes the Delaunay triangulation and extracts the complete face lattice
    to form an abstract simplicial complex. All faces of Delaunay simplices are
    included automatically.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, d)
        Point positions in d-dimensional Euclidean space.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to the ambient dimension d. Use this to truncate to lower-dimensional
        skeleton (e.g., max_dim=1 for just vertices and edges).
    metadata : dict, optional
        Additional metadata to store with the complex. Construction parameters
        are added automatically.
    validate : bool, default=True
        Whether to validate the resulting complex structure. Recommended to
        keep True unless you're certain the construction is valid.
        
    Returns
    -------
    complex : SimplicialComplex
        The Delaunay complex with complete face lattice up to max_dim.
        
    Raises
    ------
    ValueError
        If positions array is malformed or insufficient points for triangulation.
        
    Notes
    -----
    - Requires at least d+1 points in d dimensions for non-degenerate triangulation.
    - For collinear or coplanar points, scipy.spatial.Delaunay may raise QhullError.
    - The Delaunay triangulation is unique (up to degeneracies) and depends only
      on point positions, not on their ordering.
    - Vertices are indexed 0 to N-1 following the order in positions array.
      
    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.5]])
    >>> complex = build_delaunay_complex(positions)
    >>> complex.f_vector
    [3, 3, 1]  # 3 vertices, 3 edges, 1 triangle
    """
    # Validate input positions
    N, d = positions.shape
    validate_positions(positions, min_points=d + 1)
    
    # Compute input hash for provenance
    input_hash = compute_content_hash(positions)
    
    # Compute Delaunay triangulation
    delaunay = Delaunay(positions)
    
    # Extract top-dimensional simplices from Delaunay
    # delaunay.simplices is (n_simplices, d+1) array of vertex indices
    top_simplices = {d: [tuple(row) for row in delaunay.simplices]}
    
    # Compute full face closure
    simplices = compute_simplicial_closure(top_simplices)
    
    # Truncate to max_dim if specified
    if max_dim is not None:
        simplices = {k: simps for k, simps in simplices.items() if k <= max_dim}
    
    # Build metadata
    complex_metadata = metadata.copy() if metadata is not None else {}
    complex_metadata.update({
        'construction_method': 'delaunay',
        'construction_config': {
            'input_type': 'positions',
            'input_shape': (N, d),
            'max_dim': max_dim,
        },
        'input_hash': input_hash,
    })
    
    # Construct and return SimplicialComplex
    return SimplicialComplex(simplices, metadata=complex_metadata, validate=validate)


def build_alpha_complex(
    positions: np.ndarray,
    alpha: float | None = None,
    max_dim: int | None = None,
    metadata: dict | None = None,
    validate: bool = True
) -> SimplicialComplex:
    """
    Construct alpha complex from point data using GUDHI.
    
    The alpha complex is a subcomplex of the Delaunay triangulation containing
    only simplices whose circumradius is at most alpha. Provides a scale-dependent
    filtration of the Delaunay complex.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, d)
        Point positions in d-dimensional Euclidean space.
    alpha : float, optional
        Circumradius threshold. Only simplices with circumradius ≤ alpha are
        included. If None, all simplices are included (full alpha complex with
        no threshold). Must be non-negative if provided.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to the ambient dimension d.
    metadata : dict, optional
        Additional metadata to store with the complex. Construction parameters
        are added automatically.
    validate : bool, default=True
        Whether to validate the resulting complex structure.
        
    Returns
    -------
    complex : SimplicialComplex
        The alpha complex with complete face lattice up to max_dim.
        
    Raises
    ------
    ValueError
        If positions array is malformed, insufficient points, or alpha < 0.
        
    Notes
    -----
    - Uses GUDHI's AlphaComplex with precision='safe' (CGAL perturbation),
      which handles degenerate point configurations robustly.
    - GUDHI's filtration values are (circumradius)^2, so alpha is squared
      internally when passed to GUDHI.
    - The alpha complex is always a subcomplex of the Delaunay triangulation,
      and is at most d-dimensional for points in R^d. 
        - alpha = 0: only vertices
        - alpha -> infinity: recovers full Delaunay
    - Vertices are indexed 0 to N-1 following the order in positions array.
      
    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> cx = build_alpha_complex(positions)          # full complex
    >>> cx.f_vector
    [3, 3, 1]
    >>> cx_small = build_alpha_complex(positions, alpha=0.5)  # restrict by scale
    """
    # Validate input positions
    N, d = positions.shape
    validate_positions(positions, min_points=d + 1)
    
    # Validate alpha parameter
    if alpha is not None and alpha < 0:
        raise ValueError(f'alpha must be non-negative, got {alpha}')
    
    # Compute input hash for provenance
    input_hash = compute_content_hash(positions)
    
    # Compute max filtration value
    # Note: GUDHI uses (circumradius)^2; alpha=inf means no threshold
    max_filtration = alpha ** 2 if alpha is not None else float('inf')
    
    # Build GUDHI alpha complex and simplex tree
    alpha_complex = gudhi.AlphaComplex(points=positions, precision='safe')
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_filtration)
    
    # Extract simplices up to max_filtration threshold and optional max_dim
    simplices = _extract_simplices_from_gudhi_tree(
        simplex_tree, 
        max_filtration=max_filtration, 
        max_dim=max_dim
    )
    
    # Build metadata
    complex_metadata = metadata.copy() if metadata is not None else {}
    complex_metadata.update({
        'construction_method': 'alpha',
        'construction_config': {
            'input_type': 'positions',
            'input_shape': (N, d),
            'alpha': alpha,
            'max_dim': max_dim,
        },
        'input_hash': input_hash,
    })
    
    # Construct and return SimplicialComplex
    return SimplicialComplex(simplices, metadata=complex_metadata, validate=validate)


def build_vr_complex(
    *,
    positions: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    epsilon: float | None = None,
    max_dim: int | None = None,
    metadata: dict | None = None,
    validate: bool = True
) -> SimplicialComplex:
    """
    Construct Vietoris-Rips complex from positions or distance matrix.
    
    The Vietoris-Rips complex includes all simplices whose vertices are pairwise
    within distance epsilon. Defined purely from distances, independent of
    ambient geometry.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, d), optional
        Point positions in d-dimensional Euclidean space. Pairwise Euclidean
        distances are computed internally by GUDHI.
    distances : np.ndarray, shape (N, N), optional
        Pairwise distance matrix. Must be symmetric with zero diagonal.
        Use when distances are pre-computed or come from a non-Euclidean metric. 
        Cannot be used together with positions.
    epsilon : float
        Maximum edge length (distance threshold) for including an edge. A 
        simplex is included if all pairwise distances between its vertices are 
        at most epsilon.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to N-1 (clique complex). For large N, always set an explicit max_dim 
        to avoid combinatorial explosion.
    metadata : dict, optional
        Additional metadata to store with the complex. Construction parameters
        are added automatically.
    validate : bool, default=True
        Whether to validate the resulting complex structure.
        
    Returns
    -------
    complex : SimplicialComplex
        The Vietoris-Rips complex with complete face lattice up to max_dim.
        
    Raises
    ------
    ValueError
        If neither or both of positions/distances provided, if inputs are
        malformed, or if epsilon is negative.
        
    Notes
    -----
    - Uses GUDHI's RipsComplex, which builds the 1-skeleton then expands to
      the full clique complex up to max_dimension.
    - The VR complex is in general a superset of the alpha complex for the
      same point set and comparable scale parameter.
    - Vertices are indexed 0 to N-1 following the order in positions/distances.
    
    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> cx = build_vr_complex(positions=positions, epsilon=1.1, max_dim=2)
    >>> cx.f_vector
    [3, 3, 1]
    >>> # Using a pre-computed distance matrix
    >>> from scipy.spatial.distance import cdist
    >>> D = cdist(positions, positions)
    >>> cx2 = build_vr_complex(distances=D, epsilon=1.1, max_dim=2)
    >>> cx == cx2
    True
    """
    # Validate input combination
    if positions is None and distances is None:
        raise ValueError('Exactly one of positions or distances must be provided, got neither.')
    if positions is not None and distances is not None:
        raise ValueError('Exactly one of positions or distances must be provided, got both.')
    
    # Validate epsilon parameter
    if epsilon is None:
        raise ValueError(
            'epsilon is required. '
            'Set it to the maximum pairwise distance to include all edges.'
        )
    # if epsilon <= 0: # is this really needed?
    #     raise ValueError(f"epsilon must be positive, got {epsilon}")
    
    # Build GUDHI Rips complex
    if positions is not None:
        validate_positions(positions)
        N, d = positions.shape
        input_type = 'positions'
        input_shape = (N, d)
        input_hash = compute_content_hash(positions)
        rips = gudhi.RipsComplex(points=positions, max_edge_length=epsilon)
    else:
        validate_distances(distances)
        N = distances.shape[0]
        input_type = 'distances'
        input_shape = (N, N)
        input_hash = compute_content_hash(distances)
        rips = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=epsilon)
    
    simplex_tree = rips.create_simplex_tree(
        max_dimension=max_dim if max_dim is not None else N-1  # N-1 is the maximum possible dimension
    )
    
    # Extract simplices up to threshold and optional max_dim
    simplices = _extract_simplices_from_gudhi_tree(
        simplex_tree,
        max_filtration=epsilon,
        max_dim=max_dim
    )
    
    # Build metadata
    complex_metadata = metadata.copy() if metadata is not None else {}
    complex_metadata.update({
        'construction_method': 'vietoris_rips',
        'construction_config': {
            'input_type': input_type,
            'input_shape': input_shape,
            'epsilon': epsilon,
            'max_dim': max_dim,
        },
        'input_hash': input_hash,
    })
    
    # Construct and return SimplicialComplex
    return SimplicialComplex(simplices, metadata=complex_metadata, validate=validate)


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_simplices_from_gudhi_tree(
    simplex_tree,
    max_filtration: float,
    max_dim: int | None = None
) -> dict[int, list[tuple]]:
    """
    Convert GUDHI simplex tree to dict[int, list[tuple]] format.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        GUDHI simplex tree containing simplices with filtration values.
    max_filtration : float
        Maximum filtration value threshold. Interpretation depends on complex
            type: (circumradius)^2 for alpha complex, distance for VR complex.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        present in the simplex tree.
        
    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples.
        All simplices are in canonical form with vertices sorted.
        
    Notes
    -----
    - Simplices with filtration value > max_filtration are excluded.
    - Face closure is guaranteed by GUDHI: all faces of an included simplex
      appear at a lower-or-equal filtration value, so no additional closure
      step is needed.
    - Output is sorted: dimensions in ascending order, simplices within each
      dimension in lexicographic order. This ensures deterministic output
      regardless of GUDHI's internal traversal order.
    - Empty dimensions are excluded from the result.
    """
    simplices: dict[int, list[tuple]] = {}
    
    for simplex_list, filtration_value in simplex_tree.get_filtration():
        # Exclude simplices above the filtration threshold
        if filtration_value > max_filtration:
            continue
        
        # Sort vertices for canonical form
        simplex = tuple(sorted(simplex_list))
        
        # Exclude simplices above the dimension limit
        dim = len(simplex) - 1
        if max_dim is not None and dim > max_dim:
            continue
        
        # Add simplex to the appropriate dimension list
        simplices.setdefault(dim, []).append(simplex)
    
    # Return with dimensions and simplices in sorted order for determinism
    return {k: sorted(simplices[k]) for k in sorted(simplices)}

