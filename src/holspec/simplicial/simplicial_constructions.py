"""
Simplicial complex construction from point data.

Factory functions for constructing simplicial complexes from geometric data
using various construction methods (Delaunay, alpha, Vietoris-Rips).
"""

import numpy as np
from scipy.spatial import Delaunay
import gudhi

from .simplex import compute_simplicial_closure, get_faces
from holspec.point_data.validation import validate_positions, validate_distances
from holspec.utilities import format_float_str


# =============================================================================
# Main Construction Functions
# =============================================================================

def construct_delaunay_complex(
    positions: np.ndarray,
    max_dim: int | None = None,
) -> dict[int, list[tuple]]:
    """
    Construct Delaunay triangulation as a simplicial complex.
    
    Computes the Delaunay triangulation and extracts all simplices with their
    complete face closure to form an abstract simplicial complex. All faces of
    Delaunay simplices are included automatically.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, d)
        Point positions in d-dimensional Euclidean space.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to the ambient dimension d. Use this to truncate to lower-dimensional
        skeleton (e.g., max_dim=1 for just vertices and edges).
        
    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples,
        closed under taking faces, up to max_dim.
        
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
    >>> simplices = construct_delaunay_complex(positions)
    >>> [len(s) for s in simplices.values()]
    [3, 3, 1]  # 3 vertices, 3 edges, 1 triangle
    """
    # Validate input positions
    N, d = positions.shape
    validate_positions(positions, min_points=d + 1)
    
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
    
    return simplices


def construct_alpha_complex(
    positions: np.ndarray,
    alpha: float | None = None,
    max_dim: int | None = None,
) -> dict[int, list[tuple]]:
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
        
    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples,
        closed under taking faces, up to max_dim.
        
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
    >>> simplices = construct_alpha_complex(positions)          # full complex
    >>> [len(s) for s in simplices.values()]
    [3, 3, 1]
    >>> simplices_small = construct_alpha_complex(positions, alpha=0.5)  # restrict by scale
    """
    # Validate input positions
    N, d = positions.shape
    validate_positions(positions, min_points=d + 1)
    
    # Validate alpha parameter
    if alpha is not None and alpha < 0:
        raise ValueError(f'alpha must be non-negative, got {alpha}')
    
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
    
    return simplices


def construct_del_vr_complex(
    positions: np.ndarray,
    epsilon: float | None = None,
    max_dim: int | None = None,
) -> dict[int, list[tuple]]:
    """
    Construct VR-filtered Delaunay complex from point data.

    The Del-VR complex is the subcomplex of the Delaunay triangulation
    containing only simplices whose edges all have length at most epsilon.
    Equivalently, it is the intersection of the Delaunay triangulation with
    the Vietoris-Rips complex: Del_VR(epsilon) = Del(P) ∩ VR(epsilon).

    Unlike the alpha complex (which filters by circumradius), this filters
    by maximum edge length — the same criterion as Vietoris-Rips, but
    restricted to Delaunay simplices.

    Parameters
    ----------
    positions : np.ndarray, shape (N, d)
        Point positions in d-dimensional Euclidean space.
    epsilon : float
        Maximum edge length threshold. A simplex is included if all its
        edges have length at most epsilon. Must be non-negative.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to the ambient dimension d.

    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples,
        closed under taking faces, up to max_dim.

    Raises
    ------
    ValueError
        If positions array is malformed, insufficient points, epsilon is None,
        or epsilon < 0.

    Notes
    -----
    - Requires position data (not distances) since it builds on the Delaunay
      triangulation, which needs ambient coordinates.
    - Requires at least d+1 points in d dimensions for non-degenerate
      triangulation.
    - The filtration value of a simplex is its longest edge length.
    - Face closure is automatically satisfied: if all edges of a simplex pass,
      all edges of any face (a subset) also pass.
    - Limiting behavior:
        - epsilon = 0: only vertices (no edges have zero length unless points
          coincide)
        - epsilon -> infinity: recovers full Delaunay triangulation
    - Vertices are indexed 0 to N-1 following the order in positions array.

    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> simplices = construct_del_vr_complex(positions, epsilon=1.0)
    >>> [len(s) for s in simplices.values()]
    [3, 3, 1]
    >>> simplices_small = construct_del_vr_complex(positions, epsilon=0.5)
    >>> [len(s) for s in simplices_small.values()]
    [3]
    """
    # Validate input positions
    N, d = positions.shape
    validate_positions(positions, min_points=d + 1)

    # Validate epsilon parameter
    if epsilon is None:
        raise ValueError(
            'epsilon is required. '
            'Use construct_delaunay_complex for the unfiltered Delaunay triangulation.'
        )
    if epsilon < 0:
        raise ValueError(f'epsilon must be non-negative, got {epsilon}')

    # Compute full Delaunay triangulation with face closure
    delaunay = Delaunay(positions)
    top_simplices = {d: [tuple(sorted(row)) for row in delaunay.simplices]}
    full_delaunay = compute_simplicial_closure(top_simplices)

    # Extract all edges and compute lengths (vectorized)
    edges = full_delaunay.get(1, [])
    if edges:
        src = np.array([e[0] for e in edges])
        dst = np.array([e[1] for e in edges])
        lengths = np.linalg.norm(positions[src] - positions[dst], axis=1)
        passing_edges = {edges[i] for i in range(len(edges)) if lengths[i] <= epsilon}
    else:
        passing_edges = set()

    # Filter every simplex at every dimension
    filtered: dict[int, list[tuple]] = {}
    for dim in sorted(full_delaunay.keys()):
        if dim == 0:
            # Vertices always included (vacuously true — no edges)
            filtered[0] = full_delaunay[0]
        elif dim == 1:
            surviving = [e for e in full_delaunay[1] if e in passing_edges]
            if surviving:
                filtered[1] = surviving
        else:
            surviving = [
                simplex for simplex in full_delaunay[dim]
                if all(e in passing_edges for e in get_faces(simplex, 1))
            ]
            if surviving:
                filtered[dim] = surviving

    # Truncate to max_dim if specified
    if max_dim is not None:
        filtered = {k: simps for k, simps in filtered.items() if k <= max_dim}

    return filtered


def construct_vr_complex(
    *,
    positions: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    epsilon: float | None = None,
    max_dim: int | None = None,
) -> dict[int, list[tuple]]:
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
        
    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples,
        closed under taking faces, up to max_dim.
        
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
    >>> simplices = construct_vr_complex(positions=positions, epsilon=1.1, max_dim=2)
    >>> [len(s) for s in simplices.values()]
    [3, 3, 1]
    >>> # Using a pre-computed distance matrix
    >>> from scipy.spatial.distance import cdist
    >>> D = cdist(positions, positions)
    >>> simplices2 = construct_vr_complex(distances=D, epsilon=1.1, max_dim=2)
    >>> simplices == simplices2
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
        N = positions.shape[0]
        rips = gudhi.RipsComplex(points=positions, max_edge_length=epsilon)
    else:
        validate_distances(distances)
        N = distances.shape[0]
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
    
    return simplices


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


# =============================================================================
# Registry and Config-Driven Utilities
# =============================================================================

SIMPLICIAL_CONSTRUCTION_REGISTRY: dict[str, callable] = {
    'delaunay': construct_delaunay_complex,
    'alpha': construct_alpha_complex,
    'del_vr': construct_del_vr_complex,
    'vietoris_rips': construct_vr_complex,
}

# Aliases for simplicial construction methods
SIMPLICIAL_CONSTRUCTION_ALIASES: dict[str, list[str]] = {
    'delaunay': ['del'],
    'alpha': ['alp'],
    'del_vr': ['dvr'],
    'vietoris_rips': ['vr', 'rips'],
}

SIMPLICIAL_CONSTRUCTION_ALIAS_MAP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in SIMPLICIAL_CONSTRUCTION_ALIASES.items()
    for alias in aliases
}


def construct_complex_from_config(
    config: dict,
    positions: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> dict[int, list[tuple]]:
    """
    Build raw simplices from a method config and input data.

    Parameters
    ----------
    config : dict
        Must contain 'method' (str) and 'params' (dict). E.g.:
          {'method': 'delaunay', 'params': {'max_dim': 2}}
          {'method': 'alpha', 'params': {'alpha': 1.5, 'max_dim': 2}}
          {'method': 'vietoris_rips', 'params': {'epsilon': 1.2, 'max_dim': 2}}
    positions : np.ndarray, shape (N, d), optional
        Point positions. Required for 'delaunay' and 'alpha'.
    distances : np.ndarray, shape (N, N), optional
        Pairwise distance matrix. Only used with 'vietoris_rips'.

    Returns
    -------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices as sorted tuples.
    """
    # Validate config
    if 'method' not in config:
        raise ValueError("config must contain a 'method' key")

    method = SIMPLICIAL_CONSTRUCTION_ALIAS_MAP.get(config['method'], config['method'])
    if method not in SIMPLICIAL_CONSTRUCTION_REGISTRY:
        raise ValueError(
            f"Unknown method '{config['method']}'. "
            f"Available methods: {list(SIMPLICIAL_CONSTRUCTION_REGISTRY)}"
        )
    params = config.get('params', {})

    builder = SIMPLICIAL_CONSTRUCTION_REGISTRY[method]

    # vietoris_rips uses keyword-only positions/distances;
    # delaunay and alpha take positions as a positional argument.
    if method == 'vietoris_rips':
        return builder(
            positions=positions,
            distances=distances,
            **params,
        )
    else:
        if positions is None:
            raise ValueError(f"positions is required for method '{method}'")
        return builder(
            positions,
            **params,
        )


def create_simplicial_construction_label(
    config: dict,
    float_fmt: str | None = 'g',
    strip_zeros: bool = True,
) -> str:
    """
    Create a descriptive label from a complex builder config.

    Label format: method_param1_param2...
    Method names are the full canonical registry names. Aliases (e.g. 'vr', 'del') are
    resolved to their canonical form via SIMPLICIAL_CONSTRUCTION_ALIAS_MAP.
    Parameter names are abbreviated to their first two non-underscore characters.
    Floats are formatted with ``float_fmt``. None values are omitted.

    Parameters
    ----------
    config : dict
        Must contain 'method' (str) and 'params' (dict).
    float_fmt : str or None, default='g'
        Format specifier for float values (e.g. 'g', '.2e', '.3f').
    strip_zeros : bool, default=True
        If True, trailing zeros after the decimal point are removed.

    Returns
    -------
    label : str

    Examples
    --------
    >>> create_simplicial_construction_label({'method': 'delaunay', 'params': {'max_dim': 2}})
    'delaunay_md2'
    >>> create_simplicial_construction_label({'method': 'alpha', 'params': {'alpha': 1.5, 'max_dim': 2}})
    'alpha_al1p5_md2'
    >>> create_simplicial_construction_label({'method': 'vietoris_rips', 'params': {'epsilon': 1.2, 'max_dim': 2}})
    'vietoris_rips_ep1p2_md2'
    >>> create_simplicial_construction_label({'method': 'vr', 'params': {'epsilon': 1.2, 'max_dim': 2}})  # alias works too
    'vietoris_rips_ep1p2_md2'
    """
    # Validate config
    if 'method' not in config:
        raise ValueError("config must contain a 'method' key")

    method = SIMPLICIAL_CONSTRUCTION_ALIAS_MAP.get(config['method'], config['method'])
    if method not in SIMPLICIAL_CONSTRUCTION_REGISTRY:
        raise ValueError(
            f"Unknown method '{config['method']}'. "
            f"Available methods: {list(SIMPLICIAL_CONSTRUCTION_REGISTRY)}"
        )
    params = config.get('params', {})

    parts = [method]

    for key, value in params.items():
        # Skip None values — they represent 'use default / no constraint'
        if value is None:
            continue

        param_abbr = key.replace('_', '')[:2]

        if isinstance(value, bool):
            value_str = '1' if value else '0'
        elif isinstance(value, float):
            value_str = format_float_str(value, float_fmt, strip_zeros=strip_zeros)
        elif isinstance(value, str):
            value_str = value[:4].lower().replace('_', '')
        else:
            value_str = str(value).replace('.', 'p')

        parts.append(f"{param_abbr}{value_str}")

    return '_'.join(parts)

