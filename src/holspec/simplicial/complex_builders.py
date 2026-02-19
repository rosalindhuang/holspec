"""
Simplicial complex construction from point data.

Factory functions for building simplicial complexes from geometric data
using various construction methods (Delaunay, alpha, etc.).
"""

import numpy as np
from scipy.spatial import Delaunay

from .base import SimplicialComplex
from .simplex import compute_simplicial_closure, compute_circumradius
from holspec.utilities.validation import validate_positions


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
    meta = metadata.copy() if metadata is not None else {}
    meta.update({
        'construction_method': 'delaunay',
        'num_input_points': N,
        'ambient_dimension': d,
        'max_dim': max_dim if max_dim is not None else d,
    })
    
    # Construct and return SimplicialComplex
    return SimplicialComplex(simplices, metadata=meta, validate=validate)


def build_alpha_complex(
    positions: np.ndarray,
    alpha: float,
    max_dim: int | None = None,
    metadata: dict | None = None,
    validate: bool = True
) -> SimplicialComplex:
    """
    Construct alpha complex from Delaunay triangulation.
    
    The alpha complex is a subcomplex of the Delaunay triangulation containing
    only simplices whose circumradius is at most alpha. Provides a scale-dependent
    filtration of the Delaunay complex.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, d)
        Point positions in d-dimensional Euclidean space.
    alpha : float
        Radius threshold for inclusion. Simplices with circumradius ≤ alpha
        are included in the complex.
    max_dim : int, optional
        Maximum simplex dimension to include. If None, includes all dimensions
        up to the ambient dimension d.
    metadata : dict, optional
        Additional metadata to store with the complex. Construction parameters
        including alpha value are added automatically.
    validate : bool, default=True
        Whether to validate the resulting complex structure.
        
    Returns
    -------
    complex : SimplicialComplex
        The alpha complex with complete face lattice up to max_dim.
        
    Raises
    ------
    ValueError
        If positions array is malformed, alpha is negative, or insufficient points.
        
    Notes
    -----
    - Alpha complex is always a subcomplex of the Delaunay triangulation.
    - Face closure is guaranteed: if a simplex is included, all its faces are too.
    - As alpha increases, the complex grows monotonically, forming a filtration.
    - At alpha = 0, only vertices remain. At alpha → ∞, recovers full Delaunay.
    - Circumradius computation uses standard formula from computational geometry.
      
    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.5]])
    >>> complex = build_alpha_complex(positions, alpha=0.6)
    >>> # May have fewer simplices than full Delaunay depending on circumradii
    """
    pass