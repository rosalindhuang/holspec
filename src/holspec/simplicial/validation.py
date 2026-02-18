"""
Simplicial complex validation.

Functions for validating structural properties and mathematical invariants
of simplicial complexes.
"""
import numpy as np
from scipy import sparse


def validate_simplices_structure(simplices: dict[int, list[tuple]]) -> None:
    """
    Validate basic structure of simplices dictionary.
    
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension to list of simplices.
    
    Raises
    ------
    ValueError
        If any structural invariant is violated.
        
    Notes
    -----
    Performs the following checks:
    - Dictionary is not empty
    - Keys are consecutive integers starting from 0
    - Each simplex is a tuple of length k+1 for dimension k
    - Each simplex is in sorted (canonical) form
    - Vertex indices are non-negative integers
    - No duplicate simplices within each dimension

    Does NOT check face closure (use validate_face_closure() for that).
    """
    if not simplices:
        raise ValueError("Simplices dictionary cannot be empty")
    
    # Check keys are consecutive integers from 0
    keys = sorted(simplices.keys())
    if keys[0] != 0:
        raise ValueError("Simplex dimensions must start at 0")
    if keys != list(range(keys[-1] + 1)):
        raise ValueError("Simplex dimensions must be consecutive integers")
    
    # Check each dimension
    for k, k_simplices in simplices.items():
        if not k_simplices:
            raise ValueError(f"Empty simplex list at dimension {k}")
        
        for simplex in k_simplices:
            # Check is tuple
            if not isinstance(simplex, tuple):
                raise ValueError(f"Simplex must be tuple, got {type(simplex)}")
            
            # Check length
            if len(simplex) != k + 1:
                raise ValueError(
                    f"k-simplex must have k+1 vertices: "
                    f"dim {k} has {len(simplex)} vertices"
                )
            
            # Check sorted
            if simplex != tuple(sorted(simplex)):
                raise ValueError(f"Simplex {simplex} not in canonical (sorted) form")
            
            # Check valid vertex indices (non-negative integers)
            for v in simplex:
                if not isinstance(v, (int, np.integer)) or v < 0:
                    raise ValueError(f"Invalid vertex index: {v}")
        
        # Check no duplicates
        if len(k_simplices) != len(set(k_simplices)):
            raise ValueError(f"Duplicate simplices at dimension {k}")


def validate_face_closure(simplices: dict[int, list[tuple]]) -> None:
    """
    Validate that all faces of simplices are present.
    
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension to list of simplices.
    
    Raises
    ------
    ValueError
        If any face is missing from the complex.
        
    Notes
    -----
    For each k-simplex (k > 0), verifies that all (k-1)-faces exist
    in simplices[k-1]. This is a fundamental property of simplicial complexes.
    
    This check is more expensive than validate_simplices_structure() as it
    requires examining all faces of all simplices.
    """
    from .simplex import get_faces
    
    max_dim = max(simplices.keys())
    
    # Check each dimension k > 0
    for k in range(1, max_dim + 1):
        # Build set of (k-1)-faces for fast lookup
        face_set = set(simplices[k-1])
        
        # Check each k-simplex
        for simplex in simplices[k]:
            # Get all (k-1)-faces
            faces = get_faces(simplex, k - 1)
            
            # Verify each face exists
            for face in faces:
                if face not in face_set:
                    raise ValueError(
                        f"Face {face} of {simplex} (dim {k}) "
                        f"not found in dimension {k-1}"
                    )


def check_boundary_property(
    simplices: dict[int, list[tuple]], 
    tol: float = 1e-10,
    incidence_matrices: dict[int, sparse.csr_matrix] | None = None
) -> dict[int, bool]:
    """
    Verify D_k @ D_{k+1} = 0 for all applicable dimensions.
    
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension to list of simplices.
    tol : float, default=1e-10
        Numerical tolerance for zero comparison.
    incidence_matrices : dict[int, sparse.csr_matrix], optional
        Pre-computed incidence matrices. If provided, uses these instead
        of computing from scratch. Keys should be dimensions.
    
    Returns
    -------
    results : dict[int, bool]
        Dictionary mapping dimension k to boolean indicating whether
        D_k @ D_{k+1} = 0 (within tolerance).
        
    Notes
    -----
    - Fundamental property of boundary operators in simplicial complexes
    - If incidence_matrices not provided, computes all needed matrices
    - When called from SimplicialComplex.check_boundary_property(), 
      pre-computed matrices are passed to avoid redundant computation
    """
    from .incidence import compute_incidence_matrix
    
    # Helper to get incidence matrix (from cache or compute)
    def get_incidence_matrix(k: int) -> sparse.csr_matrix:
        if incidence_matrices is not None and k in incidence_matrices:
            return incidence_matrices[k]
        return compute_incidence_matrix(simplices, k)
    
    results = {}
    max_dim = max(simplices.keys())
    
    # Check D_k @ D_{k+1} = 0 for each applicable k
    for k in range(max_dim):
        D_k = get_incidence_matrix(k)
        D_kp1 = get_incidence_matrix(k + 1)
        
        # Compute product
        product = D_k @ D_kp1
        
        # Check if zero within tolerance
        if product.nnz > 0:
            max_entry = np.abs(product.data).max()
        else:
            max_entry = 0.0
        
        results[k] = (max_entry < tol)
    
    return results