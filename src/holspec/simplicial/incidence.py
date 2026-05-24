"""
Incidence matrix computation.

Functions for computing boundary operator matrices D_k from simplicial
complex structure.
"""

import numpy as np
from scipy import sparse

from holspec.simplicial.simplex import get_boundary


def compute_incidence_matrix(
    simplices: dict[int, list[tuple]], 
    k: int
) -> sparse.csr_matrix:
    """
    Compute boundary operator D_k: C_k -> C_{k-1}.
    
    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension to list of simplices.
        Each simplex is a tuple of vertex indices in sorted order.
    k : int
        Dimension of domain (k-simplices).
        
    Returns
    -------
    D_k : scipy.sparse.csr_matrix, shape (n_{k-1}, n_k)
        Boundary operator matrix with entries in {-1, 0, +1}.
        
    Notes
    -----
    - D_k[i, j] = ±1 if k-simplex j has (k-1)-face i, 0 otherwise.
    - Sign is (-1)^p where p is position of omitted vertex (assuming simplices are 
        in sorted order).
    - Returns empty matrix if k=0 or if there are no k-simplices.
    - Assumes face closure; use validation.validate_face_closure() to verify this property.
    - The boundary operator satisfies D_{k-1} @ D_k = 0 (boundary of boundary is zero).
    
    Algorithm
    ---------
    1. Build face-to-index lookup table for (k-1)-simplices
    2. For each k-simplex, iterate through its k+1 boundary faces
    3. For each boundary face, compute orientation sign and add entry to sparse matrix
    4. Construct sparse matrix in COO format, convert to CSR
    
    Examples
    --------
    >>> simplices = {
    ...     0: [(0,), (1,), (2,)],
    ...     1: [(0, 1), (0, 2), (1, 2)],
    ...     2: [(0, 1, 2)]
    ... }
    >>> D_2 = compute_incidence_matrix(simplices, 2)
    >>> D_2.toarray()
    array([[ 1],
           [-1],
           [ 1]], dtype=int8)
    """
    
    # Get simplices at relevant dimensions
    k_simplices = simplices.get(k, [])
    n_k = len(k_simplices)

    km1_simplices = simplices.get(k - 1, []) if k > 0 else []
    n_km1 = len(km1_simplices)
    
    # Edge cases: no k-simplices or boundary into empty space
    if n_k == 0 or k == 0:
        return sparse.csr_matrix((n_km1, n_k), dtype=np.int8)
    
    # Create index mapping for (k-1)-simplices (rows)
    face_to_idx = {face: i for i, face in enumerate(km1_simplices)}
    
    # Build sparse matrix in COO format
    rows = []
    cols = []
    data = []
    
    # Iterate over k-simplices (columns of D_k)
    for j, simplex in enumerate(k_simplices):
        # Compute oriented boundary of this k-simplex
        boundary = get_boundary(simplex)
        
        # Add entry for each boundary face
        for face, sign in boundary:
            i = face_to_idx[face]
            rows.append(i)
            cols.append(j)
            data.append(sign)
    
    # Construct COO sparse matrix and convert to CSR
    D_k = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_km1, n_k),
        dtype=np.int8
    ).tocsr()
    
    return D_k
