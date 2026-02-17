"""
Incidence matrix computation.

Functions for computing boundary operator matrices D_k from simplicial
complex structure.
"""

import numpy as np
from scipy import sparse


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
    D_k : sparse.csr_matrix, shape (n_{k-1}, n_k)
        Boundary operator matrix with entries in {-1, 0, +1}.
        
    Notes
    -----
    - D_k[i, j] = ±1 if k-simplex j has (k-1)-face i, 0 otherwise
    - Sign is (-1)^p where p is position of omitted vertex
    - Returns empty matrix if k=0 or k not in simplices
    - Assumes face closure is valid (all faces present in simplices[k-1])
    - Use validation.validate_face_closure() to verify this property
    
    Algorithm
    ---------
    1. Build face-to-index lookup table for (k-1)-simplices
    2. For each k-simplex, iterate through its k+1 boundary faces
    3. For each boundary face, compute orientation sign and add entry
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
           [ 1]])
    """
    pass
