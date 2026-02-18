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


# %% Demo and testing
if __name__ == "__main__":
    from holspec.simplicial.simplex import get_boundary, get_faces
    # ============================================================================
    # Example 1: Simple Triangle (2-simplex)
    # ============================================================================
    print("="*70)
    print("EXAMPLE 1: Triangle Complex")
    print("="*70)

    # Define a simple triangle: vertices 0, 1, 2
    simplices_triangle = {
        0: [(0,), (1,), (2,)],                    # 3 vertices
        1: [(0,1), (0,2), (1,2)],                 # 3 edges
        2: [(0,1,2)]                              # 1 triangle
    }

    # Compute D_1: edges -> vertices
    D_1 = compute_incidence_matrix(simplices_triangle, k=1)
    print(f"\nD_1 matrix (edges → vertices):")
    print(f"Shape: {D_1.shape} (rows=vertices, cols=edges)")
    print(D_1.toarray())

    print("\nInterpretation:")
    print("  Edge (0,1): starts at 0 (−1), ends at 1 (+1)")
    print("  Edge (0,2): starts at 0 (−1), ends at 2 (+1)")
    print("  Edge (1,2): starts at 1 (−1), ends at 2 (+1)")

    # Compute D_2: triangles -> edges
    D_2 = compute_incidence_matrix(simplices_triangle, k=2)
    print(f"\nD_2 matrix (triangles → edges):")
    print(f"Shape: {D_2.shape} (rows=edges, cols=triangles)")
    print(D_2.toarray())

    print("\nBoundary of triangle (0,1,2):")
    triangle = (0,1,2)
    boundary = get_boundary(triangle)
    for face, sign in boundary:
        sign_str = '+' if sign > 0 else ''
        print(f"  {sign_str}{sign} × {face}")

    # Verify boundary property: D_1 @ D_2 = 0
    print("\nVerify chain complex property D_1 @ D_2 = 0:")
    product = D_1 @ D_2
    print(product.toarray())
    print(f"✓ All zeros: {np.allclose(product.toarray(), 0)}")


    # ============================================================================
    # Example 2: Tetrahedron (3-simplex)  
    # ============================================================================
    print("\n" + "="*70)
    print("EXAMPLE 2: Tetrahedron Complex")
    print("="*70)

    # Build a tetrahedron: 4 vertices, 6 edges, 4 faces, 1 volume
    vertices = [(i,) for i in range(4)]
    edges = [(i,j) for i in range(4) for j in range(i+1, 4)]
    faces = get_faces((0,1,2,3), face_dim=2)
    tet = [(0,1,2,3)]

    simplices_tet = {
        0: vertices,
        1: edges,
        2: faces,
        3: tet
    }

    print(f"\nf-vector: {[len(simplices_tet[k]) for k in range(4)]}")

    # Compute all incidence matrices
    D_1 = compute_incidence_matrix(simplices_tet, k=1)
    D_2 = compute_incidence_matrix(simplices_tet, k=2)
    D_3 = compute_incidence_matrix(simplices_tet, k=3)

    print(f"\nD_1 shape: {D_1.shape} (edges → vertices)")
    print(f"D_2 shape: {D_2.shape} (faces → edges)")
    print(f"D_3 shape: {D_3.shape} (volume → faces)")

    # Verify chain complex properties
    print("\nVerifying chain complex properties:")
    print(f"  D_1 @ D_2 = 0: {np.allclose((D_1 @ D_2).toarray(), 0)}")
    print(f"  D_2 @ D_3 = 0: {np.allclose((D_2 @ D_3).toarray(), 0)}")

    # Show boundary of tetrahedron
    print("\nBoundary of tetrahedron (0,1,2,3):")
    for face, sign in get_boundary((0,1,2,3)):
        sign_str = '+' if sign > 0 else ''
        print(f"  {sign_str}{sign} × {face}")


    # ============================================================================
    # Example 3: Visualize Matrix Structure
    # ============================================================================
    print("\n" + "="*70)
    print("EXAMPLE 3: Sparsity Pattern")
    print("="*70)

    # Create a larger complex to see sparsity
    n = 6
    large_simplices = {
        0: [(i,) for i in range(n)],
        1: [(i,j) for i in range(n) for j in range(i+1, n)],
        2: [(i,j,k) for i in range(n) for j in range(i+1, n) 
            for k in range(j+1, n)]
    }

    D_2_large = compute_incidence_matrix(large_simplices, k=2)
    print(f"\nD_2 for complete 2-complex on {n} vertices:")
    print(f"  Shape: {D_2_large.shape}")
    print(f"  Non-zeros: {D_2_large.nnz}")
    print(f"  Sparsity: {D_2_large.nnz / (D_2_large.shape[0] * D_2_large.shape[1]):.2%}")
    print(f"  Average non-zeros per column: {D_2_large.nnz / D_2_large.shape[1]:.1f}")
    print(f"  (Each triangle has exactly 3 edges in its boundary)")

    print("\n" + "="*70)
    print("Summary: The implementation correctly computes boundary operators")
    print("following the standard orientation convention ∂_k = Σ (-1)^p [v̂_p]")
    print("="*70)