"""
Operations and computations on simplices.
"""

from itertools import combinations
import numpy as np


def get_faces(simplex: tuple, face_dim: int) -> list[tuple]:
    """
    Extract all face_dim-faces of simplex.
    
    Parameters
    ----------
    simplex : tuple
        A k-simplex (tuple of k+1 vertex indices in sorted order).
    face_dim : int
        Dimension of faces to extract (0 <= face_dim < k).
    
    Returns
    -------
    faces : list[tuple]
        All face_dim-faces in canonical (sorted) form.
        
    Examples
    --------
    >>> get_faces((0, 1, 2), 1)
    [(0, 1), (0, 2), (1, 2)]
    
    >>> get_faces((0, 1, 2, 3), 2)
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    """
    return [tuple(sorted(face)) for face in combinations(simplex, face_dim + 1)]


def compute_orientation_sign(face: tuple, parent: tuple) -> int:
    """
    Compute sign of face in boundary of parent.
    
    Parameters
    ----------
    face : tuple
        The (k-1)-face in canonical (sorted) form.
    parent : tuple
        The k-simplex in canonical (sorted) form.
    
    Returns
    -------
    sign : int
        +1 or -1
        
    Notes
    -----
    Sign is (-1)^j where j is the position of the omitted vertex in parent.
    Assumes valid inputs (face is subset of parent with one vertex removed) 
    for computational efficiency.
    
    Examples
    --------
    >>> compute_orientation_sign((0, 1), (0, 1, 2))
    1
    
    >>> compute_orientation_sign((0, 2), (0, 1, 2))
    -1
    """
    # Find which vertex was removed by comparing face to parent
    for j, v in enumerate(parent):
        if v not in face:
            return (-1) ** j
    
    raise ValueError(f"Face {face} is not a boundary of parent {parent}")


def get_boundary(simplex: tuple) -> list[tuple[tuple, int]]:
    """
    Get oriented boundary: list of (face, sign) pairs.
    
    Parameters
    ----------
    simplex : tuple
        A k-simplex in canonical (sorted) form.
    
    Returns
    -------
    boundary : list[tuple[tuple, int]]
        List of (face, sign) pairs where face is a (k-1)-face
        and sign is ±1 according to orientation convention.
        
    Examples
    --------
    >>> get_boundary((0, 1, 2))
    [((1, 2), 1), ((0, 2), -1), ((0, 1), 1)]
    
    Notes
    -----
    More convenient than separate get_faces + compute_orientation_sign calls.
    """
    k = len(simplex) - 1
    return [
        (tuple(v for i, v in enumerate(simplex) if i != p), (-1)**p)
        for p in range(k + 1)
    ]


def compute_simplicial_closure(
    simplices: list[tuple] | dict[int, list[tuple]]
) -> dict[int, list[tuple]]:
    """
    Compute face closure of a set of simplices.
    
    Given an arbitrary collection of simplices, returns the smallest simplicial
    complex containing all input simplices. This adds all faces of the input
    simplices to satisfy the face closure property.
    
    Parameters
    ----------
    simplices : list[tuple] or dict[int, list[tuple]]
        Input simplices. Can be either:
        - A flat list of simplices as tuples (any dimensions mixed together)
        - A dictionary mapping dimension k to list of k-simplices
        
    Returns
    -------
    closed_simplices : dict[int, list[tuple]]
        Complete simplicial complex with face closure satisfied.
        Dictionary maps dimension k to list of k-simplices as sorted tuples.
        
    Examples
    --------
    >>> # Single triangle
    >>> simplices = [(0, 1, 2)]
    >>> compute_simplicial_closure(simplices)
    {0: [(0,), (1,), (2,)], 1: [(0,1), (0,2), (1,2)], 2: [(0,1,2)]}
    
    >>> # Mixed dimensions - automatically fills in missing faces
    >>> simplices = {2: [(0, 1, 2)], 1: [(3, 4)]}
    >>> compute_simplicial_closure(simplices)
    {0: [(0,), (1,), (2,), (3,), (4,)], 1: [(0,1), (0,2), (1,2), (3,4)], 2: [(0,1,2)]}
    
    Notes
    -----
    - Vertices are automatically canonicalized (sorted within each simplex)
    - Duplicates are automatically removed
    - Empty input returns empty dictionary
    """
    # Handle empty input
    if not simplices:
        return {}
    
    # Normalize input to dictionary format with canonicalized simplices
    if isinstance(simplices, list):
        # Convert list to dict, grouping by dimension
        simplices_dict = {}
        for simplex in simplices:
            # Canonicalize: sort vertices
            canonical = tuple(sorted(simplex))
            k = len(canonical) - 1
            if k not in simplices_dict:
                simplices_dict[k] = []
            simplices_dict[k].append(canonical)
    elif isinstance(simplices, dict):
        # Canonicalize dict input
        simplices_dict = {
            k: [tuple(sorted(s)) for s in simps]
            for k, simps in simplices.items()
        }
    else:
        raise TypeError(
            f"simplices must be a list or dict, got {type(simplices).__name__}"
        )
    
    # Use sets to store simplices at each dimension (automatically handles duplicates)
    simplex_sets = {k: set(simps) for k, simps in simplices_dict.items()}
    
    # Determine max dimension
    max_dim = max(simplex_sets.keys())
    
    # Initialize sets for all dimensions (including those that might be empty initially)
    for k in range(max_dim + 1):
        if k not in simplex_sets:
            simplex_sets[k] = set()
    
    # Extract faces iteratively from high to low dimension
    for k in range(max_dim, 0, -1):  # k = max_dim down to 1
        for simplex in simplex_sets[k]:
            # Extract all (k-1)-faces and add to lower dimension set
            faces = get_faces(simplex, k - 1)
            simplex_sets[k - 1].update(faces)
    
    # Convert sets back to sorted lists, excluding empty dimensions
    result = {
        k: sorted(list(simplex_sets[k]))
        for k in range(max_dim + 1)
        if simplex_sets[k]  # Only include non-empty dimensions
    }
    
    return result


# =============================================================================
# Combinatorial utilities
# =============================================================================

def compute_permutation_sign(seq_a: tuple, seq_b: tuple) -> int:
    """
    Compute the sign of the permutation that maps seq_a to seq_b.

    Parameters
    ----------
    seq_a : tuple
        Reference ordering of elements.
    seq_b : tuple
        Target ordering; must be a permutation of seq_a.

    Returns
    -------
    sign : int
        +1 if the permutation is even, -1 if odd.

    Notes
    -----
    Uses cycle decomposition: sign = (-1)^(n - c) where n is the number
    of elements and c is the number of cycles. O(n) time and space.

    Raises
    ------
    ValueError
        If seq_b is not a permutation of seq_a.

    """
    # Validate that seq_b is a permutation of seq_a
    if sorted(seq_a) != sorted(seq_b):
        raise ValueError(f"{seq_b} is not a permutation of {seq_a}")
    
    index = {v: i for i, v in enumerate(seq_a)}
    perm = [index[v] for v in seq_b]
    visited = [False] * len(perm)
    n_cycles = 0
    for i in range(len(perm)):
        if not visited[i]:
            n_cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    return (-1) ** (len(perm) - n_cycles)


# =============================================================================
# Testing and demo
# =============================================================================
if __name__ == "__main__":
    
    # Demo: Simplex operations
    import numpy as np
    rng = np.random.default_rng(42)

    # Define simplex
    k = 4
    n = 10
    simplex = tuple(sorted(rng.choice(n, size=k+1, replace=False).tolist()))

    print("="*80)
    print(f"Properties and operations for a {k}-simplex")
    print("="*80)
    print("Vertices:")
    print(simplex)
    print()

    # Faces of the simplex
    print("Faces:")
    print(f"{'Dim':<5} {'Count':<8} {'Faces'}")
    print("-" * 60)
    for j in range(k+1):
        faces = get_faces(simplex, j)
        num_faces = len(faces)
        print(f"{j:<5} {num_faces:<8} {faces}")
    print()


    # Boundary
    boundary = get_boundary(simplex)
    print("Boundary:")
    print(f"{'Sign':<6} {'Face'}")
    print("-" * 60)
    for face, sign in boundary:
        sign_str = '+' if sign == 1 else '-'
        print(f"{sign_str:<6} {face}")
    print()

    # Orientation sign
    print("Orientations:")
    print(f"{'Sign':<6} {'Permutation'}")
    print("-" * 60)
    for i in range(3):
        perm = tuple(rng.permutation(simplex).tolist())
        sign = compute_permutation_sign(perm, simplex)
        sign_str = '+' if sign == 1 else '-'
        print(f"{sign_str:<6} {perm}")
    print()

    # Simplicial closure
    print("Simplicial closure:")
    print(f"{'Dim':<5} {'Count':<8} {'Simplices'}")
    print("-" * 60)
    closed_simplices = compute_simplicial_closure([simplex])
    for k, simplices in closed_simplices.items():
        print(f"{k:<5} {len(simplices):<8} {simplices}")
