"""
Pure simplex operations.

Functions for manipulating individual simplices without reference
to the full complex structure.
"""

from itertools import combinations


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
    [((0, 1), 1), ((0, 2), -1), ((1, 2), 1)]
    
    Notes
    -----
    More convenient than separate get_faces + compute_orientation_sign calls.
    """
    k = len(simplex) - 1
    return [
        (tuple(v for i, v in enumerate(simplex) if i != p), (-1)**p)
        for p in range(k + 1)
    ]
