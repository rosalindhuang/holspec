# holspec/point_data/validation.py
"""
Input validation for point cloud data.

Functions for validating positions arrays and pairwise distance matrices,
used by PointData and downstream construction functions.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# Point Data Validation
# =============================================================================

def validate_positions(positions: np.ndarray, min_points: int | None = None) -> None:
    """
    Validate positions array for geometric computations.
    
    Parameters
    ----------
    positions : np.ndarray
        Array to validate, expected shape (N, d).
    min_points : int, optional
        Minimum number of points required. If None, no minimum is enforced.
        Common values: d+1 for Delaunay triangulation in d dimensions.
        
    Raises
    ------
    ValueError
        If array is malformed (wrong shape, contains NaN/inf, insufficient points).
        
    Notes
    -----
    Validation checks:
    - Input is a numpy array
    - Array is 2D with shape (N, d)
    - No NaN or inf values present
    - Sufficient number of points if min_points specified
    
    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 0], [0, 1]])
    >>> validate_positions(positions, min_points=3)  # Passes
    
    >>> validate_positions(positions, min_points=4)  # Raises ValueError
    """
    # Check input is numpy array
    if not isinstance(positions, np.ndarray):
        raise ValueError(f"positions must be numpy array, got {type(positions)}")
    
    # Check array is 2D
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array with shape (N, d), got shape {positions.shape}"
        )
    
    N, d = positions.shape
    
    # Check for NaN or inf
    if not np.isfinite(positions).all():
        raise ValueError("positions array contains NaN or inf values")
    
    # Check sufficient points if specified
    if min_points is not None and N < min_points:
        raise ValueError(
            f"Insufficient points: need at least {min_points}, got {N}"
        )


def validate_distances(distances: np.ndarray, min_points: int | None = None) -> None:
    """
    Validate distance matrix for geometric computations.
    
    Parameters
    ----------
    distances : np.ndarray
        Array to validate, expected shape (N, N).
    min_points : int, optional
        Minimum number of points required. If None, no minimum is enforced.
        
    Raises
    ------
    ValueError
        If array is malformed (wrong shape, non-square, contains NaN/inf,
        negative entries, non-zero diagonal, asymmetric, or insufficient points).
        
    Notes
    -----
    Validation checks:
    - Input is a numpy array
    - Array is 2D with square shape (N, N)
    - No NaN or inf values present
    - All entries are non-negative
    - Diagonal entries are zero (d(x, x) = 0)
    - Matrix is symmetric (d(x, y) = d(y, x))
    - Sufficient number of points if min_points specified
    
    Floating-point comparisons for diagonal and symmetry use tolerance 1e-10.
    
    Examples
    --------
    >>> distances = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    >>> validate_distances(distances)  # Passes
    
    >>> validate_distances(distances, min_points=4)  # Raises ValueError
    """
    # Check input is numpy array
    if not isinstance(distances, np.ndarray):
        raise ValueError(f"distances must be numpy array, got {type(distances)}")
    
    # Check array is 2D
    if distances.ndim != 2:
        raise ValueError(
            f"distances must be 2D array with shape (N, N), got shape {distances.shape}"
        )
    
    # Check array is square
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"distances must be square array, got shape {distances.shape}"
        )
    
    N = distances.shape[0]
    
    # Check for NaN or inf
    if not np.isfinite(distances).all():
        raise ValueError("distances array contains NaN or inf values")
    
    # Check non-negative
    if np.any(distances < 0):
        raise ValueError("distances array must be non-negative")
    
    # Check zero diagonal
    if not np.allclose(np.diag(distances), 0, atol=1e-10):
        raise ValueError("distances array diagonal must be zero (d(x, x) = 0)")
    
    # Check symmetry
    if not np.allclose(distances, distances.T, rtol=1e-10, atol=1e-10):
        raise ValueError("distances array must be symmetric (d(x, y) = d(y, x))")
    
    # Check sufficient points if specified
    if min_points is not None and N < min_points:
        raise ValueError(
            f"Insufficient points: need at least {min_points}, got {N}"
        )
