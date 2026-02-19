"""
Input validation utilities.

Functions for validating inputs to ensure correctness and fail early
with clear error messages.
"""

import numpy as np


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
