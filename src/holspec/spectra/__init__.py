"""
Spectral computation for Hodge Laplacians.

This subpackage defines the spectral layer of the holspec pipeline:
eigendecomposition routines, spectrum containers, and validation for computed
eigenvalues and eigenvectors.
"""

from .base import HodgeLaplacianSpectra
from .spectrum import Spectrum
from .eigensolvers import (
    VALID_SOLVER_NAMES,
    compute_eigendecomposition,
    validate_solver_params,
)
from .validation import validate_spectrum, ZERO_EIGENVALUE_TOL

__all__ = [
    "HodgeLaplacianSpectra",
    "Spectrum",
    "compute_eigendecomposition",
    "validate_solver_params",
    "VALID_SOLVER_NAMES",
    "validate_spectrum",
    "ZERO_EIGENVALUE_TOL",
]
