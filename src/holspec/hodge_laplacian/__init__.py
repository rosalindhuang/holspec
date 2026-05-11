"""
Hodge Laplacians and discrete differential operators.

This subpackage defines the operator layer of the holspec pipeline: coboundary
operators, metric adjoints, lower/upper/full Hodge Laplacians, and validation
checks for Laplacian properties.
"""

from .base import HodgeLaplacian, LAPLACIAN_COMPONENT_NAMES
from .operators import (
    compute_coboundary_matrix,
    compute_dual_coboundary_matrix,
    compute_laplacian_lower_matrix,
    compute_laplacian_upper_matrix,
    compute_laplacian_matrix,
    symmetrize_matrix,
)
from .validation import (
    LAPLACIAN_PROPERTY_TOL,
    validate_hodge_laplacian_inputs,
    validate_laplacian_properties,
)

__all__ = [
    "HodgeLaplacian",
    "LAPLACIAN_COMPONENT_NAMES",
    "compute_coboundary_matrix",
    "compute_dual_coboundary_matrix",
    "compute_laplacian_lower_matrix",
    "compute_laplacian_upper_matrix",
    "compute_laplacian_matrix",
    "symmetrize_matrix",
    "LAPLACIAN_PROPERTY_TOL",
    "validate_hodge_laplacian_inputs",
    "validate_laplacian_properties",
]
