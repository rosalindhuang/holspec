"""
Hodge Laplacian and discrete differential operators.

Provides HodgeLaplacian, the primary Stage 3 pipeline output, along with
pure operator construction functions and validation utilities.
"""

from .base import HodgeLaplacian
from .operators import (
    compute_coboundary_matrix,
    compute_dual_coboundary_matrix,
    compute_laplacian_lower_matrix,
    compute_laplacian_upper_matrix,
    compute_laplacian_matrix,
    symmetrize_matrix,
)
from .validation import (
    validate_hodge_laplacian_inputs,
    validate_laplacian_properties,
)
