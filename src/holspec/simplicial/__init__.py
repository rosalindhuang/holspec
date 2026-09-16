# holspec/simplicial/__init__.py
"""
Simplicial complex construction and representation.

This subpackage defines the topological layer of the holspec pipeline:
oriented simplicial complexes, incidence matrices, simplex operations, and
point-cloud-to-complex construction routines.
"""

from .base import SimplicialComplex
from .simplex import (
    compute_orientation_sign,
    compute_permutation_sign,
    compute_simplicial_closure,
    get_boundary,
    get_faces,
)
from .incidence import compute_incidence_matrix
from .simplicial_constructions import (
    construct_alpha_complex,
    construct_complex_from_config,
    construct_del_vr_complex,
    construct_delaunay_complex,
    construct_vr_complex,
    create_simplicial_construction_label,
)
from .validation import (
    BOUNDARY_PROPERTY_TOL,
    validate_boundary_property,
    validate_face_closure,
    validate_simplices_structure,
)

# Define public API of the module
__all__ = [
    # Core classes
    "SimplicialComplex",
    # Computations
    # simplex.py
    "get_faces",
    "compute_orientation_sign",
    "get_boundary",
    "compute_simplicial_closure",
    "compute_permutation_sign",
    # incidence.py
    "compute_incidence_matrix",
    # simplicial_constructions.py
    "construct_delaunay_complex",
    "construct_alpha_complex",
    "construct_del_vr_complex",
    "construct_vr_complex",
    "construct_complex_from_config",
    "create_simplicial_construction_label",
    # validation.py
    "BOUNDARY_PROPERTY_TOL",
    "validate_simplices_structure",
    "validate_face_closure",
    "validate_boundary_property",
]
