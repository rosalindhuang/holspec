# holspec/simplicial/__init__.py
"""
Simplicial complex construction and representation.
"""

from .base import SimplicialComplex
from .simplex import *
from .incidence import *
from .simplicial_constructions import *
from .validation import *

# Define public API of the module
__all__ = [
    # Core classes
    'SimplicialComplex',

    # Computations
    # simplex.py
    'get_faces',
    'compute_orientation_sign',
    'get_boundary',
    'compute_simplicial_closure',
    'compute_permutation_sign',

    # incidence.py
    'compute_incidence_matrix',

    # simplicial_constructions.py
    'construct_delaunay_complex',
    'construct_alpha_complex',
    'construct_vr_complex',
    'construct_complex_from_config',
    'create_simplicial_construction_label',

    # validation.py
    'BOUNDARY_PROPERTY_TOL',
    'validate_simplices_structure',
    'validate_face_closure',
    'validate_boundary_property',
    
]
