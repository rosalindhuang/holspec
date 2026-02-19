# holspec/simplicial/__init__.py
"""
Simplicial complex construction and representation.
"""

from .base import SimplicialComplex
from .simplex import *
from .incidence import *
from .complex_builders import *
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
    'compute_circumradius',
    'compute_permutation_sign',

    # incidence.py
    'compute_incidence_matrix',

    # complex_builders.py
    'build_delaunay_complex',
    'build_alpha_complex',
    'build_vr_complex',

    # validation.py
    'validate_simplices_structure',
    'validate_face_closure',
    'check_boundary_property',
    
]
