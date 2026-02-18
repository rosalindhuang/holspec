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
    'compute_permutation_sign',

    # incidence.py
    'compute_incidence_matrix',

    # Utilities
    # complex_builders.py

    # validation.py
    'validate_simplices_structure',
    'validate_face_closure',
    'check_boundary_property',
    
]
