"""
Module for point data representation and generation.
"""

from .base import PointData
from .generators import *

# # Define public API of the module
__all__ = [
    # base.py
    'PointData',

    # generators.py
    'generate_triangular_lattice_hex',
    'generate_triangular_lattice_rect',
    'generate_square_lattice',
    'generate_bcc_lattice',
    'generate_random_uniform',
    'generate_regular_polygon',
    'generate_tetrahedron',
    'generate_cube_vertices',

    'add_random_perturbation',
    
    'GENERATOR_MAP',
    'generate_from_config',
    'create_config_label'
]