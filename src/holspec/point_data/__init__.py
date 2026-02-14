"""
Point cloud data containers and utilities.

Core classes (used by pipeline):
- PointData: Container for single point cloud
- PointDataEnsemble: Container for ensemble of related clouds

Utilities (for data creation/loading):  
- generators.py: Synthetic point cloud generators
"""

from .base import PointData
from .ensemble import PointDataEnsemble
from .generators import *
from .loaders import *


# Define public API of the module
__all__ = [
    # Core classes (used in pipeline)
    'PointData',
    'PointDataEnsemble',

    # Utilities (for data creation/loading)
    # generators.py
    'generate_triangular_lattice_hex',
    'generate_triangular_lattice_rect',
    'generate_square_lattice',
    'generate_bcc_lattice',
    'generate_random_uniform',
    'generate_regular_polygon',
    'generate_tetrahedron',
    'generate_cube_vertices',

    'GENERATOR_MAP',
    'generate_from_config',
    'create_config_label'

    # loaders.py
    # TODO: Add loader functions to __all__ when implemented

]