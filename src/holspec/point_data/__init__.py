# holspec/point_data/__init__.py
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
from .point_generators import *
from .data_loaders import *
from .data_generation import make_ensemble_config, create_ensemble_label, run_data_generation
from .validation import validate_positions, validate_distances


# Define public API of the module
__all__ = [
    # Core classes (used in pipeline)
    'PointData',
    'PointDataEnsemble',

    # validation.py
    'validate_positions',
    'validate_distances',

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

    'POINT_GENERATOR_REGISTRY',
    'generate_points_from_config',
    'create_point_generator_label',

    # data_generation.py
    'make_ensemble_config',
    'create_ensemble_label',
    'run_data_generation',

    # loaders.py
    # TODO: Add loader functions to __all__ when implemented

]