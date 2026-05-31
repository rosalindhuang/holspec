# holspec/point_data/__init__.py
"""
Point cloud data containers and input preparation.

This subpackage defines the input layer of the holspec pipeline: point data,
point data ensembles, configurable data generators, external data loaders, and
orchestration helpers for reproducible input datasets.
"""

from .base import PointData
from .ensemble import PointDataEnsemble
from .data_generators import (
    POINT_GENERATOR_REGISTRY,
    create_point_generator_label,
    generate_bcc_lattice,
    generate_cube_vertices,
    generate_points_from_config,
    generate_random_uniform,
    generate_regular_polygon,
    generate_square_lattice,
    generate_tetrahedron,
    generate_triangular_lattice_hex,
    generate_triangular_lattice_rect,
)
from .input_preparation import (
    create_ensemble_label,
    make_ensemble_config,
    run_data_generation,
)
from .data_loaders import (
    SUPPORTED_FILE_FORMATS,
    load_array_from_config,
    load_distances_array,
    load_positions_array,
)
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
    # data_generators.py
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

    # input_preparation.py
    'make_ensemble_config',
    'create_ensemble_label',
    'run_data_generation',

    # data_loaders.py
    'SUPPORTED_FILE_FORMATS',
    'load_array_from_config',
    'load_positions_array',
    'load_distances_array',
]
