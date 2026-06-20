# holspec/utilities/__init__.py
"""
Shared implementation utilities for holspec.

This subpackage collects common I/O, validation, numerical, formatting,
hashing, and helper routines used across the framework.
"""
from .io import (
    get_keys_h5,
    initialize_h5,
    inspect_h5,
    join_h5_group,
    read_h5,
    repack_h5,
    save_h5,
)
from natsort import natsorted
from .validation import check_raises
from .numerical import add_noise, compute_content_hash, create_noise_label
from .helpers import (
    convert_notebook,
    convert_numpy_to_python,
    convert_paths_to_relative,
    convert_relative_to_paths,
    count_text_lines,
    export_notebook_outputs,
    format_float_str,
    format_text,
    inspect_dict,
    print_dict,
    print_pipeline_config,
    timed,
)

# Define public API of the module
__all__ = [
    # io.py
    'join_h5_group',
    'save_h5',
    'read_h5',
    'initialize_h5',
    'get_keys_h5',
    'repack_h5',
    'inspect_h5',
    
    # validation.py
    'check_raises',

    # numerical.py
    'compute_content_hash',
    'add_noise',
    'create_noise_label',

    # helpers.py
    'natsorted',
    'convert_numpy_to_python',
    'convert_paths_to_relative',
    'convert_relative_to_paths',
    'count_text_lines',
    'format_float_str',
    'format_text',
    'inspect_dict',
    'print_dict',
    'print_pipeline_config',
    'timed',
    'export_notebook_outputs',
    'convert_notebook',
]
