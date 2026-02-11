"""
Helper functions for point data generation and storage.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Any, Optional

from holspec.utilities import save_h5, read_h5, compute_content_hash
from .generators import GENERATOR_MAP

# %% Generate point data

def generate_from_config(config: dict) -> np.ndarray:
    """
    Generate point data from configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain 'generator' (str) and 'params' (dict).

    Returns
    -------
    positions : np.ndarray
        Generated point positions.
    """
    # Validate config
    if 'generator' not in config:
        raise ValueError("Config must contain 'generator' key")
    if 'params' not in config:
        raise ValueError("Config must contain 'params' key")

    generator_name = config['generator']
    params = config['params']

    # Get generator function
    if generator_name not in GENERATOR_MAP:
        raise ValueError(f"Unknown generator: {generator_name}")

    generator_func = GENERATOR_MAP[generator_name]

    # Generate positions
    return generator_func(**params)


# %% I/O and metadata handling for point data
def save_point_data(
    filepath: str | Path,
    positions: np.ndarray,
    config: dict,
    group: Optional[str] = None,
    mode: str = 'replace',
    hdf5_options: Optional[dict] = None
) -> str:
    """
    Save point data with metadata to HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file to save data.
    positions : np.ndarray
        Point positions array of shape (n_points, dimension).
    config : dict
        Configuration dictionary with 'generator' and 'params' keys.
    
    Returns
    -------
    content_hash : str
        SHA256 hash of saved positions for verification.
    """
    # Validate config
    if 'generator' not in config:
        raise ValueError("Config must contain 'generator' key")
    if 'params' not in config:
        raise ValueError("Config must contain 'params' key")
    
    # Validate positions
    if positions.ndim != 2:
        raise ValueError(f"Positions must be 2D array, got shape {positions.shape}")
    
    # Compute content hash
    content_hash = compute_content_hash(positions)

    # Generate config label
    
    # Build metadata
    metadata = {
        'generator': config['generator'],
        'generator_params': json.dumps(config['params']),
        'shape': positions.shape,
        'content_hash': content_hash,
        'config_label': create_config_label(config),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save with all options passed through
    save_h5(
        filepath,
        datasets={'positions': positions},
        attributes=metadata,
        mode=mode,
        group=group,
        hdf5_options=hdf5_options if hdf5_options else None
    )
    
    return content_hash


def read_point_data(
    filepath: str | Path,
    group: Optional[str] = None,
    validate_hash: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """
    Read point data and metadata from HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file.
    
    Returns
    -------
    positions : np.ndarray
        Point positions array.
    config : dict
        Reconstructed configuration with 'generator' and 'params' keys.
    metadata : dict
        Metadata associated with the point data (generation config, 
        content_hash, timestamp, shape).
    validate_hash : bool
        Whether to validate content hash against positions data.
    """    
    # Read all datasets and attributes
    datasets, metadata = read_h5(filepath, group=group)
    
    # Extract positions
    if 'positions' not in datasets:
        raise ValueError(f"No 'positions' dataset found in {filepath}")
    positions = datasets['positions']
    
    # Reconstruct config
    if 'generator' not in metadata or 'generator_params' not in metadata:
        raise ValueError(f"Missing generator metadata in {filepath}")
    config = {
        'generator': metadata['generator'],
        'params': metadata['generator_params']
    }

    # Validate content hash if requested
    if validate_hash:
        if 'content_hash' not in metadata:
            print("Warning: Metadata missing 'content_hash' for validation, skipping hash check")
        stored_hash = metadata['content_hash']
        computed_hash = compute_content_hash(positions, length=len(stored_hash))
        if computed_hash != stored_hash:
            raise ValueError(f"Content hash mismatch: expected {stored_hash}, got {computed_hash}")
    
    return positions, config, metadata


def create_config_label(config):
    """
    Create a unique label from configuration.
    Format: generator_param1_param2_...
    """
    generator = config['generator']
    params = config['params']
    
    # Start with generator name
    parts = [generator]
    
    # Iterate through parameters in original order
    for key, value in params.items():
        # Get abbreviated parameter name (first 2 letters, strip underscores)
        param_abbr = key.replace('_', '')[:2]
        
        # Format value based on type
        if isinstance(value, bool):
            value_str = '1' if value else '0'
        elif isinstance(value, float):
            value_str = f"{value:.2f}".rstrip('0').rstrip('.').replace('.', 'p')
        elif isinstance(value, str):
            value_str = value[:4].lower().replace('_', '')
        else:
            # Handle int and other types
            value_str = str(value).replace('.', 'p')
        
        parts.append(f"{param_abbr}{value_str}")
    
    return '_'.join(parts)

