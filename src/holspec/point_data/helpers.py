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


def create_config_label(config):
    """
    Create a unique label from point data generation config.
    Format: generator_param1_param2_...
    """
    # Validate config
    if 'generator' not in config:
        raise ValueError("Config must contain 'generator' key")
    if 'params' not in config:
        raise ValueError("Config must contain 'params' key")

    generator_name = config['generator']
    params = config['params']
    
    # Start with generator name
    parts = [generator_name]
    
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


# %% I/O and metadata handling for point data

def save_point_data(
    filepath: str | Path,
    positions: np.ndarray,
    config: Optional[dict] = None,
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
    config : dict, optional
        Configuration dictionary (e.g., with 'generator' and 'params' keys).
        If None, only basic metadata is stored.
    
    Returns
    -------
    content_hash : str
        SHA256 hash of saved positions for verification.
    """
    # Validate positions
    if positions.ndim != 2:
        raise ValueError(f"Positions must be 2D array, got shape {positions.shape}")
    
    # Compute content hash
    content_hash = compute_content_hash(positions)

    # Build metadata - always include these three
    metadata = {
        'shape': positions.shape,
        'content_hash': content_hash,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add config as single JSON string if provided
    if config is not None:
        metadata['config'] = json.dumps(config)
    
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
) -> tuple[np.ndarray, dict]:
    """
    Read point data and metadata from HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file.
    validate_hash : bool
        Whether to validate content hash against positions data.
    
    Returns
    -------
    positions : np.ndarray
        Point positions array.
    metadata : dict
        Metadata associated with the point data (shape, content_hash, 
        timestamp, and optionally config).
    """    
    # Read all datasets and attributes
    datasets, metadata = read_h5(filepath, group=group)
    
    # Extract positions
    if 'positions' not in datasets:
        raise ValueError(f"No 'positions' dataset found in {filepath}")
    positions = datasets['positions']

    # Validate content hash if requested
    if validate_hash:
        if 'content_hash' not in metadata:
            print("Warning: Metadata missing 'content_hash' for validation, skipping hash check")
        else:
            stored_hash = metadata['content_hash']
            computed_hash = compute_content_hash(positions, length=len(stored_hash))
            if computed_hash != stored_hash:
                raise ValueError(f"Content hash mismatch: expected {stored_hash}, got {computed_hash}")
    
    return positions, metadata


