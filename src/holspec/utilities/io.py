"""
Functions for saving and reading data and configs.
"""

import numpy as np
import h5py
import os
import json
from pathlib import Path
from typing import Any, Optional
import shutil
from natsort import natsorted


# %% Saving and reading to HDF5
def save_h5(
    filepath: str | Path,
    datasets: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, Any]] = None,
    group: Optional[str] = None,
    mode: str = 'replace',
    hdf5_options: Optional[dict] = None
):
    """
    Save datasets and attributes to an HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    datasets : dict[str, array-like], optional
        Dictionary of (name: data) pairs to save as datasets.
    attributes : dict, optional
        Dictionary of (name: value) pairs to save as HDF5 attributes. 
    group : str, optional
        If specified, datasets and attributes are saved under this group.
        If None, saved at root level.
    mode : {'update', 'create', 'replace'}, default 'replace'
        How to handle existing groups:
        - 'update': Merge with existing group (overwrite conflicting datasets/attributes)
        - 'create': Raise error if group already exists
        - 'replace': Delete existing group and recreate
    hdf5_options : dict, optional
        Additional options for dataset creation (e.g., {'compression': 'gzip'}).
        If compression is specified without chunks, chunks=True is set automatically.
    
    Raises
    ------
    ValueError
        If mode is not one of {'update', 'create', 'replace'}.
    FileExistsError
        If mode='create' and the group already exists.
    
    Notes
    -----
    - Parent directories are created automatically if they don't exist.
    - For `attributes`, values must be HDF5-compatible types (str, int, float, bool, 
      list, small numpy arrays). For dicts or complex structures, serialize with json.dumps() first.
    
    Examples
    --------
    >>> # Save to root level
    >>> save_h5('data.h5', datasets={'x': [1, 2, 3]}, attributes={'info': 'test'})
    
    >>> # Save to a group, replacing if exists
    >>> save_h5('data.h5', datasets={'x': data}, group='experiment1', mode='replace')
    
    >>> # Create a new group (error if exists)
    >>> save_h5('data.h5', datasets={'x': data}, group='experiment2', mode='create')
    """
    # Validate mode
    valid_modes = {'update', 'create', 'replace'}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
    
    # Ensure parent directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Set default HDF5 options if not provided
    if hdf5_options is None: 
        hdf5_options = {}
    if 'compression' in hdf5_options and 'chunks' not in hdf5_options:
        hdf5_options['chunks'] = True
    
    with h5py.File(str(filepath), 'a') as f:
        
        # Determine target location (root or group)
        if group is None:
            target = f  # Save at root level
        else:
            group_exists = group in f
            
            if mode == 'create':
                if group_exists:
                    raise FileExistsError(
                        f"Group '{group}' already exists in {filepath}. "
                        f"Use mode='update' to merge or mode='replace' to overwrite."
                    )
                target = f.create_group(group)
            
            elif mode == 'replace':
                if group_exists:
                    del f[group]
                target = f.create_group(group)
            
            else:  # mode == 'update'
                if group_exists:
                    target = f[group]
                else:
                    target = f.create_group(group)
        
        # Save each dataset under target (always overwrite existing datasets)
        if datasets is not None:
            for key, value in datasets.items():
                if key in target:  # Remove existing dataset
                    del target[key]
                target.create_dataset(key, data=np.asarray(value), **hdf5_options)
        
        # Save each attribute (always overwrite existing attributes)
        if attributes is not None:
            for key, value in attributes.items():
                target.attrs[key] = to_serializable(value)


def read_h5(
    filepath: str | Path,
    group: Optional[str] = None,
    dataset_names: Optional[list[str]] = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Read datasets and attributes from an HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    group : str, optional
        If specified, reads from this group. If None, reads from root level.
    dataset_names : list[str], optional
        If specified, only loads these datasets by name.
        If None, loads all datasets in the group/root.
    
    Returns
    -------
    datasets : dict[str, np.ndarray]
        Dictionary of loaded datasets.
    attributes : dict[str, Any]
        Dictionary of attributes.
    
    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    KeyError
        If the specified group does not exist, or if a requested dataset is not found.
    
    Examples
    --------
    >>> # Read all datasets and attributes from root
    >>> datasets, attrs = read_h5('data.h5')
    
    >>> # Read from a specific group
    >>> datasets, attrs = read_h5('data.h5', group='experiment1')
    
    >>> # Read only specific datasets
    >>> datasets, attrs = read_h5('data.h5', group='experiment1', dataset_names=['x', 'y'])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")
    
    with h5py.File(str(filepath), 'r') as f:
        
        # Navigate to target location (root or group)
        if group is None:
            source = f
        else:
            if group not in f:
                raise KeyError(f"Group '{group}' not found in {filepath}")
            source = f[group]
        
        # Load datasets
        datasets = {}
        if dataset_names is None:
            # Load all datasets
            for key in source.keys():
                if isinstance(source[key], h5py.Dataset):
                    datasets[key] = source[key][...]
        else:
            # Load only specified datasets
            for key in dataset_names:
                if key not in source:
                    raise KeyError(f"Dataset '{key}' not found in group '{group or 'root'}'")
                if not isinstance(source[key], h5py.Dataset):
                    raise ValueError(f"'{key}' is not a dataset (it's a {type(source[key]).__name__})")
                datasets[key] = source[key][...]
        
        # Load attributes
        attributes = {}
        for key in source.attrs.keys():
            value = source.attrs[key]
            # Try to parse JSON strings back to dicts/lists
            if isinstance(value, str):
                try:
                    attributes[key] = json.loads(value)
                except json.JSONDecodeError:
                    attributes[key] = value
            else:
                attributes[key] = value
    
    return datasets, attributes


# %% Additional functions for HDF5 handling

def initialize_h5(
    path: str | Path, 
    overwrite: bool = False, 
    verbose: bool = True
) -> Path:
    """
    Initialize an HDF5 file, creating parent directories if needed.
    
    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.
    overwrite : bool, default False
        If True and file exists, clear all contents. If False, open for appending.
    verbose : bool, default True
        If True, print file operation message.
    
    Returns
    -------
    Path
        Resolved path to the initialized file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'w' if overwrite else 'a'
    
    if verbose:
        if path.exists() and overwrite:
            print(f'Initializing HDF5 file (overwrite): {path}')
        elif path.exists():
            print(f'Initializing HDF5 file (append): {path}')
        else:
            print(f'Creating HDF5 file: {path}')
    
    with h5py.File(path, mode):
        pass
    
    return path


def get_keys_h5(
    h5_path: str | Path, 
    group: Optional[str] = None, 
    sort: bool = True
) -> list[str]:
    """
    Return list of dataset and group names from an HDF5 file.
    If group is specified, returns keys from that group. If sort=True, uses natural sorting.
    """
    with h5py.File(h5_path, 'r') as f:
        if group is None:
            keys = list(f.keys())
        else:
            if group not in f:
                raise KeyError(f"Group '{group}' not found in {h5_path}")
            keys = list(f[group].keys())
        
        return natsorted(keys) if sort else keys


def inspect_h5(filepath: str | Path) -> None:
    """Print the structure and attributes of an HDF5 file."""
    def summarize_attr(value):
        if isinstance(value, (str, bytes)):
            return f'(type={type(value).__name__}, len={len(value)})'
        elif isinstance(value, np.ndarray):
            return f'(type=ndarray, shape={value.shape}, dtype={value.dtype})'
        elif isinstance(value, (int, float, bool)):
            return f'(type={type(value).__name__})'
        elif hasattr(value, 'shape'):
            return f'(type={type(value).__name__}, shape={value.shape})'
        else:
            return f'(type={type(value).__name__})'

    def print_attrs(obj, prefix):
        for key, val in obj.attrs.items():
            summary = summarize_attr(val)
            print(f"{prefix}@{key} {summary}")

    def print_h5_structure(name, obj, prefix=""):
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{name}/")
            print_attrs(obj, prefix + "    ")
            for key in obj:
                print_h5_structure(key, obj[key], prefix + "    ")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}{name} (shape={obj.shape}, dtype={obj.dtype})")
            print_attrs(obj, prefix + "    ")
    
    with h5py.File(filepath, 'r') as f:
        print(f"{filepath}/")
        print_attrs(f, prefix="    ")
        for key in f:
            print_h5_structure(key, f[key], prefix="    ")


def repack_h5(
    path: str | Path, 
    output_path: Optional[str | Path] = None
) -> Path:
    """
    Repack an HDF5 file to optimize storage. Useful after deleting datasets/groups.
    If output_path is None, repacks in-place.
    """
    path = Path(path)
    output_path = Path(output_path) if output_path is not None else path
    tmp = Path(f"{output_path}.tmp")
    
    try:
        with h5py.File(path, 'r') as src, h5py.File(tmp, 'w') as dst:
            dst.attrs.update(src.attrs)
            for name in src:
                src.copy(name, dst)
            dst.flush()
        shutil.copystat(path, tmp)
        os.replace(tmp, output_path)
        return output_path
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise


def to_serializable(obj):
    """
    Convert object to HDF5-compatible attribute type.
    
    Handles numpy types and basic Python types. For complex structures
    (dicts, nested objects), use json.dumps() before passing to save_h5().
    
    Parameters
    ----------
    obj : any
        Object to convert.
    
    Returns
    -------
    serializable
        HDF5-compatible representation.
    
    Raises
    ------
    TypeError
        If object type is not supported.
    """
    if isinstance(obj, np.ndarray):
        # Convert small arrays to lists, reject large ones
        if obj.size > 100:
            raise TypeError(
                f"Large arrays ({obj.size} elements) should be datasets, not attributes"
            )
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        raise TypeError(
            f"Dict attributes not supported. Use json.dumps() to serialize dicts as JSON strings."
        )
    else:
        raise TypeError(
            f"Attribute type {type(obj).__name__} not supported. "
            f"Supported types: str, int, float, bool, list, numpy types. "
            f"For complex objects, serialize with json.dumps()."
        )


# %% Test
if __name__ == "__main__":
    pass