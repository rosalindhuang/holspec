"""
Functions for saving and reading data and configs.
"""

import numpy as np
import pandas as pd
import h5py
import os
import json
from pathlib import Path
import shutil
from natsort import natsorted

# %% Saving and reading to HDF5
def save_h5(
    file_path: str | Path,
    datasets: dict[str, any] = None,
    attributes: dict[str, any] = None,
    group: str = None,
    overwrite_group: bool = False,
    hdf5_options: dict | None = None
):
    '''
    Save datasets and attributes to an HDF5 file.
    - `datasets`: dict of (name: array-like data) to save as datasets (optional)
    - `attributes`: dict of (name: value) to save as attributes (optional)
    - `group`: if specified, datasets and attributes are saved under that group. 
    - `overwrite_group`: if True, existing groups will be deleted and recreated.
       If False, will append to existing groups. Datasets and attributes are always overwritten.
    - `hdf5_options`: dict of options for dataset creation (e.g., compression)
    '''
    
    # Set default HDF5 options if not provided
    if hdf5_options is None: 
        hdf5_options = {}
    if 'compression' in hdf5_options and 'chunks' not in hdf5_options:
        hdf5_options['chunks'] = True
    
    with h5py.File(str(file_path), 'a') as f:
        
        # Determine target location (root or group)
        if group is None:
            target = f  # Save at root level
        else:
            # Handle group creation based on overwrite_group setting
            if group in f:
                if overwrite_group:
                    del f[group]
                    target = f.create_group(group)
                else:
                    target = f[group]  # Use existing group
            else:
                target = f.create_group(group)
        
        # Save each dataset under target (always overwrite existing datasets)
        if datasets is not None:
            for key, value in datasets.items():
                if key in target:  # Remove existing dataset
                    del target[key]
                target.create_dataset(key, data=np.asarray(value), **hdf5_options)
        
        # Save each attribute as serializable under target (always overwrite existing attributes)
        if attributes is not None:
            for key, value in attributes.items():
                target.attrs[key] = to_serializable(value)

def read_h5(
    file_path: str | Path,
    group: str = None
) -> tuple[dict[str, np.ndarray], dict[str, any]]:
    '''
    Read datasets and attributes from an HDF5 file, optionally from a specified group.
    - `group`: if specified, read datasets and attributes from that group.
    - Returns:
        - `datasets`: dict of (name: np.ndarray) for each dataset
        - `attributes`: dict of (name: value) for each attribute
    '''
    
    with h5py.File(str(file_path), 'r') as f:
        
        # Determine source location (root or group)
        if group is None:
            source = f  # Read from root level
        else:
            if group not in f:
                return {}, {}
            source = f[group]
        
        # Read each dataset (skip subgroups)
        datasets = {}
        for key in source.keys():
            if isinstance(source[key], h5py.Dataset):
                datasets[key] = source[key][...]
        
        # Read attributes and parse JSON strings back to dicts
        attributes = {}
        for key, value in source.attrs.items():
            if isinstance(value, (str, bytes)):
                # Try to parse as JSON in case it's a serialized dict
                try:
                    attributes[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    attributes[key] = value
            else:
                attributes[key] = value
    
    return datasets, attributes


# %% Additional functions for HDF5 handling
def initialize_h5(path: Path, overwrite = False, verbose = True):
    """
    Initialize an HDF5 file at the given path.
    If `overwrite` = True, any existing file will be cleared.
    """
    if path.exists():
        if overwrite:
            if verbose: print(f'Writing to file (overwrite): {path}\n')
            with h5py.File(path, 'w'): pass
        else:
            if verbose: print(f'Writing to file (append): {path}\n')
            with h5py.File(path, 'a'): pass
    else:
        if verbose: print(f'Writing to file: {path}\n')
        with h5py.File(path, 'a'): pass

def get_keys_h5(h5_path, group=None, sort=True):
    '''Return a list of dataset and group names at the root of an HDF5 file or within a specific group.
    
    Parameters:
    -----------
    h5_path : str or Path
        Path to the HDF5 file
    group : str, optional
        Group path within the HDF5 file. If None, returns keys from root.
    sort : bool, default True
        Whether to sort the keys naturally
    '''
    with h5py.File(h5_path, 'r') as f:
        if group is None:
            keys = list(f.keys())
        else:
            keys = list(f[group].keys())
        
        if sort: 
            return natsorted(keys)
        else: 
            return keys

def inspect_h5(file_path):
    '''Print the structure and attributes of an HDF5 file.'''
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
    
    with h5py.File(file_path, 'r') as f:
        print(f"{file_path}/")
        print_attrs(f, prefix="    ")
        for key in f:
            print_h5_structure(key, f[key], prefix="    ")
            
def repack_h5(path: str, in_place: bool = True):
    """Repack an HDF5 file to optimize storage."""
    tmp = f"{path}.tmp"
    try:
        with h5py.File(path,'r') as src, h5py.File(tmp,'w') as dst:
            dst.attrs.update(src.attrs)
            for name in src: src.copy(name, dst)
            dst.flush()
        shutil.copystat(path, tmp)
        if in_place: os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        raise
   
def to_serializable(obj):
    '''Convert an object to a serializable format for JSON or HDF5 storage.'''
    if isinstance(obj, dict): 
        # Convert dict to JSON string for HDF5 attribute compatibility
        return json.dumps({k: to_serializable(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)): return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)): return round(obj.item(), 12)
    else: return obj

