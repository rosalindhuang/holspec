# holspec/utilities/io.py
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

from holspec.utilities.helpers import convert_numpy_to_python


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
    - For `attributes`, values are automatically converted to HDF5-compatible types.
      Dicts are automatically serialized to JSON strings. Numpy types are converted to native Python types.
    
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
                target.attrs[key] = to_h5_attribute(value)


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


def to_h5_attribute(obj):
    """
    Convert object to HDF5-compatible attribute type.
    
    HDF5 attributes have limitations compared to datasets. This function handles
    conversion of Python/numpy types to HDF5-storable attributes.
    
    Parameters
    ----------
    obj : any
        Object to convert.
    
    Returns
    -------
    serializable
        HDF5-compatible representation. Dicts become JSON strings, numpy types
        become native Python types.
    
    Raises
    ------
    TypeError
        If object type is not supported.
    
    Notes
    -----
    - Dicts are serialized to JSON strings (can be parsed back with json.loads)
    - Numpy types are converted to native Python types
    - Lists/tuples are recursively processed
    - On read, json.loads() is automatically applied to string attributes in read_h5()
    
    See Also
    --------
    convert_numpy_to_python : Recursively converts numpy types without JSON serialization
    """
    if isinstance(obj, dict): # Dicts must be serialized as JSON strings for HDF5
        return json.dumps(convert_numpy_to_python(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_h5_attribute(item) for item in obj]
    else:
        raise TypeError(
            f"{type(obj).__name__} not supported for HDF5 attributes. "
            f"Supported types: str, int, float, bool, list, dict (as JSON), numpy types."
        )


# %% Test
if __name__ == "__main__":
    import tempfile
    
    print("Testing nested dict serialization with HDF5 attributes")
    print("=" * 60)
    
    # Create a test dict with nested structures and numpy types
    original_dict = {
        'simple_int': 42,
        'numpy_int': np.int64(100),
        'numpy_float': np.float32(3.14),
        'numpy_bool': np.bool_(True),
        'nested': {
            'level2_int': np.int32(50),
            'level2_str': 'hello',
            'level2_list': [1, 2, np.int64(3)],
            'deeply_nested': {
                'level3_value': np.float64(2.718),
                'level3_array': np.array([1, 2, 3])
            }
        },
        'list_with_numpy': [np.int64(10), np.float32(20.5), 'text'],
    }
    
    print("\nOriginal dict:")
    print(original_dict)
    print(f"\nType of nested dict: {type(original_dict['nested'])}")
    
    # Test serialization
    print("\n" + "-" * 60)
    print("Testing to_h5_attribute()...")
    serialized = to_h5_attribute(original_dict)
    print(f"Serialized: {serialized}")
    print(f"Type of serialized: {type(serialized)}")
    if isinstance(serialized, str):
        print("✓ Dict was converted to JSON string (as expected for HDF5)")
    
    # Test HDF5 round-trip
    print("\n" + "-" * 60)
    print("Testing HDF5 save/load round-trip...")
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save to HDF5
        save_h5(
            tmp_path,
            attributes={'test_dict': original_dict}
        )
        print(f"Saved to {tmp_path}")
        
        # Load from HDF5
        datasets, attributes = read_h5(tmp_path)
        loaded_dict = attributes['test_dict']
        
        print(f"\nLoaded dict: {loaded_dict}")
        print(f"Type of loaded dict: {type(loaded_dict)}")
        print(f"Type of loaded nested: {type(loaded_dict.get('nested', 'MISSING'))}")
        
        # Compare
        print("\n" + "-" * 60)
        print("Comparison:")
        
        def compare_values(orig, loaded, path=""):
            """Recursively compare values."""
            if type(orig) != type(loaded):
                # Allow numpy types to match Python types
                if isinstance(orig, (np.integer, np.int64, np.int32)) and isinstance(loaded, int):
                    if int(orig) != loaded:
                        print(f"  ❌ {path}: values differ: {orig} != {loaded}")
                    else:
                        print(f"  ✓ {path}: {orig} == {loaded} (type conversion ok)")
                elif isinstance(orig, (np.floating, np.float32, np.float64)) and isinstance(loaded, float):
                    if not np.isclose(float(orig), loaded):
                        print(f"  ❌ {path}: values differ: {orig} != {loaded}")
                    else:
                        print(f"  ✓ {path}: {orig} ≈ {loaded} (type conversion ok)")
                elif isinstance(orig, np.bool_) and isinstance(loaded, bool):
                    if bool(orig) != loaded:
                        print(f"  ❌ {path}: values differ: {orig} != {loaded}")
                    else:
                        print(f"  ✓ {path}: {orig} == {loaded} (type conversion ok)")
                elif isinstance(orig, np.ndarray) and isinstance(loaded, list):
                    if not np.array_equal(orig, loaded):
                        print(f"  ❌ {path}: array/list differ")
                    else:
                        print(f"  ✓ {path}: array matches list (type conversion ok)")
                else:
                    print(f"  ❌ {path}: type mismatch: {type(orig).__name__} != {type(loaded).__name__}")
                return
            
            if isinstance(orig, dict):
                if set(orig.keys()) != set(loaded.keys()):
                    print(f"  ❌ {path}: keys differ")
                    print(f"      Original: {set(orig.keys())}")
                    print(f"      Loaded: {set(loaded.keys())}")
                else:
                    print(f"  ✓ {path}: dict keys match")
                    for key in orig.keys():
                        compare_values(orig[key], loaded[key], f"{path}.{key}" if path else key)
            elif isinstance(orig, (list, tuple)):
                if len(orig) != len(loaded):
                    print(f"  ❌ {path}: length differs")
                else:
                    for i, (o, l) in enumerate(zip(orig, loaded)):
                        compare_values(o, l, f"{path}[{i}]")
            elif isinstance(orig, np.ndarray):
                if not np.array_equal(orig, loaded):
                    print(f"  ❌ {path}: arrays differ")
                else:
                    print(f"  ✓ {path}: arrays match")
            else:
                if orig != loaded:
                    print(f"  ❌ {path}: {orig} != {loaded}")
                else:
                    print(f"  ✓ {path}: {orig} == {loaded}")
        
        compare_values(original_dict, loaded_dict, "test_dict")
        
    finally:
        # Cleanup
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"\nCleaned up {tmp_path}")
    
    print("\n" + "=" * 60)
    print("Test complete!")