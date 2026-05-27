"""
Array loaders for external point data files.

This module provides low-level helpers for reading coordinate arrays and
distance matrices from simple file formats. The helpers intentionally return
plain NumPy arrays; ``PointData`` owns shape validation and framework-level
object construction.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_FILE_FORMATS = {"csv", "tsv", "txt", "npy"}
_DEFAULT_DELIMITERS = {"csv": ",", "tsv": "\t", "txt": None}


# =============================================================================
# Public Loading API
# =============================================================================


def load_array_from_config(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Load an external point data array from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration with keys ``data_type`` and ``filepath``. Optional keys
        are ``file_format`` and ``params``. ``data_type`` must be either
        ``"positions"`` or ``"distances"``. ``params`` are passed to
        ``load_positions_array`` or ``load_distances_array``.
    base_dir : str or Path, optional
        Base directory for resolving relative filepaths.

    Returns
    -------
    array : ndarray
        Loaded numeric array.
    """
    if "data_type" not in config:
        raise ValueError("Config must contain 'data_type' key")
    if "filepath" not in config:
        raise ValueError("Config must contain 'filepath' key")

    data_type = config["data_type"]
    params = dict(config.get("params", {}))
    file_format = config.get("file_format")

    if data_type == "positions":
        return load_positions_array(
            config["filepath"],
            file_format=file_format,
            base_dir=base_dir,
            **params,
        )
    if data_type == "distances":
        return load_distances_array(
            config["filepath"],
            file_format=file_format,
            base_dir=base_dir,
            **params,
        )

    raise ValueError(
        "Config 'data_type' must be either 'positions' or 'distances', "
        f"got {data_type!r}"
    )


def load_positions_array(
    filepath: str | Path,
    file_format: str | None = None,
    base_dir: str | Path | None = None,
    columns: list[str] | list[int] | None = None,
    delimiter: str | None = None,
    header: int | str | None = "infer",
    dtype: type | str | None = float,
    **kwargs: Any,
) -> np.ndarray:
    """
    Load point positions from a simple external file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.
    file_format : {'csv', 'tsv', 'txt', 'npy'}, optional
        File format. If omitted, inferred from the file suffix.
    base_dir : str or Path, optional
        Base directory for resolving relative filepaths.
    columns : list of str or int, optional
        Columns to select for CSV or TSV files. Passed to ``pandas.read_csv`` as
        ``usecols``.
    delimiter : str, optional
        Delimiter override. Defaults to comma for CSV, tab for TSV, and
        whitespace for TXT.
    header : int, str, or None, default='infer'
        Header argument for CSV or TSV files.
    dtype : type, str, or None, default=float
        Numeric dtype conversion. If None, keep the loaded dtype.
    **kwargs
        Additional file-format-specific options passed to ``pandas.read_csv``,
        ``numpy.loadtxt``, or ``numpy.load``.

    Returns
    -------
    positions : ndarray
        Loaded positions array.
    """
    return _load_array(
        filepath=filepath,
        file_format=file_format,
        base_dir=base_dir,
        columns=columns,
        delimiter=delimiter,
        header=header,
        dtype=dtype,
        **kwargs,
    )


def load_distances_array(
    filepath: str | Path,
    file_format: str | None = None,
    base_dir: str | Path | None = None,
    columns: list[str] | list[int] | None = None,
    delimiter: str | None = None,
    header: int | str | None = None,
    dtype: type | str | None = float,
    **kwargs: Any,
) -> np.ndarray:
    """
    Load a pairwise distance matrix from a simple external file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.
    file_format : {'csv', 'tsv', 'txt', 'npy'}, optional
        File format. If omitted, inferred from the file suffix.
    base_dir : str or Path, optional
        Base directory for resolving relative filepaths.
    columns : list of str or int, optional
        Columns to select for CSV or TSV files. Mainly useful for unusual
        distance-matrix files with extra columns.
    delimiter : str, optional
        Delimiter override. Defaults to comma for CSV, tab for TSV, and
        whitespace for TXT.
    header : int, str, or None, default=None
        Header argument for CSV or TSV files. Distance matrices are assumed
        headerless by default.
    dtype : type, str, or None, default=float
        Numeric dtype conversion. If None, keep the loaded dtype.
    **kwargs
        Additional file-format-specific options passed to ``pandas.read_csv``,
        ``numpy.loadtxt``, or ``numpy.load``.

    Returns
    -------
    distances : ndarray
        Loaded distance matrix.
    """
    return _load_array(
        filepath=filepath,
        file_format=file_format,
        base_dir=base_dir,
        columns=columns,
        delimiter=delimiter,
        header=header,
        dtype=dtype,
        **kwargs,
    )


# =============================================================================
# Helpers
# =============================================================================

def _load_array(
    filepath: str | Path,
    file_format: str | None,
    base_dir: str | Path | None,
    columns: list[str] | list[int] | None,
    delimiter: str | None,
    header: int | str | None,
    dtype: type | str | None,
    **kwargs: Any,
) -> np.ndarray:
    """Load an array from a supported file format."""
    filepath = _resolve_filepath(filepath, base_dir=base_dir)
    file_format = _normalize_file_format(file_format, filepath)

    if file_format in {"csv", "tsv"}:
        sep = delimiter if delimiter is not None else _DEFAULT_DELIMITERS[file_format]
        dataframe = pd.read_csv(
            filepath,
            sep=sep,
            header=header,
            usecols=columns,
            **kwargs,
        )
        array = dataframe.to_numpy()
    elif file_format == "txt":
        array = np.loadtxt(
            filepath,
            delimiter=delimiter,
            dtype=dtype if dtype is not None else float,
            **kwargs,
        )
    elif file_format == "npy":
        array = np.load(filepath, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format {file_format!r}. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if dtype is not None and file_format != "txt":
        array = array.astype(dtype)

    return np.asarray(array)


def _resolve_filepath(
    filepath: str | Path,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve a data filepath, optionally relative to a base directory."""
    filepath = Path(filepath)
    if not filepath.is_absolute() and base_dir is not None:
        filepath = Path(base_dir) / filepath
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return filepath


def _normalize_file_format(
    file_format: str | None,
    filepath: Path,
) -> str:
    """Normalize or infer a supported file format."""
    if file_format is None:
        if not filepath.suffix:
            raise ValueError(
                f"Could not infer file format from path without suffix: {filepath}"
            )
        file_format = filepath.suffix.lstrip(".")

    file_format = file_format.lower().lstrip(".")
    if file_format not in SUPPORTED_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file format {file_format!r}. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )
    return file_format


__all__ = [
    "SUPPORTED_FILE_FORMATS",
    "load_array_from_config",
    "load_positions_array",
    "load_distances_array",
]
