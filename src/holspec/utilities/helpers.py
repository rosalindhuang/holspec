# holspec/utilities/helpers.py
"""
Helper functions including:
- Inspecting dictionary structures
- Exporting Jupyter notebook outputs and converting notebooks to other formats
- Formatting text for printing
- Timing function execution
"""

import numpy as np
from pathlib import Path
import time
import textwrap
import re
import yaml
import shutil
import subprocess


# =============================================================================
# Type conversion
# =============================================================================


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy and path-like types to native Python types.

    Parameters
    ----------
    obj : any
        Object to convert. Can be a dict, list, numpy array, numpy scalar, Path,
        or any other type.

    Returns
    -------
    any
        Object with all numpy types converted to native Python types.

    Notes
    -----
    - numpy integers → int
    - numpy floats → float
    - numpy bool → bool
    - numpy arrays → list (recursively)
    - pathlib Path objects → str
    - dict values → recursively converted
    - list items → recursively converted
    - other types → unchanged

    Examples
    --------
    >>> import numpy as np
    >>> config = {'n': np.int64(100), 'value': np.float32(1.5)}
    >>> convert_numpy_to_python(config)
    {'n': 100, 'value': 1.5}
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def convert_paths_to_relative(obj, root: Path):
    """
    Recursively convert Path objects to relative path strings.

    Intended for serializing file path structures into YAML configs, where
    paths are stored as strings relative to the project root.

    Parameters
    ----------
    obj : dict, list, or Path
        Nested structure (dicts and/or lists) with Path objects at the leaves.
        String leaves are passed through unchanged.
    root : Path
        Root path to make paths relative to.

    Returns
    -------
    dict, list, or str
        Same structure with Path leaves replaced by relative path strings.

    Raises
    ------
    TypeError
        If a leaf value is neither a Path nor a str.

    Examples
    --------
    >>> from pathlib import Path
    >>> root = Path('/project')
    >>> obj = {'a': Path('/project/data/a.h5'), 'b': {'c': Path('/project/data/b.h5')}}
    >>> convert_paths_to_relative(obj, root)
    {'a': 'data/a.h5', 'b': {'c': 'data/b.h5'}}
    """
    if isinstance(obj, dict):
        return {
            key: convert_paths_to_relative(value, root) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_paths_to_relative(item, root) for item in obj]
    elif isinstance(obj, Path):
        return str(obj.relative_to(root))
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(
            f"Expected Path, str, dict, or list; got {type(obj).__name__!r}."
        )


def convert_relative_to_paths(obj, root: Path):
    """
    Recursively convert relative path strings to absolute Path objects.

    Inverse of convert_paths_to_relative. Intended for reconstructing file path
    structures read back from YAML configs.

    Parameters
    ----------
    obj : dict, list, or str
        Nested structure (dicts and/or lists) with relative path strings at the leaves.
    root : Path
        Root path to resolve relative paths against.

    Returns
    -------
    dict, list, or Path
        Same structure with string leaves replaced by absolute Path objects.

    Raises
    ------
    TypeError
        If a leaf value is not a str.

    Examples
    --------
    >>> from pathlib import Path
    >>> root = Path('/project')
    >>> obj = {'a': 'data/a.h5', 'b': {'c': 'data/b.h5'}}
    >>> convert_relative_to_paths(obj, root)
    {'a': PosixPath('/project/data/a.h5'), 'b': {'c': PosixPath('/project/data/b.h5')}}
    """
    if isinstance(obj, dict):
        return {
            key: convert_relative_to_paths(value, root) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_relative_to_paths(item, root) for item in obj]
    elif isinstance(obj, str):
        return root / obj
    else:
        raise TypeError(f"Expected str, dict, or list; got {type(obj).__name__!r}.")


# =============================================================================
# File inspection utilities
# =============================================================================


def count_text_lines(paths, method="auto"):
    """
    Count physical newline-delimited lines in text files.

    Parameters
    ----------
    paths : path-like or iterable of path-like
        Text file path(s) to count.
    method : {'auto', 'wc', 'python'}, default='auto'
        Counting backend. ``'wc'`` shells out to ``wc -l`` and is fast on
        Unix-like systems. ``'python'`` uses a portable binary chunk reader.
        ``'auto'`` uses ``wc`` when available and otherwise falls back to
        ``'python'``.

    Returns
    -------
    dict[Path, int]
        Mapping from input path to line count.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(path) for path in paths]
    if not paths:
        return {}

    if method not in {"auto", "wc", "python"}:
        raise ValueError("method must be one of {'auto', 'wc', 'python'}")

    if method == "auto":
        method = "wc" if shutil.which("wc") is not None else "python"

    if method == "wc":
        result = subprocess.run(
            ["wc", "-l", *[str(path) for path in paths]],
            capture_output=True,
            text=True,
            check=True,
        )
        line_counts = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            count, filename = parts
            if filename == "total":
                continue
            line_counts[Path(filename)] = int(count)
        return {path: line_counts[path] for path in paths}

    line_counts = {}
    for path in paths:
        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        line_counts[path] = count
    return line_counts


# =============================================================================
# String utilities
# =============================================================================


def format_float_str(
    value: float, fmt: str | None = "g", strip_zeros: bool = True
) -> str:
    """
    Format a float for use in labels with trailing zeros removed and dots replaced.

    Parameters
    ----------
    value : float
        The float value to format.
    fmt : str or None, optional
        Format specifier (e.g., 'g', '.2e', '.0e', '.3f'). Default is 'g'.
        If None, defaults to 'g'.
    strip_zeros : bool, default=True
        If True, trailing zeros after the decimal point are removed.
        Set to False to preserve them (e.g. '1.50' with '.2f').

    Returns
    -------
    str
        Formatted string with trailing zeros optionally removed and '.' replaced by 'p'.
    """
    # Handle None fmt
    if fmt is None:
        fmt = "g"

    # Format the value
    formatted = f"{value:{fmt}}"

    # Remove trailing zeros after decimal point
    if "." in formatted and strip_zeros:
        if "e" in formatted.lower():
            # Handle scientific notation: split at 'e', trim mantissa, rejoin
            parts = formatted.lower().split("e")
            parts[0] = parts[0].rstrip("0").rstrip(".")
            formatted = "e".join(parts)
        else:
            # Handle regular decimal
            formatted = formatted.rstrip("0").rstrip(".")

    # Replacements decimal point with 'p'
    formatted = formatted.replace(".", "p")

    return formatted


def format_text(
    text: str, max_width: int = 80, preserve_paragraphs: bool = True
) -> str:
    """
    Format text with intelligent line wrapping that preserves indentation.

    Handles bullet points with proper hanging indentation and avoids breaking words.

    Parameters
    ----------
    text : str
        The text to format.
    max_width : int, default=80
        Maximum line width before wrapping.
    preserve_paragraphs : bool, default=True
        If True, preserve empty lines to maintain paragraph separation.

    Returns
    -------
    str
        Formatted text with proper line wrapping and preserved indentation.
    """
    lines = text.split("\n")
    formatted_lines = []

    # Pattern to match bullet points (-, *, +, or numbered like 1., 2., etc.)
    bullet_pattern = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")

    for line in lines:
        # Preserve empty lines if requested
        if not line.strip() and preserve_paragraphs:
            formatted_lines.append("")
            continue

        # Detect basic indentation (spaces or tabs at the beginning)
        indent = ""
        for char in line:
            if char in " \t":
                indent += char
            else:
                break

        # Get the content without leading whitespace
        content = line.lstrip()

        if not content:  # Line was only whitespace
            formatted_lines.append("")
            continue

        # Check if this is a bullet point
        bullet_match = bullet_pattern.match(line)
        if bullet_match:
            # For bullet points, create hanging indentation
            bullet_indent = bullet_match.group(1)  # Initial spaces/tabs
            bullet_marker = bullet_match.group(2)  # The bullet character(s)
            bullet_text = line[bullet_match.end() :]  # Text after bullet

            # First line uses the original indent + bullet
            first_line_prefix = bullet_indent + bullet_marker + " "
            # Continuation lines align with the text after the bullet
            continuation_indent = bullet_indent + " " * len(bullet_marker + " ")

            # Calculate available width for content
            available_width = max_width - len(continuation_indent)
            if available_width < 20:
                available_width = 20

            # Wrap the bullet text
            if bullet_text.strip():
                wrapped_lines = textwrap.fill(
                    bullet_text,
                    width=available_width,
                    break_long_words=False,
                    break_on_hyphens=True,
                ).split("\n")

                # Add the first line with bullet
                formatted_lines.append(first_line_prefix + wrapped_lines[0])

                # Add continuation lines with hanging indent
                for wrapped_line in wrapped_lines[1:]:
                    formatted_lines.append(continuation_indent + wrapped_line)
            else:
                # Empty bullet point
                formatted_lines.append(first_line_prefix)

        else:
            # Regular line (not a bullet point)
            # Calculate available width for content (accounting for indentation)
            available_width = max_width - len(indent)
            if available_width < 20:
                available_width = 20

            # Wrap the content
            wrapped_lines = textwrap.fill(
                content,
                width=available_width,
                break_long_words=False,
                break_on_hyphens=True,
            ).split("\n")

            # Add the original indentation to each wrapped line
            for wrapped_line in wrapped_lines:
                formatted_lines.append(indent + wrapped_line)

    return "\n".join(formatted_lines)


# =============================================================================
# Printing utilities
# =============================================================================


def inspect_dict(
    data_dict: dict,
    dict_name: str = "dict",
    max_depth: int | None = None,
    prefix: str = "",
    indent: str = "  ",
    _current_depth: int = 0,
    _indent: str = "",
) -> None:
    """
    Print the structure and contents of a nested dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary to inspect.
    dict_name : str, default="dict"
        Name to display for the root dictionary.
    max_depth : int, optional
        Maximum depth to traverse. If None, traverse all levels.
    prefix : str, default=''
        String prepended to every printed line (e.g., '    ' to indent the whole block).
    indent : str, default='  '
        String used for each level of indentation (e.g., '  ' for two spaces).
    _current_depth : int, optional
        Internal parameter for tracking recursion depth (do not use).
    _indent : str, optional
        Internal parameter for accumulating indentation (do not use).

    Returns
    -------
    None
    """

    def summarize_value(value):
        """Summarize the type and properties of a value."""
        if isinstance(value, dict):
            return f"(type=dict, keys={len(value)})"
        elif isinstance(value, np.ndarray):
            return f"(type=ndarray, shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, list):
            return f"(type=list, len={len(value)})"
        elif isinstance(value, tuple):
            return f"(type=tuple, len={len(value)})"
        elif isinstance(value, str):
            return f"(type=str, len={len(value)})"
        elif isinstance(value, (int, float, bool)):
            return f"(type={type(value).__name__}, value={value})"
        elif hasattr(value, "shape"):
            return f"(type={type(value).__name__}, shape={value.shape})"
        elif hasattr(value, "__len__"):
            return f"(type={type(value).__name__}, len={len(value)})"
        else:
            return f"(type={type(value).__name__})"

    # Print root header
    if _current_depth == 0:
        print(f"{prefix}{dict_name}/")

    # Check depth limit
    if max_depth is not None and _current_depth >= max_depth:
        return

    # Iterate through dictionary items
    for key, value in data_dict.items():
        if isinstance(value, dict):
            print(f"{prefix}{_indent}{indent}{key}/")
            inspect_dict(
                value,
                dict_name=key,
                max_depth=max_depth,
                prefix=prefix,
                indent=indent,
                _current_depth=_current_depth + 1,
                _indent=_indent + indent,
            )
        else:
            summary = summarize_value(value)
            print(f"{prefix}{_indent}{indent}{key} {summary}")


def print_dict(
    data_dict: dict,
    header: str | None = None,
    print_items: bool = True,
) -> None:
    """
    Print a formatted section-based summary of a nested dictionary.

    Treats each top-level key whose value is a dict as a named section.
    Within each section, prints key-value pairs with aligned columns.
    Non-dict top-level values are skipped.

    - Scalar values are printed as ``key: value``.
    - List values are printed as ``key (N items):`` followed by each item indented.
    - Dict values are printed as ``key (N entries):`` followed by each key indented.

    Label widths are auto-computed per section from the longest key name.

    Parameters
    ----------
    data_dict : dict
        Dictionary to print.
    header : str, optional
        If provided, prints a header block with ``=`` separators.
    print_items : bool, default=True
        If True, print individual items within list/dict values.
    """
    # Header
    if header is not None:
        print("=" * 60)
        print(header)
        print("=" * 60)

    # Sections
    sep = "-" * 60
    for section_name, section in data_dict.items():
        if not isinstance(section, dict):
            continue

        print(sep)
        print(section_name.capitalize())
        print(sep)

        def _collection_hint(v):
            """Return a concise type-hint string for dict/list values."""
            if isinstance(v, dict):
                if v:
                    k0, v0 = next(iter(v.items()))
                    return f"dict[{type(k0).__name__}, {type(v0).__name__}], {len(v)}"
                return f"dict, {len(v)}"
            elif isinstance(v, list):
                if v:
                    return f"list[{type(v[0]).__name__}], {len(v)}"
                return f"list, {len(v)}"

        lw = max((len(k) for k in section), default=8)

        for key, value in section.items():
            if isinstance(value, dict):
                print(f"{key:{lw}s}: ({_collection_hint(value)})")
                if print_items:
                    for entry_key in value:
                        print(f"  {entry_key}")
            elif isinstance(value, list):
                print(f"{key:{lw}s}: ({_collection_hint(value)})")
                if print_items:
                    for item in value:
                        print(f"  {item}")
            else:
                print(f"{key:{lw}s}: {value}")

        print()


def print_pipeline_config(
    config_path: str | Path,
    print_items: bool = True,
) -> None:
    """
    Print a formatted summary of a pipeline config loaded from a YAML file.

    Convenience wrapper around ``print_dict`` that handles file loading.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    print_items : bool, default=True
        If True, print individual items within list/dict values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print_dict(config, print_items=print_items)


# =============================================================================
# Timing utilities
# =============================================================================


def timed(fun, args, repeats=1) -> float:
    """
    Time the execution of a function.

    Parameters
    ----------
    fun : callable
        Function to time.
    args : tuple
        Arguments to pass to the function.
    repeats : int, default=1
        Number of times to repeat the function call.

    Returns
    -------
    float
        Execution time in seconds, averaged over repeats.
    """
    start = time.time()
    for _ in range(repeats):
        fun(*args)
    return (time.time() - start) / repeats


# =============================================================================
# Notebook utilities
# =============================================================================


def export_notebook_outputs(notebook_name: str, output_filename: str = None):
    """
    Export outputs from code cells in a Jupyter Notebook to a text file.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook file (with or without .ipynb extension).
    output_filename : str, optional
        Output text filename. If None, uses '{notebook_name}_outputs.txt'.

    Returns
    -------
    None
    """
    try:
        import nbformat
    except ImportError as exc:
        raise ImportError(
            "export_notebook_outputs requires the notebook dependencies. "
            'Install them with: python -m pip install "holspec[notebook]"'
        ) from exc

    # Set defaults
    if "." in notebook_name:
        notebook_name = notebook_name.split(".")[0]
    if output_filename is None:
        output_filename = f"{notebook_name}_outputs.txt"

    nb = nbformat.read(open(f"{notebook_name}.ipynb"), as_version=4)
    with open(output_filename, "w") as out:
        for cell in nb.cells:
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        out.write(output.text)
                    elif output.output_type == "execute_result":
                        out.write(str(output["data"].get("text/plain", "")) + "\n")
                    elif output.output_type == "error":
                        out.write("\n".join(output["traceback"]) + "\n")
    print(f"Notebook outputs from {notebook_name} exported to {output_filename}")


def convert_notebook(
    notebook_name, output_format="html", exclude=("input",), output_name=None
):
    """
    Convert a Jupyter notebook to a specified format with exclusion options.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook file (with or without .ipynb extension).
    output_format : str, default='html'
        Output format (html, pdf, latex, slides, etc.).
    exclude : tuple of str, default=('input',)
        Elements to exclude from output. Options: 'input', 'output',
        'markdown', 'raw', 'empty', 'code_cell'.
    output_name : str, optional
        Custom output filename. If None, uses '{notebook_name}.{format}'.

    Returns
    -------
    str or None
        Path to converted file if successful, None if conversion failed.
    """

    # Set defaults
    if "." in notebook_name:
        notebook_name = notebook_name.split(".")[0]
    if output_name is None:
        output_name = f"{notebook_name}.{output_format}"
    if exclude is None:
        exclude = ()

    # Build the conversion command
    cmd = [
        "jupyter",
        "nbconvert",
        f"{notebook_name}.ipynb",
        "--to",
        output_format,
        "--output",
        output_name,
    ]

    # Add exclusion options based on tuple contents
    if "input" in exclude:
        cmd.extend(["--TemplateExporter.exclude_input=True"])
    if "output" in exclude:
        cmd.extend(["--TemplateExporter.exclude_output=True"])
    if "markdown" in exclude:
        cmd.extend(["--TemplateExporter.exclude_markdown=True"])
    if "raw" in exclude:
        cmd.extend(["--TemplateExporter.exclude_raw=True"])
    if "empty" in exclude:
        cmd.extend(["--TemplateExporter.exclude_empty=True"])
    if "code_cell" in exclude:
        cmd.extend(["--TemplateExporter.exclude_code_cell=True"])

    try:
        # Execute the conversion
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(
            f"Converted {notebook_name}.ipynb to {output_name}"
            + (f" excluding: {', '.join(exclude)}" if exclude else "")
        )

        return str(Path(output_name).resolve())
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None
