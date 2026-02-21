# holspec/utilities/helpers.py
"""
Helper functions including:
- Inspecting dictionary structures
- Exporting Jupyter notebook outputs and converting notebooks to other formats
- Formatting text for printing
- Timing function execution
"""

import numpy as np
import pandas as pd
import h5py
import os
import json
from pathlib import Path
import shutil
from natsort import natsorted
import time
import textwrap
import re
import nbformat
import subprocess


# %% Type conversion

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types.
    
    Parameters
    ----------
    obj : any
        Object to convert. Can be a dict, list, numpy array, numpy scalar, or any other type.
    
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
    else:
        return obj


# %% String utilities

def format_float_str(value: float, fmt: str | None = 'g', strip_zeros: bool = True) -> str:
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
        fmt = 'g'
    
    # Format the value
    formatted = f"{value:{fmt}}"
    
    # Remove trailing zeros after decimal point
    if '.' in formatted and strip_zeros:
        if 'e' in formatted.lower():
            # Handle scientific notation: split at 'e', trim mantissa, rejoin
            parts = formatted.lower().split('e')
            parts[0] = parts[0].rstrip('0').rstrip('.')
            formatted = 'e'.join(parts)
        else:
            # Handle regular decimal
            formatted = formatted.rstrip('0').rstrip('.')
    
    # Replacements decimal point with 'p'
    formatted = formatted.replace('.', 'p')
    
    return formatted


def format_text(text: str, max_width: int = 80, preserve_paragraphs: bool = True) -> str:
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
    lines = text.split('\n')
    formatted_lines = []
    
    # Pattern to match bullet points (-, *, +, or numbered like 1., 2., etc.)
    bullet_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')
    
    for line in lines:
        # Preserve empty lines if requested
        if not line.strip() and preserve_paragraphs:
            formatted_lines.append('')
            continue
        
        # Detect basic indentation (spaces or tabs at the beginning)
        indent = ''
        for char in line:
            if char in ' \t':
                indent += char
            else:
                break
        
        # Get the content without leading whitespace
        content = line.lstrip()
        
        if not content:  # Line was only whitespace
            formatted_lines.append('')
            continue
        
        # Check if this is a bullet point
        bullet_match = bullet_pattern.match(line)
        if bullet_match:
            # For bullet points, create hanging indentation
            bullet_indent = bullet_match.group(1)  # Initial spaces/tabs
            bullet_marker = bullet_match.group(2)  # The bullet character(s)
            bullet_text = line[bullet_match.end():]  # Text after bullet
            
            # First line uses the original indent + bullet
            first_line_prefix = bullet_indent + bullet_marker + ' '
            # Continuation lines align with the text after the bullet
            continuation_indent = bullet_indent + ' ' * len(bullet_marker + ' ')
            
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
                    break_on_hyphens=True
                ).split('\n')
                
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
                break_on_hyphens=True
            ).split('\n')
            
            # Add the original indentation to each wrapped line
            for wrapped_line in wrapped_lines:
                formatted_lines.append(indent + wrapped_line)
    
    return '\n'.join(formatted_lines)


#%% Printing utilities

def inspect_dict(data_dict, dict_name="dict", max_depth=None, _current_depth=0, _prefix=""):
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
    _current_depth : int, optional
        Internal parameter for tracking recursion depth (do not use).
    _prefix : str, optional
        Internal parameter for indentation (do not use).

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
        elif hasattr(value, 'shape'):
            return f"(type={type(value).__name__}, shape={value.shape})"
        elif hasattr(value, '__len__'):
            return f"(type={type(value).__name__}, len={len(value)})"
        else:
            return f"(type={type(value).__name__})"
    
    # Print current level
    if _current_depth == 0:
        print(f"{dict_name}/")

    # Check depth limit
    if max_depth is not None and _current_depth >= max_depth:
        # print(f"{_prefix}    ...")
        return
    
    # Iterate through dictionary items
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Nested dictionary
            print(f"{_prefix}    {key}/")
            inspect_dict(
                value, 
                dict_name=key,
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
                _prefix=_prefix + "    "
            )
        else:
            # Regular value
            summary = summarize_value(value)
            print(f"{_prefix}    {key} {summary}")


# %% Timing utilities

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


# %% Notebook utilities

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
    # Set defaults
    if '.' in notebook_name:
        notebook_name = notebook_name.split('.')[0]
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


def convert_notebook(notebook_name, output_format='html', exclude=('input',), output_name=None):
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
    if '.' in notebook_name:
        notebook_name = notebook_name.split('.')[0]
    if output_name is None:
        output_name = f"{notebook_name}.{output_format}"
    if exclude is None:
        exclude = ()
    
    # Build the conversion command
    cmd = [
        'jupyter', 'nbconvert', 
        f'{notebook_name}.ipynb',
        '--to', output_format,
        '--output', output_name
    ]
    
    # Add exclusion options based on tuple contents
    if 'input' in exclude:
        cmd.extend(['--TemplateExporter.exclude_input=True'])
    if 'output' in exclude:
        cmd.extend(['--TemplateExporter.exclude_output=True'])
    if 'markdown' in exclude:
        cmd.extend(['--TemplateExporter.exclude_markdown=True'])
    if 'raw' in exclude:
        cmd.extend(['--TemplateExporter.exclude_raw=True'])
    if 'empty' in exclude:
        cmd.extend(['--TemplateExporter.exclude_empty=True'])
    if 'code_cell' in exclude:
        cmd.extend(['--TemplateExporter.exclude_code_cell=True'])
    
    try:
        # Execute the conversion
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Converted {notebook_name}.ipynb to {output_name}" + (f" excluding: {', '.join(exclude)}" if exclude else ""))
        
        return str(Path(output_name).resolve())
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None


