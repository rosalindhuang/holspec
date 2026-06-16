# holspec/paths.py
"""
Project path conventions and directory utilities.

The top-level configs, data, notebooks, and outputs paths are local workspace
conventions for development and exploratory workflows.
"""
from pathlib import Path
from fnmatch import fnmatch
import shutil
from functools import lru_cache

# =============================================================================
# Project paths
# =============================================================================

# --- Project root ---

@lru_cache(maxsize=1)
def get_project_root(start: Path | None = None, root_markers = ("pyproject.toml", ".git")) -> Path:
    """
    Resolve the project root directory by searching upward for common project markers.
    Walks upward from `start` (defaults to this file's location) until it finds a directory
    containing one of the root marker files.

    Raises
    ------
    RuntimeError if no root marker is found.
    """
    p = (start or Path(__file__)).resolve()
    for parent in (p, *p.parents):
        if any((parent / m).exists() for m in root_markers):
            return parent
    raise RuntimeError(f"Could not find project root from {p} based on root markers {root_markers}.")

PROJECT_ROOT = get_project_root()


# --- Directories ---

CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_GENERATED_DIR = DATA_RAW_DIR / "generated"
DATA_RAW_IMPORTED_DIR = DATA_RAW_DIR / "imported"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


# =============================================================================
# Utilities for directory management
# =============================================================================

def prepare_directory(
    path: Path,
    clear_mode: str | bool | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    create_missing: bool = True,
    verbose: bool = True,
) -> None:
    """
    Prepare a directory: optionally create it if missing, and optionally clear its contents.

    Parameters
    ----------
    path : Path
        Directory path to prepare.
    clear_mode : {'all', 'files', 'dirs', 'none'}, bool, or None, default=None
        What to clear if directory exists:
        - None, False, or 'none': Don't clear anything
        - True or 'all': Remove all files and subdirectories
        - 'files': Remove only files, keep subdirectories
        - 'dirs': Remove only subdirectories, keep files
    include_patterns : list of str, optional
        Glob patterns for item names (e.g. ``['pipeline_*.yml', 'pipeline_*_stages']``).
        If given, only items whose name matches at least one pattern are
        considered for clearing. If None or empty, all items are candidates
        (subject to ``clear_mode``). Exact names are valid patterns.
    exclude_patterns : list of str, optional
        Glob patterns for items to preserve. Items whose name matches any
        pattern are skipped, even if ``clear_mode`` and ``include_patterns``
        would otherwise remove them. Exact names are valid patterns.
    create_missing : bool, default=True
        If True, create the directory when it does not exist. If False,
        skip creation (clearing still applies if the directory exists).
    verbose : bool, default=True
        If True, print status messages.

    Returns
    -------
    None
    """
    if path is None:
        return

    # Normalize boolean and string inputs for clear_mode
    if clear_mode is True:
        clear_mode = 'all'
    elif clear_mode is False or clear_mode == 'none':
        clear_mode = None

    # Validate clear_mode
    valid_modes = {None, 'all', 'files', 'dirs'}
    if clear_mode not in valid_modes:
        raise ValueError(f"clear_mode must be one of {valid_modes} or a boolean, got '{clear_mode}'")

    if path.exists() and path.is_dir():
        if verbose:
            print(f"Directory exists: {path}")

        if clear_mode is not None:
            items = list(path.iterdir())
            if not items:
                return

            # Apply pattern filters
            def _pattern_match(name):
                if include_patterns and not any(fnmatch(name, p) for p in include_patterns):
                    return False
                if exclude_patterns and any(fnmatch(name, p) for p in exclude_patterns):
                    return False
                return True

            items = [item for item in items if _pattern_match(item.name)]
            if not items:
                return

            if verbose:
                print(f"Clearing directory ({clear_mode}): {path}")

            files = [item for item in items if item.is_file()]
            dirs = [item for item in items if item.is_dir()]

            # Clear based on mode
            if clear_mode in ('all', 'files'):
                for item in files:
                    item.unlink()
                if verbose and files:
                    print(f"  Removed {len(files)} file(s): {[f.name for f in files]}")

            if clear_mode in ('all', 'dirs'):
                for item in dirs:
                    shutil.rmtree(item, ignore_errors=True)
                if verbose and dirs:
                    print(f"  Removed {len(dirs)} dir(s): {[d.name for d in dirs]}")

            if verbose:
                print()
    elif create_missing:
        if verbose:
            print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


def print_directory_tree(
    root_path, 
    include_files=True, 
    ignore_dotfiles=True, 
    ignore_patterns=None, 
    ignore_exact=None, 
    _prefix="", 
    _is_last=True
):
    """
    Print directory structure in tree format with filtering options.

    Parameters
    ----------
    root_path : str or Path
        Root directory to start from.
    include_files : bool, default=True
        If True, include files in output; if False, show only directories.
    ignore_dotfiles : bool, default=True
        If True, ignore items starting with '.'.
    ignore_patterns : list of str, optional
        Ignore items containing any of these substrings.
    ignore_exact : list of str, optional
        Ignore items with names exactly matching these values.
    _prefix : str, optional
        Internal parameter for formatting indentation (do not use).
    _is_last : bool, optional
        Internal parameter for formatting tree connectors (do not use).

    Returns
    -------
    None
    """
    root_path = Path(root_path)
    
    # Set defaults for ignore lists
    if ignore_patterns is None:
        ignore_patterns = []
    if ignore_exact is None:
        ignore_exact = []
    
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return
    
    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory")
        return
    
    def should_ignore(item_name):
        """Check if an item should be ignored based on ignore rules."""
        # Check dotfiles
        if ignore_dotfiles and item_name.startswith('.'):
            return True
        
        # Check exact matches
        if item_name in ignore_exact:
            return True
        
        # Check patterns (substring matching)
        for pattern in ignore_patterns:
            if pattern in item_name:
                return True
        
        return False
    
    # Print the root directory name
    if _prefix == "":
        print(f"{root_path.name}/")
    
    try:
        # Get all items in the directory
        items = list(root_path.iterdir())
        
        # Filter out ignored items
        items = [item for item in items if not should_ignore(item.name)]
        
        # Filter based on include_files parameter
        if include_files:
            # Sort: directories first, then files, both alphabetically
            directories = sorted([item for item in items if item.is_dir()])
            files = sorted([item for item in items if item.is_file()])
            all_items = directories + files
        else:
            # Only directories
            all_items = sorted([item for item in items if item.is_dir()])
        
        # Print each item
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            
            # Determine the connector character
            if is_last_item:
                connector = "└── "
                next_prefix = _prefix + "    "
            else:
                connector = "├── "
                next_prefix = _prefix + "│   "
            
            # Print the item
            if item.is_dir():
                print(f"{_prefix}{connector}{item.name}/")
                # Recursively print subdirectory with same ignore rules
                print_directory_tree(item, include_files, ignore_dotfiles, 
                                   ignore_patterns, ignore_exact, next_prefix, is_last_item)
            else:
                print(f"{_prefix}{connector}{item.name}")
                
    except PermissionError:
        print(f"{_prefix}[Permission Denied]")
    except Exception as e:
        print(f"{_prefix}[Error: {e}]")


# =============================================================================
# Test code
# =============================================================================

if __name__ == "__main__":
    print("Project paths:")
    print("  PROJECT_ROOT:".ljust(15), PROJECT_ROOT)
    print("  CONFIGS_DIR:".ljust(15), CONFIGS_DIR)
    print("  DATA_DIR:".ljust(15), DATA_DIR)
    print("  NOTEBOOKS_DIR:".ljust(15), NOTEBOOKS_DIR)
    print("  OUTPUTS_DIR:".ljust(15), OUTPUTS_DIR)
    print("  SRC_DIR:".ljust(15), SRC_DIR)
    print("  TESTS_DIR:".ljust(15), TESTS_DIR)
    
    print()
    print("Project directory tree:")
    print_directory_tree(PROJECT_ROOT)
