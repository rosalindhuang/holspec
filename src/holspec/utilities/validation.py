"""
Validation utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Testing Utilities
# =============================================================================

def check_raises(
    label: str,
    func: Callable,
    expected_exception: type[Exception],
    label_width: int = 0,
    prefix: str = "",
) -> bool:
    """
    Check that calling func raises expected_exception, printing a PASS/FAIL result.

    Useful for interactively verifying that validation functions raise the correct
    exceptions in notebooks and exploratory scripts.

    Parameters
    ----------
    label : str
        Description of the case being tested, printed as the row label.
    func : callable
        Zero-argument callable to invoke.
    expected_exception : type[Exception]
        Exception type expected to be raised.
    label_width : int, default=35
        Field width for left-aligning label in printed output.
    prefix : str, default=""
        String prepended to each output line (e.g. for indentation).

    Returns
    -------
    bool
        True if expected_exception was raised, False otherwise.

    Examples
    --------
    >>> check_raises("negative value", lambda: validate_weights([-1.]), ValueError)
      negative value                    : PASS (ValueError: ...)
    """
    try:
        func()
        print(f"{prefix}{label:{label_width}s}: FAIL (no error raised)")
        return False
    except expected_exception as exc:
        print(f"{prefix}{label:{label_width}s}: PASS ({type(exc).__name__}: {exc})")
        return True
    except Exception as exc:
        print(f"{prefix}{label:{label_width}s}: FAIL (unexpected {type(exc).__name__}: {exc})")
        return False

