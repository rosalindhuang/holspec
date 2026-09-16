# holspec/utilities/numerical.py
"""
Core numerical utilities including hashing and noise generation.
"""

import numpy as np
import hashlib

from .helpers import format_float_str


def compute_content_hash(data: np.ndarray, length: int = 12) -> str:
    """
    Compute SHA256 hash of array data.
    Uses data.tobytes() for deterministic, platform-independent hashing.
    """
    data_bytes = np.asarray(data).tobytes()
    hash_full = hashlib.sha256(data_bytes).hexdigest()
    return hash_full[:length]


def add_noise(
    array: np.ndarray,
    scale: float,
    distribution: str = "normal",
    axis: int | tuple | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add random noise to an array.

    Parameters
    ----------
    array : ndarray
        Input array of any shape.
    scale : float
        Noise amplitude/standard deviation.
    distribution : {'normal', 'uniform'}, default='normal'
        Noise distribution:
        - 'normal': Gaussian noise with std = scale
        - 'uniform': Uniform noise in [-scale/2, scale/2]
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to add noise. If None, add noise to all elements.
        If specified, noise shape will broadcast along other axes.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_array : ndarray
        Array with added noise, same shape as input.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> add_noise(x, scale=0.1, seed=42)
    array([1.04967142, 1.98617357, 2.86475726])

    >>> # Add noise only along first axis of 2D array
    >>> x = np.ones((3, 4))
    >>> noisy = add_noise(x, scale=0.1, axis=0, seed=42)
    >>> noisy[:, 0]  # Same noise broadcast across columns
    array([1.04967142, 0.98617357, 0.86475726])
    """
    rng = np.random.default_rng(seed)

    # Determine noise shape
    if axis is None:
        noise_shape = array.shape
    else:
        # Create shape for broadcasting
        noise_shape = list(array.shape)
        axes = (axis,) if isinstance(axis, int) else axis
        for ax in axes:
            if ax < 0 or ax >= len(noise_shape):
                raise ValueError(
                    f"axis {ax} out of bounds for array of dimension {len(noise_shape)}"
                )
        # Set non-noise axes to size 1 for broadcasting
        for i in range(len(noise_shape)):
            if i not in axes:
                noise_shape[i] = 1
        noise_shape = tuple(noise_shape)

    # Generate noise
    if distribution == "normal":
        noise = scale * rng.standard_normal(noise_shape)
    elif distribution == "uniform":
        noise = scale * (rng.random(noise_shape) - 0.5)
    else:
        raise ValueError(
            f"Unknown distribution: '{distribution}'. Use 'normal' or 'uniform'."
        )

    return array + noise


def create_noise_label(
    noise_config: dict,
    float_fmt: str | None = "g",
    strip_zeros: bool = True,
) -> str:
    """
    Create a unique label from noise configuration.

    Parameters
    ----------
    noise_config : dict
        Noise configuration with 'scale' and optional 'distribution' keys.
    float_fmt : str or None, optional
        Format specifier for floats (e.g., 'g', '.2e', '.0e', '.3f'). Default is 'g'.
        If None, defaults to 'g'.

    Returns
    -------
    label : str
        Descriptive label. Format: noise_{dist}_s{scale}

    Examples
    --------
    >>> create_noise_label({'scale': 0.1, 'distribution': 'uniform'})
    'noise_unif_s0p1'
    >>> create_noise_label({'scale': 0.01, 'distribution': 'normal'}, float_fmt='.0e')
    'noise_norm_s1e-02'
    """
    # Validate noise_config
    if "scale" not in noise_config:
        raise ValueError("noise_config must contain 'scale' key")

    # Format scale for label
    scale = noise_config["scale"]
    scale_str = format_float_str(scale, float_fmt, strip_zeros=strip_zeros)

    # Format distribution for label
    dist = noise_config.get("distribution")
    if dist is None:
        return f"noise_s{scale_str}"
    dist_str = dist[:1]

    return f"noise_{dist_str}_s{scale_str}"
