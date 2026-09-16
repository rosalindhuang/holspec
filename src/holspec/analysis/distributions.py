"""
Distribution estimation and comparison for 1D samples.

Standalone mathematical functions used by EnsembleSpectraAnalysis and
available for direct use on arbitrary eigenvalue or scalar arrays.
"""

from __future__ import annotations

import numpy as np


_HISTOGRAM_DEFAULTS = {
    "bins": 50,
    "range": None,
    "density": True,
}


def empirical_distribution(
    values: np.ndarray,
    method: str = "histogram",
    method_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the empirical distribution of a 1D sample.

    Parameters
    ----------
    values : ndarray
        1D array of sample values.
    method : {'histogram'}, default='histogram'
        Distribution estimation method. 'kde' is planned but not yet
        implemented.
    method_params : dict, optional
        Method-specific parameters. If None, method defaults are used.
        Keys not present in the dict fall back to their defaults.

        Histogram parameters:
            bins : int or ndarray, default=50
                Number of bins or explicit bin edges.
            range : tuple of float, optional
                Lower and upper value range. None uses data range.
                Ignored when bins is an ndarray.
            density : bool, default=True
                If True, return probability density; if False, return
                counts.

    Returns
    -------
    x : ndarray, shape (m,)
        Support points. For histogram, these are bin centers.
    density : ndarray, shape (m,)
        Estimated density (or counts if density=False).
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {values.shape}.")

    if method == "histogram":
        params = {**_HISTOGRAM_DEFAULTS, **(method_params or {})}
        hist, bin_edges = np.histogram(
            values,
            bins=params["bins"],
            range=params["range"],
            density=params["density"],
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, hist

    elif method == "kde":
        raise NotImplementedError(
            "KDE distribution estimation is planned but not yet implemented."
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'histogram'.")


def distribution_distance(
    density_a: np.ndarray,
    density_b: np.ndarray,
    x: np.ndarray | None = None,
    metric: str = "l2",
    p: float | None = None,
) -> float:
    """
    Compute distance between two distributions on a shared support.

    Parameters
    ----------
    density_a, density_b : ndarray
        1D density arrays of equal length, defined on a shared support.
    x : ndarray, optional
        Shared support points (e.g., bin centers). Required for
        'wasserstein'; ignored for Lp metrics.
    metric : {'l1', 'l2', 'lp', 'wasserstein'}, default='l2'
        Distance metric.
    p : float, optional
        Exponent for the Lp distance. Required when metric='lp'.
        Ignored for 'l1', 'l2', and 'wasserstein'.

    Returns
    -------
    float
        Non-negative distance.
    """
    density_a = np.asarray(density_a, dtype=np.float64)
    density_b = np.asarray(density_b, dtype=np.float64)

    if density_a.ndim != 1 or density_b.ndim != 1:
        raise ValueError(
            f"Densities must be 1D. Got shapes {density_a.shape} and {density_b.shape}."
        )
    if len(density_a) != len(density_b):
        raise ValueError(
            f"Densities must have the same length. Got {len(density_a)} "
            f"and {len(density_b)}."
        )

    if metric in ("l1", "l2", "lp"):
        if metric == "l1":
            p_val = 1.0
        elif metric == "l2":
            p_val = 2.0
        else:
            if p is None:
                raise ValueError("Parameter p is required when metric='lp'.")
            p_val = float(p)

        diff = np.abs(density_a - density_b)
        return float(np.sum(diff**p_val) ** (1.0 / p_val))

    elif metric == "wasserstein":
        if x is None:
            raise ValueError(
                "Parameter x (support points) is required for metric='wasserstein'."
            )
        from scipy.stats import wasserstein_distance

        x = np.asarray(x, dtype=np.float64)
        return float(
            wasserstein_distance(x, x, u_weights=density_a, v_weights=density_b)
        )

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Supported metrics: 'l1', 'l2', 'lp', 'wasserstein'."
        )
