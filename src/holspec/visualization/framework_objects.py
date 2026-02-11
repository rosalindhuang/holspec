"""
Visualization functions for objects in the holspec framework:
1. Point data (2D/3D)
2. Simplicial complexes (vertices, edges, triangles)
3. Hodge Laplacian spectra (eigenvalues, eigenvectors)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union, List

from holspec.visualization import plot_circles, plot_spheres

# %% Point data

def plot_point_data(
    positions: np.ndarray,
    radius: float = 0.1,
    # Coloring
    colors: Optional[Union[str, List, np.ndarray]] = None,
    color_values: Optional[np.ndarray] = None,
    cmap: Optional[str] = None,
    # Figure
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot point data as circles (2D) or spheres (3D).
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, d) where d is 2 or 3.
    radius : float, default=0.1
        Radius for circles/spheres.
    colors : str, list, or array, optional
        Colors for points. Can be:
        - Single color string: all points same color
        - List/array of colors: explicit color per point
    color_values : np.ndarray, optional
        Values to map to colormap. Used only if colors is None.
    cmap : Optional[str]
        Colormap name for use with color_values.
    ax : Axes, optional
        Existing axes. If None, creates new figure.
    **kwargs
        Passed to plot_circles or plot_spheres.
    
    Returns
    -------
    fig, ax : Figure, Axes
    
    Examples
    --------
    >>> # All same color
    >>> plot_point_data(positions, colors='red')
    
    >>> # Colormap
    >>> plot_point_data(positions, color_values=np.arange(n))
    
    >>> # Explicit control with overrides
    >>> colors = [plt.get_cmap('viridis')(i/n) for i in range(n)]
    >>> colors[3] = 'red'
    >>> colors[10] = 'blue'
    >>> plot_point_data(positions, colors=colors)
    """
    dimension = positions.shape[1]
    if dimension not in [2, 3]:
        raise ValueError(f"positions must be 2D or 3D, got {dimension}")
    
    n_points = len(positions)
    
    # Determine colors
    if colors is not None:
        if isinstance(colors, str):
            # Single color for all
            colors_list = [colors] * n_points
        else:
            # Explicit list/array
            colors_list = list(colors)
            if len(colors_list) != n_points:
                raise ValueError(f"colors length must match positions ({n_points})")
    
    elif color_values is not None and cmap is not None:
        # Map values to colormap
        color_values = np.asarray(color_values)
        if len(color_values) != n_points:
            raise ValueError(f"color_values length must match positions ({n_points})")
        
        vmin, vmax = color_values.min(), color_values.max()
        if vmax > vmin:
            norm = (color_values - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(color_values)
        
        cmap_obj = plt.get_cmap(cmap)
        colors_list = [cmap_obj(v) for v in norm]
    
    else:
        # Use defaults from plot_circles/plot_spheres
        colors_list = None
    
    # Plot
    if colors_list is not None:
        if dimension == 2:
            props = [{'facecolor': c} for c in colors_list]
            fig, ax = plot_circles(positions, radius, circle_props=props, ax=ax, **kwargs)
        else:
            props = [{'c': c} for c in colors_list]
            fig, ax = plot_spheres(positions, radius, sphere_props=props, ax=ax, **kwargs)
    else:
        if dimension == 2:
            fig, ax = plot_circles(positions, radius, ax=ax, **kwargs)
        else:
            fig, ax = plot_spheres(positions, radius, ax=ax, **kwargs)
    
    return fig, ax


# %% Simplicial complex