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
from typing import Optional, Tuple, Union, List, Dict

from holspec.visualization import (
    plot_circles, 
    plot_spheres,
    plot_vertices_2d,
    plot_edges_2d,
    plot_triangles_2d,
    add_edge_arrows_2d,
    add_triangle_orientation_2d,
    add_simplex_labels_2d
)

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

def plot_simplicial_complex_2d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    # Color control
    colors: Optional[Dict[int, str]] = None,
    # Labeling control
    show_labels: Optional[Union[bool, Dict[int, bool]]] = None,
    # Orientation control
    show_orientation: Optional[Union[bool, Dict[int, bool]]] = None,
    # Figure
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D simplicial complex with vertices, edges, and triangles.
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
        Keys: 0 (vertices), 1 (edges), 2 (triangles).
        Each simplex is a tuple of vertex indices.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) vertex coordinates.
    colors : Dict[int, str], optional
        Colors per dimension. Default: {0: 'C0', 1: 'C1', 2: 'C2'}.
        Example: {0: 'red', 1: 'blue', 2: 'green'}
    show_labels : bool or Dict[int, bool], optional
        Control labeling by dimension.
        - None or False: no labels
        - True: label all dimensions
        - {0: True, 1: False, 2: True}: per-dimension control
    show_orientation : bool or Dict[int, bool], optional
        Control orientation markers by dimension.
        - None or False: no orientation markers
        - True: show for all applicable dimensions (1, 2)
        - {1: True, 2: False}: per-dimension control
        Note: Dimension 0 (vertices) has no orientation.
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional styling arguments (advanced use).
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    
    Examples
    --------
    >>> # Basic plot with default colors
    >>> simplices = {
    ...     0: [(0,), (1,), (2,)],
    ...     1: [(0, 1), (0, 2), (1, 2)],
    ...     2: [(0, 1, 2)]
    ... }
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> fig, ax = plot_simplicial_complex_2d(simplices, positions)
    
    >>> # Custom colors
    >>> fig, ax = plot_simplicial_complex_2d(
    ...     simplices, positions,
    ...     colors={0: 'red', 1: 'blue', 2: 'yellow'}
    ... )
    
    >>> # With labels and orientation
    >>> fig, ax = plot_simplicial_complex_2d(
    ...     simplices, positions,
    ...     show_labels=True,
    ...     show_orientation=True
    ... )
    
    >>> # Selective labeling and orientation
    >>> fig, ax = plot_simplicial_complex_2d(
    ...     simplices, positions,
    ...     show_labels={0: True, 1: False, 2: True},
    ...     show_orientation={1: True, 2: False}
    ... )
    
    Notes
    -----
    - For fine-grained control over styling, use the basic plotting
      functions directly: plot_triangles_2d(), plot_edges_2d(), plot_vertices_2d()
    - Orientation only applies to dimensions 1 (arrows) and 2 (+/− signs)
    - Labels show simplex indices in order of appearance in simplices dict
    - Default colors use matplotlib's color cycle: C0, C1, C2
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Set up default colors
    default_colors = {0: 'C0', 1: 'C1', 2: 'C2'}
    color_map = {**default_colors, **(colors or {})}
    
    # Plot simplices (triangles → edges → vertices for proper layering)
    if 2 in simplices and simplices[2]:
        fig, ax = plot_triangles_2d(simplices[2], positions, color=color_map[2], ax=ax, zorder=1)
    
    if 1 in simplices and simplices[1]:
        fig, ax = plot_edges_2d(simplices[1], positions, color=color_map[1], ax=ax, zorder=2)
    
    if 0 in simplices and simplices[0]:
        fig, ax = plot_vertices_2d(simplices[0], positions, color=color_map[0], ax=ax, zorder=3)
    
    # Add labels if requested
    if show_labels:
        # Parse bool or dict to list of dimensions
        if show_labels is True:
            dims_to_label = [0, 1, 2]
        elif isinstance(show_labels, dict):
            dims_to_label = [d for d in [0, 1, 2] if show_labels.get(d, False)]
        else:
            dims_to_label = []
        
        if dims_to_label:
            add_simplex_labels_2d(simplices, positions, dims_to_label, color_map, ax=ax)
    
    # Add orientation if requested (only for edges and triangles)
    if show_orientation:
        # Parse bool or dict to list of dimensions
        if show_orientation is True:
            dims_to_orient = [1, 2]
        elif isinstance(show_orientation, dict):
            dims_to_orient = [d for d in [1, 2] if show_orientation.get(d, False)]
        else:
            dims_to_orient = []
        
        # Only call if dimension exists and is non-empty
        if 1 in dims_to_orient and 1 in simplices and simplices[1]:
            add_edge_arrows_2d(simplices[1], positions, color=color_map[1], ax=ax)
        if 2 in dims_to_orient and 2 in simplices and simplices[2]:
            add_triangle_orientation_2d(simplices[2], positions, color=color_map[2], ax=ax)
    
    return fig, ax
