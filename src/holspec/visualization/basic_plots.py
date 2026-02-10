"""
Basic plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from typing import Optional, Tuple, List, Union, Dict, Any

# %% Line plots

def plot_lines(
    xy_list: List[Tuple[np.ndarray, np.ndarray]],
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5, 4),
    # Line styling
    color: Optional[str] = None,
    colors_list: Optional[List[str]] = None,
    # Legend
    legend_labels: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot multiple lines on a single axis.
    
    Parameters
    ----------
    xy_list : list of (x, y) tuples
        List of (x_array, y_array) tuples to plot.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple of float, default (5, 4)
        Figure size as (width, height) in inches.
    color : str, optional
        Single color for all lines (overrides colors_list).
    colors_list : list of str, optional
        List of colors for each line. If shorter than xy_list, extended with 'k'.
    legend_labels : list of str, optional
        List of legend labels for each line. If provided, legend will be displayed.
    **kwargs
        Additional keyword arguments passed to ax.plot().
    
    Returns
    -------
    tuple
        (fig, ax) - The figure and axes objects.
    
    Raises
    ------
    ValueError
        If xy_list is empty.
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Resolve colors
    if color is not None:
        colors = [color] * len(xy_list)
    elif colors_list is not None:
        colors = _extend_list(colors_list, len(xy_list), 'k')
    else:
        colors = ['k'] * len(xy_list)
    
    # Resolve legend labels
    labels = None
    if legend_labels is not None:
        labels = _extend_list(legend_labels, len(xy_list), None)
    
    # Plot lines
    for i, (x, y) in enumerate(xy_list):
        label = labels[i] if labels is not None else None
        ax.plot(x, y, color=colors[i], label=label, **kwargs)
    
    # Add legend if labels were provided
    if legend_labels is not None:
        ax.legend()
    
    return fig, ax


def plot_lines_stack(
    xy_list: List[Tuple[np.ndarray, np.ndarray]],
    # Figure properties
    figsize: Tuple[float, float] = (6, 6),
    stack_direction: str = 'vertical',
    sharex: bool = False,
    sharey: bool = False,
    # Line styling
    color: Optional[str] = None,
    colors_list: Optional[List[str]] = None,
    # Legend
    legend_labels: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Plot multiple lines on stacked subplots.
    
    Parameters
    ----------
    xy_list : list of (x, y) tuples
        List of (x_array, y_array) tuples to plot, one line per subplot.
    figsize : tuple of float, default (6, 6)
        Figure size as (width, height) in inches.
    stack_direction : str, default 'vertical'
        Direction to stack subplots ('vertical' or 'horizontal').
    sharex : bool, default False
        Whether to share x-axis between subplots.
    sharey : bool, default False
        Whether to share y-axis between subplots.
    color : str, optional
        Single color for all lines (overrides colors_list).
    colors_list : list of str, optional
        List of colors for each line/subplot. If shorter than xy_list, extended with 'k'.
    legend_labels : list of str, optional
        List of legend labels for each line/subplot. If provided, legends will be displayed.
    **kwargs
        Additional keyword arguments passed to ax.plot().
    
    Returns
    -------
    tuple
        (fig, axes) - The figure and axes objects. 
        axes is an ndarray for multiple subplots, single Axes for one subplot.
    
    Raises
    ------
    ValueError
        If xy_list is empty or stack_direction is invalid.
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    if stack_direction not in ['vertical', 'horizontal']:
        raise ValueError("stack_direction must be 'vertical' or 'horizontal'")
    
    # Resolve colors
    if color is not None:
        colors = [color] * len(xy_list)
    elif colors_list is not None:
        colors = _extend_list(colors_list, len(xy_list), 'k')
    else:
        colors = ['k'] * len(xy_list)
    
    # Resolve legend labels
    labels = None
    if legend_labels is not None:
        labels = _extend_list(legend_labels, len(xy_list), None)
    
    # Create subplots
    if stack_direction == 'vertical':
        fig, axes = plt.subplots(len(xy_list), 1, figsize=figsize, sharex=sharex, sharey=sharey)
    else:  # horizontal
        fig, axes = plt.subplots(1, len(xy_list), figsize=figsize, sharex=sharex, sharey=sharey)
    
    axes = np.atleast_1d(axes)
    
    # Plot lines
    for i, (x, y) in enumerate(xy_list):
        ax = axes[i]
        label = labels[i] if labels is not None else None
        ax.plot(x, y, color=colors[i], label=label, **kwargs)
        
        # Add legend if label exists
        if label is not None:
            ax.legend()
    
    return fig, axes if len(xy_list) > 1 else axes[0]


# %% Scatter plots

def plot_scatters(
    xy_list: List[Tuple[np.ndarray, np.ndarray]],
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5, 4),
    # Color styling
    color: Optional[str] = None,
    colors_list: Optional[List[str]] = None,
    c: Optional[np.ndarray] = None,
    c_list: Optional[List[np.ndarray]] = None,
    cmap: Optional[str] = None,
    # Legend and colorbar
    legend_labels: Optional[List[str]] = None,
    colorbar: bool = False,
    **kwargs
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Axes, Colorbar]]:
    """
    Plot multiple scatter plots on a single axis.
    
    Parameters
    ----------
    xy_list : list of (x, y) tuples
        List of (x_array, y_array) tuples to plot.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple of float, default (5, 4)
        Figure size as (width, height) in inches.
    color : str, optional
        Single color for all scatter plots (highest priority, overrides other color options).
    colors_list : list of str, optional
        List of colors for each scatter plot. If shorter than xy_list, extended with 'k'.
    c : array-like, optional
        Color values for colormap (applied to all scatter plots).
    c_list : list of array-like, optional
        List of color values for each scatter plot (for use with colormap).
    cmap : str or Colormap, optional
        Colormap to use with c/c_list.
    legend_labels : list of str, optional
        List of legend labels for each scatter plot.
    colorbar : bool, default False
        Whether to add a colorbar (only applicable when using colormap).
    **kwargs
        Additional keyword arguments passed to ax.scatter().
    
    Returns
    -------
    tuple
        (fig, ax) if no colorbar, or (fig, ax, cbar) if colorbar is added.
    
    Raises
    ------
    ValueError
        If xy_list is empty.
    
    Notes
    -----
    Color priority (highest to lowest): color > colors_list > colormap (c/c_list).
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    
    n_plots = len(xy_list)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Determine color strategy (priority: color > colors_list > colormap)
    use_colormap = False
    if color is not None:
        # Single color for all
        plot_colors = [color] * n_plots
        plot_c_values = [None] * n_plots
    elif colors_list is not None:
        # Different colors for each scatter
        plot_colors = _extend_list(colors_list, n_plots, 'k')
        plot_c_values = [None] * n_plots
    elif c is not None:
        # Use colormap with single c values for all
        plot_colors = [None] * n_plots
        plot_c_values = [c] * n_plots
        use_colormap = True
    elif c_list is not None:
        # Use colormap with different c values for each scatter
        plot_colors = [None] * n_plots
        plot_c_values = _extend_list(c_list, n_plots, None)
        use_colormap = True
    else:
        # Default colors
        plot_colors = ['k'] * n_plots
        plot_c_values = [None] * n_plots
    
    # Resolve legend labels
    labels = None
    if legend_labels is not None:
        labels = _extend_list(legend_labels, n_plots, None)
    
    # Plot scatter plots
    scatter_plots = []
    for i, (x, y) in enumerate(xy_list):
        label = labels[i] if labels is not None else None
        
        if use_colormap and plot_c_values[i] is not None:
            sc = ax.scatter(x, y, c=plot_c_values[i], cmap=cmap, label=label, **kwargs)
        else:
            sc = ax.scatter(x, y, color=plot_colors[i], label=label, **kwargs)
        
        scatter_plots.append(sc)
    
    # Add colorbar if requested
    cbar = None
    if colorbar and use_colormap:
        cbar = fig.colorbar(scatter_plots[-1], ax=ax)
    
    # Add legend if labels exist
    if legend_labels is not None:
        ax.legend()
    
    if cbar is None:
        return fig, ax
    return fig, ax, cbar


# %% Basic shapes plots

def plot_circles(
    positions: np.ndarray,
    radius: Union[float, np.ndarray],
    # Circle styling
    circle_props: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (6, 6),
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot circles in 2D at specified positions with given radius/radii.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) coordinates of circle centers.
    radius : float or np.ndarray
        Circle radius. Can be a single value for all circles or an array of 
        length N for individual radii.
    circle_props : dict or list of dicts, optional
        Circle properties.
        - None: use defaults (lightgrey face, black edge, alpha=0.8)
        - dict: apply same properties to all circles
        - list of dicts: individual properties for each circle
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple of float, default (6, 6)
        Figure size as (width, height) in inches.
    **kwargs
        Additional keyword arguments passed to matplotlib Circle patches.
        These serve as additional defaults that can be overridden by circle_props.
    
    Returns
    -------
    tuple
        (fig, ax) - The figure and axes objects.
    
    Raises
    ------
    ValueError
        If length of radius array doesn't match number of positions, or if
        circle_props format is invalid.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert positions to numpy array
    positions = np.asarray(positions)
    
    # Handle single radius vs array of radii
    if np.isscalar(radius):
        radii = np.full(len(positions), radius)
    else:
        radii = np.asarray(radius)
        if len(radii) != len(positions):
            raise ValueError("Length of radius array must match number of positions")
    
    # Set up default properties
    default_props = {
        'facecolor': 'lightgrey',
        'edgecolor': 'black',
        'alpha': 0.8,
        'zorder': 0
    }
    default_props.update(kwargs)
    
    # Handle circle_props
    if circle_props is None:
        # Use defaults for all circles
        props_list = [default_props.copy() for _ in positions]
    elif isinstance(circle_props, dict):
        # Single dict for all circles
        single_props = default_props.copy()
        single_props.update(circle_props)
        props_list = [single_props.copy() for _ in positions]
    elif isinstance(circle_props, (list, tuple)):
        # List of dicts, one per circle
        if len(circle_props) != len(positions):
            raise ValueError("Length of circle_props list must match number of positions")
        props_list = []
        for props in circle_props:
            circle_prop = default_props.copy()
            if props is not None:
                circle_prop.update(props)
            props_list.append(circle_prop)
    else:
        raise ValueError("circle_props must be None, dict, or list of dicts")
    
    # Add circles to the plot
    for pos, r, props in zip(positions, radii, props_list):
        circle = plt.Circle(pos, r, **props)
        ax.add_patch(circle)
    
    return fig, ax


def plot_spheres(
    positions: np.ndarray,
    radius: Union[float, np.ndarray],
    # Sphere styling
    sphere_props: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (6, 6),
    # Sphere rendering
    size_scale: float = 100.0,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot spheres in 3D at specified positions with given radius/radii.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates of sphere centers.
    radius : float or np.ndarray
        Sphere radius. Can be a single value for all spheres or an array of 
        length N for individual radii.
    sphere_props : dict or list of dicts, optional
        Sphere properties.
        - None: use defaults (lightgrey color, black edges, alpha=0.8)
        - dict: apply same properties to all spheres
        - list of dicts: individual properties for each sphere
    ax : Axes, optional
        3D axes to plot on (must have projection='3d'). If None, creates new figure.
    figsize : tuple of float, default (6, 6)
        Figure size as (width, height) in inches.
    size_scale : float, default 100.0
        Scaling factor for sphere sizes. The scatter size is calculated as (radius * size_scale)^2.
        Adjust this if spheres appear too large or too small.
    **kwargs
        Additional keyword arguments passed to ax.scatter().
        These serve as additional defaults that can be overridden by sphere_props.
    
    Returns
    -------
    tuple
        (fig, ax) - The figure and 3D axes objects.
    
    Raises
    ------
    ValueError
        If length of radius array doesn't match number of positions, or if
        sphere_props format is invalid.
    
    Notes
    -----
    Spheres are rendered using scatter plot with sizes proportional to radius squared.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure and axes if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Convert positions to numpy array
    positions = np.asarray(positions)
    
    # Handle single radius vs array of radii
    if np.isscalar(radius):
        radii = np.full(len(positions), radius)
    else:
        radii = np.asarray(radius)
        if len(radii) != len(positions):
            raise ValueError("Length of radius array must match number of positions")
    
    # Set up default properties
    default_props = {
        'c': 'lightgrey',
        'edgecolors': 'black',
        'alpha': 0.8
    }
    default_props.update(kwargs)
    
    # Handle sphere_props
    if sphere_props is None:
        # Use defaults for all spheres
        props_list = [default_props.copy() for _ in positions]
    elif isinstance(sphere_props, dict):
        # Single dict for all spheres
        single_props = default_props.copy()
        single_props.update(sphere_props)
        props_list = [single_props.copy() for _ in positions]
    elif isinstance(sphere_props, (list, tuple)):
        # List of dicts, one per sphere
        if len(sphere_props) != len(positions):
            raise ValueError("Length of sphere_props list must match number of positions")
        props_list = []
        for props in sphere_props:
            sphere_prop = default_props.copy()
            if props is not None:
                sphere_prop.update(props)
            props_list.append(sphere_prop)
    else:
        raise ValueError("sphere_props must be None, dict, or list of dicts")
    
    # Plot spheres using scatter with size proportional to radius squared
    for pos, r, props in zip(positions, radii, props_list):
        size = (r * size_scale) ** 2
        ax.scatter(pos[0], pos[1], pos[2], s=size, **props)
    
    return fig, ax


# %% Helper functions

def _extend_list(lst: List, target_length: int, default_value) -> List:
    """
    Extend list to target length with default value.
    
    Parameters
    ----------
    lst : list
        List to extend.
    target_length : int
        Target length for the list.
    default_value : any
        Value to use for extending the list.
    
    Returns
    -------
    list
        Extended list.
    """
    if len(lst) < target_length:
        return lst + [default_value] * (target_length - len(lst))
    return lst