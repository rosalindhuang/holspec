# holspec/visualization/basic_plots.py
"""
Basic plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from typing import Optional, Tuple, List, Union, Dict, Any
from itertools import combinations

# =============================================================================
# Line plots
# =============================================================================

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

    Notes
    -----
    For stacked subplots, use the ``ax`` parameter in a loop::

        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        for ax, (x, y), c, lbl in zip(axes, data_list, colors, labels):
            plot_lines([(x, y)], ax=ax, color=c, legend_labels=[lbl], **kwargs)
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


# =============================================================================
# Stem plots
# =============================================================================

def plot_stems(
    xy_list: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    y_list: Optional[List[np.ndarray]] = None,
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5, 4),
    # Stem styling
    color: Optional[str] = None,
    colors_list: Optional[List[str]] = None,
    # Legend
    legend_labels: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot multiple stem (lollipop) plots on a single axis.

    Parameters
    ----------
    xy_list : list of (x, y) tuples, optional
        List of (x_array, y_array) tuples to plot.
    y_list : list of ndarray, optional
        List of y arrays to plot. x values default to np.arange(len(y)).
        Mutually exclusive with xy_list.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple of float, default (5, 4)
        Figure size as (width, height) in inches.
    color : str, optional
        Single color for all stems (overrides colors_list).
    colors_list : list of str, optional
        List of colors for each stem series. If shorter than data, extended with 'k'.
    legend_labels : list of str, optional
        List of legend labels. If provided, legend will be displayed.
    **kwargs
        Stem-specific keyword arguments popped before passing to ax.stem():
        - linefmt (str): format string for stem lines; if set, overrides color for lines.
        - markerfmt (str): format string for markers; if set, overrides color for markers.
        - basefmt (str, default 'k-'): format string for the baseline.
        - bottom (float, default 0): y-value for the baseline.
        - markersize (float): marker size for stem tips (applied via StemContainer).
        - linewidth (float): line width for stem lines (applied via StemContainer).
        Remaining kwargs are passed to ax.stem().

    Returns
    -------
    tuple
        (fig, ax) - The figure and axes objects.

    Raises
    ------
    ValueError
        If both xy_list and y_list are provided, or neither is provided.

    Notes
    -----
    For stacked subplots, use the ``ax`` parameter in a loop::

        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        for ax, y, c, lbl in zip(axes, y_list, colors, labels):
            plot_stems(y_list=[y], ax=ax, color=c, legend_labels=[lbl], **kwargs)
    """
    if xy_list is not None and y_list is not None:
        raise ValueError("Provide either xy_list or y_list, not both.")
    if xy_list is None and y_list is None:
        raise ValueError("Must provide either xy_list or y_list.")

    # Convert y_list to xy_list
    if y_list is not None:
        xy_list = [(np.arange(len(y)), np.asarray(y)) for y in y_list]

    if len(xy_list) == 0:
        raise ValueError("Data list must contain at least one entry.")

    # Pop stem-specific kwargs that need special routing
    linefmt = kwargs.pop('linefmt', None)
    markerfmt = kwargs.pop('markerfmt', None)
    basefmt = kwargs.pop('basefmt', 'k-')
    bottom = kwargs.pop('bottom', 0)
    markersize = kwargs.pop('markersize', None)
    linewidth = kwargs.pop('linewidth', None)

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

    # Plot stems
    for i, (x, y) in enumerate(xy_list):
        label = labels[i] if labels is not None else None
        bfmt = basefmt if i == 0 else ''
        container = ax.stem(x, y, linefmt=linefmt or '', markerfmt=markerfmt or 'o',
                            basefmt=bfmt, bottom=bottom, label=label, **kwargs)

        # Apply color via container when format strings don't encode color
        if linefmt is None:
            container.stemlines.set_color(colors[i])
        if markerfmt is None:
            container.markerline.set_color(colors[i])

        # Apply optional sizing
        if markersize is not None:
            container.markerline.set_markersize(markersize)
        if linewidth is not None:
            container.stemlines.set_linewidth(linewidth)

    # Add legend if labels were provided
    if legend_labels is not None:
        ax.legend()

    return fig, ax


# =============================================================================
# Scatter plots
# =============================================================================

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


# =============================================================================
# Basic shapes plots
# =============================================================================

def plot_circles(
    positions: np.ndarray,
    radius: Union[float, np.ndarray],
    # Circle styling
    circle_props: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    # Figure properties
    ax: Optional[Axes] = None,
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
        fig, ax = plt.subplots()
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
        fig = plt.figure()
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
        
        # Fix matplotlib warning: convert 'c' to 'color' if it's an RGBA tuple
        if 'c' in props and isinstance(props['c'], (tuple, list)) and len(props['c']) in [3, 4]:
            props['color'] = props.pop('c')
        
        ax.scatter(pos[0], pos[1], pos[2], s=size, **props)
    
    return fig, ax


# =============================================================================
# Simplicial complex primitives (2D)
# =============================================================================

def plot_vertices_2d(
    vertices: List[Tuple],
    positions: np.ndarray,
    color: str = 'C0',
    radius: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 0-simplices (vertices) as circles in 2D.
    
    Parameters
    ----------
    vertices : List[Tuple]
        List of vertex simplices, each a tuple with 1 int index.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) vertex coordinates.
    color : str, default='C0'
        Fill color for vertices.
    radius : float, optional
        Radius of circles. If None, auto-computed from data range.
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to plot_circles().
        Can override circle_props, e.g., edgecolor='black'.
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    
    Examples
    --------
    >>> fig, ax = plot_vertices_2d(
    ...     vertices, positions, 
    ...     color='blue', 
    ...     circle_props={'edgecolor': 'black', 'linewidth': 2}
    ... )
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Extract vertex indices and get their positions
    vertex_indices = [v[0] for v in vertices]
    vertex_positions = positions[vertex_indices]
    
    # Auto-compute radius if not provided
    if radius is None:
        data_range = np.ptp(positions, axis=0).max()
        radius = 0.02 * data_range if data_range > 0 else 0.05
    
    # Set up circle properties
    default_circle_props = {
        'facecolor': color,
        'edgecolor': 'black',
        'linewidth': 1,
        'alpha': 1
    }
    
    # Merge with user-provided circle_props if present
    if 'circle_props' in kwargs:
        user_props = kwargs.pop('circle_props')
        if isinstance(user_props, dict):
            default_circle_props.update(user_props)
        else:
            # If user provided list, use it directly
            kwargs['circle_props'] = user_props
            return plot_circles(vertex_positions, radius, ax=ax, **kwargs)
    
    # Plot using plot_circles
    fig, ax = plot_circles(
        vertex_positions,
        radius,
        circle_props=default_circle_props,
        ax=ax,
        **kwargs
    )
    
    return fig, ax


def plot_edges_2d(
    edges: List[Tuple],
    positions: np.ndarray,
    color: str = 'C1',
    linewidth: float = 1.5,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 1-simplices (edges) as line segments in 2D.
    
    Parameters
    ----------
    edges : List[Tuple]
        List of edge simplices, each a tuple of 2 vertex indices.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) vertex coordinates.
    color : str, default='C1'
        Line color for all edges.
    linewidth : float, default=1.5
        Line width for edges.
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to ax.plot().
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    
    Examples
    --------
    >>> fig, ax = plot_edges_2d(edges, positions, color='blue', linewidth=2)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Validate positions
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be shape (N, 2), got {positions.shape}")
    
    # Plot each edge
    for edge in edges:
        edge_positions = positions[list(edge)]
        x, y = edge_positions.T
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=1, **kwargs)
    
    return fig, ax


def plot_triangles_2d(
    triangles: List[Tuple],
    positions: np.ndarray,
    color: str = 'C2',
    alpha: float = 0.4,
    edgecolor: str = 'none',
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 2-simplices (triangles) as filled polygons in 2D.
    
    Parameters
    ----------
    triangles : List[Tuple]
        List of triangle simplices, each a tuple of 3 vertex indices.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) vertex coordinates.
    color : str, default='C2'
        Fill color for all triangles.
    alpha : float, default=0.4
        Transparency for triangles (0=transparent, 1=opaque).
    edgecolor : str, default='none'
        Edge color for triangle outlines.
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to Polygon patches.
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    
    Examples
    --------
    >>> fig, ax = plot_triangles_2d(triangles, positions)
    """
    from matplotlib.patches import Polygon as PolygonPatch
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Validate positions
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be shape (N, 2), got {positions.shape}")
    
    # Plot each triangle
    for triangle in triangles:
        tri_positions = positions[list(triangle)]
        polygon = PolygonPatch(
            tri_positions, 
            closed=True,
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            **kwargs
        )
        ax.add_patch(polygon)
    
    return fig, ax


# =============================================================================
# Simplicial complex primitives (3D)
# =============================================================================

def plot_vertices_3d(
    vertices: List[Tuple],
    positions: np.ndarray,
    color: str = 'C0',
    radius: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 0-simplices (vertices) as spheres in 3D.
    
    Parameters
    ----------
    vertices : List[Tuple]
        List of vertex simplices, each a tuple with 1 int index.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    color : str, default='C0'
        Fill color for vertices.
    radius : float, optional
        Radius of spheres. If None, auto-computed from data range.
    ax : Axes, optional
        3D axes to plot on. If None, creates new figure.
    **kwargs
        Passed to plot_spheres(). Can override sphere_props.
    
    Returns
    -------
    fig, ax : Figure, Axes
    """
    # Create 3D figure if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Extract vertex indices and get their positions
    vertex_indices = [v[0] for v in vertices]
    vertex_positions = positions[vertex_indices]
    
    # Auto-compute radius if not provided
    if radius is None:
        data_range = np.ptp(positions, axis=0).max()
        radius = 0.06 * data_range if data_range > 0 else 0.05
    
    # Set up sphere properties
    default_sphere_props = {
        'c': color,
        'edgecolors': 'black',
        'linewidths': 1,
        'alpha': 1
    }
    
    # Merge with user-provided sphere_props if present
    if 'sphere_props' in kwargs:
        user_props = kwargs.pop('sphere_props')
        if isinstance(user_props, dict):
            default_sphere_props.update(user_props)
        else:
            # If user provided list, use it directly
            kwargs['sphere_props'] = user_props
            return plot_spheres(vertex_positions, radius, ax=ax, **kwargs)
    
    # Plot using plot_spheres
    fig, ax = plot_spheres(
        vertex_positions,
        radius,
        sphere_props=default_sphere_props,
        ax=ax,
        **kwargs
    )
    
    return fig, ax


def plot_edges_3d(
    edges: List[Tuple],
    positions: np.ndarray,
    color: str = 'C1',
    linewidth: float = 2.0,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 1-simplices (edges) as line segments in 3D.
    
    Parameters
    ----------
    edges : List[Tuple]
        List of edge simplices, each a tuple of 2 vertex indices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    color : str, default='C1'
        Line color.
    linewidth : float, default=2.0
        Line width.
    ax : Axes, optional
        3D axes to plot on. If None, creates new figure.
    **kwargs
        Passed to ax.plot().
    
    Returns
    -------
    fig, ax : Figure, Axes
    """
    # Create 3D figure if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Validate positions
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be shape (N, 3), got {positions.shape}")
    
    # Plot each edge
    for edge in edges:
        p0 = positions[edge[0]]
        p1 = positions[edge[1]]
        ax.plot(
            [p0[0], p1[0]], 
            [p0[1], p1[1]], 
            [p0[2], p1[2]],
            color=color, 
            linewidth=linewidth, 
            alpha=1,
            **kwargs
        )
    
    return fig, ax


def plot_triangles_3d(
    triangles: List[Tuple],
    positions: np.ndarray,
    color: str = 'C2',
    alpha: float = 0.4,
    edgecolor: str = 'none',
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 2-simplices (triangles) as filled polygons in 3D.
    
    Parameters
    ----------
    triangles : List[Tuple]
        List of triangle simplices, each a tuple of 3 vertex indices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    color : str, default='C2'
        Fill color.
    alpha : float, default=0.4
        Transparency (0=transparent, 1=opaque).
    edgecolor : str, default='none'
        Edge color for outlines.
    ax : Axes, optional
        3D axes to plot on. If None, creates new figure.
    **kwargs
        Passed to Poly3DCollection.
    
    Returns
    -------
    fig, ax : Figure, Axes
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create 3D figure if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Validate positions
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be shape (N, 3), got {positions.shape}")
    
    # Plot each triangle
    for triangle in triangles:
        tri_positions = positions[list(triangle)]
        poly = Poly3DCollection(
            [tri_positions],
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            **kwargs
        )
        ax.add_collection3d(poly)
    
    return fig, ax


def plot_tetrahedra_3d(
    tetrahedra: List[Tuple],
    positions: np.ndarray,
    color: str = 'C3',
    alpha: float = 0.2,
    edgecolor: str = 'none',
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot 3-simplices (tetrahedra) as collections of triangular facets in 3D.
    
    Each tetrahedron is rendered as 4 triangular faces. Very low alpha recommended.
    
    Parameters
    ----------
    tetrahedra : List[Tuple]
        List of tetrahedron simplices, each a tuple of 4 vertex indices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    color : str, default='C3'
        Fill color.
    alpha : float, default=0.2
        Transparency (much lower than triangles for visibility).
    edgecolor : str, default='none'
        Edge color for facet outlines.
    ax : Axes, optional
        3D axes to plot on. If None, creates new figure.
    **kwargs
        Passed to Poly3DCollection.
    
    Returns
    -------
    fig, ax : Figure, Axes
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create 3D figure if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Validate positions
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be shape (N, 3), got {positions.shape}")
    
    # Plot each tetrahedron as 4 triangular facets
    for tet in tetrahedra:
        # Generate all 4 facets (combinations of 3 vertices from 4)
        facets = list(combinations(tet, 3))
        for facet in facets:
            facet_positions = positions[list(facet)]
            poly = Poly3DCollection(
                [facet_positions],
                facecolor=color,
                edgecolor=edgecolor,
                alpha=alpha,
                **kwargs
            )
            ax.add_collection3d(poly)
    
    return fig, ax


def add_simplex_orientation_2d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    dims: List[int],
    colors: Optional[Dict[int, str]] = None,
    arrow_scale: float = 0.02,
    fontsize: int = 10,
    ax: Optional[Axes] = None
) -> None:
    """
    Add orientation markers to simplices in 2D.
    
    For edges (dim 1): draws arrows at midpoints.
    For triangles (dim 2): draws +/− signs based on winding order.
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) coordinates.
    dims : List[int]
        Which dimensions to add orientation (e.g., [1, 2]).
    colors : Dict[int, str], optional
        Colors per dimension. Default: {1: 'C1', 2: 'C2'}.
    arrow_scale : float, default=0.02
        Arrow size as fraction of axis range (for edges).
    fontsize : int, default=10
        Font size for triangle orientation signs.
    ax : Axes
        Axes to draw on. Required.
    
    Examples
    --------
    >>> simplices = {1: [(0, 1), (1, 2)], 2: [(0, 1, 2)]}
    >>> positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> fig, ax = plt.subplots()
    >>> add_simplex_orientation_2d(simplices, positions, [1, 2], ax=ax)
    """
    if ax is None:
        raise ValueError("ax parameter is required for add_simplex_orientation_2d")
    
    # Default colors
    if colors is None:
        colors = {1: 'C1', 2: 'C2'}
    
    # Get axis range for scaling
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    axis_scale = max(np.ptp(xlim), np.ptp(ylim))
    
    # Add orientation for each requested dimension
    for dim in dims:
        if dim not in simplices or not simplices[dim]:
            continue
        
        color = colors.get(dim, 'black')
        
        if dim == 1:
            # Add arrows to edges
            arrow_size = arrow_scale * axis_scale
            for edge in simplices[1]:
                p0 = positions[edge[0]]
                p1 = positions[edge[1]]
                
                # Compute direction
                vec = p1 - p0
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    direction = vec / vec_norm
                else:
                    continue
                
                # Position arrow at midpoint
                midpoint = 0.5 * (p0 + p1)
                arrow_start = midpoint - arrow_size * direction
                arrow_dxdy = 2 * arrow_size * direction
                
                # Draw arrow
                ax.arrow(
                    arrow_start[0], arrow_start[1],
                    arrow_dxdy[0], arrow_dxdy[1],
                    head_width=arrow_size,
                    head_length=arrow_size * 1.3,
                    fc=color,
                    ec=color,
                    length_includes_head=True,
                    zorder=12,
                    alpha=1
                )
        
        elif dim == 2:
            # Add +/− signs to triangles
            circle_radius = 0.02 * axis_scale
            for triangle in simplices[2]:
                # Get triangle vertices
                v0, v1, v2 = positions[list(triangle)]
                
                # Compute signed area (determinant)
                det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
                sign = np.sign(det).astype(int)
                
                if sign != 0:
                    centroid = np.mean(positions[list(triangle)], axis=0)
                    symbol = '+' if sign > 0 else '−'
                    
                    # Add text symbol
                    ax.text(
                        *centroid, symbol,
                        fontsize=fontsize,
                        weight='bold',
                        ha='center',
                        va='center',
                        color=color,
                        zorder=15
                    )
                    
                    # Add circle around symbol
                    circle = plt.Circle(
                        centroid,
                        radius=circle_radius,
                        color='none',
                        ec=color,
                        lw=1.2,
                        zorder=14
                    )
                    ax.add_patch(circle)


def add_simplex_labels_2d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    dims: List[int],
    colors: Optional[Dict[int, str]] = None,
    fontsize: int = 9,
    ax: Optional[Axes] = None
) -> None:
    """
    Add index labels to simplices in 2D.
    
    Labels are placed at the centroid of each simplex (vertex position for
    0-simplices, edge midpoint for 1-simplices, triangle centroid for
    2-simplices).
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
    positions : np.ndarray
        Array of shape (N, 2) with (x, y) vertex coordinates.
    dims : List[int]
        Which dimensions to label (e.g., [0, 1, 2]).
    colors : Dict[int, str], optional
        Colors per dimension. If None, uses {0: 'C0', 1: 'C1', 2: 'C2'}.
    fontsize : int, default=9
        Font size for labels.
    ax : Axes
        Axes to draw labels on. Must be provided.
    
    Examples
    --------
    >>> simplices = {0: [(0,), (1,)], 1: [(0, 1)]}
    >>> positions = np.array([[0, 0], [1, 0]])
    >>> fig, ax = plt.subplots()
    >>> add_simplex_labels_2d(simplices, positions, [0, 1], ax=ax)
    """
    if ax is None:
        raise ValueError("ax parameter is required for add_simplex_labels_2d")
    
    # Default colors
    if colors is None:
        colors = {0: 'C0', 1: 'C1', 2: 'C2'}
    
    # Get axis range for offset scaling
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    axis_scale = max(np.ptp(xlim), np.ptp(ylim))
    offset = 0.03 * axis_scale
    
    # Add labels for each requested dimension
    for dim in dims:
        if dim not in simplices or not simplices[dim]:
            continue
        
        color = colors.get(dim, 'black')
        
        for i, simplex in enumerate(simplices[dim]):
            # Compute centroid/position
            simplex_positions = positions[list(simplex)]
            centroid = np.mean(simplex_positions, axis=0)
            
            # Add vertical offset for all dimensions
            label_pos = centroid + np.array([0, offset])
            
            # Add text label
            ax.text(
                *label_pos, str(i),
                fontsize=fontsize,
                ha='center',
                va='center',
                color=color,
                zorder=20
            )


# =============================================================================
# Simplicial complex annotation (3D)
# =============================================================================

def add_simplex_orientation_3d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    dims: List[int],
    colors: Optional[Dict[int, str]] = None,
    arrow_scale: float = 0.02,
    fontsize: int = 10,
    ax: Optional[Axes] = None
) -> None:
    """
    Add orientation markers to simplices in 3D.
    
    For edges (dim 1): draws arrows along edge direction.
    For triangles (dim 2): draws arrows along surface normal.
    Dimensions 0 and 3 have no orientation visualization.
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    dims : List[int]
        Which dimensions to add orientation (e.g., [1, 2]).
    colors : Dict[int, str], optional
        Colors per dimension. Default: {1: 'C1', 2: 'C2'}.
    arrow_scale : float, default=0.02
        Arrow size as fraction of axis range.
    fontsize : int, default=10
        Font size (unused in 3D, kept for API consistency).
    ax : Axes
        3D axes to draw on. Required.
    
    Examples
    --------
    >>> simplices = {1: [(0, 1), (1, 2)], 2: [(0, 1, 2)]}
    >>> positions = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> add_simplex_orientation_3d(simplices, positions, [1, 2], ax=ax)
    """
    if ax is None:
        raise ValueError("ax parameter is required for add_simplex_orientation_3d")
    
    # Default colors
    if colors is None:
        colors = {1: 'C1', 2: 'C2'}
    
    # Get axis range for scaling
    data_range = np.ptp(positions, axis=0)
    axis_scale = data_range.max()
    
    # Add orientation for each requested dimension
    for dim in dims:
        if dim not in simplices or not simplices[dim]:
            continue
        
        color = colors.get(dim, 'black')
        
        if dim == 1:
            # Add arrows to edges
            arrow_size = arrow_scale * axis_scale
            for edge in simplices[1]:
                p0 = positions[edge[0]]
                p1 = positions[edge[1]]
                
                # Compute direction
                vec = p1 - p0
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    direction = vec / vec_norm
                else:
                    continue
                
                # Position arrow at midpoint
                midpoint = 0.5 * (p0 + p1)
                arrow_length = arrow_size
                
                # Draw arrow using quiver
                ax.quiver(
                    midpoint[0], midpoint[1], midpoint[2],
                    direction[0], direction[1], direction[2],
                    length=arrow_length,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    color=color,
                    linewidth=1.5,
                    alpha=1
                )
        
        elif dim == 2:
            # Add normal arrows to triangles
            arrow_size = 0.15 * axis_scale
            for triangle in simplices[2]:
                # Get triangle vertices
                v0, v1, v2 = positions[list(triangle)]
                
                # Compute surface normal via cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_norm = np.linalg.norm(normal)
                
                if normal_norm > 0:
                    normal = normal / normal_norm
                else:
                    continue
                
                # Position arrow at centroid
                centroid = np.mean(positions[list(triangle)], axis=0)
                
                # Draw normal arrow
                ax.quiver(
                    centroid[0], centroid[1], centroid[2],
                    normal[0], normal[1], normal[2],
                    length=arrow_size,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    color=color,
                    linewidth=1.5,
                    alpha=1
                )


def add_simplex_labels_3d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    dims: List[int],
    colors: Optional[Dict[int, str]] = None,
    fontsize: int = 8,
    ax: Optional[Axes] = None
) -> None:
    """
    Add index labels to simplices in 3D.
    
    Labels placed at simplex centroids with small offset for visibility.
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) coordinates.
    dims : List[int]
        Which dimensions to label (e.g., [0, 1, 2, 3]).
    colors : Dict[int, str], optional
        Colors per dimension. Default: {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}.
    fontsize : int, default=8
        Font size.
    ax : Axes
        3D axes to draw on. Required.
    
    Examples
    --------
    >>> simplices = {0: [(0,), (1,)], 1: [(0, 1)]}
    >>> positions = np.array([[0, 0, 0], [1, 0, 0]])
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> add_simplex_labels_3d(simplices, positions, [0, 1], ax=ax)
    """
    if ax is None:
        raise ValueError("ax parameter is required for add_simplex_labels_3d")
    
    # Default colors
    if colors is None:
        colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}
    
    # Get axis range for offset scaling
    data_range = np.ptp(positions, axis=0)
    axis_scale = data_range.max()
    offset = 0.03 * axis_scale
    
    # Add labels for each requested dimension
    for dim in dims:
        if dim not in simplices or not simplices[dim]:
            continue
        
        color = colors.get(dim, 'black')
        
        for i, simplex in enumerate(simplices[dim]):
            # Compute centroid
            simplex_positions = positions[list(simplex)]
            centroid = np.mean(simplex_positions, axis=0)
            
            # Add offset for visibility
            label_pos = centroid + np.array([offset, offset, offset])
            
            # Add text label using text3D
            ax.text(
                label_pos[0], label_pos[1], label_pos[2], str(i),
                fontsize=fontsize,
                ha='center',
                va='center',
                color=color,
                zorder=20
            )


# =============================================================================
# Matrix plots
# =============================================================================

def plot_matrix(
    matrix: np.ndarray,
    # Figure properties
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    # Colormap
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric: bool = False,
    # Zero masking
    zero_color: Optional[str] = None,
    zero_thresh: float = 1e-8,
    # Colorbar
    colorbar: bool = False,
    **kwargs
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Axes, Colorbar]]:
    """
    Visualize a matrix as a color-mapped grid using pcolormesh.

    Each cell in the plot corresponds to one entry in the matrix. Row 0 is
    placed at the top, matching standard matrix convention. The axes aspect
    ratio is set to equal so every cell is square.

    Parameters
    ----------
    matrix : array-like
        2-D array of shape (nrows, ncols) to visualize.
    ax : Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches. If None, auto-sized based on
        the matrix shape at 0.65 inches per cell.
    cmap : str, default 'RdBu_r'
        Matplotlib colormap name. The default places red at positive values
        and blue at negative values.
    vmin : float, optional
        Lower bound for colormap normalization. Defaults to data minimum.
    vmax : float, optional
        Upper bound for colormap normalization. Defaults to data maximum.
    symmetric : bool, default False
        If True, automatically sets ``vmin = -abs_max`` and ``vmax = abs_max``
        where ``abs_max`` is the maximum absolute value in the data. This
        ensures the colormap is centered on zero. Overrides any explicit
        ``vmin``/``vmax`` when True.
    zero_color : str, optional
        Color for entries satisfying ``|value| <= zero_thresh``. If None,
        zero masking is disabled and zeros are colored by the colormap.
    zero_thresh : float, default 1e-8
        Threshold below which entries are considered zero. Only used when
        ``zero_color`` is not None.
    colorbar : bool, default False
        Whether to add a colorbar to the figure.
    **kwargs
        Additional keyword arguments passed to ``ax.pcolormesh()``.

    Returns
    -------
    tuple
        ``(fig, ax)`` if ``colorbar=False``, or ``(fig, ax, cbar)`` if
        ``colorbar=True``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.random.randn(4, 6)
    >>> fig, ax = plot_matrix(M, vmin=-3, vmax=3)

    >>> # Highlight zeros in a sparse matrix
    >>> fig, ax, cbar = plot_matrix(M, zero_color='lightgray', colorbar=True)
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")

    nrows, ncols = matrix.shape

    # Build colormap object; copy it when we need to set a bad (masked) color
    # so the global colormap registry is not mutated.
    if zero_color is not None:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(zero_color)
        data = np.ma.masked_where(np.abs(matrix) <= zero_thresh, matrix)
    else:
        cmap_obj = cmap
        data = matrix

    # Symmetric normalization: center colormap on zero
    if symmetric:
        filled = np.ma.filled(data, 0) if np.ma.is_masked(data) else data
        abs_max = np.nanmax(np.abs(filled))
        vmin, vmax = -abs_max, abs_max

    # Auto figsize: 0.65 in per cell, extra width for colorbar
    if ax is None:
        if figsize is None:
            cell = 0.65
            w = ncols * cell + (1.5 if colorbar else 0.4)
            h = nrows * cell + 0.4
            figsize = (w, h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Cell edge coordinates
    X = np.arange(ncols + 1)
    Y = np.arange(nrows + 1)

    mesh = ax.pcolormesh(X, Y, data, cmap=cmap_obj, vmin=vmin, vmax=vmax, **kwargs)

    # Row 0 at top
    ax.invert_yaxis()

    # Equal aspect so each cell is square
    ax.set_aspect('equal')

    # Ticks at cell centres, labelled with integer indices
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(np.arange(nrows))
    ax.tick_params(length=0)

    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax)
        return fig, ax, cbar

    return fig, ax


# =============================================================================
# Helper functions
# =============================================================================

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