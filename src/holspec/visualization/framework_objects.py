# holspec/visualization/framework_objects.py
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
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, Normalize
from typing import Optional, Sequence, Tuple, Union, List, Dict

from holspec.analysis import OBSERVABLE_LABELS
from holspec.visualization import (
    format_axis,
    format_cbar,
    plot_bars,
    plot_lines,
    plot_circles,
    plot_spheres,
    plot_vertices_2d,
    plot_edges_2d,
    plot_triangles_2d,
    add_simplex_orientation_2d,
    add_simplex_labels_2d,
    plot_vertices_3d,
    plot_edges_3d,
    plot_triangles_3d,
    plot_tetrahedra_3d,
    add_simplex_orientation_3d,
    add_simplex_labels_3d,
    values_to_colors,
)

# =============================================================================
# Labels
# =============================================================================

def make_Lk_label(k, comp):
    """LaTeX label for Hodge Laplacian: L^k or L^{k,comp}."""
    if comp == 'full':
        return f'L^{{{k}}}'
    return f'L^{{{k},\\mathrm{{{comp}}}}}'


# =============================================================================
# Point data
# =============================================================================

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


# =============================================================================
# Simplicial complex
# =============================================================================

def _parse_feature_flags(flags: Union[bool, Dict[int, bool]], all_keys: List[int]) -> List[int]:
    """
    Helper: Convert bool or dict flags to list of enabled keys.
        
    Parameters
    ----------
    flags : bool or Dict[int, bool]
        Feature flags specification:
        - True: enable for all keys
        - False/None: enable for no keys
        - Dict: enable for keys where value is True
    all_keys : List[int]
        Complete list of valid keys.
    
    Returns
    -------
    List[int]
        List of keys where feature is enabled.
    
    Examples
    --------
    >>> _parse_feature_flags(True, [0, 1, 2])
    [0, 1, 2]
    >>> _parse_feature_flags({0: True, 1: False, 2: True}, [0, 1, 2])
    [0, 2]
    >>> _parse_feature_flags(False, [0, 1, 2])
    []
    """
    if flags is True:
        return all_keys
    elif isinstance(flags, dict):
        return [k for k in all_keys if flags.get(k, False)]
    return []


def plot_simplicial_complex_2d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    # Color control
    simplex_colors: Optional[Dict[int, str]] = None,
    # Styling per dimension
    simplex_kwargs: Optional[Dict[int, Dict]] = None,
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
    simplex_colors : Dict[int, str], optional
        Colors per dimension. Default: {0: 'C0', 1: 'C1', 2: 'C2'}.
        Example: {0: 'red', 1: 'blue', 2: 'green'}
    simplex_kwargs : Dict[int, Dict], optional
        Additional keyword arguments per dimension, passed to primitive functions.
        Example: {0: {'radius': 0.1}, 1: {'linewidth': 2}, 2: {'alpha': 0.3}}
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
        Additional styling arguments applied to all dimensions.
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    
    Examples
    --------
    >>> fig, ax = plot_simplicial_complex_2d(
    ...     simplices, positions,
    ...     simplex_colors={0: 'red', 1: 'blue', 2: 'yellow'},
    ...     simplex_kwargs={0: {'radius': 0.08}, 1: {'linewidth': 3}, 2: {'alpha': 0.6}}
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
    color_map = {**default_colors, **(simplex_colors or {})}
    
    # Set up per-dimension kwargs
    dim_kwargs = simplex_kwargs or {}
    
    # Plot simplices (triangles -> edges -> vertices for proper layering)
    if 2 in simplices and simplices[2]:
        kw = {**kwargs, **dim_kwargs.get(2, {})}
        fig, ax = plot_triangles_2d(simplices[2], positions, color=color_map[2], ax=ax, zorder=10, **kw)
    
    if 1 in simplices and simplices[1]:
        kw = {**kwargs, **dim_kwargs.get(1, {})}
        fig, ax = plot_edges_2d(simplices[1], positions, color=color_map[1], ax=ax, zorder=20, **kw)
    
    if 0 in simplices and simplices[0]:
        kw = {**kwargs, **dim_kwargs.get(0, {})}
        fig, ax = plot_vertices_2d(simplices[0], positions, color=color_map[0], ax=ax, zorder=30, **kw)
    
    # Add labels if requested
    if show_labels:
        dims_to_label = _parse_feature_flags(show_labels, [0, 1, 2])
        if dims_to_label:
            add_simplex_labels_2d(simplices, positions, dims_to_label, color_map, ax=ax)
    
    # Add orientation if requested
    if show_orientation:
        dims_to_orient = _parse_feature_flags(show_orientation, [1, 2])
        if dims_to_orient:
            add_simplex_orientation_2d(simplices, positions, dims_to_orient, color_map, ax=ax)
    
    return fig, ax


def plot_simplicial_complex_3d(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    # Color control
    simplex_colors: Optional[Dict[int, str]] = None,
    # Styling per dimension
    simplex_kwargs: Optional[Dict[int, Dict]] = None,
    # Labeling control
    show_labels: Optional[Union[bool, Dict[int, bool]]] = None,
    # Orientation control
    show_orientation: Optional[Union[bool, Dict[int, bool]]] = None,
    # Figure
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a 3D simplicial complex with vertices, edges, triangles, and tetrahedra.
    
    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Dictionary mapping dimension to list of simplices.
        Keys: 0 (vertices), 1 (edges), 2 (triangles), 3 (tetrahedra).
        Each simplex is a tuple of vertex indices.
    positions : np.ndarray
        Array of shape (N, 3) with (x, y, z) vertex coordinates.
    simplex_colors : Dict[int, str], optional
        Colors per dimension. Default: {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}.
        Example: {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    simplex_kwargs : Dict[int, Dict], optional
        Additional keyword arguments per dimension, passed to primitive functions.
        Example: {0: {'radius': 0.05}, 1: {'linewidth': 2.5}, 2: {'alpha': 0.5}}
    show_labels : bool or Dict[int, bool], optional
        Control labeling by dimension.
        - None or False: no labels
        - True: label all dimensions
        - {0: True, 1: False, 2: True, 3: False}: per-dimension control
    show_orientation : bool or Dict[int, bool], optional
        Control orientation markers by dimension.
        - None or False: no orientation markers
        - True: show for all applicable dimensions (1, 2)
        - {1: True, 2: False}: per-dimension control
        Note: Only dimensions 1 (arrows) and 2 (normals) have orientation.
        Dimensions 0 (vertices) and 3 (tetrahedra) have no orientation visualization.
    ax : Axes, optional
        3D axes to plot on. If None, creates new figure.
    **kwargs
        Additional styling arguments applied to all dimensions.
    
    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib 3D axes object.
    
    Examples
    --------
    >>> fig, ax = plot_simplicial_complex_3d(
    ...     simplices, positions,
    ...     simplex_colors={0: 'red', 1: 'blue', 2: 'yellow', 3: 'purple'},
    ...     simplex_kwargs={0: {'radius': 0.04}, 1: {'linewidth': 2}, 3: {'alpha': 0.05}},
    ...     show_labels={0: True, 1: False, 2: True, 3: False},
    ...     show_orientation={1: True, 2: False}
    ... )    
    
    Notes
    -----
    - For fine-grained control over styling, use the basic plotting
      functions directly: plot_tetrahedra_3d(), plot_triangles_3d(), 
      plot_edges_3d(), plot_vertices_3d()
    - Orientation only applies to dimensions 1 (edge arrows) and 2 (face normals)
    - Tetrahedra rendered as 4 triangular facets with low alpha (0.1)
    - Rendering order: back-to-front (tetrahedra -> triangles -> edges -> vertices)
    - Labels show simplex indices in order of appearance in simplices dict
    - Default colors use matplotlib's color cycle: C0, C1, C2, C3
    - Equal aspect ratio automatically set based on data range
    """
    # Create 3D figure if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Set up default colors
    default_colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}
    color_map = {**default_colors, **(simplex_colors or {})}
    
    # Set up per-dimension kwargs
    dim_kwargs = simplex_kwargs or {}
    
    # Plot simplices (back-to-front: tetrahedra -> triangles -> edges -> vertices)
    if 3 in simplices and simplices[3]:
        kw = {**kwargs, **dim_kwargs.get(3, {})}
        kw.setdefault('edgecolor', color_map[1])
        fig, ax = plot_tetrahedra_3d(simplices[3], positions, color=color_map[3], ax=ax, **kw)
    
    if 2 in simplices and simplices[2]:
        kw = {**kwargs, **dim_kwargs.get(2, {})}
        kw.setdefault('edgecolor', color_map[1])
        fig, ax = plot_triangles_3d(simplices[2], positions, color=color_map[2], ax=ax, **kw)
    
    if 1 in simplices and simplices[1]:
        kw = {**kwargs, **dim_kwargs.get(1, {})}
        fig, ax = plot_edges_3d(simplices[1], positions, color=color_map[1], ax=ax, **kw)
    
    if 0 in simplices and simplices[0]:
        kw = {**kwargs, **dim_kwargs.get(0, {})}
        fig, ax = plot_vertices_3d(simplices[0], positions, color=color_map[0], ax=ax, **kw)
    
    # Re-plot vertices on top (same trick as old repo: scatter drawn last wins depth sort)
    if 0 in simplices and simplices[0]:
        kw = {**kwargs, **dim_kwargs.get(0, {})}
        fig, ax = plot_vertices_3d(simplices[0], positions, color=color_map[0], ax=ax, **kw)
    
    # Set equal aspect ratio
    ax.set_box_aspect(np.ptp(positions, axis=0))
    
    # Add labels if requested
    if show_labels:
        dims_to_label = _parse_feature_flags(show_labels, [0, 1, 2, 3])
        if dims_to_label:
            add_simplex_labels_3d(simplices, positions, dims_to_label, color_map, ax=ax)
    
    # Add orientation if requested (only dims 1 and 2)
    if show_orientation:
        dims_to_orient = _parse_feature_flags(show_orientation, [1, 2])
        if dims_to_orient:
            add_simplex_orientation_3d(simplices, positions, dims_to_orient, color_map, ax=ax)
    
    return fig, ax


# =============================================================================
# Cochain visualization
# =============================================================================

# Dispatch dicts: map simplex dimension to primitive plotting function
_PLOT_SIMPLEX_2D = {0: plot_vertices_2d, 1: plot_edges_2d, 2: plot_triangles_2d}
_PLOT_SIMPLEX_3D = {
    0: plot_vertices_3d, 1: plot_edges_3d,
    2: plot_triangles_3d, 3: plot_tetrahedra_3d,
}

# Default kwargs for reference geometry (lower-dim simplices drawn in background)
_DEFAULT_COMPLEX_KWARGS_2D = {
    0: {'radius_scale': 0.6},
    1: {'linewidth': 1.0},
}
_DEFAULT_COMPLEX_KWARGS_3D = {
    0: {'radius_scale': 0.6},
    1: {'linewidth': 1.0},
    2: {'alpha': 0.3, 'edgecolor': 'none'},
}


def plot_cochain(
    simplices: Dict[int, List[Tuple]],
    positions: np.ndarray,
    k: int,
    values: np.ndarray,
    *,
    cmap: str = 'coolwarm',
    clim: Optional[Tuple[float, float]] = None,
    show_complex: bool = True,
    complex_color: str = 'k',
    complex_kwargs: Optional[Dict[int, Dict]] = None,
    simplex_kwargs: Optional[Dict] = None,
    cbar: bool = True,
    cbar_kwargs: Optional[Dict] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Optional[Colorbar]]:
    """
    Visualize a scalar-valued cochain on a simplicial complex.

    Colors k-simplices by cochain values using a colormap, with optional
    lower-dimensional simplices drawn as structural reference. Works for
    both 2D and 3D ambient spaces, dispatching automatically based on
    the point positions.

    Parameters
    ----------
    simplices : Dict[int, List[Tuple]]
        Full simplicial complex. Keys are dimensions (0, 1, 2, ...),
        values are lists of vertex-index tuples.
    positions : np.ndarray
        Vertex coordinates, shape (N, 2) or (N, 3).
    k : int
        Degree of the cochain (dimension of simplices to color).
    values : np.ndarray
        Cochain values, one per k-simplex. Must have length
        ``len(simplices[k])``.
    cmap : str, default='coolwarm'
        Colormap name.
    clim : tuple of (float, float), optional
        Color limits (vmin, vmax). If None, uses symmetric limits
        centered at zero: ``(-max(|values|), +max(|values|))``.
    show_complex : bool, default=True
        Whether to draw lower-dimensional simplices (dim < k) as
        structural reference geometry. For k=0 nothing is drawn
        regardless of this setting.
    complex_color : str, default='k'
        Color for reference geometry.
    complex_kwargs : Dict[int, Dict], optional
        Per-dimension kwargs for reference geometry, keyed by simplex
        dimension. Merged with defaults (user values take precedence).
    simplex_kwargs : Dict, optional
        Kwargs for the colored k-simplices (e.g. linewidth, alpha).
    cbar : bool, default=True
        Whether to create a colorbar.
    cbar_kwargs : Dict, optional
        Kwargs passed to ``fig.colorbar()`` (e.g. shrink, pad).
    ax : Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : Figure
    ax : Axes
    cbar : Colorbar or None
        The colorbar object if ``cbar=True``, else None. Use
        ``format_cbar(cbar, ...)`` to style it.
    """
    values = np.asarray(values)
    ambient_dim = positions.shape[1]

    # Select 2D or 3D dispatch and defaults
    if ambient_dim == 2:
        plot_simplex = _PLOT_SIMPLEX_2D
        default_complex_kw = _DEFAULT_COMPLEX_KWARGS_2D
    elif ambient_dim == 3:
        plot_simplex = _PLOT_SIMPLEX_3D
        default_complex_kw = _DEFAULT_COMPLEX_KWARGS_3D
    else:
        raise ValueError(
            f"positions must be 2D or 3D, got shape {positions.shape}"
        )

    # Create figure/axes if needed
    if ax is None:
        if ambient_dim == 2:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Color normalization
    if clim is not None:
        vmin, vmax = clim
    else:
        absmax = np.max(np.abs(values)) if values.size > 0 else 1.0
        vmin, vmax = -absmax, absmax

    colors, sm = values_to_colors(values, cmap=cmap, vmin=vmin, vmax=vmax)

    # Draw reference geometry (lower-dim simplices)
    if show_complex:
        merged_complex_kw = {
            d: {**default_complex_kw.get(d, {}), **(complex_kwargs or {}).get(d, {})}
            for d in range(k)
        }
        # zorder: higher-dim reference behind colored simplices,
        # vertices (d=0) always on top for visibility
        for d in sorted(merged_complex_kw):
            if d not in simplices or not simplices[d]:
                continue
            if d not in plot_simplex:
                continue
            ref_zorder = 50 if d == 0 else 10 + d * 10
            kw = merged_complex_kw[d]
            plot_simplex[d](
                simplices[d], positions,
                color=complex_color, ax=ax, zorder=ref_zorder, **kw,
            )

    # Draw colored k-simplices
    kw = dict(simplex_kwargs or {})
    plot_simplex[k](
        simplices[k], positions,
        color=colors, ax=ax, zorder=40, **kw,
    )

    # Colorbar
    cbar_obj = None
    if cbar:
        cbar_obj = fig.colorbar(sm, ax=ax, **(cbar_kwargs or {}))

    return fig, ax, cbar_obj


# =============================================================================
# Spectra visualization
# =============================================================================

def plot_eigval_distribution(
    esa,
    analysis_config: dict,
    *,
    axis_config: Optional[Dict] = None,
    bar_config: Optional[Dict] = None,
    ylim_ranges: Optional[Dict[Tuple, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 4),
    float_fmt: str = '.4g',
) -> Figure:
    """
    Plot eigenvalue distribution bar charts for one spectra analysis,
    with one subplot per degree.

    Parameters
    ----------
    esa : EnsembleSpectraAnalysis
        The spectra analysis object (with cached distributions).
    analysis_config : dict
        Per-dataset analysis config with keys 'degrees', 'components',
        'nonzero', and optionally 'distribution_params'.
    axis_config : dict, optional
        Keyword arguments passed to format_axis for each subplot.
    bar_config : dict, optional
        Keyword arguments passed to plot_bars for each subplot.
    ylim_ranges : dict, optional
        Shared y-ranges as {(k, comp): (ylo, yhi)} for matching ylims
        across experiment series. If None, ylims are auto-scaled.
    title : str, optional
        Figure suptitle.
    figsize : tuple of float, default (14, 4)
        Figure size.
    float_fmt : str, default '.4g'
        Format string for annotation numbers.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if axis_config is None:
        axis_config = {}
    if bar_config is None:
        bar_config = {}

    degrees = analysis_config['degrees']
    components = analysis_config['components']
    analysis_keys = [(k, comp) for k in degrees for comp in components]
    nonzero = analysis_config.get('nonzero', False)
    dstrb_params = analysis_config.get('distribution_params', {})
    dims = esa.dimensions

    margins = axis_config.get('margins', (0.05, 0.05))
    x_margin = margins[0] if isinstance(margins, (tuple, list)) else margins
    y_margin = margins[1] if isinstance(margins, (tuple, list)) else margins

    fig, axes = plt.subplots(1, len(degrees), figsize=figsize)
    if len(degrees) == 1:
        axes = [axes]

    for col, (k, comp) in enumerate(analysis_keys):
        ax = axes[col]
        dist = esa.eigenvalue_distribution(k, comp, nonzero=nonzero)
        x = dist['x']
        density = dist['density_mean']
        bar_width = (x[1] - x[0]) if len(x) > 1 else 0.5

        plot_bars(xy_list=[(x, density)], ax=ax,
                  color=f'C{k}', width=bar_width, **bar_config)
        format_axis(ax, title=f"${make_Lk_label(k, comp)}$", **axis_config)

        # Set x-limits from distribution range config
        xrange = dstrb_params.get(k, {}).get('range')
        if xrange is not None:
            xspan = xrange[1] - xrange[0]
            xpad = xspan * x_margin
            ax.set_xlim(xrange[0] - xpad, xrange[1] + xpad)

        # Apply shared y-limits
        if ylim_ranges is not None:
            yrange = ylim_ranges.get((k, comp))
            if yrange is not None:
                ax.set_ylim(0, yrange[1] * (1 + y_margin))

        # Annotate with dimension, nullity, and eigenvalue range
        summary = esa.observables_summary(k, comp)
        N_k = dims.get(k, 0)
        null_mean = summary['dim_ker']['mean']
        lam_min = summary['eigval_min_nz']['mean']
        lam_max = summary['eigval_max']['mean']
        Lk = make_Lk_label(k, comp)
        ann = (
            f'$\\operatorname{{dim}} C_{{{k}}} = {N_k}$\n'
            f'$\\operatorname{{null}} {Lk} = {null_mean:{float_fmt}}$\n'
            f'$\\lambda_{{\\min}} = {lam_min:{float_fmt}}$\n'
            f'$\\lambda_{{\\max}} = {lam_max:{float_fmt}}$'
        )
        ax.text(0.025, 0.95, ann, transform=ax.transAxes,
                fontsize=9, color=f'C{k}', va='top', ha='left')

    if title is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_observable_vs_parameter(
    series: dict,
    obs_name: str,
    *,
    axis_config: Optional[Dict] = None,
    line_config: Optional[Dict] = None,
    fill_alpha: float = 0.2,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 4),
) -> Figure:
    """
    Plot an observable vs experiment parameter, one subplot per degree.

    Parameters
    ----------
    series : dict
        Experiment series dict with keys 'exp_param', 'exp_values',
        'analysis_keys', 'observable_series'.
    obs_name : str
        Observable name to plot (e.g. 'dim_ker', 'eigval_mean_nz').
    axis_config : dict, optional
        Keyword arguments passed to format_axis.
    line_config : dict, optional
        Keyword arguments passed to plot_lines (marker, linewidth, etc.).
    fill_alpha : float, default 0.2
        Alpha for the mean +/- std shading.
    title : str, optional
        Figure suptitle.
    figsize : tuple of float, default (14, 4)
        Figure size.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if axis_config is None:
        axis_config = {}
    if line_config is None:
        line_config = {}

    exp_param = series['exp_param']
    exp_values = series['exp_values']
    analysis_keys = series['analysis_keys']
    obs_label = OBSERVABLE_LABELS.get(obs_name, obs_name)
    n_deg = len(analysis_keys)

    fig, axes = plt.subplots(1, n_deg, figsize=figsize)
    if n_deg == 1:
        axes = [axes]

    for col, (k, comp) in enumerate(analysis_keys):
        ax = axes[col]
        obs = series['observable_series'][(k, comp)][obs_name]
        mean = obs['mean']
        std = obs['std']

        plot_lines([(exp_values, mean)], ax=ax, color=f'C{k}', **line_config)
        ax.fill_between(exp_values, mean - std, mean + std,
                        alpha=fill_alpha, color=f'C{k}')

        format_axis(ax, title=f"${make_Lk_label(k, comp)}$",
                    ylabel=obs_label, xlabel=exp_param, **axis_config)

        # Legend on first subplot only
        if col == 0:
            ax.plot([], [], color=f'C{k}', label='mean', **line_config)
            ax.fill_between([], [], [], alpha=fill_alpha,
                            color=f'C{k}', label=r'mean $\pm$ std')
            ax.legend(fontsize=8, loc='best')

    if title is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_distribution_heatmap(
    series: dict,
    *,
    cmap: str = 'magma',
    log: bool = False,
    axis_config: Optional[Dict] = None,
    cbar_config: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 4),
) -> Figure:
    """
    Plot eigenvalue distribution heatmaps, one subplot per degree.

    Parameters
    ----------
    series : dict
        Experiment series dict with keys 'exp_param', 'exp_values',
        'analysis_keys', 'distribution_series'.
    cmap : str, default 'magma'
        Colormap name.
    log : bool, default False
        If True, apply LogNorm to the density values.
    axis_config : dict, optional
        Keyword arguments passed to format_axis.
    cbar_config : dict, optional
        Keyword arguments passed to format_cbar.
    title : str, optional
        Figure suptitle.
    figsize : tuple of float, default (16, 4)
        Figure size.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if axis_config is None:
        axis_config = {}
    if cbar_config is None:
        cbar_config = {}

    exp_param = series['exp_param']
    exp_values = series['exp_values']
    analysis_keys = series['analysis_keys']
    n_deg = len(analysis_keys)

    fig, axes = plt.subplots(1, n_deg, figsize=figsize)
    if n_deg == 1:
        axes = [axes]

    for col, (k, comp) in enumerate(analysis_keys):
        ax = axes[col]
        ds = series['distribution_series'][(k, comp)]
        x = ds['x']
        density_stack = ds['density_stack']

        # Reconstruct bin edges from centers for pcolormesh
        bin_width = x[1] - x[0] if len(x) > 1 else 1.0
        x_edges = np.concatenate([[x[0] - bin_width / 2], x + bin_width / 2])

        # Parameter edges from midpoints
        if len(exp_values) > 1:
            param_mids = 0.5 * (exp_values[:-1] + exp_values[1:])
            param_step = exp_values[1] - exp_values[0]
        else:
            param_mids = np.array([])
            param_step = 1.0
        param_edges = np.concatenate([
            [exp_values[0] - param_step / 2],
            param_mids,
            [exp_values[-1] + param_step / 2],
        ])

        # Optional log normalization
        norm = None
        if log:
            z_max = np.nanmax(density_stack)
            if z_max > 0:
                z_floor = z_max * 0.01
                density_plot = np.where(density_stack > z_floor, density_stack, z_floor)
                norm = LogNorm(vmin=z_floor, vmax=z_max)
            else:
                density_plot = density_stack
        else:
            density_plot = density_stack

        im = ax.pcolormesh(x_edges, param_edges, density_plot,
                           shading='flat', cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        format_cbar(cbar, **cbar_config)
        format_axis(ax, title=f"${make_Lk_label(k, comp)}$",
                    ylabel=exp_param, **axis_config)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_distribution_lines(
    series: dict,
    *,
    cmap: str = 'viridis',
    linewidth: float = 1.5,
    axis_config: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 4),
) -> Figure:
    """
    Overlay eigenvalue distributions as colored lines, one subplot per degree.

    Parameters
    ----------
    series : dict
        Experiment series dict with keys 'exp_param', 'exp_values',
        'analysis_keys', 'distribution_series'.
    cmap : str, default 'viridis'
        Colormap name for line colors.
    linewidth : float, default 1.5
        Line width.
    axis_config : dict, optional
        Keyword arguments passed to format_axis.
    title : str, optional
        Figure suptitle.
    figsize : tuple of float, default (16, 4)
        Figure size.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if axis_config is None:
        axis_config = {}

    exp_param = series['exp_param']
    exp_values = series['exp_values']
    analysis_keys = series['analysis_keys']
    n_deg = len(analysis_keys)
    n_levels = len(exp_values)
    colormap = plt.get_cmap(cmap)

    fig, axes = plt.subplots(1, n_deg, figsize=figsize)
    if n_deg == 1:
        axes = [axes]

    for col, (k, comp) in enumerate(analysis_keys):
        ax = axes[col]
        ds = series['distribution_series'][(k, comp)]
        x = ds['x']
        density_stack = ds['density_stack']

        for i in range(n_levels):
            c = (i + 1) / (n_levels + 1)
            ax.plot(x, density_stack[i], color=colormap(c),
                    linewidth=linewidth, zorder=i)

        # Colorbar showing parameter values
        sm = plt.cm.ScalarMappable(
            cmap=colormap,
            norm=Normalize(vmin=exp_values[0], vmax=exp_values[-1]),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        format_cbar(cbar, label=exp_param)

        format_axis(ax, title=f"${make_Lk_label(k, comp)}$", **axis_config)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_distribution_distance(
    series: dict,
    *,
    distance_metric: str = 'distance',
    axis_config: Optional[Dict] = None,
    line_config: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 4),
) -> Figure:
    """
    Plot successive distribution distance vs experiment parameter,
    one subplot per degree.

    Parameters
    ----------
    series : dict
        Experiment series dict with keys 'exp_param', 'exp_values',
        'analysis_keys', 'distance_series'.
    distance_metric : str, default 'distance'
        Name of the distance metric (used for y-axis label).
    axis_config : dict, optional
        Keyword arguments passed to format_axis.
    line_config : dict, optional
        Keyword arguments passed to plot_lines.
    title : str, optional
        Figure suptitle.
    figsize : tuple of float, default (14, 4)
        Figure size.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if axis_config is None:
        axis_config = {}
    if line_config is None:
        line_config = {}

    exp_param = series['exp_param']
    exp_values = series['exp_values']
    analysis_keys = series['analysis_keys']
    n_deg = len(analysis_keys)
    midpoints = 0.5 * (exp_values[:-1] + exp_values[1:])

    fig, axes = plt.subplots(1, n_deg, figsize=figsize)
    if n_deg == 1:
        axes = [axes]

    for col, (k, comp) in enumerate(analysis_keys):
        ax = axes[col]
        dists = series['distance_series'][(k, comp)]

        plot_lines([(midpoints, dists)], ax=ax, color=f'C{k}', **line_config)

        format_axis(ax, title=f"${make_Lk_label(k, comp)}$",
                    xlabel=exp_param, ylabel=f'{distance_metric} distance',
                    **axis_config)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig

