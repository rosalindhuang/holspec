"""
Functions for plotting lines, scatter plots, and bar plots using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib as mpl
import os
import shutil
import subprocess

# %% Helper functions

def extend_list(lst, target_length, default_value):
    """Extend list to target length with default value."""
    if len(lst) < target_length:
        return lst + [default_value] * (target_length - len(lst))
    return lst

def format_axis(ax, figsize=None, grid=False, margins=None, aspect=None,
                xlims=None, ylims=None, xscale=None, yscale=None,
                xlabel=None, ylabel=None, title=None, scilimits=None,
                legend_kwargs=None,
                **kwargs):
    """
    Apply formatting to a single axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    grid : bool, default False
    xlims, ylims : tuple, optional
    xscale, yscale : str, optional
    xlabel, ylabel : str, optional
    title : str, optional
    scilimits : tuple, optional
    margins : tuple, optional
    legend_kwargs : dict, optional
    **kwargs : grid_alpha (float, default 0.5), 
               title_fontsize (int, default 11),
               sci_axis (str, default 'both')
    """
    # Figure size
    if figsize is not None:
        fig = ax.get_figure()
        fig.set_size_inches(figsize)
    
    # Grid
    if grid:
        grid_alpha = kwargs.get('grid_alpha', 0.5)
        ax.grid(alpha=grid_alpha)
    
    # Limits
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    
    # Aspect ratio
    if aspect is not None:
        ax.set_aspect(aspect)
        
    # Scales
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    
    # Scientific notation
    if scilimits is not None:
        sci_axis = kwargs.get('sci_axis', 'both')
        ax.ticklabel_format(style='sci', axis=sci_axis, scilimits=scilimits)
    
    # Margins
    if margins is not None:
        ax.margins(*margins)
    
    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # Title
    if title is not None:
        title_fontsize = kwargs.get('title_fontsize', 11)
        ax.set_title(title, fontsize=title_fontsize)
    
    # Legend
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)

def format_cbar(cbar, 
                label=None, scilimits=None,
                ticks=None, ticklabels=None,
                orientation=None,
                **kwargs):
    """
    Apply formatting to a colorbar.
    
    Parameters:
    -----------
    cbar : matplotlib.colorbar.Colorbar
    label : str, optional
    scilimits : tuple, optional
    ticks : array-like, optional
    ticklabels : list, optional
    orientation : str, optional ('vertical' or 'horizontal')
    **kwargs : label_fontsize (int, default 10),
               tick_fontsize (int, optional)
    """
    # Label
    if label is not None:
        label_fontsize = kwargs.get('label_fontsize', 10)
        cbar.set_label(label, fontsize=label_fontsize)
    
    # Scientific notation
    if scilimits is not None:
        cbar.formatter.set_powerlimits(scilimits)
        cbar.update_ticks()
    
    # Ticks
    if ticks is not None:
        cbar.set_ticks(ticks)
    
    # Tick labels and font size
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    
    # Tick label font size
    if 'tick_fontsize' in kwargs:
        cbar.ax.tick_params(labelsize=kwargs['tick_fontsize'])
    
    # Orientation (tick label rotation)
    if orientation == 'horizontal':
        cbar.ax.tick_params(rotation=45)

def save_figure(fig, fig_path, transparent=True, **kwargs):
    """
    Save figure with standard settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    fig_path : str or None
        Path to save figure, if None does nothing
    transparent : bool, default True
        Whether to save with transparent background
    **kwargs : additional arguments passed to savefig
    """
    if fig_path is None:
        return
    
    # Set default savefig parameters
    save_kwargs = {
        'dpi': 200,
        'transparent': transparent,
        'bbox_inches': 'tight'
    }
    save_kwargs.update(kwargs)
    
    fig.savefig(fig_path, **save_kwargs)


# %% Line plots
def plot_lines(xy_list, figsize=(5, 4), 
               color=None, colors_list=None, legend_labels=None, **kwargs):
    """
    Plot multiple lines on a single axis.
    
    Parameters:
    -----------
    xy_list : list of (x, y) tuples
        Data to plot
    figsize : tuple, default (5, 4)
        Figure size
    color : str, optional
        Single color for all lines (overrides colors_list)
    colors_list : list, optional
        List of colors for each line
    legend_labels : list, optional
        List of legend labels for each line
    **kwargs : additional arguments passed to plot()
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    
    # Resolve colors and labels
    if color is not None:
        colors_list = [color] * len(xy_list)
    else:
        if colors_list is None:
            colors_list = []
        colors_list = extend_list(colors_list, len(xy_list), 'k')

    if legend_labels is not None:
        legend_labels = extend_list(legend_labels, len(xy_list), None)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (x, y) in enumerate(xy_list):
        label = legend_labels[i] if legend_labels is not None else None
        ax.plot(x, y, color=colors_list[i], label=label, **kwargs)
    
    # Add legend if labels were provided
    if legend_labels is not None:
        ax.legend()
    
    return fig, ax

def plot_lines_stack(xy_list, figsize=(6, 6), stack_direction='vertical', sharex=False, sharey=False,
                     color=None, colors_list=None, legend_labels=None, **kwargs):
    """
    Plot multiple lines on stacked subplots.
    
    Parameters:
    -----------
    xy_list : list of (x, y) tuples
        Data to plot, one line per subplot
    figsize : tuple, default (6, 6)
        Figure size
    stack_direction : str, default 'vertical'
        Direction to stack subplots ('vertical' or 'horizontal')
    sharex, sharey : bool, default False
        Whether to share x/y axes between subplots
    color : str, optional
        Single color for all lines (overrides colors_list)
    colors_list : list, optional
        List of colors for each line/subplot
    legend_labels : list, optional
        List of legend labels for each line/subplot
    **kwargs : additional arguments passed to plot()
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects (axes is array for multiple subplots)
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    
    # Resolve colors and legend
    if color is not None:
        colors_list = [color] * len(xy_list)
    else:
        if colors_list is None:
            colors_list = []
        colors_list = extend_list(colors_list, len(xy_list), 'k')
    if legend_labels is not None:
        legend_labels = extend_list(legend_labels, len(xy_list), None)
    
    # Create subplots
    if stack_direction == 'vertical':
        fig, axes = plt.subplots(len(xy_list), 1, figsize=figsize, sharex=sharex, sharey=sharey)
    elif stack_direction == 'horizontal':
        fig, axes = plt.subplots(1, len(xy_list), figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        raise ValueError("stack_direction must be 'vertical' or 'horizontal'")
    
    axes = np.atleast_1d(axes)
    
    # Plot lines
    for i, (x, y) in enumerate(xy_list):
        ax = axes[i]
        label = legend_labels[i] if legend_labels is not None else None
        ax.plot(x, y, color=colors_list[i], label=label, **kwargs)
        
        # Add legend
        if label is not None:
            ax.legend()
    
    return fig, axes


# %% Scatter plots

def plot_scatters(xy_list, ax=None, figsize=(5, 4), 
                  color=None, colors_list=None, c=None, c_list=None, 
                  cmap=None, legend_labels=None, colorbar=False, **kwargs):
    """
    Plot multiple scatter plots on a single axis.
    
    Parameters:
    -----------
    xy_list : list of (x, y) tuples
        Data to plot
    figsize : tuple, default (5, 4)
        Figure size
    color : str, optional
        Single color for all scatter plots (overrides other color options)
    colors_list : list, optional
        List of colors for each scatter plot (overrides colormap options)
    c : array-like, optional
        Color values for colormap (applied to all scatter plots, overrides c_list)
    c_list : list of array-like, optional
        List of color values for each scatter plot (for colormap)
    cmap : str or Colormap, optional
        Colormap to use with c/c_list
    legend_labels : list, optional
        List of legend labels for each scatter plot
    colorbar : bool, default False
        Whether to add a colorbar (only with colormap)
    **kwargs : additional arguments passed to scatter()
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if len(xy_list) == 0:
        raise ValueError("xy_list must contain at least one (x, y) pair.")
    
    n_plots = len(xy_list)
    
    # Determine color strategy (priority: color > colors_list > colormap)
    if color is not None:
        # Single color for all
        plot_colors = [color] * n_plots
        plot_c_values = [None] * n_plots
        use_colormap = False
    elif colors_list is not None:
        # Different colors for each scatter
        plot_colors = extend_list(colors_list, n_plots, 'k')
        plot_c_values = [None] * n_plots
        use_colormap = False
    elif c is not None:
        # Use colormap with single c values for all
        plot_colors = [None] * n_plots
        plot_c_values = [c] * n_plots
        use_colormap = True
    elif c_list is not None:
        # Use colormap with different c values for each scatter
        plot_colors = [None] * n_plots
        plot_c_values = extend_list(c_list, n_plots, None)
        use_colormap = True
    else:
        # Default colors
        plot_colors = ['k'] * n_plots
        plot_c_values = [None] * n_plots
        use_colormap = False
    
    # Extend legend labels
    if legend_labels is not None:
        legend_labels = extend_list(legend_labels, n_plots, None)
    
    # Create figure and plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    scatter_plots = []
    
    for i, (x, y) in enumerate(xy_list):
        label = legend_labels[i] if legend_labels is not None else None
        
        if use_colormap and plot_c_values[i] is not None:
            sc = ax.scatter(x, y, c=plot_c_values[i], cmap=cmap, label=label, **kwargs)
        else:
            sc = ax.scatter(x, y, color=plot_colors[i], label=label, **kwargs)
        
        scatter_plots.append(sc)
    
    # Add colorbar and legend
    cbar = None
    if colorbar and use_colormap:
        cbar = fig.colorbar(scatter_plots[-1], ax=ax)
    
    if legend_labels is not None:
        ax.legend()
    
    if cbar is None:
        return fig, ax
    
    return fig, ax, cbar

# %% Animation functions

def animate_turn(fig, ax, fig_path, 
                 initial_view_angles = (30,45), n_frames = 180, frame_interval = 50, 
                 output_config = None):
    if fig_path is None: return

    default_output_config = {'writer': "ffmpeg", 'fps': 12, 'dpi': 200, 'savefig_kwargs': {'transparent': True}}
    if output_config is not None:
        output_config = default_output_config | output_config
    else:
        output_config = default_output_config

    def update_frame(frame):
        ax.view_init(elev = initial_view_angles[0], azim = frame)
        return ax,
    fig.tight_layout()
    ani = FuncAnimation(fig, update_frame, frames=np.linspace(0, 360, n_frames)+initial_view_angles[1], interval = frame_interval)
    ani.save(str(fig_path) + '.mp4', **output_config)
    ani.event_source.stop()

def make_movie(image_paths, output_path = "output.mp4", fps = 30):
    """
    Create an MP4 movie from a sequence of images using ffmpeg.
    
    Parameters:
    ----------
    image_paths: List of paths to image files (png, jpeg, etc.)
    output_path: Path for the output MP4 file (default: "output.mp4")
    fps: Frames per second for the output video (default: 30)
    
    Returns:
    -------
    bool: True if successful, False otherwise
    """
    if not image_paths:
        print("Error: No image paths provided")
        return False
    
    # Create temporary directory with sequential copies
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Copy images with sequential names
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"Error: Image file not found: {img_path}")
                return False
            shutil.copy2(img_path, f"{temp_dir}/frame_{i:05d}.png")
        
        # Run ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{temp_dir}/frame_%05d.png",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # print(f"Movie created successfully: {output_path}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Always clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)