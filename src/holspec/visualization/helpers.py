# holspec/visualization/helpers.py
"""
Helper functions for visualizations.
"""

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
import os
import shutil
import subprocess
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path

# %% Formatting

def format_axis(
    ax: Axes,
    # Figure properties
    figsize: Optional[Tuple[float, float]] = None,
    # Axis limits and scales
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    zlims: Optional[Tuple[float, float]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    zscale: Optional[str] = None,
    aspect: Optional[str] = None,
    margins: Optional[Tuple[float, ...]] = None,
    # View angles (3D only)
    view_angles: Optional[Tuple[float, float]] = None,
    # Labels and title
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    title: Optional[str] = None,
    title_fontsize: int = 11,
    # Grid
    grid: bool = False,
    grid_alpha: float = 0.5,
    # Scientific notation
    scilimits: Optional[Tuple[int, int]] = None,
    sci_axis: str = 'both',
    # Legend
    legend_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Apply formatting to a matplotlib axis (2D or 3D).
    
    Parameters
    ----------
    ax : Axes
        The matplotlib axis to format (2D or 3D).
    figsize : tuple of float, optional
        Figure size as (width, height) in inches.
    xlims : tuple of float, optional
        x-axis limits as (min, max).
    ylims : tuple of float, optional
        y-axis limits as (min, max).
    zlims : tuple of float, optional
        z-axis limits as (min, max). Only for 3D axes.
    xscale : str, optional
        Scale for x-axis (e.g., 'linear', 'log').
    yscale : str, optional
        Scale for y-axis (e.g., 'linear', 'log').
    zscale : str, optional
        Scale for z-axis (e.g., 'linear', 'log'). Only for 3D axes.
    aspect : str, optional
        Aspect ratio (e.g., 'equal', 'auto').
        For 3D: 'equal' is implemented manually due to matplotlib limitations.
    margins : tuple of float, optional
        Margins as (x_margin, y_margin) for 2D or (x_margin, y_margin, z_margin) for 3D.
    view_angles : tuple of float, optional
        View angles as (elevation, azimuth) in degrees. Only for 3D axes.
        Elevation is the angle above the horizontal plane (typically 0-90).
        Azimuth is the rotation angle around the z-axis (0-360).
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    zlabel : str, optional
        Label for z-axis. Only for 3D axes.
    title : str, optional
        Title for the plot.
    title_fontsize : int, default 11
        Font size for the title.
    grid : bool, default False
        Whether to display grid lines.
    grid_alpha : float, default 0.5
        Transparency of grid lines (0=transparent, 1=opaque).
    scilimits : tuple of int, optional
        Power limits for scientific notation as (min_exp, max_exp).
    sci_axis : str, default 'both'
        Which axis to apply scientific notation ('x', 'y', or 'both').
    legend_kwargs : dict, optional
        Keyword arguments to pass to ax.legend().
    
    Notes
    -----
    The function automatically detects whether the axis is 2D or 3D.
    For 3D axes with aspect='equal', all axes are set to the same range, centered 
    on the data, providing an approximate equal aspect ratio.
    """
    # Detect if 3D axis
    is_3d = hasattr(ax, 'zaxis')
    
    # Figure size
    if figsize is not None:
        fig: Figure = ax.get_figure()
        fig.set_size_inches(figsize)
    
    # Handle aspect ratio (3D equal aspect needs special handling)
    if aspect == 'equal' and is_3d:
        xlims, ylims, zlims = _set_equal_aspect_3d(ax, xlims, ylims, zlims)
    elif aspect is not None and not is_3d:
        ax.set_aspect(aspect)
    
    # Axis limits
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if zlims is not None and is_3d:
        ax.set_zlim(zlims)
    
    # Axis scales
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if zscale is not None and is_3d:
        ax.set_zscale(zscale)
    
    # Margins
    if margins is not None:
        if is_3d:
            # 3D doesn't support margins() method, ignore for now
            pass
        else:
            ax.margins(*margins)
    
    # View angles (3D only)
    if view_angles is not None and is_3d:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None and is_3d:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    
    # Grid
    if grid:
        ax.grid(alpha=grid_alpha)
    
    # Scientific notation
    if scilimits is not None:
        if is_3d:
            ax.ticklabel_format(style='sci', scilimits=scilimits)
        else:
            ax.ticklabel_format(style='sci', axis=sci_axis, scilimits=scilimits)
    
    # Legend
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)


def _set_equal_aspect_3d(
    ax: Axes,
    xlims: Optional[Tuple[float, float]],
    ylims: Optional[Tuple[float, float]],
    zlims: Optional[Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate equal aspect ratio limits for 3D axes.
    
    Parameters
    ----------
    ax : Axes
        3D matplotlib axis.
    xlims : tuple of float, optional
        Desired x-axis limits, or None to use current limits.
    ylims : tuple of float, optional
        Desired y-axis limits, or None to use current limits.
    zlims : tuple of float, optional
        Desired z-axis limits, or None to use current limits.
    
    Returns
    -------
    tuple
        (xlims, ylims, zlims) with equal ranges centered on data.
    """
    # Get current axis limits if not provided
    xlims_current = ax.get_xlim() if xlims is None else xlims
    ylims_current = ax.get_ylim() if ylims is None else ylims
    zlims_current = ax.get_zlim() if zlims is None else zlims
    
    # Calculate ranges
    x_range = xlims_current[1] - xlims_current[0]
    y_range = ylims_current[1] - ylims_current[0]
    z_range = zlims_current[1] - zlims_current[0]
    max_range = max(x_range, y_range, z_range)
    
    # Calculate centers
    x_center = (xlims_current[1] + xlims_current[0]) / 2
    y_center = (ylims_current[1] + ylims_current[0]) / 2
    z_center = (zlims_current[1] + zlims_current[0]) / 2
    
    # Set equal limits
    xlims = (x_center - max_range / 2, x_center + max_range / 2)
    ylims = (y_center - max_range / 2, y_center + max_range / 2)
    zlims = (z_center - max_range / 2, z_center + max_range / 2)
    
    return xlims, ylims, zlims


def format_cbar(
    cbar: Colorbar,
    # Label
    label: Optional[str] = None,
    label_fontsize: int = 10,
    # Ticks
    ticks: Optional[Union[np.ndarray, List[float]]] = None,
    ticklabels: Optional[List[str]] = None,
    tick_fontsize: Optional[int] = None,
    # Scientific notation
    scilimits: Optional[Tuple[int, int]] = None,
    # Orientation
    orientation: Optional[str] = None
) -> None:
    """
    Apply formatting to a matplotlib colorbar.
    
    Parameters
    ----------
    cbar : Colorbar
        The matplotlib colorbar to format.
    label : str, optional
        Label for the colorbar.
    label_fontsize : int, default 10
        Font size for the colorbar label.
    ticks : array-like, optional
        Tick positions for the colorbar.
    ticklabels : list of str, optional
        Labels for the colorbar ticks.
    tick_fontsize : int, optional
        Font size for tick labels.
    scilimits : tuple of int, optional
        Power limits for scientific notation as (min_exp, max_exp).
    orientation : str, optional
        Orientation of the colorbar ('vertical' or 'horizontal').
        If 'horizontal', tick labels will be rotated 45 degrees.
    """
    # Label
    if label is not None:
        cbar.set_label(label, fontsize=label_fontsize)
    
    # Scientific notation
    if scilimits is not None:
        cbar.formatter.set_powerlimits(scilimits)
        cbar.update_ticks()
    
    # Ticks
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    
    # Tick label font size
    if tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # Orientation-specific formatting
    if orientation == 'horizontal':
        cbar.ax.tick_params(rotation=45)


# %% Animation functions

def animate_3d_turn(
    fig: Figure,
    ax: Axes,
    output_path: Union[str, Path],
    # View angles
    initial_view_angles: Tuple[float, float] = (30, 45),
    # Animation parameters
    n_frames: int = 180,
    frame_interval: int = 50,
    # Output configuration
    writer: str = "ffmpeg",
    fps: int = 12,
    dpi: int = 200,
    transparent: bool = True
) -> None:
    """
    Create a 3D rotation animation of a matplotlib 3D plot.
    
    Parameters
    ----------
    fig : Figure
        The matplotlib figure containing the 3D plot.
    ax : Axes
        The 3D axes to animate (must be created with projection='3d').
    output_path : str or Path
        Path where the animation will be saved (without extension, '.mp4' will be added).
    initial_view_angles : tuple of float, default (30, 45)
        Initial view angles as (elevation, azimuth) in degrees.
    n_frames : int, default 180
        Number of frames in the animation.
    frame_interval : int, default 50
        Delay between frames in milliseconds.
    writer : str, default "ffmpeg"
        Animation writer to use.
    fps : int, default 12
        Frames per second for the output video.
    dpi : int, default 200
        Dots per inch for the output video.
    transparent : bool, default True
        Whether to save with transparent background.
    """
    def update_frame(azim: float) -> Tuple[Axes]:
        """Update function for animation frame."""
        ax.view_init(elev=initial_view_angles[0], azim=azim)
        return (ax,)
    
    # Prepare figure layout
    fig.tight_layout()
    
    # Create animation
    azim_angles = np.linspace(0, 360, n_frames) + initial_view_angles[1]
    ani = FuncAnimation(
        fig,
        update_frame,
        frames=azim_angles,
        interval=frame_interval,
        blit=False
    )
    
    # Save animation
    output_config = {
        'writer': writer,
        'fps': fps,
        'dpi': dpi,
        'savefig_kwargs': {'transparent': transparent}
    }
    ani.save(f"{output_path}.mp4", **output_config)
    ani.event_source.stop()


def make_movie(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path] = "output.mp4",
    fps: int = 30
) -> bool:
    """
    Create an MP4 movie from a sequence of images using ffmpeg.
    
    Parameters
    ----------
    image_paths : list of str or Path
        List of paths to image files (png, jpeg, etc.).
    output_path : str or Path, default "output.mp4"
        Path for the output MP4 file.
    fps : int, default 30
        Frames per second for the output video.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    
    Notes
    -----
    This function requires ffmpeg to be installed and available in the system PATH.
    A temporary directory 'temp_frames' is created and cleaned up automatically.
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
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
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