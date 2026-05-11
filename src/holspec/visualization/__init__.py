# holspec/visualization/__init__.py
"""
Visualization for holspec objects and analysis outputs.

This subpackage contains plotting helpers for point data, simplicial complexes,
cochain fields, matrices, spectra, eigenvectors, and experiment-series results.
"""
from .helpers import (
    animate_3d_turn,
    combine_images,
    format_axis,
    format_cbar,
    make_fig_path,
    make_movie,
)
from .basic_plots import (
    add_simplex_labels_2d,
    add_simplex_labels_3d,
    add_simplex_orientation_2d,
    add_simplex_orientation_3d,
    plot_bars,
    plot_circles,
    plot_edges_2d,
    plot_edges_3d,
    plot_lines,
    plot_matrix,
    plot_scatters,
    plot_spheres,
    plot_stems,
    plot_tetrahedra_3d,
    plot_triangles_2d,
    plot_triangles_3d,
    plot_vertices_2d,
    plot_vertices_3d,
    values_to_colors,
)
from .framework_objects import (
    make_Lk_label,
    make_cochain_kwargs,
    plot_cochain,
    plot_cochain_grid,
    plot_distribution_distance,
    plot_distribution_heatmap,
    plot_distribution_lines,
    plot_eigval_distribution,
    plot_observable_vs_parameter,
    plot_point_data,
    plot_simplicial_complex_2d,
    plot_simplicial_complex_3d,
)


# Define public API of the module
__all__ = [
    # From helpers.py
    'format_axis',
    'format_cbar',
    'animate_3d_turn',
    'make_movie',
    'make_fig_path',
    'combine_images',

    # From basic_plots.py
    'plot_lines',
    'plot_scatters',
    'plot_bars',
    'plot_stems',
    'plot_matrix',
    'plot_circles',
    'plot_spheres',
    'plot_vertices_2d',
    'plot_edges_2d',
    'plot_triangles_2d',
    'add_simplex_orientation_2d',
    'add_simplex_labels_2d',
    'plot_vertices_3d',
    'plot_edges_3d',
    'plot_triangles_3d',
    'plot_tetrahedra_3d',
    'add_simplex_orientation_3d',
    'add_simplex_labels_3d',
    'values_to_colors',

    # From framework_objects.py
    'make_Lk_label',
    'plot_eigval_distribution',
    'plot_observable_vs_parameter',
    'plot_distribution_heatmap',
    'plot_distribution_lines',
    'plot_distribution_distance',
    'plot_point_data',
    'plot_simplicial_complex_2d',
    'plot_simplicial_complex_3d',
    'plot_cochain',
    'plot_cochain_grid',
    'make_cochain_kwargs',
]
