"""
Point data generation functions.

This module provides functions to generate point data for lattices, random 
distributions, and simple shapes in 2D and 3D.
"""

import numpy as np

from holspec.utilities import format_float_str

# %% 2D Lattice Generators

def generate_triangular_lattice_hex(
    n_rings: int,
    spacing: float = 1.0
) -> np.ndarray:
    """
    Generate triangular lattice in hexagonal domain. The lattice has 
    `n_rings` concentric rings around a central point.
    
    Parameters
    ----------
    n_rings : int
        Number of rings around center. n_rings=0 gives 1 point, n_rings=1 
        gives 7 points. Total points: N = 1 + 3*n_rings*(n_rings+1)
    spacing : float, default=1.0
        Nearest-neighbor distance.
    
    Returns
    -------
    positions : (N, 2) ndarray
        Particle positions, centered at origin.
    
    """
    # Lattice basis vectors
    a1 = spacing * np.array([1.0, 0.0])
    a2 = spacing * np.array([0.5, np.sqrt(3) / 2])
    
    positions = []
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i + j) <= n_rings:
                pos = i * a1 + j * a2
                positions.append(pos)
    
    return np.array(positions)


def generate_triangular_lattice_rect(
    nx: int,
    ny: int,
    spacing: float = 1.0
) -> np.ndarray:
    """
    Generate triangular lattice in rectangular domain.

    Parameters
    ----------
    nx : int
        Number of particles along x-direction.
    ny : int
        Number of particles along y-direction (number of rows).
    spacing : float, default=1.0
        Nearest-neighbor distance.
    
    Returns
    -------
    positions : (nx*ny, 2) ndarray
        Particle positions, centered at origin.
    
    """
    a1 = spacing * np.array([1.0, 0.0])
    a2 = spacing * np.array([0.5, np.sqrt(3) / 2])
    
    positions = []
    for i in range(nx):
        for j in range(ny):
            pos = i * a1 + j * a2
            positions.append(pos)
    
    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    
    return positions - center


def generate_square_lattice(
    nx: int,
    ny: int,
    spacing: float = 1.0
) -> np.ndarray:
    """
    Generate square lattice in 2D.
    
    Parameters
    ----------
    nx : int
        Number of particles along x-direction.
    ny : int
        Number of particles along y-direction.
    spacing : float, default=1.0
        Lattice spacing.
    
    Returns
    -------
    positions : (nx*ny, 2) ndarray
        Particle positions, centered at origin.
    
    """
    positions = []
    for i in range(nx):
        for j in range(ny):
            positions.append([i * spacing, j * spacing])
    
    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    
    return positions - center


# %% 3D Lattice Generators

def generate_bcc_lattice(
    nx: int,
    ny: int,
    nz: int,
    spacing: float = 1.0
) -> np.ndarray:
    """
    Generate body-centered cubic (BCC) lattice.
    
    Parameters
    ----------
    nx : int
        Number of unit cells along x-direction.
    ny : int
        Number of unit cells along y-direction.
    nz : int
        Number of unit cells along z-direction.
    spacing : float, default=1.0
        Conventional cubic cell lattice constant.
    
    Returns
    -------
    positions : (N, 3) ndarray
        Particle positions, centered at origin.
        N = nx*ny*nz + (nx-1)*(ny-1)*(nz-1)
    
    Notes
    -----
    The BCC lattice has 2 atoms per conventional cubic unit cell.
    The nearest-neighbor distance is spacing * sqrt(3)/2.

    """
    positions = []
    
    # Corner atoms
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions.append([i * spacing, j * spacing, k * spacing])
    
    # Body-center atoms
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                positions.append([
                    (i + 0.5) * spacing,
                    (j + 0.5) * spacing,
                    (k + 0.5) * spacing
                ])
    
    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    
    return positions - center


# %% Random Point Generators

def generate_random_uniform(
    n_points: int,
    dimension: int = 2,
    box_size: float = 1.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate uniformly distributed random points in a hyperrectangle 
    [0, box_size]^d, centered at the origin.
    
    Parameters
    ----------
    n_points : int
        Number of random points.
    dimension : int, default=2
        Dimension of the space.
    box_size : float, default=1.0
        Side length of the hyperrectangle.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    positions : (n_points, dimension) ndarray
        Random point positions, centered at origin.
    
    """
    rng = np.random.default_rng(seed)
    positions = box_size * rng.random((n_points, dimension))
    center = np.mean(positions, axis=0)
    
    return positions - center



# %% Simple Shape Generators

def generate_regular_polygon(
    n_sides: int,
    radius: float = 1.0
) -> np.ndarray:
    """
    Generate vertices of a regular n-sided polygon inscribed in a circle
    of given radius, centered at the origin.
    
    Parameters
    ----------
    n_sides : int
        Number of sides (must be >= 3).
    radius : float, default=1.0
        Circumradius (distance from center to vertex).
    
    Returns
    -------
    positions : (n_sides, 2) ndarray
        Vertex positions, centered at origin.
    """
    if n_sides < 3:
        raise ValueError("n_sides must be at least 3")
    
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    return np.column_stack((x, y))


def generate_tetrahedron(
    side_length: float = 1.0
) -> np.ndarray:
    """
    Generate vertices of a regular tetrahedron with specified
    side length, centered at the origin.
    
    Parameters
    ----------
    side_length : float, default=1.0
        Edge length of the tetrahedron.
    
    Returns
    -------
    positions : (4, 3) ndarray
        Vertex positions, centered at origin.
    
    """
    # Construct tetrahedron with one face in xy-plane
    a = side_length
    height = np.sqrt(2/3) * a
    
    # Base triangle (equilateral)
    R = a / np.sqrt(3)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    z = np.full(3, -height / 4)
    
    base = np.column_stack((x, y, z))
    apex = np.array([[0, 0, 3 * height / 4]])
    
    return np.vstack((base, apex))


def generate_cube_vertices(
    side_length: float = 1.0
) -> np.ndarray:
    """
    Generate vertices of a cube with specified side length,
    centered at the origin.
    
    Parameters
    ----------
    side_length : float, default=1.0
        Edge length of the cube.
    
    Returns
    -------
    positions : (8, 3) ndarray
        Vertex positions, centered at origin.
    
    """
    a = side_length / 2
    vertices = np.array([
        [-a, -a, -a],
        [ a, -a, -a],
        [-a,  a, -a],
        [ a,  a, -a],
        [-a, -a,  a],
        [ a, -a,  a],
        [-a,  a,  a],
        [ a,  a,  a]
    ])
    
    return vertices



# %% Utilities

GENERATOR_MAP = {
    'trilatthex': generate_triangular_lattice_hex,
    'trilattrect': generate_triangular_lattice_rect,
    'sqrlatt': generate_square_lattice,
    'bcclatt': generate_bcc_lattice,
    'randunif': generate_random_uniform,
    'regpoly': generate_regular_polygon,
    'tetrahedron': generate_tetrahedron,
    'cubevert': generate_cube_vertices,
}


def generate_from_config(config: dict) -> np.ndarray:
    """
    Generate point data from configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain 'generator' (str) and 'params' (dict).

    Returns
    -------
    positions : np.ndarray
        Generated point positions.
    """
    # Validate config
    if 'generator' not in config:
        raise ValueError("Config must contain 'generator' key")
    if 'params' not in config:
        raise ValueError("Config must contain 'params' key")

    generator_name = config['generator']
    params = config['params']

    # Get generator function
    if generator_name not in GENERATOR_MAP:
        raise ValueError(f"Unknown generator: {generator_name}")

    generator_func = GENERATOR_MAP[generator_name]

    # Generate positions
    return generator_func(**params)


def create_config_label(config: dict, dimension: int | None = None, float_fmt: str | None = 'g') -> str:
    """
    Create a unique label from point data generation config.
    
    Parameters
    ----------
    config : dict
        Configuration with 'generator' and 'params' keys.
    dimension : int, optional
        Spatial dimension. If provided, prepends '{d}D_' to label.
    float_fmt : str or None, optional
        Format specifier for floats (e.g., 'g', '.2e', '.0e', '.3f'). Default is 'g'.
        If None, defaults to 'g'.
    
    Returns
    -------
    label : str
        Descriptive label. Format: [dimension_]generator_param1_param2...
    
    Examples
    --------
    >>> config = {'generator': 'trilatthex', 'params': {'n_rings': 4, 'spacing': 1.0}}
    >>> create_config_label(config)
    'trilatthex_nr4_sp1'
    >>> create_config_label(config, dimension=2)
    '2D_trilatthex_nr4_sp1'
    >>> create_config_label(config, dimension=2, float_fmt='.2f')
    '2D_trilatthex_nr4_sp1p00'
    """
    # Validate config
    if 'generator' not in config:
        raise ValueError("Config must contain 'generator' key")
    if 'params' not in config:
        raise ValueError("Config must contain 'params' key")

    generator_name = config['generator']
    params = config['params']
    
    # Start with generator name
    parts = [generator_name]
    
    # Iterate through parameters in original order
    for key, value in params.items():
        # Skip 'dimension' if present in params
        if dimension is not None and key == 'dimension':
            continue
        
        # Get abbreviated parameter name (first 2 letters, strip underscores)
        param_abbr = key.replace('_', '')[:2]
        
        # Format value based on type
        if isinstance(value, bool):
            value_str = '1' if value else '0'
        elif isinstance(value, float):
            value_str = format_float_str(value, float_fmt)
        elif isinstance(value, str):
            value_str = value[:4].lower().replace('_', '')
        else:
            # Handle int and other types
            value_str = str(value).replace('.', 'p')
        
        parts.append(f"{param_abbr}{value_str}")
    
    base_label = '_'.join(parts)
    
    # Prepend dimension if provided
    if dimension is not None:
        return f"{dimension}d_{base_label}"
    
    return base_label


