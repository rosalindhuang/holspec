"""
Point data generation functions.

This module provides functions to generate point data for lattices, random 
distributions, and simple shapes in 2D and 3D.
"""

import numpy as np


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

def add_random_perturbation(
    positions: np.ndarray,
    amplitude: float,
    mode: str = 'uniform',
    seed: int | None = None
) -> np.ndarray:
    """
    Add random displacement to point positions.
    
    Parameters
    ----------
    positions : (N, d) ndarray
        Original particle positions.
    amplitude : float
        Displacement scale.
    mode : {'uniform', 'normal'}, default='uniform'
        Distribution type:
        - 'uniform': displacement in [-amplitude/2, amplitude/2]
        - 'normal': Gaussian with std=amplitude
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    positions_perturbed : (N, d) ndarray
        Perturbed particle positions.
    """
    rng = np.random.default_rng(seed)
    
    if mode == 'uniform':
        displacement = amplitude * (rng.random(positions.shape) - 0.5)
    elif mode == 'normal':
        displacement = amplitude * rng.standard_normal(positions.shape)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return positions + displacement